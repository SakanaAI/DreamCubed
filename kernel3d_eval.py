"""
Voxel-native KID and MMD for generated Minecraft chunks.

This script is the low-sample companion to ``fid3d_eval.py``.  It reuses the
same frozen 3D-CNN and cached per-biome reference features, but replaces the
Gaussian assumption of FID with kernel two-sample distances:

* ``kid3d`` is the unbiased polynomial-kernel MMD estimator conventionally
  called Kernel Inception Distance (KID).  The canonical kernel is
  ``(x @ y / feature_dim + 1) ** 3``.
* ``mmd3d`` is the same unbiased estimator with an RBF kernel.  Its bandwidth
  is fixed per biome from real features only using the median pairwise-distance
  heuristic, so evaluating a new generator cannot change the metric.

Both metrics are distances: lower is better, zero is equality in population,
and an unbiased finite-sample estimate may be slightly negative.

Protocol decisions that matter
------------------------------
The generated side has only 100 chunks per biome.  By default each estimate
therefore compares all 100 generated chunks with a matched subset of 100 real
chunks.  Fifty deterministic real subsets are averaged, and the same subsets
are reused for every model in one invocation.  This is computationally cheap,
keeps model comparisons paired on the reference side, and avoids fitting any
model-dependent preprocessing.  It is not a full bootstrap: when exactly 100
generated chunks are available, the generated side is fixed across draws.

Features are left raw by default, matching conventional KID.  ``--feature_norm
l2`` is available as a sensitivity analysis but changes both metrics and must
not be mixed with raw-feature results.  RBF bandwidths and every protocol
choice are written to the JSON and NPZ artifacts.

One command can evaluate several sample directories.  Each directory receives:

* ``per_biome_kid3d.{json,npz}``
* ``per_biome_mmd3d.{json,npz}``

Example:

    python kernel3d_eval.py \
      --samples_dir renders/per_biome_gen/model_a renders/per_biome_gen/model_b \
      --checkpoint results/fid_defaults/exp_x/feature_extractor_best.pt \
      --ref_dataset data/balanced_dataset/32_biome_parts_dir_balanced_cc_val_dir
"""

import argparse
import glob
import hashlib
import json
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from fid3d_eval import (
    NOISE_KINDS,
    build_index_lut,
    group_samples_by_biome,
    load_extractor_and_converter,
    normalize_biome,
    reference_cache_path,
    reference_features_by_biome,
    structure_free_features,
    to_index_volume,
)
from fid3d_model import extract_features


KID3D_NAME = 'kid3d'
MMD3D_NAME = 'mmd3d'
DEFAULT_SUBSET_SIZE = 100
DEFAULT_NUM_DRAWS = 50
DEFAULT_BANDWIDTH_SAMPLE_SIZE = 1000
DEFAULT_KID_DEGREE = 3
DEFAULT_KID_COEF0 = 1.0


def _stable_seed(seed: int, *parts) -> int:
    text = '|'.join([str(int(seed)), *(str(part) for part in parts)])
    digest = hashlib.sha256(text.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'little', signed=False)


def _sample_rows(total: int, n: int, seed: int, *parts) -> np.ndarray:
    if int(n) > int(total):
        raise ValueError(f'cannot draw {n} rows without replacement from {total}')
    rng = np.random.default_rng(_stable_seed(seed, *parts))
    return rng.choice(int(total), size=int(n), replace=False).astype(np.int64)


def _transform_features(features: np.ndarray, norm: str) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError(f'expected at least two feature rows, got {values.shape}')
    if norm == 'none':
        return values
    if norm != 'l2':
        raise ValueError(f'unknown feature norm: {norm}')
    lengths = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(lengths, np.finfo(np.float64).eps)


def _kernel_matrices(
    a: np.ndarray,
    b: np.ndarray,
    *,
    rbf_bandwidth: float,
    kid_degree: int = DEFAULT_KID_DEGREE,
    kid_coef0: float = DEFAULT_KID_COEF0,
    kid_gamma: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return canonical KID and RBF kernels for two feature matrices."""
    if a.shape[1] != b.shape[1]:
        raise ValueError(f'feature dimensions differ: {a.shape[1]} vs {b.shape[1]}')
    gamma = (1.0 / float(a.shape[1])) if kid_gamma is None else float(kid_gamma)
    dots = np.asarray(a, dtype=np.float64) @ np.asarray(b, dtype=np.float64).T
    polynomial = np.power(gamma * dots + float(kid_coef0), int(kid_degree))

    a_sq = np.sum(np.square(a), axis=1, keepdims=True)
    b_sq = np.sum(np.square(b), axis=1, keepdims=True).T
    squared_distances = np.maximum(a_sq + b_sq - 2.0 * dots, 0.0)
    bandwidth_sq = float(rbf_bandwidth) ** 2
    if not np.isfinite(bandwidth_sq) or bandwidth_sq <= 0:
        raise ValueError(f'RBF bandwidth must be positive, got {rbf_bandwidth}')
    rbf = np.exp(-squared_distances / (2.0 * bandwidth_sq))
    return polynomial, rbf


def unbiased_kid_mmd(
    p_features: np.ndarray,
    q_features: np.ndarray,
    *,
    rbf_bandwidth: float,
    kid_degree: int = DEFAULT_KID_DEGREE,
    kid_coef0: float = DEFAULT_KID_COEF0,
    kid_gamma: Optional[float] = None,
) -> Tuple[float, float]:
    """Unbiased MMD^2 for the polynomial KID kernel and an RBF kernel.

    Diagonal self-similarities are excluded.  The estimator is unbiased for
    fixed distributions and may be negative at finite sample sizes.
    """
    p = np.asarray(p_features, dtype=np.float64)
    q = np.asarray(q_features, dtype=np.float64)
    n_p, n_q = int(p.shape[0]), int(q.shape[0])
    if n_p < 2 or n_q < 2:
        raise ValueError('unbiased MMD requires at least two samples per side')

    pp_kid, pp_rbf = _kernel_matrices(
        p, p, rbf_bandwidth=rbf_bandwidth, kid_degree=kid_degree,
        kid_coef0=kid_coef0, kid_gamma=kid_gamma)
    qq_kid, qq_rbf = _kernel_matrices(
        q, q, rbf_bandwidth=rbf_bandwidth, kid_degree=kid_degree,
        kid_coef0=kid_coef0, kid_gamma=kid_gamma)
    pq_kid, pq_rbf = _kernel_matrices(
        p, q, rbf_bandwidth=rbf_bandwidth, kid_degree=kid_degree,
        kid_coef0=kid_coef0, kid_gamma=kid_gamma)

    def estimate(k_pp, k_qq, k_pq):
        p_term = (k_pp.sum() - np.trace(k_pp)) / (n_p * (n_p - 1))
        q_term = (k_qq.sum() - np.trace(k_qq)) / (n_q * (n_q - 1))
        cross_term = k_pq.mean()
        return float(p_term + q_term - 2.0 * cross_term)

    return estimate(pp_kid, qq_kid, pq_kid), estimate(pp_rbf, qq_rbf, pq_rbf)


def median_rbf_bandwidth(
    real_features: np.ndarray,
    *,
    max_samples: int = DEFAULT_BANDWIDTH_SAMPLE_SIZE,
    seed: int = 0,
    tag: str = '',
) -> float:
    """Median pairwise Euclidean distance on real features only.

    Defining sigma from reference data keeps the RBF metric fixed when models
    are added or removed.  Zero distances are omitted to tolerate duplicates.
    """
    real = np.asarray(real_features, dtype=np.float64)
    n = min(int(real.shape[0]), max(2, int(max_samples)))
    rows = _sample_rows(real.shape[0], n, seed, tag, 'rbf-bandwidth')
    sample = real[rows]
    dots = sample @ sample.T
    squared = np.maximum(
        np.sum(sample * sample, axis=1, keepdims=True)
        + np.sum(sample * sample, axis=1, keepdims=True).T
        - 2.0 * dots,
        0.0,
    )
    upper = squared[np.triu_indices(n, k=1)]
    positive = upper[upper > np.finfo(np.float64).eps]
    if positive.size == 0:
        raise ValueError(f'{tag}: all sampled real features are identical')
    return float(np.sqrt(np.median(positive)))


def _reference_draws(
    real_features: np.ndarray,
    *,
    subset_size: int,
    num_draws: int,
    seed: int,
    tag: str,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Matched real subsets (P, Q) used for scoring and real-vs-real calibration."""
    needed = 2 * int(subset_size)
    if int(real_features.shape[0]) < needed:
        return []
    draws = []
    for draw in range(int(num_draws)):
        rows = _sample_rows(real_features.shape[0], needed, seed, tag, 'real', draw)
        draws.append((
            real_features[rows[:int(subset_size)]],
            real_features[rows[int(subset_size):]],
        ))
    return draws


def prepare_reference_protocol(
    ref_features: Dict[str, np.ndarray],
    *,
    subset_size: int,
    num_draws: int,
    feature_norm: str,
    rbf_bandwidth: Optional[float],
    bandwidth_sample_size: int,
    seed: int,
) -> Dict[str, Dict]:
    """Transform reference features and precompute shared draws and baselines."""
    prepared: Dict[str, Dict] = {}
    for biome_name in sorted(ref_features):
        real = _transform_features(ref_features[biome_name], feature_norm)
        bandwidth = (
            float(rbf_bandwidth)
            if rbf_bandwidth not in (None, 0)
            else median_rbf_bandwidth(
                real, max_samples=bandwidth_sample_size, seed=seed, tag=biome_name)
        )
        draws = _reference_draws(
            real, subset_size=subset_size, num_draws=num_draws,
            seed=seed, tag=biome_name)
        if not draws:
            print(f'[kernel3d][warn] {biome_name} has {real.shape[0]} real chunks; '
                  f'need {2 * subset_size} for matched baseline - skipping')
            continue

        baseline_kid = []
        baseline_mmd = []
        for real_query, real_control in draws:
            kid, mmd = unbiased_kid_mmd(
                real_query, real_control, rbf_bandwidth=bandwidth)
            baseline_kid.append(kid)
            baseline_mmd.append(mmd)

        prepared[biome_name] = {
            'features': real,
            'bandwidth': bandwidth,
            'draws': draws,
            'baseline_kid3d_values': np.asarray(baseline_kid, dtype=np.float64),
            'baseline_mmd3d_values': np.asarray(baseline_mmd, dtype=np.float64),
        }
        print(f'[kernel3d] reference {biome_name:<20} real={real.shape[0]} '
              f'sigma={bandwidth:.6g}   baseline KID={np.mean(baseline_kid):.6g} '
              f'MMD={np.mean(baseline_mmd):.6g}')
    if not prepared:
        raise RuntimeError('no biome had enough reference features for the requested protocol')
    return prepared


def _generated_draw(
    gen_features: np.ndarray,
    subset_size: int,
    *,
    seed: int,
    biome_name: str,
    model_tag: str,
    draw: int,
) -> np.ndarray:
    if gen_features.shape[0] == int(subset_size):
        return gen_features
    rows = _sample_rows(
        gen_features.shape[0], subset_size, seed,
        biome_name, model_tag, 'generated', draw)
    return gen_features[rows]


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    return mean, std


def _macro_replicates(per_biome_values: Dict[str, np.ndarray]) -> np.ndarray:
    names = sorted(per_biome_values)
    lengths = {len(per_biome_values[name]) for name in names}
    if len(lengths) != 1:
        raise ValueError(f'replicate counts differ across biomes: {lengths}')
    return np.stack([per_biome_values[name] for name in names], axis=0).mean(axis=0)


@torch.no_grad()
def compute_model_metrics(
    samples_dir: str,
    *,
    model,
    converter,
    prepared_reference: Dict[str, Dict],
    samples_glob: str,
    samples_format: str,
    samples_per_biome: Optional[int],
    subset_size: int,
    feature_norm: str,
    batch_size: int,
    device: str,
    seed: int,
) -> Dict:
    if not os.path.isdir(str(samples_dir)):
        raise FileNotFoundError(f'samples_dir not found: {samples_dir}')
    files = sorted(glob.glob(os.path.join(str(samples_dir), samples_glob or '*.pt')))
    if not files:
        raise FileNotFoundError(
            f'no sample files matched {os.path.join(str(samples_dir), samples_glob)}')
    print(f'\n[kernel3d] === {samples_dir} ({len(files)} sample files) ===')

    lut = build_index_lut(converter) if samples_format == 'block_ids' else None
    ref_by_key = {normalize_biome(name): name for name in prepared_reference}
    grouped = group_samples_by_biome(files, converter, ref_by_key)
    model_tag = os.path.basename(os.path.normpath(str(samples_dir)))

    kid_values: Dict[str, np.ndarray] = {}
    mmd_values: Dict[str, np.ndarray] = {}
    counts: Dict[str, int] = {}

    for biome_name in sorted(grouped):
        if biome_name not in prepared_reference:
            continue
        samples = grouped[biome_name]
        if samples_per_biome not in (None, 0):
            samples = samples[:int(samples_per_biome)]
        n_gen = int(samples.shape[0])
        if n_gen < int(subset_size):
            print(f'[kernel3d][warn] {biome_name} has {n_gen} generated chunks; '
                  f'need subset_size={subset_size} - skipping')
            continue

        gen = extract_features(
            model,
            (to_index_volume(chunk, lut) for chunk in torch.split(samples, batch_size)),
            device=device,
        )
        gen = _transform_features(gen, feature_norm)
        reference = prepared_reference[biome_name]
        bandwidth = float(reference['bandwidth'])

        biome_kid = []
        biome_mmd = []
        for draw, (real_query, _) in enumerate(reference['draws']):
            gen_draw = _generated_draw(
                gen, subset_size, seed=seed, biome_name=biome_name,
                model_tag=model_tag, draw=draw)
            kid, mmd = unbiased_kid_mmd(
                real_query, gen_draw, rbf_bandwidth=bandwidth)
            biome_kid.append(kid)
            biome_mmd.append(mmd)

        kid_values[biome_name] = np.asarray(biome_kid, dtype=np.float64)
        mmd_values[biome_name] = np.asarray(biome_mmd, dtype=np.float64)
        counts[biome_name] = n_gen
        kid_mean, kid_std = _mean_std(biome_kid)
        mmd_mean, mmd_std = _mean_std(biome_mmd)
        print(f'[kernel3d] {biome_name:<20} KID {kid_mean:.6g} +/- {kid_std:.3g}   '
              f'MMD {mmd_mean:.6g} +/- {mmd_std:.3g}   gen={n_gen}')

    if not kid_values:
        raise RuntimeError(f'no biomes were scored in {samples_dir}')
    return {
        'biome_names': sorted(kid_values),
        'kid3d_values': kid_values,
        'mmd3d_values': mmd_values,
        'counts': counts,
    }


def _build_metric_result(
    metric_name: str,
    values_by_biome: Dict[str, np.ndarray],
    prepared_reference: Dict[str, Dict],
    counts: Dict[str, int],
    *,
    samples_dir: str,
    checkpoint: str,
    ref_dataset: str,
    mappings_path: str,
    weight_hash: str,
    feature_dim: int,
    subset_size: int,
    num_draws: int,
    feature_norm: str,
    rbf_bandwidth: Optional[float],
    bandwidth_sample_size: int,
    seed: int,
) -> Dict:
    biome_names = sorted(values_by_biome)
    baseline_key = f'baseline_{metric_name}_values'
    means = {}
    stds = {}
    baselines = {}
    baseline_stds = {}
    replicate_values = {}
    baseline_replicates = {}
    for biome in biome_names:
        values = np.asarray(values_by_biome[biome], dtype=np.float64)
        baseline = np.asarray(prepared_reference[biome][baseline_key], dtype=np.float64)
        means[biome], stds[biome] = _mean_std(values)
        baselines[biome], baseline_stds[biome] = _mean_std(baseline)
        replicate_values[biome] = values.tolist()
        baseline_replicates[biome] = baseline.tolist()

    macro = _macro_replicates(values_by_biome)
    macro_baseline = _macro_replicates({
        biome: prepared_reference[biome][baseline_key] for biome in biome_names
    })
    overall_mean, overall_std = _mean_std(macro)
    baseline_overall_mean, baseline_overall_std = _mean_std(macro_baseline)

    protocol = {
        'estimator': 'unbiased_mmd_squared_u_statistic',
        'direction': 'lower_is_better',
        'subset_size_per_side': int(subset_size),
        'num_reference_draws': int(num_draws),
        'generated_sampling': (
            'all rows each draw when count equals subset_size; otherwise '
            'deterministic subsets without replacement'
        ),
        'reference_sampling': 'deterministic matched subsets without replacement',
        'full_generated_bootstrap': False,
        'feature_norm': feature_norm,
        'seed': int(seed),
    }
    if metric_name == KID3D_NAME:
        protocol['kernel'] = {
            'name': 'polynomial',
            'formula': '(x_dot_y / feature_dim + 1) ** 3',
            'degree': DEFAULT_KID_DEGREE,
            'gamma': '1 / feature_dim',
            'coef0': DEFAULT_KID_COEF0,
        }
    else:
        protocol['kernel'] = {
            'name': 'rbf',
            'formula': 'exp(-squared_distance / (2 * sigma ** 2))',
            'bandwidth': (
                float(rbf_bandwidth)
                if rbf_bandwidth not in (None, 0)
                else 'per-biome median pairwise Euclidean distance on real features'
            ),
            'bandwidth_sample_size': int(bandwidth_sample_size),
        }

    return {
        'metric': metric_name,
        'direction': 'lower_is_better',
        'biome_names': biome_names,
        metric_name: means,
        f'{metric_name}_std': stds,
        f'reference_{metric_name}': baselines,
        f'reference_{metric_name}_std': baseline_stds,
        f'{metric_name}_replicates': replicate_values,
        f'reference_{metric_name}_replicates': baseline_replicates,
        'overall_mean': overall_mean,
        'overall_std': overall_std,
        'reference_overall_mean': baseline_overall_mean,
        'reference_overall_std': baseline_overall_std,
        'counts': {b: int(counts[b]) for b in biome_names},
        'ref_counts': {
            b: int(prepared_reference[b]['features'].shape[0]) for b in biome_names
        },
        'rbf_bandwidth': {
            b: float(prepared_reference[b]['bandwidth']) for b in biome_names
        },
        'protocol': protocol,
        'feature_dim': int(feature_dim),
        'weight_hash': weight_hash,
        'checkpoint': str(checkpoint),
        'ref_dataset': str(ref_dataset),
        'mappings_path': str(mappings_path),
        'samples_dir': str(samples_dir),
    }


def _write_metric_artifacts(result: Dict, samples_dir: str) -> Tuple[str, str]:
    metric_name = result['metric']
    json_path = os.path.join(str(samples_dir), f'per_biome_{metric_name}.json')
    npz_path = os.path.join(str(samples_dir), f'per_biome_{metric_name}.npz')
    result['timestamp'] = int(time.time())
    result['metrics_path'] = npz_path

    with open(json_path, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)

    biome_names = result['biome_names']
    payload = {}
    for biome in biome_names:
        payload[f'replicates__{biome}'] = np.asarray(
            result[f'{metric_name}_replicates'][biome], dtype=np.float64)
        payload[f'reference_replicates__{biome}'] = np.asarray(
            result[f'reference_{metric_name}_replicates'][biome], dtype=np.float64)
    np.savez_compressed(
        npz_path,
        biome_names=np.asarray(biome_names),
        **{
            metric_name: np.asarray(
                [result[metric_name][b] for b in biome_names], dtype=np.float64),
            f'{metric_name}_std': np.asarray(
                [result[f'{metric_name}_std'][b] for b in biome_names], dtype=np.float64),
            f'reference_{metric_name}': np.asarray(
                [result[f"reference_{metric_name}"][b] for b in biome_names],
                dtype=np.float64),
            f'reference_{metric_name}_std': np.asarray(
                [result[f"reference_{metric_name}_std"][b] for b in biome_names],
                dtype=np.float64),
        },
        counts=np.asarray([result['counts'][b] for b in biome_names], dtype=np.int32),
        ref_counts=np.asarray([result['ref_counts'][b] for b in biome_names], dtype=np.int32),
        rbf_bandwidth=np.asarray(
            [result['rbf_bandwidth'][b] for b in biome_names], dtype=np.float64),
        overall_mean=np.asarray([result['overall_mean']], dtype=np.float64),
        overall_std=np.asarray([result['overall_std']], dtype=np.float64),
        reference_overall_mean=np.asarray(
            [result['reference_overall_mean']], dtype=np.float64),
        reference_overall_std=np.asarray(
            [result['reference_overall_std']], dtype=np.float64),
        protocol_json=np.asarray(json.dumps(result['protocol'], sort_keys=True)),
        feature_dim=np.asarray([result['feature_dim']], dtype=np.int32),
        weight_hash=np.asarray(result['weight_hash']),
        checkpoint=np.asarray(result['checkpoint']),
        ref_dataset=np.asarray(result['ref_dataset']),
        **payload,
    )
    print(f'[kernel3d] wrote {json_path}')
    print(f'[kernel3d] wrote {npz_path}')
    return json_path, npz_path


def _score_kernel_query(
    query_features: np.ndarray,
    reference: Dict,
    *,
    subset_size: int,
    feature_norm: str,
    seed: int,
    tag: str,
) -> Dict[str, object]:
    """Score one fixed query against every matched draw of a target biome."""
    query = _transform_features(query_features, feature_norm)
    if query.shape[0] < int(subset_size):
        raise ValueError(
            f'{tag}: query has {query.shape[0]} rows; need {subset_size}')
    if query.shape[0] > int(subset_size):
        rows = _sample_rows(
            query.shape[0], subset_size, seed, tag, 'calibration-query')
        query = query[rows]

    kid_values = []
    mmd_values = []
    for real_query, _ in reference['draws']:
        kid, mmd = unbiased_kid_mmd(
            query,
            real_query,
            rbf_bandwidth=float(reference['bandwidth']),
        )
        kid_values.append(kid)
        mmd_values.append(mmd)
    kid_mean, kid_std = _mean_std(kid_values)
    mmd_mean, mmd_std = _mean_std(mmd_values)
    return {
        'kid3d': kid_mean,
        'kid3d_std': kid_std,
        'mmd3d': mmd_mean,
        'mmd3d_std': mmd_std,
        'kid3d_replicates': [float(value) for value in kid_values],
        'mmd3d_replicates': [float(value) for value in mmd_values],
    }


@torch.no_grad()
def run_calibration(
    *,
    checkpoint: str,
    ref_dataset: str,
    mappings_path: Optional[str] = None,
    ref_values: str = 'indices',
    ref_per_biome: Optional[int] = None,
    subset_size: int = DEFAULT_SUBSET_SIZE,
    num_draws: int = DEFAULT_NUM_DRAWS,
    feature_norm: str = 'none',
    rbf_bandwidth: Optional[float] = None,
    bandwidth_sample_size: int = DEFAULT_BANDWIDTH_SAMPLE_SIZE,
    kinds: Sequence[str] = NOISE_KINDS,
    cross_biome: bool = True,
    batch_size: int = 64,
    device: Optional[str] = None,
    cache_dir: Optional[str] = 'fid3d_cache',
    refresh_cache: bool = False,
    seed: int = 0,
    out_dir: str = 'analysis/kernel3d_calibration',
) -> Dict:
    """Calibrate KID/MMD with same-biome, cross-biome, and noise queries.

    Cross-biome cells use exactly ``subset_size`` chunks from the row biome and
    the target biome's fixed reference draws/bandwidth.  Noise cells destroy
    geometry in real chunks while retaining either exact or expected block
    histograms, testing whether the metric reads more than block frequencies.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model, converter, resolved_mappings, weight_hash = load_extractor_and_converter(
        str(checkpoint), mappings_path, device=device)
    ref_features = reference_features_by_biome(
        str(ref_dataset), model, converter,
        ref_values=str(ref_values), per_biome=ref_per_biome,
        batch_size=int(batch_size), device=device, seed=int(seed),
        cache_path=reference_cache_path(
            cache_dir, str(ref_dataset), weight_hash, ref_per_biome, int(seed)),
        refresh_cache=bool(refresh_cache),
    )
    prepared = prepare_reference_protocol(
        ref_features,
        subset_size=int(subset_size),
        num_draws=int(num_draws),
        feature_norm=str(feature_norm),
        rbf_bandwidth=rbf_bandwidth,
        bandwidth_sample_size=int(bandwidth_sample_size),
        seed=int(seed),
    )
    names = sorted(prepared)
    cross: Dict[str, Dict[str, Dict[str, object]]] = {}
    if cross_biome:
        print(f'[kernel3d][cal] --- {len(names)} x {len(names)} real-biome matrix ---')
        for source in names:
            source_query = ref_features[source]
            cross[source] = {}
            for target in names:
                if source == target:
                    reference = prepared[target]
                    kid_values = reference['baseline_kid3d_values']
                    mmd_values = reference['baseline_mmd3d_values']
                    kid_mean, kid_std = _mean_std(kid_values)
                    mmd_mean, mmd_std = _mean_std(mmd_values)
                    cell = {
                        'kid3d': kid_mean,
                        'kid3d_std': kid_std,
                        'mmd3d': mmd_mean,
                        'mmd3d_std': mmd_std,
                        'kid3d_replicates': kid_values.astype(float).tolist(),
                        'mmd3d_replicates': mmd_values.astype(float).tolist(),
                    }
                else:
                    cell = _score_kernel_query(
                        source_query,
                        prepared[target],
                        subset_size=int(subset_size),
                        feature_norm=str(feature_norm),
                        seed=int(seed),
                        tag=f'cross|{source}|{target}',
                    )
                cross[source][target] = cell
            other_kid = [cross[source][target]['kid3d']
                         for target in names if target != source]
            other_mmd = [cross[source][target]['mmd3d']
                         for target in names if target != source]
            print(f'[kernel3d][cal] {source:<20} '
                  f"floor KID={cross[source][source]['kid3d']:.5g} "
                  f'mean-other={np.mean(other_kid):.5g}   '
                  f"floor MMD={cross[source][source]['mmd3d']:.5g} "
                  f'mean-other={np.mean(other_mmd):.5g}')

    noise: Dict[str, Dict[str, Dict[str, object]]] = {}
    if kinds:
        print('[kernel3d][cal] --- structure-free queries ---')
        noise_features = structure_free_features(
            str(ref_dataset), model, converter, ref_features,
            kinds=tuple(kinds), n=int(subset_size), ref_values=str(ref_values),
            batch_size=int(batch_size), device=device, seed=int(seed))
        for kind in kinds:
            noise[kind] = {}
            for biome_name in names:
                features = noise_features.get(kind, {}).get(biome_name)
                if features is None:
                    continue
                noise[kind][biome_name] = _score_kernel_query(
                    features,
                    prepared[biome_name],
                    subset_size=int(subset_size),
                    feature_norm=str(feature_norm),
                    seed=int(seed),
                    tag=f'noise|{kind}|{biome_name}',
                )
            if noise[kind]:
                print(
                    f'[kernel3d][cal] mean {kind:<8} '
                    f"KID={np.mean([v['kid3d'] for v in noise[kind].values()]):.5g} "
                    f"MMD={np.mean([v['mmd3d'] for v in noise[kind].values()]):.5g}"
                )

    def cross_macro(metric: str, diagonal: bool) -> Optional[float]:
        values = [
            cross[source][target][metric]
            for source in names for target in names
            if (source == target) == diagonal
        ] if cross else []
        return float(np.mean(values)) if values else None

    result = {
        'n': int(subset_size),
        'biomes': names,
        'cross_biome': cross,
        'noise': noise,
        'macro': {
            'same_biome_kid3d': cross_macro('kid3d', True),
            'different_biome_kid3d': cross_macro('kid3d', False),
            'same_biome_mmd3d': cross_macro('mmd3d', True),
            'different_biome_mmd3d': cross_macro('mmd3d', False),
            'noise': {
                kind: {
                    'kid3d': float(np.mean([v['kid3d'] for v in cells.values()])),
                    'mmd3d': float(np.mean([v['mmd3d'] for v in cells.values()])),
                }
                for kind, cells in noise.items() if cells
            },
        },
        'rbf_bandwidth': {
            name: float(prepared[name]['bandwidth']) for name in names},
        'protocol': {
            'subset_size_per_side': int(subset_size),
            'num_reference_draws': int(num_draws),
            'feature_norm': str(feature_norm),
            'rbf_bandwidth': (
                float(rbf_bandwidth) if rbf_bandwidth not in (None, 0) else 'real_median'),
            'bandwidth_sample_size': int(bandwidth_sample_size),
            'noise_kinds': list(kinds),
            'seed': int(seed),
        },
        'weight_hash': weight_hash,
        'checkpoint': str(checkpoint),
        'ref_dataset': str(ref_dataset),
        'mappings_path': resolved_mappings,
    }
    tag = (
        f'{os.path.basename(os.path.normpath(str(ref_dataset)))}_'
        f'{weight_hash[:8]}_n{int(subset_size)}'
    )
    os.makedirs(str(out_dir), exist_ok=True)
    json_path = os.path.join(str(out_dir), f'calibration_{tag}.json')
    with open(json_path, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
    if cross:
        kid_matrix = np.asarray([
            [cross[source][target]['kid3d'] for target in names] for source in names])
        mmd_matrix = np.asarray([
            [cross[source][target]['mmd3d'] for target in names] for source in names])
        np.savez_compressed(
            os.path.join(str(out_dir), f'cross_biome_{tag}.npz'),
            biome_names=np.asarray(names),
            kid3d=kid_matrix,
            mmd3d=mmd_matrix,
        )
    print(f'[kernel3d][cal] wrote {json_path}')
    return result


def run_selftest() -> None:
    rng = np.random.default_rng(0)
    p = rng.normal(size=(1000, 32))
    q_same = rng.normal(size=(1000, 32))
    q_shifted = rng.normal(size=(1000, 32)) + 0.75
    sigma = median_rbf_bandwidth(p, max_samples=500, seed=0, tag='selftest')

    same_kid, same_mmd = unbiased_kid_mmd(p, q_same, rbf_bandwidth=sigma)
    shifted_kid, shifted_mmd = unbiased_kid_mmd(p, q_shifted, rbf_bandwidth=sigma)
    reverse_kid, reverse_mmd = unbiased_kid_mmd(q_shifted, p, rbf_bandwidth=sigma)
    print(f'[selftest] same distribution: KID={same_kid:.6g} MMD={same_mmd:.6g}')
    print(f'[selftest] shifted: KID={shifted_kid:.6g} MMD={shifted_mmd:.6g}')
    assert shifted_kid > same_kid
    assert shifted_mmd > same_mmd
    assert np.isclose(shifted_kid, reverse_kid, atol=1e-12)
    assert np.isclose(shifted_mmd, reverse_mmd, atol=1e-12)
    print('[selftest] ok')


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Compute per-biome 3D-KID and RBF-MMD from voxel samples.')
    parser.add_argument('--samples_dir', nargs='+', default=None,
                        help='One or more generated-sample directories.')
    parser.add_argument('--checkpoint', default=None,
                        help='Frozen feature_extractor_best.pt used by 3D-FID.')
    parser.add_argument('--ref_dataset', default=None,
                        help='Held-out real validation dataset.')
    parser.add_argument('--mappings_path', default=None)
    parser.add_argument('--samples_glob', default='*.pt')
    parser.add_argument('--samples_format', choices=['block_ids', 'indices'],
                        default='block_ids')
    parser.add_argument('--ref_values', choices=['indices', 'block_ids'], default='indices')
    parser.add_argument('--samples_per_biome', type=int, default=0,
                        help='Cap generated chunks before featurization; 0 uses all.')
    parser.add_argument('--ref_per_biome', type=int, default=0,
                        help='Cap cached real features; 0 uses the full split.')
    parser.add_argument('--subset_size', type=int, default=DEFAULT_SUBSET_SIZE,
                        help='Rows per side of each unbiased estimate (default: 100).')
    parser.add_argument('--num_draws', type=int, default=DEFAULT_NUM_DRAWS,
                        help='Matched reference subsets averaged per biome (default: 50).')
    parser.add_argument('--feature_norm', choices=['none', 'l2'], default='none',
                        help='Raw matches conventional KID; changing this changes both metrics.')
    parser.add_argument('--rbf_bandwidth', type=float, default=0.0,
                        help='RBF sigma. 0 uses a per-biome real-only median heuristic.')
    parser.add_argument('--bandwidth_sample_size', type=int,
                        default=DEFAULT_BANDWIDTH_SAMPLE_SIZE,
                        help='Maximum real rows used by the median bandwidth heuristic.')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', default=None)
    parser.add_argument('--cache_dir', default='fid3d_cache')
    parser.add_argument('--refresh_cache', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--selftest', action='store_true')
    calibration = parser.add_argument_group('calibration')
    calibration.add_argument(
        '--calibrate', action='store_true',
        help='Score no generator; compare real biomes to each other and score '
             'structure-free shuffled/marginal/uniform chunks.')
    calibration.add_argument(
        '--calibration_kinds', default=','.join(NOISE_KINDS),
        help="Comma-separated noise types, or 'none'.")
    calibration.add_argument(
        '--no_cross_biome', action='store_true',
        help='Skip the real-biome-vs-real-biome matrices.')
    calibration.add_argument(
        '--calibration_out', default='analysis/kernel3d_calibration')
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.selftest:
        run_selftest()
        return
    required = ('checkpoint', 'ref_dataset')
    if not args.calibrate:
        required = ('samples_dir',) + required
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        raise SystemExit('missing required arguments: '
                         + ', '.join(f'--{name}' for name in missing))
    if int(args.subset_size) < 2:
        raise SystemExit('--subset_size must be at least 2')
    if int(args.num_draws) < 1:
        raise SystemExit('--num_draws must be at least 1')
    if float(args.rbf_bandwidth) < 0:
        raise SystemExit('--rbf_bandwidth must be 0 (auto) or positive')

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    cache_dir = None if args.cache_dir in (None, '', 'none') else args.cache_dir
    ref_per_biome = int(args.ref_per_biome) or None
    if args.calibrate:
        kinds = [
            kind.strip()
            for kind in str(args.calibration_kinds).split(',')
            if kind.strip()
        ]
        if kinds == ['none']:
            kinds = []
        unknown = sorted(set(kinds) - set(NOISE_KINDS))
        if unknown:
            raise SystemExit(
                f"unknown --calibration_kinds: {', '.join(unknown)}; "
                f"expected {', '.join(NOISE_KINDS)}, or none")
        run_calibration(
            checkpoint=str(args.checkpoint),
            ref_dataset=str(args.ref_dataset),
            mappings_path=args.mappings_path,
            ref_values=str(args.ref_values),
            ref_per_biome=ref_per_biome,
            subset_size=int(args.subset_size),
            num_draws=int(args.num_draws),
            feature_norm=str(args.feature_norm),
            rbf_bandwidth=(
                float(args.rbf_bandwidth) if float(args.rbf_bandwidth) > 0 else None),
            bandwidth_sample_size=int(args.bandwidth_sample_size),
            kinds=kinds,
            cross_biome=not bool(args.no_cross_biome),
            batch_size=int(args.batch_size),
            device=device,
            cache_dir=cache_dir,
            refresh_cache=bool(args.refresh_cache),
            seed=int(args.seed),
            out_dir=str(args.calibration_out),
        )
        return

    model, converter, resolved_mappings, weight_hash = load_extractor_and_converter(
        str(args.checkpoint), args.mappings_path, device=device)
    ref_features = reference_features_by_biome(
        str(args.ref_dataset), model, converter,
        ref_values=str(args.ref_values), per_biome=ref_per_biome,
        batch_size=int(args.batch_size), device=device, seed=int(args.seed),
        cache_path=reference_cache_path(
            cache_dir, str(args.ref_dataset), weight_hash,
            ref_per_biome, int(args.seed)),
        refresh_cache=bool(args.refresh_cache),
    )
    prepared_reference = prepare_reference_protocol(
        ref_features,
        subset_size=int(args.subset_size),
        num_draws=int(args.num_draws),
        feature_norm=str(args.feature_norm),
        rbf_bandwidth=(
            float(args.rbf_bandwidth) if float(args.rbf_bandwidth) > 0 else None),
        bandwidth_sample_size=int(args.bandwidth_sample_size),
        seed=int(args.seed),
    )

    summaries = []
    for samples_dir in args.samples_dir:
        computed = compute_model_metrics(
            str(samples_dir),
            model=model,
            converter=converter,
            prepared_reference=prepared_reference,
            samples_glob=str(args.samples_glob),
            samples_format=str(args.samples_format),
            samples_per_biome=(int(args.samples_per_biome) or None),
            subset_size=int(args.subset_size),
            feature_norm=str(args.feature_norm),
            batch_size=int(args.batch_size),
            device=device,
            seed=int(args.seed),
        )
        common = dict(
            prepared_reference=prepared_reference,
            counts=computed['counts'],
            samples_dir=str(samples_dir),
            checkpoint=str(args.checkpoint),
            ref_dataset=str(args.ref_dataset),
            mappings_path=resolved_mappings,
            weight_hash=weight_hash,
            feature_dim=int(model.feature_dim),
            subset_size=int(args.subset_size),
            num_draws=int(args.num_draws),
            feature_norm=str(args.feature_norm),
            rbf_bandwidth=(
                float(args.rbf_bandwidth) if float(args.rbf_bandwidth) > 0 else None),
            bandwidth_sample_size=int(args.bandwidth_sample_size),
            seed=int(args.seed),
        )
        kid_result = _build_metric_result(
            KID3D_NAME, computed['kid3d_values'], **common)
        mmd_result = _build_metric_result(
            MMD3D_NAME, computed['mmd3d_values'], **common)
        kid_json, kid_npz = _write_metric_artifacts(kid_result, str(samples_dir))
        mmd_json, mmd_npz = _write_metric_artifacts(mmd_result, str(samples_dir))
        summaries.append({
            'samples_dir': str(samples_dir),
            'kid3d_overall_mean': kid_result['overall_mean'],
            'kid3d_reference_overall_mean': kid_result['reference_overall_mean'],
            'mmd3d_overall_mean': mmd_result['overall_mean'],
            'mmd3d_reference_overall_mean': mmd_result['reference_overall_mean'],
            'kid3d_json': kid_json,
            'kid3d_npz': kid_npz,
            'mmd3d_json': mmd_json,
            'mmd3d_npz': mmd_npz,
        })

    print('\n' + json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
