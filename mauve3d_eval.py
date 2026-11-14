"""
Voxel-native MAUVE: a divergence-frontier companion to the 3D-FID in `fid3d_eval.py`.

    python mauve3d_eval.py \
      --samples_dir renders/per_biome_gen/MD4_p4_.../ \
      --checkpoint results/fid_defaults/exp_.../feature_extractor_best.pt \
      --ref_dataset data/balanced_dataset/32_biome_parts_dir_balanced_cc_val_dir

Why this exists.  A Frechet distance summarizes two point clouds by a mean and a
covariance and returns one number, so it cannot say *how* a model failed: a model
that emits plausible-but-repetitive chunks and a model that emits varied-but-
implausible ones can land on the same FID.  MAUVE quantizes the feature space and
compares the two cluster histograms along a mixture path, which splits the error
into the two directions that actually matter here:

  * `fidelity` drops when the model puts mass where the real chunks never go --
    volumes that do not look like Minecraft terrain at all.
  * `coverage` drops when the model never reaches parts of the real distribution
    -- collapsed or missing modes.

These are the Type I and Type II errors of Pillutla et al., read from opposite
ends of that curve rather than from its midpoint -- the midpoint gives the two
Jensen-Shannon components instead, which order the two failures backwards.  See
the comment in `mauve_from_embedding`, and `--selftest`, which pins the direction
with a matched pair of mirrored failures.

Which of the two moves is what distinguishes a model that generates junk from one
that generates the same handful of plausible chunks forever, and a Frechet
distance assigns those two failures the same kind of number.

MAUVE itself is the area under the frontier traced by those two quantities, in
[0, 1], higher is better.  The full curve is written out so the frontier can be
plotted per biome.

Features come from the same frozen extractor as the 3D-FID, through the same
cached `reference_features_by_biome`, so the two metrics are two readings of one
feature space rather than two unrelated numbers.  Run this with the same
`--checkpoint`, `--ref_dataset`, `--ref_per_biome` and `--seed` you used for
`fid3d_eval.py` and the reference cache is reused verbatim.

Reading the numbers
-------------------
The quantization is k-means, and the official heuristic picks k = min(|P|,|Q|)/10.
A histogram over k bins estimated from n samples carries an error of roughly
(k-1)/n even when the two distributions are identical, which biases MAUVE
*downward*.  The bias is the same for every model evaluated at the same N, so
rankings stay fair, but a raw value is not a percentage of anything.

So the real-vs-real reference is not optional here.  It draws a query set of the
same size from the held-out real chunks and scores it against the remainder,
measuring what this protocol returns when the two distributions are genuinely
identical.  That is the ceiling, `mauve_normalized` is the ratio to it, and the
two should always be quoted together.  It depends only on the real features, the
query size and k -- never on a model -- so `--baseline_only` computes it on its
own, without needing `--samples_dir`:

    python mauve3d_eval.py --baseline_only \
      --checkpoint ... --ref_dataset ... --baseline_query_size 100

If the ceiling at some k sits near 1.0, that k cannot rank anything: every model
is pinned against it and the ordering is decided by replicate noise.  Choose k
from the baseline before reading any model's score.

The ceiling also decides whether the metric can say anything at all, and at 100
generated chunks per biome that is a live problem rather than a formality.  The
standard rule gives k = 10 there, and 10 bins from 100 samples puts the ceiling
near 0.99 -- so every model scores in the high nineties and MAUVE becomes exactly
as uninformative as the FID it was meant to supplement.  A finer quantization
costs ceiling but buys resolution.  `--cluster_sweep` is therefore on by default:
it reports, for each k, the ceiling, the headroom below it, and the gap between
model and ceiling in units of replicate noise, so k is chosen from evidence
instead of from a heuristic that was tuned for sample sizes we do not have.

The MAUVE computation follows Pillutla et al. (2021) and matches the defaults of
the reference `mauve-text` implementation: l2-normalized features, PCA to 90%
explained variance, k-means quantization, a 25-point mixture path, and
exp(-5 * KL) scaling.  `--selftest` checks it against those expectations, and
against the `mauve` package when it happens to be installed.
"""

import glob
import json
import os
import random
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

MAUVE3D_LABEL = 'mauve3d'

# Defaults of the reference implementation; changing any of them changes the
# numbers, so they are recorded in the output rather than left implicit.
DEFAULT_SCALING_FACTOR = 5.0
DEFAULT_CURVE_POINTS = 25
# How far inside the ends of the divergence curve the two error types are read.
# The ends themselves (0 and 1) are infinite whenever one support misses the
# other, which is the norm at 100 generated chunks.
DEFAULT_ENDPOINT_LAMBDA = 0.9
DEFAULT_EXPLAINED_VARIANCE = 0.9
DEFAULT_KMEANS_REDO = 5
DEFAULT_KMEANS_MAX_ITER = 500

# numpy 2 renamed trapz; both spellings have to work because the cluster and the
# laptop are not always on the same numpy.
_trapezoid = getattr(np, 'trapezoid', None) or np.trapz


# ---------------------------------------------------------------------------
# filenames, kept parallel to fid3d_eval.py's helpers
# ---------------------------------------------------------------------------


def metrics_filename_for_label(label: str) -> str:
    safe = str(label).strip().lower().replace('-', '_')
    return f'per_biome_mauve_{safe}.npz'


def summary_filename_for_label(label: str) -> str:
    safe = str(label).strip().lower().replace('-', '_')
    return f'per_biome_mauve_{safe}.json'


# ---------------------------------------------------------------------------
# MAUVE core
# ---------------------------------------------------------------------------


def auto_num_clusters(n_p: int, n_q: int) -> int:
    """The reference implementation's rule: one cluster per ten samples.

    Driven by the *smaller* set, because that is the one whose histogram has to
    be estimated, and it is the generated set here.
    """
    return max(2, int(round(min(int(n_p), int(n_q)) / 10)))


def _reduce_joint(p_features: np.ndarray, q_features: np.ndarray, *,
                  feature_norm: str = 'l2',
                  explained_variance: float = DEFAULT_EXPLAINED_VARIANCE,
                  whiten: bool = False) -> Tuple[np.ndarray, int]:
    """Stack, normalize and PCA-reduce both feature sets into one space.

    Split out from the clustering because the PCA here is a full deterministic
    SVD -- it does not depend on the k-means seed, so recomputing it per seed (as
    the reference implementation does) would only waste time on the ~6.6k real
    chunks per biome.  Returns the reduced matrix with the generated rows first,
    matching the reference implementation's ordering, plus that row count.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize

    q = np.asarray(q_features, dtype=np.float64)
    p = np.asarray(p_features, dtype=np.float64)
    data = np.vstack([q, p])
    if feature_norm in ('l1', 'l2'):
        data = normalize(data, norm=feature_norm, axis=1)

    pca = PCA(n_components=None, whiten=bool(whiten), random_state=0)
    pca.fit(data)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    keep = int(np.argmax(cumulative >= float(explained_variance))) + 1
    reduced = pca.transform(data)[:, :keep].astype(np.float32)
    return reduced, int(q.shape[0])


def _kmeans_labels(reduced: np.ndarray, num_clusters: int, seed: int, *,
                   n_init: int = DEFAULT_KMEANS_REDO,
                   max_iter: int = DEFAULT_KMEANS_MAX_ITER) -> np.ndarray:
    from sklearn.cluster import KMeans

    k = int(min(int(num_clusters), reduced.shape[0]))
    kmeans = KMeans(n_clusters=max(2, k), n_init=int(n_init), max_iter=int(max_iter),
                    random_state=int(seed))
    return kmeans.fit_predict(reduced)


def _histogram(labels: np.ndarray, num_clusters: int) -> np.ndarray:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=int(num_clusters))
    counts = counts.astype(np.float64)
    total = counts.sum()
    return counts / total if total > 0 else counts


def _kl_multinomial(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) for two histograms over the same support."""
    if np.logical_and(p != 0, q == 0).any():
        return float('inf')
    support = p > 0
    return float(np.sum(p[support] * np.log(p[support] / q[support])))


def _divergence_curve(p: np.ndarray, q: np.ndarray, *, scaling_factor: float,
                      curve_points: int) -> np.ndarray:
    """The frontier traced by mixtures R = w*P + (1-w)*Q.

    x is exp(-c * KL(Q || R)) and y is exp(-c * KL(P || R)), so both axes live in
    [0, 1] and (1, 1) is the ideal corner.  The endpoints are the degenerate
    mixtures R = Q and R = P, which pin the curve to (1, 0) and (0, 1).
    """
    weights = np.linspace(1e-6, 1.0 - 1e-6, int(curve_points))
    curve = [[0.0, float('inf')]]
    for w in weights:
        r = w * p + (1.0 - w) * q
        curve.append([_kl_multinomial(q, r), _kl_multinomial(p, r)])
    curve.append([float('inf'), 0.0])
    with np.errstate(over='ignore'):
        return np.exp(-float(scaling_factor) * np.asarray(curve, dtype=np.float64))


def _area_under_curve(curve: np.ndarray) -> float:
    """Area under the frontier.

    The path runs from (1, 0) to (0, 1), so x is decreasing and the trapezoid
    rule comes out negative; the magnitude is the area.  Integrating along the
    path rather than sorting by x matters, because the two endpoints share an x
    coordinate with their neighbours when the distributions are identical and
    sorting would slice the area in half.
    """
    return float(abs(_trapezoid(curve[:, 1], curve[:, 0])))


def mauve_from_embedding(reduced: np.ndarray, n_q: int, *, num_clusters: int, seed: int,
                         scaling_factor: float = DEFAULT_SCALING_FACTOR,
                         curve_points: int = DEFAULT_CURVE_POINTS,
                         endpoint_lambda: float = DEFAULT_ENDPOINT_LAMBDA,
                         n_init: int = DEFAULT_KMEANS_REDO,
                         max_iter: int = DEFAULT_KMEANS_MAX_ITER) -> Dict:
    """One MAUVE replicate over an already-reduced joint embedding."""
    labels = _kmeans_labels(reduced, num_clusters, seed, n_init=n_init, max_iter=max_iter)
    k = int(labels.max()) + 1
    q_hist = _histogram(labels[:int(n_q)], k)
    p_hist = _histogram(labels[int(n_q):], k)

    curve = _divergence_curve(p_hist, q_hist, scaling_factor=scaling_factor,
                              curve_points=curve_points)

    # Pillutla et al. identify the two error types with the *ends* of the curve:
    # Type I is KL(Q || P), the first coordinate as lambda -> 1, and Type II is
    # KL(P || Q), the second coordinate as lambda -> 0.  Each end has to be read
    # at its own lambda -- the other coordinate goes to zero there, so no single
    # lambda reports both.  The exact ends are unusable because either KL becomes
    # infinite as soon as one support fails to contain the other, which at 100
    # generated chunks happens in nearly every biome, so each is read just inside
    # its end at `endpoint_lambda`.
    #
    # Reading both at lambda = 0.5 instead gives the two Jensen-Shannon
    # components, which is a different quantity and, worse, orders the two
    # failures backwards: a model reaching only half the real modes has an
    # infinite Type II error but a *larger* first coordinate at the midpoint.
    lam = float(endpoint_lambda)
    r_fidelity = lam * p_hist + (1.0 - lam) * q_hist
    r_coverage = (1.0 - lam) * p_hist + lam * q_hist
    kl_type1 = _kl_multinomial(q_hist, r_fidelity)
    kl_type2 = _kl_multinomial(p_hist, r_coverage)

    mixture = 0.5 * (p_hist + q_hist)
    js = 0.5 * (_kl_multinomial(p_hist, mixture) + _kl_multinomial(q_hist, mixture))

    return {
        'mauve': _area_under_curve(curve),
        'fidelity': float(np.exp(-float(scaling_factor) * kl_type1)),
        'coverage': float(np.exp(-float(scaling_factor) * kl_type2)),
        'js_divergence': float(js),
        'num_clusters': k,
        'curve': curve,
    }


def mauve_from_features(p_features: np.ndarray, q_features: np.ndarray, *,
                        num_clusters: Optional[int] = None, seed: int = 0,
                        feature_norm: str = 'l2',
                        explained_variance: float = DEFAULT_EXPLAINED_VARIANCE,
                        **kwargs) -> Dict:
    """Single-shot MAUVE between two feature matrices. P is real, Q is generated."""
    reduced, n_q = _reduce_joint(p_features, q_features, feature_norm=feature_norm,
                                 explained_variance=explained_variance)
    k = int(num_clusters) if num_clusters else auto_num_clusters(
        p_features.shape[0], q_features.shape[0])
    return mauve_from_embedding(reduced, n_q, num_clusters=k, seed=seed, **kwargs)


# ---------------------------------------------------------------------------
# replicate protocol
# ---------------------------------------------------------------------------

SCALAR_KEYS = ('mauve', 'fidelity', 'coverage', 'js_divergence')


def _shuffled_order(total: int, seed: int, tag: str) -> List[int]:
    order = list(range(int(total)))
    random.Random(f'{seed}|{tag}').shuffle(order)
    return order


def generated_pairs(real: np.ndarray, gen: np.ndarray, *, real_side: str,
                    num_draws: int, seed: int, tag: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    """(P, Q) pairs for scoring a model.

    With `real_side='all'` nothing on either side is random, so there is exactly
    one pair and the only replicate axis left is the k-means seed.  With
    `real_side='matched'` the real side is subsampled to the generated count and
    each draw is a different subsample.
    """
    if real_side == 'all':
        return [(real, gen)]
    n_gen = int(gen.shape[0])
    if int(real.shape[0]) < n_gen:
        return [(real, gen)]
    pairs = []
    for draw in range(int(num_draws)):
        order = _shuffled_order(real.shape[0], seed, f'{tag}|gen|draw{draw}')
        pairs.append((real[np.array(order[:n_gen], dtype=np.int64)], gen))
    return pairs


def reference_pairs(real: np.ndarray, n_query: int, *, real_side: str, num_draws: int,
                    seed: int, tag: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    """(P, Q) pairs where Q is also real, giving the ceiling for this protocol.

    Q always has `n_query` rows so the reference sits at the same sample count as
    the generated set -- comparing against a ceiling measured at a different N
    would be meaningless, for the same reason the 3D-FID floor is quoted at the
    generated count.
    """
    total = int(real.shape[0])
    needed = 2 * int(n_query) if real_side == 'matched' else int(n_query) + 2
    if total < needed:
        return []
    pairs = []
    for draw in range(int(num_draws)):
        order = _shuffled_order(total, seed, f'{tag}|ref|draw{draw}')
        q_rows = np.array(order[:int(n_query)], dtype=np.int64)
        p_rows = np.array(order[int(n_query):2 * int(n_query)] if real_side == 'matched'
                          else order[int(n_query):], dtype=np.int64)
        pairs.append((real[p_rows], real[q_rows]))
    return pairs


def _aggregate(records: List[Dict]) -> Dict:
    out: Dict = {}
    for key in SCALAR_KEYS:
        values = np.array([r[key] for r in records], dtype=np.float64)
        out[key] = float(values.mean())
        out[f'{key}_std'] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        out[f'{key}_values'] = values.tolist()
    out['num_clusters'] = int(records[0]['num_clusters'])
    out['num_replicates'] = len(records)
    out['curve'] = np.mean(np.stack([r['curve'] for r in records], axis=0), axis=0)
    return out


def build_embeddings(pairs: Sequence[Tuple[np.ndarray, np.ndarray]], *, feature_norm: str,
                     explained_variance: float) -> List[Tuple[np.ndarray, int]]:
    """Reduce every draw once.

    Kept separate from the scoring so a cluster sweep pays for the PCA a single
    time; on the full real side that is a 6.6k x 512 SVD per draw and it does not
    depend on k or on the k-means seed.
    """
    return [_reduce_joint(p, q, feature_norm=feature_norm,
                          explained_variance=explained_variance) for p, q in pairs]


def mauve_over_embeddings(embeddings: Sequence[Tuple[np.ndarray, int]], *, num_seeds: int,
                          num_clusters: Optional[int], seed: int, scaling_factor: float,
                          curve_points: int,
                          endpoint_lambda: float = DEFAULT_ENDPOINT_LAMBDA) -> Optional[Dict]:
    """Every (draw, k-means seed) replicate, aggregated to mean and std."""
    if not embeddings:
        return None
    records = []
    for draw, (reduced, n_q) in enumerate(embeddings):
        n_p = int(reduced.shape[0]) - int(n_q)
        k = int(num_clusters) if num_clusters else auto_num_clusters(n_p, n_q)
        for s in range(max(1, int(num_seeds))):
            records.append(mauve_from_embedding(
                reduced, n_q, num_clusters=k, seed=int(seed) + draw * 1009 + s,
                scaling_factor=scaling_factor, curve_points=curve_points,
                endpoint_lambda=endpoint_lambda))
    return _aggregate(records)


# ---------------------------------------------------------------------------
# choosing the cluster count
# ---------------------------------------------------------------------------


def _sweep_entry(gen: Dict, ref: Optional[Dict]) -> Dict[str, float]:
    """One (biome, k) cell of the sweep.

    The quantity to read is `ceiling_headroom`, how far the real-vs-real ceiling
    sits below 1.  A k whose ceiling is 0.99 cannot resolve anything: every model
    is pinned against it and the ranking is decided by replicate noise.

    A gap divided by its pooled std is deliberately not reported.  At small k
    that ratio is large for the wrong reason -- the replicate spread collapses
    faster than the gap does, so a k with no usable range scores best on it.
    """
    entry = {
        'mauve': float(gen['mauve']),
        'mauve_std': float(gen['mauve_std']),
        'fidelity': float(gen['fidelity']),
        'coverage': float(gen['coverage']),
        'reference_mauve': None if ref is None else float(ref['mauve']),
        'reference_mauve_std': None if ref is None else float(ref['mauve_std']),
        'reference_fidelity': None if ref is None else float(ref['fidelity']),
        'reference_coverage': None if ref is None else float(ref['coverage']),
        'ceiling_headroom': None if ref is None else float(1.0 - ref['mauve']),
    }
    return entry


def _summarize_sweep(sweep: Dict[int, Dict[str, Dict[str, float]]]) -> Dict[str, Dict]:
    """Macro-average each sweep column so the k choice can be read off one table."""
    summary = {}
    for k in sorted(sweep):
        cells = list(sweep[k].values())
        if not cells:
            continue
        refs = [c['reference_mauve'] for c in cells if c['reference_mauve'] is not None]
        mean_mauve = float(np.mean([c['mauve'] for c in cells]))
        mean_ref = float(np.mean(refs)) if refs else None
        summary[str(k)] = {
            'mauve': mean_mauve,
            'reference_mauve': mean_ref,
            'gap_to_ceiling': None if mean_ref is None else mean_ref - mean_mauve,
            'ceiling_headroom': None if mean_ref is None else 1.0 - mean_ref,
        }
    return summary


def _print_sweep(summary: Dict[str, Dict]) -> None:
    if not summary:
        return
    print('[mauve3d] cluster-count sweep (macro-averaged over biomes)')
    print(f"[mauve3d]   {'k':>5}  {'mauve':>8}  {'real-vs-real':>13}  "
          f"{'ceiling hr':>11}  {'gap':>8}")
    for k, row in summary.items():
        ref = '    n/a' if row['reference_mauve'] is None else f"{row['reference_mauve']:13.4f}"
        hr = ('        n/a' if row['ceiling_headroom'] is None
              else f"{row['ceiling_headroom']:11.4f}")
        gap = '     n/a' if row['gap_to_ceiling'] is None else f"{row['gap_to_ceiling']:8.4f}"
        print(f"[mauve3d]   {k:>5}  {row['mauve']:8.4f}  {ref}  {hr}  {gap}")
    print('[mauve3d]   pick the smallest k whose real-vs-real ceiling is clear of 1.0;')
    print('[mauve3d]   a ceiling above ~0.95 leaves no range for models to differ in.')


# ---------------------------------------------------------------------------
# per-biome evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_per_biome_mauve3d(
    samples_dir: str,
    *,
    checkpoint: str,
    ref_dataset: str,
    mappings_path: Optional[str] = None,
    samples_glob: str = '*.pt',
    samples_format: str = 'block_ids',
    ref_values: str = 'indices',
    gen_out_base_dir: str = 'renders/per_biome_gen',
    run_id: Optional[str] = None,
    samples_per_biome: Optional[int] = None,
    ref_per_biome: Optional[int] = None,
    real_side: str = 'all',
    num_draws: int = 5,
    num_seeds: int = 5,
    num_clusters: Optional[int] = None,
    cluster_sweep: Sequence[int] = (),
    feature_norm: str = 'l2',
    explained_variance: float = DEFAULT_EXPLAINED_VARIANCE,
    scaling_factor: float = DEFAULT_SCALING_FACTOR,
    curve_points: int = DEFAULT_CURVE_POINTS,
    endpoint_lambda: float = DEFAULT_ENDPOINT_LAMBDA,
    batch_size: int = 64,
    device: Optional[str] = None,
    label: str = MAUVE3D_LABEL,
    cache_dir: Optional[str] = 'fid3d_cache',
    refresh_cache: bool = False,
    seed: int = 0,
    model=None,
    converter=None,
    ref_features: Optional[Dict[str, np.ndarray]] = None,
    save_metrics_path: Optional[str] = None,
) -> Dict:
    """Per-biome MAUVE for a directory of generated voxel dumps.

    `model`, `converter` and `ref_features` can be passed in by a caller that is
    scoring several models, so the extractor is loaded and the reference is
    featurized once instead of once per model.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.isdir(str(samples_dir)):
        raise FileNotFoundError(f'samples_dir not found: {samples_dir}')
    if ref_dataset in (None, ''):
        raise ValueError('ref_dataset is required: point it at the val split of the '
                         'dataset the extractor was trained on')

    if model is None or converter is None:
        model, converter, resolved_mappings, weight_hash = load_extractor_and_converter(
            checkpoint, mappings_path, device=device)
    else:
        resolved_mappings = str(mappings_path or '')
        weight_hash = model.weight_hash()

    if ref_features is None:
        ref_features = reference_features_by_biome(
            str(ref_dataset), model, converter,
            ref_values=ref_values, per_biome=ref_per_biome, batch_size=batch_size,
            device=device, seed=seed,
            cache_path=reference_cache_path(cache_dir, str(ref_dataset), weight_hash,
                                            ref_per_biome, seed),
            refresh_cache=refresh_cache,
        )
    if not ref_features:
        raise RuntimeError(f'no reference features built from {ref_dataset}')

    files = sorted(glob.glob(os.path.join(str(samples_dir), samples_glob or '*.pt')))
    if not files:
        raise FileNotFoundError(
            f'no sample files matched {os.path.join(str(samples_dir), samples_glob)}. '
            'This metric needs the raw voxel dumps, not rendered images.')
    print(f'[mauve3d] found {len(files)} sample files in {samples_dir}')

    resolved_run_id = run_id or os.path.basename(os.path.normpath(str(samples_dir)))
    out_root = os.path.join(str(gen_out_base_dir), str(resolved_run_id))
    os.makedirs(out_root, exist_ok=True)

    lut = build_index_lut(converter) if samples_format == 'block_ids' else None
    ref_by_key = {normalize_biome(name): name for name in ref_features}
    grouped = group_samples_by_biome(files, converter, ref_by_key)

    reduce_kw = dict(feature_norm=feature_norm, explained_variance=float(explained_variance))
    score_kw = dict(num_seeds=int(num_seeds), seed=int(seed),
                    scaling_factor=float(scaling_factor), curve_points=int(curve_points),
                    endpoint_lambda=float(endpoint_lambda))

    scores: Dict[str, Dict] = {}
    references: Dict[str, Dict] = {}
    curves: Dict[str, np.ndarray] = {}
    ref_curves: Dict[str, np.ndarray] = {}
    counts: Dict[str, int] = {}
    ref_counts: Dict[str, int] = {}
    sweep: Dict[int, Dict[str, Dict[str, float]]] = {int(k): {} for k in cluster_sweep}

    for biome_name in sorted(grouped):
        samples = grouped[biome_name]
        if samples_per_biome not in (None, 0):
            samples = samples[:int(samples_per_biome)]
        n_gen = int(samples.shape[0])
        if n_gen < 10:
            # Below ten chunks the auto rule collapses to the two-cluster minimum
            # and the histogram carries almost no information.
            print(f'[mauve3d][warn] {biome_name} has {n_gen} samples - skipping')
            continue

        gen_feats = extract_features(
            model,
            (to_index_volume(chunk, lut) for chunk in torch.split(samples, batch_size)),
            device=device,
        )
        real_feats = ref_features[biome_name]

        gen_emb = build_embeddings(
            generated_pairs(real_feats, gen_feats, real_side=real_side,
                            num_draws=int(num_draws), seed=int(seed), tag=biome_name),
            **reduce_kw)
        ref_emb = build_embeddings(
            reference_pairs(real_feats, n_gen, real_side=real_side,
                            num_draws=int(num_draws), seed=int(seed), tag=biome_name),
            **reduce_kw)

        gen_res = mauve_over_embeddings(gen_emb, num_clusters=num_clusters, **score_kw)
        ref_res = mauve_over_embeddings(ref_emb, num_clusters=num_clusters, **score_kw)

        scores[biome_name] = gen_res
        curves[biome_name] = gen_res.pop('curve')
        counts[biome_name] = n_gen
        ref_counts[biome_name] = int(real_feats.shape[0])
        if ref_res is not None:
            references[biome_name] = ref_res
            ref_curves[biome_name] = ref_res.pop('curve')

        for k in sweep:
            g = mauve_over_embeddings(gen_emb, num_clusters=int(k), **score_kw)
            r = mauve_over_embeddings(ref_emb, num_clusters=int(k), **score_kw)
            sweep[k][biome_name] = _sweep_entry(g, r)

        ceiling = references.get(biome_name, {}).get('mauve')
        normalized = (gen_res['mauve'] / ceiling) if ceiling else float('nan')
        ref_txt = (f"ref {ceiling:.3f}   norm {normalized:.3f}" if ceiling
                   else 'ref n/a')
        print(f"[mauve3d] {biome_name:<20} mauve {gen_res['mauve']:.3f} "
              f"+/- {gen_res['mauve_std']:.3f}   {ref_txt}   "
              f"fidelity {gen_res['fidelity']:.3f}   coverage {gen_res['coverage']:.3f}   "
              f"k={gen_res['num_clusters']}   gen={n_gen} real={real_feats.shape[0]}")

    if not scores:
        raise RuntimeError('no biomes were scored')

    biome_names = sorted(scores)
    result = _summarize(
        biome_names, scores, references, counts, ref_counts,
        model=model, weight_hash=weight_hash, checkpoint=str(checkpoint),
        ref_dataset=str(ref_dataset), mappings_path=resolved_mappings,
        real_side=real_side, num_draws=int(num_draws), num_seeds=int(num_seeds),
        feature_norm=feature_norm, explained_variance=float(explained_variance),
        scaling_factor=float(scaling_factor), curve_points=int(curve_points),
        endpoint_lambda=float(endpoint_lambda), seed=int(seed),
    )
    result['cluster_sweep'] = {str(k): v for k, v in sweep.items()}
    result['cluster_sweep_summary'] = _summarize_sweep(sweep)
    result['out_root'] = out_root

    metrics_path = save_metrics_path or os.path.join(out_root, metrics_filename_for_label(label))
    result['metrics_path'] = metrics_path
    _write_npz(metrics_path, result, biome_names, scores, references, curves, ref_curves)

    print(f"[mauve3d] macro-average over {len(biome_names)} biomes: "
          f"{result['overall_mean']:.4f} +/- {result['overall_std']:.4f}"
          + (f"   (real-vs-real reference {result['reference_overall_mean']:.4f}, "
             f"normalized {result['overall_normalized']:.4f})"
             if result['reference_overall_mean'] is not None else ''))
    _print_sweep(result['cluster_sweep_summary'])

    ceiling = result['reference_overall_mean']
    if ceiling is not None and ceiling > 0.98:
        # k scales with the generated count, so a small generated set buys a
        # coarse quantization, and a coarse quantization cannot tell two
        # plausible models apart no matter how different they are.
        print(f'[mauve3d][warn] the real-vs-real ceiling is {ceiling:.4f}, so this '
              f"quantization (k={min(result['num_clusters'].values())}) leaves almost no "
              'headroom and will not separate models. Raise --num_clusters -- the sweep '
              'above shows which value is most discriminative at this sample count.')
    return result


def _summarize(biome_names, scores, references, counts, ref_counts, *, model,
               weight_hash, checkpoint, ref_dataset, mappings_path, real_side,
               num_draws, num_seeds, feature_norm, explained_variance,
               scaling_factor, curve_points, endpoint_lambda, seed) -> Dict:
    """Macro-average across biomes, with the spread taken the right way round.

    The std of the macro-average is computed by averaging across biomes *within*
    each replicate and then taking the spread across replicates, not by averaging
    the per-biome stds -- replicate r means the same draw and k-means seed in
    every biome, so the aggregate is a real quantity with a real variance.
    """
    per_biome = {b: float(scores[b]['mauve']) for b in biome_names}
    overall_mean = float(np.mean([per_biome[b] for b in biome_names]))

    replicate_counts = {len(scores[b]['mauve_values']) for b in biome_names}
    if len(replicate_counts) == 1:
        matrix = np.array([scores[b]['mauve_values'] for b in biome_names], dtype=np.float64)
        macro = matrix.mean(axis=0)
        overall_std = float(macro.std(ddof=1)) if macro.size > 1 else 0.0
    else:
        overall_std = float(np.mean([scores[b]['mauve_std'] for b in biome_names]))

    ref_mean = None
    overall_normalized = None
    if references:
        ref_mean = float(np.mean([references[b]['mauve'] for b in references]))
        if ref_mean > 0:
            overall_normalized = overall_mean / ref_mean

    normalized = {}
    for b in biome_names:
        ceiling = references.get(b, {}).get('mauve')
        if ceiling:
            normalized[b] = float(per_biome[b] / ceiling)

    return {
        'biome_names': biome_names,
        'mauve': per_biome,
        'mauve_std': {b: float(scores[b]['mauve_std']) for b in biome_names},
        'fidelity': {b: float(scores[b]['fidelity']) for b in biome_names},
        'coverage': {b: float(scores[b]['coverage']) for b in biome_names},
        'js_divergence': {b: float(scores[b]['js_divergence']) for b in biome_names},
        'num_clusters': {b: int(scores[b]['num_clusters']) for b in biome_names},
        'reference_mauve': {b: float(references[b]['mauve']) for b in references},
        'reference_mauve_std': {b: float(references[b]['mauve_std']) for b in references},
        # The reference is real-vs-real, so its two terms are drawn from the same
        # distribution and should come out level.  A model whose fidelity and
        # coverage differ is only interesting against that symmetric baseline.
        'reference_fidelity': {b: float(references[b]['fidelity']) for b in references},
        'reference_coverage': {b: float(references[b]['coverage']) for b in references},
        'mauve_normalized': normalized,
        'overall_mean': overall_mean,
        'overall_std': overall_std,
        'reference_overall_mean': ref_mean,
        'overall_normalized': overall_normalized,
        'counts': {b: int(counts[b]) for b in biome_names},
        'ref_counts': {b: int(ref_counts[b]) for b in biome_names},
        'num_replicates': {b: int(scores[b]['num_replicates']) for b in biome_names},
        'protocol': {
            'real_side': real_side,
            'num_draws': num_draws,
            'num_seeds': num_seeds,
            'feature_norm': feature_norm,
            'explained_variance': explained_variance,
            'scaling_factor': scaling_factor,
            'curve_points': curve_points,
            'endpoint_lambda': endpoint_lambda,
            'seed': seed,
        },
        'feature_dim': int(model.feature_dim),
        'weight_hash': weight_hash,
        'checkpoint': checkpoint,
        'ref_dataset': ref_dataset,
        'mappings_path': mappings_path,
    }


def _write_npz(path, result, biome_names, scores, references, curves, ref_curves):
    """Arrays too bulky for JSON: per-replicate values and the frontier curves."""
    try:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        payload = {}
        for b in biome_names:
            payload[f'curve__{b}'] = curves[b].astype(np.float32)
            payload[f'replicates__{b}'] = np.array(scores[b]['mauve_values'], dtype=np.float32)
            if b in ref_curves:
                payload[f'ref_curve__{b}'] = ref_curves[b].astype(np.float32)
                payload[f'ref_replicates__{b}'] = np.array(
                    references[b]['mauve_values'], dtype=np.float32)
        np.savez_compressed(
            path,
            biome_names=np.array(biome_names),
            mauve=np.array([result['mauve'][b] for b in biome_names], dtype=np.float32),
            mauve_std=np.array([result['mauve_std'][b] for b in biome_names], dtype=np.float32),
            fidelity=np.array([result['fidelity'][b] for b in biome_names], dtype=np.float32),
            coverage=np.array([result['coverage'][b] for b in biome_names], dtype=np.float32),
            reference_mauve=np.array([result['reference_mauve'].get(b, np.nan)
                                      for b in biome_names], dtype=np.float32),
            reference_fidelity=np.array([result['reference_fidelity'].get(b, np.nan)
                                         for b in biome_names], dtype=np.float32),
            reference_coverage=np.array([result['reference_coverage'].get(b, np.nan)
                                         for b in biome_names], dtype=np.float32),
            counts=np.array([result['counts'][b] for b in biome_names], dtype=np.int32),
            ref_counts=np.array([result['ref_counts'][b] for b in biome_names], dtype=np.int32),
            overall_mean=np.array([result['overall_mean']], dtype=np.float32),
            overall_std=np.array([result['overall_std']], dtype=np.float32),
            weight_hash=np.array(result['weight_hash']),
            checkpoint=np.array(result['checkpoint']),
            ref_dataset=np.array(result['ref_dataset']),
            **payload,
        )
    except Exception as e:
        print(f'[mauve3d][warn] failed to write {path}: {e}')


def write_summary_json(res: Dict, samples_dir: str, label: str) -> str:
    """JSON summary alongside the npz, mirroring per_biome_fid_<label>.json."""
    path = os.path.join(os.path.dirname(res['metrics_path']), summary_filename_for_label(label))
    body = {k: v for k, v in res.items() if k not in ('out_root',)}
    body['timestamp'] = int(time.time())
    body['samples_dir'] = str(samples_dir)
    body['metric_mode'] = str(label)
    try:
        with open(path, 'w') as f:
            json.dump(body, f, indent=2)
    except Exception as e:
        print(f'[mauve3d][warn] failed to write {path}: {e}')
    return path


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------


def run_selftest() -> None:
    """Check the implementation behaves the way the definition says it should."""
    rng = np.random.default_rng(0)
    dim, n = 64, 400

    identical_a = rng.normal(size=(n, dim))
    identical_b = rng.normal(size=(n, dim))
    same = mauve_from_features(identical_a, identical_b, seed=0)

    shifted = rng.normal(size=(n, dim)) + 4.0
    far = mauve_from_features(identical_a, shifted, seed=0)

    # Two mirror-image failures, which together pin down which coordinate reports
    # which. First: a model reaching only one of two real modes. Everything it
    # makes is realistic, but half the real distribution is unreachable.
    mode_a = rng.normal(size=(n // 2, dim)) - 3.0
    mode_b = rng.normal(size=(n // 2, dim)) + 3.0
    bimodal_real = np.vstack([mode_a, mode_b])
    collapsed = rng.normal(size=(n, dim)) - 3.0
    collapse = mauve_from_features(bimodal_real, collapsed, seed=0)

    # Second: a model that reaches everything real but puts half its mass
    # somewhere the real data never goes. This is the failure a Frechet distance
    # cannot tell apart from the first one.
    unimodal_real = rng.normal(size=(n, dim))
    scattered = np.vstack([rng.normal(size=(n // 2, dim)),
                           rng.normal(size=(n // 2, dim)) + 8.0])
    off_manifold = mauve_from_features(unimodal_real, scattered, seed=0)

    for name, res in (('same distribution', same), ('shifted distribution', far),
                      ('mode collapse', collapse), ('off-manifold mass', off_manifold)):
        print(f"[selftest] {name:<22} mauve {res['mauve']:.4f}  "
              f"fidelity {res['fidelity']:.3f}  coverage {res['coverage']:.3f}")

    assert same['mauve'] > 0.9, same['mauve']
    # Endpoint terms fluctuate more than the frontier area across sklearn
    # versions because k-means can split finite samples differently.
    assert min(same['fidelity'], same['coverage']) > 0.5, same
    assert far['mauve'] < 0.1, far['mauve']
    assert collapse['mauve'] < same['mauve']
    assert off_manifold['mauve'] < same['mauve']
    assert collapse['coverage'] < collapse['fidelity'], \
        'a model missing half the real modes must lose more coverage than fidelity'
    assert off_manifold['fidelity'] < off_manifold['coverage'], \
        'a model generating off-manifold chunks must lose more fidelity than coverage'

    # The two cases are constructed as exact mirrors of each other, so reading the
    # error types from opposite ends of the curve has to mirror them too. Without
    # this the pair of inequalities above could both hold while the two terms
    # measured different things at different magnitudes.
    assert abs(collapse['coverage'] - off_manifold['fidelity']) < 0.1, \
        (collapse['coverage'], off_manifold['fidelity'])
    assert abs(collapse['fidelity'] - off_manifold['coverage']) < 0.1, \
        (collapse['fidelity'], off_manifold['coverage'])

    # Real-vs-real is symmetric by construction, so a lopsided decomposition here
    # would mean the asymmetry comes from the estimator and not from the model.
    assert abs(same['fidelity'] - same['coverage']) < 0.1, same

    try:
        import mauve as mauve_pkg
    except ImportError:
        print('[selftest] mauve package not installed; skipped the cross-check')
    else:
        for name, (p, q) in (('same', (identical_a, identical_b)),
                             ('shifted', (identical_a, shifted))):
            ours = mauve_from_features(p, q, seed=25)['mauve']
            theirs = mauve_pkg.compute_mauve(
                p_features=p, q_features=q, verbose=False,
                num_buckets=auto_num_clusters(p.shape[0], q.shape[0]),
                mauve_scaling_factor=DEFAULT_SCALING_FACTOR).mauve
            print(f'[selftest] {name}: ours {ours:.4f}  mauve package {theirs:.4f}')
            assert abs(ours - theirs) < 0.05, (ours, theirs)

    print('[selftest] ok')


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------


def build_argparser():
    import argparse

    p = argparse.ArgumentParser(
        description='Per-biome voxel-native MAUVE using the trained 3D feature extractor.')
    p.add_argument('--samples_dir', nargs='+', default=None,
                   help="One or more directories of voxel dumps ('generated_<biome>.pt', "
                        "'<biome>_samples_*.pt', or labelled trainer dumps). Passing "
                        'several scores them all against one featurization of the '
                        'reference set. Not needed with --selftest.')
    p.add_argument('--checkpoint', default=None,
                   help='feature_extractor_best.pt from train_fid3d.py. Use the same '
                        'checkpoint as the 3D-FID numbers in the paper.')
    p.add_argument('--ref_dataset', default=None,
                   help='Reference chunks: a memmap dataset dir or a .pt dataset. '
                        'Use the val split of the extractor training composition.')
    p.add_argument('--mappings_path', default=None,
                   help='Defaults to the path recorded in the checkpoint.')
    p.add_argument('--samples_glob', default='*.pt')
    p.add_argument('--samples_format', choices=['block_ids', 'indices'], default='block_ids')
    p.add_argument('--ref_values', choices=['indices', 'block_ids'], default='indices')
    p.add_argument('--out_dir', default='renders/per_biome_gen',
                   help='Results go to <out_dir>/<run_id>/.')
    p.add_argument('--run_id', default=None,
                   help='Only valid with a single --samples_dir.')
    p.add_argument('--samples_per_biome', type=int, default=0)
    p.add_argument('--ref_per_biome', type=int, default=0,
                   help='Cap real chunks per biome; 0 uses the whole split. Match the '
                        'value used for the 3D-FID run so the feature cache is reused.')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--device', default=None)
    p.add_argument('--label', default=MAUVE3D_LABEL,
                   help='Filename tag: per_biome_mauve_<label>.{npz,json}.')
    p.add_argument('--cache_dir', default='fid3d_cache',
                   help='Shared with fid3d_eval.py; identical settings reuse its cache.')
    p.add_argument('--refresh_cache', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--save_metrics_path', default=None)
    p.add_argument('--selftest', action='store_true',
                   help='Run the MAUVE implementation checks and exit.')

    m = p.add_argument_group('mauve protocol')
    m.add_argument('--real_side', choices=['all', 'matched'], default='all',
                   help="'all' scores the generated set against every real chunk of that "
                        "biome, matching the 3D-FID protocol. 'matched' subsamples the "
                        'real side to the generated count, matching the original MAUVE '
                        'setup. The real-vs-real reference follows whichever is chosen.')
    m.add_argument('--num_draws', type=int, default=5,
                   help='Random subsamples of the real chunks. Always applies to the '
                        "real-vs-real reference; applies to the model only under "
                        "--real_side matched, where the real side is resampled.")
    m.add_argument('--num_seeds', type=int, default=5,
                   help='k-means initializations per draw.')
    m.add_argument('--num_clusters', type=int, default=0,
                   help='0 uses the standard rule, min(|real|,|gen|)/10.')
    m.add_argument('--cluster_sweep', default='5,10,20,50',
                   help='Comma-separated cluster counts to additionally report, or "none". '
                        'On by default because k is the one free hyperparameter and the '
                        'standard rule (k = |gen|/10 = 10 here) is coarse enough at 100 '
                        'samples that the real-vs-real ceiling can sit near 1.0, leaving '
                        'nothing to measure. The embeddings are shared, so each extra k '
                        'only costs its k-means fits.')
    m.add_argument('--feature_norm', choices=['l2', 'l1', 'none'], default='l2',
                   help="Feature normalization before PCA; 'l2' matches the reference "
                        'MAUVE implementation.')
    m.add_argument('--explained_variance', type=float, default=DEFAULT_EXPLAINED_VARIANCE,
                   help='PCA components are kept up to this cumulative explained variance.')
    m.add_argument('--scaling_factor', type=float, default=DEFAULT_SCALING_FACTOR,
                   help='c in exp(-c * KL); 5 is the published default.')
    m.add_argument('--curve_points', type=int, default=DEFAULT_CURVE_POINTS,
                   help='Mixture weights along the divergence curve.')
    m.add_argument('--endpoint_lambda', type=float, default=DEFAULT_ENDPOINT_LAMBDA,
                   help='How far inside the ends of the curve the fidelity (Type I) and '
                        'coverage (Type II) terms are read. The ends themselves are '
                        'infinite whenever one support misses the other.')
    m.add_argument('--baseline_only', action='store_true',
                   help='Score only the real-vs-real reference and exit, without any '
                        'model. Needs --ref_dataset and --checkpoint but not '
                        '--samples_dir. Answers what this protocol returns when both '
                        'sides are genuinely real, which is the ceiling any model is '
                        'read against.')
    m.add_argument('--baseline_query_size', type=int, default=100,
                   help='Chunks on the query side of the real-vs-real baseline. Set it to '
                        'the generated count so the baseline carries the same '
                        'small-sample optimism as the model scores.')
    calibration = p.add_argument_group('calibration')
    calibration.add_argument(
        '--calibrate', action='store_true',
        help='Score no generator; compare real biomes to each other and score '
             'structure-free shuffled/marginal/uniform chunks.')
    calibration.add_argument(
        '--calibration_n', type=int, default=100,
        help='Query chunks per biome; match the generated sample count.')
    calibration.add_argument(
        '--calibration_kinds', default=','.join(NOISE_KINDS),
        help="Comma-separated noise types, or 'none'.")
    calibration.add_argument(
        '--no_cross_biome', action='store_true',
        help='Skip the real-biome-vs-real-biome matrices.')
    calibration.add_argument(
        '--calibration_out', default='analysis/mauve3d_calibration')
    return p


def run_baseline_only(ref_features: Dict[str, np.ndarray], *, query_size: int,
                      num_clusters: Optional[int], cluster_sweep: Sequence[int],
                      real_side: str, num_draws: int, num_seeds: int, feature_norm: str,
                      explained_variance: float, scaling_factor: float, curve_points: int,
                      endpoint_lambda: float, seed: int, out_path: str,
                      weight_hash: str, checkpoint: str, ref_dataset: str) -> Dict:
    """Score real-vs-real on its own: `query_size` real chunks against the rest.

    This is the same reference every model run already computes, so the numbers
    reproduce those `reference_mauve` entries exactly given the same protocol.  It
    is worth having standalone because the ceiling depends only on the real
    features, the query size, and k -- never on a model -- so it can be swept over
    k once instead of being recomputed inside every model's run.

    A ceiling well below 1 is not a bug.  It is the score two genuinely
    identical distributions earn at this sample count and quantization, and it is
    the number a model's score has to be read against.
    """
    ks: List[Optional[int]] = [num_clusters]
    ks += [int(k) for k in cluster_sweep if int(k) != num_clusters]

    reduce_kw = dict(feature_norm=feature_norm, explained_variance=float(explained_variance))
    score_kw = dict(num_seeds=int(num_seeds), seed=int(seed),
                    scaling_factor=float(scaling_factor), curve_points=int(curve_points),
                    endpoint_lambda=float(endpoint_lambda))

    per_k: Dict[str, Dict[str, Dict[str, float]]] = {}
    skipped: List[str] = []

    for biome_name in sorted(ref_features):
        real_feats = ref_features[biome_name]
        pairs = reference_pairs(real_feats, int(query_size), real_side=real_side,
                                num_draws=int(num_draws), seed=int(seed), tag=biome_name)
        if not pairs:
            print(f'[mauve3d][warn] {biome_name} has only {real_feats.shape[0]} real '
                  f'chunks, too few to split at {query_size} - skipping')
            skipped.append(biome_name)
            continue

        emb = build_embeddings(pairs, **reduce_kw)
        n_p = int(emb[0][0].shape[0]) - int(emb[0][1])
        print(f'[mauve3d] {biome_name:<20} real-vs-real  query={query_size} '
              f'rest={n_p}')
        for k in ks:
            res = mauve_over_embeddings(emb, num_clusters=k, **score_kw)
            key = str(int(res['num_clusters']))
            per_k.setdefault(key, {})[biome_name] = {
                'mauve': float(res['mauve']),
                'mauve_std': float(res['mauve_std']),
                'fidelity': float(res['fidelity']),
                'coverage': float(res['coverage']),
            }
            print(f"[mauve3d]     k={key:>4}  mauve {res['mauve']:.4f} "
                  f"+/- {res['mauve_std']:.4f}   headroom {1.0 - res['mauve']:.4f}   "
                  f"fidelity {res['fidelity']:.3f}   coverage {res['coverage']:.3f}")

    if not per_k:
        raise RuntimeError('no biome had enough real chunks for the baseline')

    macro = {}
    for key in sorted(per_k, key=lambda s: int(s)):
        cells = list(per_k[key].values())
        macro[key] = {
            'mauve': float(np.mean([c['mauve'] for c in cells])),
            'headroom': float(1.0 - np.mean([c['mauve'] for c in cells])),
            'fidelity': float(np.mean([c['fidelity'] for c in cells])),
            'coverage': float(np.mean([c['coverage'] for c in cells])),
            'num_biomes': len(cells),
        }

    print('\n[mauve3d] real-vs-real ceiling, macro-averaged over biomes')
    print(f"[mauve3d]   {'k':>5}  {'ceiling':>8}  {'headroom':>9}  "
          f"{'fidelity':>9}  {'coverage':>9}")
    for key, row in macro.items():
        print(f"[mauve3d]   {key:>5}  {row['mauve']:8.4f}  {row['headroom']:9.4f}  "
              f"{row['fidelity']:9.4f}  {row['coverage']:9.4f}")

    usable = [key for key, row in macro.items() if row['mauve'] < 0.95]
    if usable:
        print(f"[mauve3d]   k with a ceiling under 0.95 (usable range): "
              f"{', '.join(usable)}")
    else:
        print('[mauve3d][warn] every k leaves the ceiling above 0.95; models will be '
              'pinned against it and the ranking will be replicate noise. Raise k.')

    result = {
        'baseline_query_size': int(query_size),
        'per_biome_by_k': per_k,
        'macro_by_k': macro,
        'skipped_biomes': skipped,
        'protocol': {
            'real_side': real_side,
            'num_draws': int(num_draws),
            'num_seeds': int(num_seeds),
            'feature_norm': feature_norm,
            'explained_variance': float(explained_variance),
            'scaling_factor': float(scaling_factor),
            'curve_points': int(curve_points),
            'endpoint_lambda': float(endpoint_lambda),
            'seed': int(seed),
        },
        'weight_hash': weight_hash,
        'checkpoint': checkpoint,
        'ref_dataset': ref_dataset,
    }

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'[mauve3d] wrote {out_path}')
    return result


def _calibration_cell(result: Dict) -> Dict[str, object]:
    """Drop the averaged curve but retain scalar replicate distributions."""
    cell: Dict[str, object] = {
        'num_clusters': int(result['num_clusters']),
        'num_replicates': int(result['num_replicates']),
    }
    for key in SCALAR_KEYS:
        cell[key] = float(result[key])
        cell[f'{key}_std'] = float(result[f'{key}_std'])
        cell[f'{key}_values'] = [
            float(value) for value in result[f'{key}_values']]
    return cell


def _score_calibration_pairs(
    pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    ks: Sequence[Optional[int]],
    feature_norm: str,
    explained_variance: float,
    num_seeds: int,
    seed: int,
    scaling_factor: float,
    curve_points: int,
    endpoint_lambda: float,
) -> Dict[str, Dict[str, object]]:
    embeddings = build_embeddings(
        pairs,
        feature_norm=feature_norm,
        explained_variance=float(explained_variance),
    )
    scores: Dict[str, Dict[str, object]] = {}
    for requested_k in ks:
        result = mauve_over_embeddings(
            embeddings,
            num_seeds=int(num_seeds),
            num_clusters=requested_k,
            seed=int(seed),
            scaling_factor=float(scaling_factor),
            curve_points=int(curve_points),
            endpoint_lambda=float(endpoint_lambda),
        )
        if result is None:
            continue
        scores[str(int(result['num_clusters']))] = _calibration_cell(result)
    return scores


@torch.no_grad()
def run_calibration(
    ref_features: Dict[str, np.ndarray],
    *,
    model,
    converter,
    ref_dataset: str,
    ref_values: str,
    query_size: int,
    num_clusters: Optional[int],
    cluster_sweep: Sequence[int],
    real_side: str,
    num_draws: int,
    num_seeds: int,
    feature_norm: str,
    explained_variance: float,
    scaling_factor: float,
    curve_points: int,
    endpoint_lambda: float,
    kinds: Sequence[str],
    cross_biome: bool,
    batch_size: int,
    device: str,
    seed: int,
    out_dir: str,
    weight_hash: str,
    checkpoint: str,
    mappings_path: Optional[str],
) -> Dict:
    """Calibrate MAUVE with real-biome substitutions and destroyed geometry."""
    ks: List[Optional[int]] = [num_clusters]
    ks += [int(k) for k in cluster_sweep if int(k) != num_clusters]
    names = sorted(ref_features)

    baseline_by_k: Dict[str, Dict[str, Dict[str, object]]] = {}
    for biome_name in names:
        pairs = reference_pairs(
            ref_features[biome_name], int(query_size),
            real_side=real_side, num_draws=int(num_draws),
            seed=int(seed), tag=f'cal|{biome_name}')
        if not pairs:
            print(f'[mauve3d][cal][warn] {biome_name} has too few reference rows')
            continue
        for key, cell in _score_calibration_pairs(
                pairs, ks=ks, feature_norm=feature_norm,
                explained_variance=explained_variance, num_seeds=num_seeds,
                seed=seed, scaling_factor=scaling_factor,
                curve_points=curve_points, endpoint_lambda=endpoint_lambda).items():
            baseline_by_k.setdefault(key, {})[biome_name] = cell

    if not baseline_by_k:
        raise RuntimeError('no biome had enough reference rows for calibration')
    valid_names = set(names)
    for cells in baseline_by_k.values():
        valid_names &= set(cells)
    names = sorted(valid_names)
    if not names:
        raise RuntimeError('no biome is available under every requested k')

    cross_by_k: Dict[str, Dict[str, Dict[str, Dict[str, object]]]] = {}
    if cross_biome:
        print(f'[mauve3d][cal] --- {len(names)} x {len(names)} real-biome matrix ---')
        for source in names:
            order = _shuffled_order(
                ref_features[source].shape[0], int(seed), f'cal|query|{source}')
            rows = np.asarray(order[:int(query_size)], dtype=np.int64)
            source_query = ref_features[source][rows]
            for target in names:
                if source == target:
                    scored = {
                        key: cells[source]
                        for key, cells in baseline_by_k.items()
                        if source in cells
                    }
                else:
                    pairs = generated_pairs(
                        ref_features[target], source_query,
                        real_side=real_side, num_draws=int(num_draws),
                        seed=int(seed), tag=f'cal|cross|{source}|{target}')
                    scored = _score_calibration_pairs(
                        pairs, ks=ks, feature_norm=feature_norm,
                        explained_variance=explained_variance,
                        num_seeds=num_seeds, seed=seed,
                        scaling_factor=scaling_factor,
                        curve_points=curve_points,
                        endpoint_lambda=endpoint_lambda)
                for key, cell in scored.items():
                    cross_by_k.setdefault(key, {}).setdefault(source, {})[target] = cell

            available_keys = [
                key for key in cross_by_k if source in cross_by_k[key]
                and source in cross_by_k[key][source]]
            if available_keys:
                detail = []
                for key in sorted(available_keys, key=int):
                    cells = cross_by_k[key][source]
                    other = [cell['mauve'] for target, cell in cells.items()
                             if target != source]
                    detail.append(
                        f"k={key} floor={cells[source]['mauve']:.4f} "
                        f"other={np.mean(other):.4f}")
                print(f"[mauve3d][cal] {source:<20} {'   '.join(detail)}")

    noise_by_k: Dict[str, Dict[str, Dict[str, Dict[str, object]]]] = {}
    if kinds:
        print('[mauve3d][cal] --- structure-free queries ---')
        noise_features = structure_free_features(
            str(ref_dataset), model, converter, ref_features,
            kinds=tuple(kinds), n=int(query_size), ref_values=str(ref_values),
            batch_size=int(batch_size), device=device, seed=int(seed))
        for kind in kinds:
            for biome_name in names:
                query = noise_features.get(kind, {}).get(biome_name)
                if query is None:
                    continue
                pairs = generated_pairs(
                    ref_features[biome_name], query,
                    real_side=real_side, num_draws=int(num_draws),
                    seed=int(seed), tag=f'cal|noise|{kind}|{biome_name}')
                scored = _score_calibration_pairs(
                    pairs, ks=ks, feature_norm=feature_norm,
                    explained_variance=explained_variance,
                    num_seeds=num_seeds, seed=seed,
                    scaling_factor=scaling_factor, curve_points=curve_points,
                    endpoint_lambda=endpoint_lambda)
                for key, cell in scored.items():
                    noise_by_k.setdefault(key, {}).setdefault(kind, {})[
                        biome_name] = cell
            for key in sorted(noise_by_k, key=int):
                cells = noise_by_k[key].get(kind, {})
                if cells:
                    print(
                        f'[mauve3d][cal] k={key:<4} mean {kind:<8} '
                        f"MAUVE={np.mean([v['mauve'] for v in cells.values()]):.4f}"
                    )

    all_keys = sorted(
        set(baseline_by_k) | set(cross_by_k) | set(noise_by_k), key=int)
    macro_by_k: Dict[str, Dict[str, object]] = {}
    for key in all_keys:
        baseline = baseline_by_k.get(key, {})
        cross = cross_by_k.get(key, {})
        off_diagonal = [
            cell['mauve']
            for source, row in cross.items()
            for target, cell in row.items()
            if source != target
        ]
        macro_by_k[key] = {
            'same_biome_mauve': (
                float(np.mean([cell['mauve'] for cell in baseline.values()]))
                if baseline else None),
            'different_biome_mauve': (
                float(np.mean(off_diagonal)) if off_diagonal else None),
            'noise_mauve': {
                kind: float(np.mean([cell['mauve'] for cell in cells.values()]))
                for kind, cells in noise_by_k.get(key, {}).items() if cells
            },
        }

    result = {
        'query_size': int(query_size),
        'biomes': names,
        'baseline_by_k': baseline_by_k,
        'cross_biome_by_k': cross_by_k,
        'noise_by_k': noise_by_k,
        'macro_by_k': macro_by_k,
        'protocol': {
            'real_side': real_side,
            'num_draws': int(num_draws),
            'num_seeds': int(num_seeds),
            'feature_norm': feature_norm,
            'explained_variance': float(explained_variance),
            'scaling_factor': float(scaling_factor),
            'curve_points': int(curve_points),
            'endpoint_lambda': float(endpoint_lambda),
            'noise_kinds': list(kinds),
            'seed': int(seed),
        },
        'weight_hash': weight_hash,
        'checkpoint': checkpoint,
        'ref_dataset': ref_dataset,
        'mappings_path': mappings_path,
    }
    tag = (
        f'{os.path.basename(os.path.normpath(str(ref_dataset)))}_'
        f'{weight_hash[:8]}_n{int(query_size)}'
    )
    os.makedirs(str(out_dir), exist_ok=True)
    json_path = os.path.join(str(out_dir), f'calibration_{tag}.json')
    with open(json_path, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
    for key, matrix_rows in cross_by_k.items():
        matrix = np.asarray([
            [matrix_rows[source][target]['mauve'] for target in names]
            for source in names
        ], dtype=np.float64)
        np.savez_compressed(
            os.path.join(str(out_dir), f'cross_biome_{tag}_k{key}.npz'),
            biome_names=np.asarray(names),
            mauve=matrix,
        )
    print('[mauve3d][cal] macro calibration')
    for key, row in macro_by_k.items():
        print(
            f"[mauve3d][cal] k={key:<4} same={row['same_biome_mauve']!s:<8} "
            f"different={row['different_biome_mauve']!s:<8} "
            f"noise={row['noise_mauve']}"
        )
    print(f'[mauve3d][cal] wrote {json_path}')
    return result


def main():
    args = build_argparser().parse_args()

    if args.selftest:
        run_selftest()
        return

    if args.baseline_only and args.calibrate:
        raise SystemExit('--baseline_only and --calibrate are mutually exclusive')

    required = ('checkpoint', 'ref_dataset')
    if not args.baseline_only and not args.calibrate:
        required = ('samples_dir',) + required
    missing = [n for n in required if not getattr(args, n)]
    if missing:
        raise SystemExit('missing required arguments: '
                         + ', '.join(f'--{n}' for n in missing)
                         + ' (or pass --selftest)')
    if not args.baseline_only and not args.calibrate and len(args.samples_dir) > 1:
        for flag in ('run_id', 'save_metrics_path'):
            if getattr(args, flag):
                raise SystemExit(f'--{flag} would collide across models; '
                                 'it only applies to a single --samples_dir')

    sweep_arg = str(args.cluster_sweep).strip().lower()
    sweep = () if sweep_arg in ('', 'none') else tuple(
        int(k) for k in sweep_arg.split(',') if k.strip())
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    cache_dir = None if args.cache_dir in (None, '', 'none') else args.cache_dir
    ref_per_biome = int(args.ref_per_biome) or None

    # Load and featurize once, then reuse across every model being scored.
    model, converter, resolved_mappings, weight_hash = load_extractor_and_converter(
        str(args.checkpoint), args.mappings_path, device=device)
    ref_features = reference_features_by_biome(
        str(args.ref_dataset), model, converter,
        ref_values=args.ref_values, per_biome=ref_per_biome,
        batch_size=int(args.batch_size), device=device, seed=int(args.seed),
        cache_path=reference_cache_path(cache_dir, str(args.ref_dataset), weight_hash,
                                        ref_per_biome, int(args.seed)),
        refresh_cache=bool(args.refresh_cache),
    )

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
        if int(args.calibration_n) < 2:
            raise SystemExit('--calibration_n must be at least 2')
        run_calibration(
            ref_features,
            model=model,
            converter=converter,
            ref_dataset=str(args.ref_dataset),
            ref_values=str(args.ref_values),
            query_size=int(args.calibration_n),
            num_clusters=(int(args.num_clusters) or None),
            cluster_sweep=sweep,
            real_side=str(args.real_side),
            num_draws=int(args.num_draws),
            num_seeds=int(args.num_seeds),
            feature_norm=str(args.feature_norm),
            explained_variance=float(args.explained_variance),
            scaling_factor=float(args.scaling_factor),
            curve_points=int(args.curve_points),
            endpoint_lambda=float(args.endpoint_lambda),
            kinds=kinds,
            cross_biome=not bool(args.no_cross_biome),
            batch_size=int(args.batch_size),
            device=device,
            seed=int(args.seed),
            out_dir=str(args.calibration_out),
            weight_hash=weight_hash,
            checkpoint=str(args.checkpoint),
            mappings_path=resolved_mappings,
        )
        return

    if args.baseline_only:
        out_path = args.save_metrics_path or os.path.join(
            str(args.out_dir), 'mauve3d_real_vs_real_baseline.json')
        run_baseline_only(
            ref_features,
            query_size=int(args.baseline_query_size),
            num_clusters=(int(args.num_clusters) or None),
            cluster_sweep=sweep,
            real_side=str(args.real_side),
            num_draws=int(args.num_draws),
            num_seeds=int(args.num_seeds),
            feature_norm=str(args.feature_norm),
            explained_variance=float(args.explained_variance),
            scaling_factor=float(args.scaling_factor),
            curve_points=int(args.curve_points),
            endpoint_lambda=float(args.endpoint_lambda),
            seed=int(args.seed),
            out_path=out_path,
            weight_hash=weight_hash,
            checkpoint=str(args.checkpoint),
            ref_dataset=str(args.ref_dataset),
        )
        return

    summaries = []
    for samples_dir in args.samples_dir:
        print(f'\n[mauve3d] === {samples_dir} ===')
        res = compute_per_biome_mauve3d(
            samples_dir,
            checkpoint=str(args.checkpoint),
            ref_dataset=str(args.ref_dataset),
            mappings_path=resolved_mappings,
            samples_glob=str(args.samples_glob or '*.pt'),
            samples_format=str(args.samples_format),
            ref_values=str(args.ref_values),
            gen_out_base_dir=args.out_dir,
            run_id=args.run_id,
            samples_per_biome=(int(args.samples_per_biome) or None),
            ref_per_biome=ref_per_biome,
            real_side=str(args.real_side),
            num_draws=int(args.num_draws),
            num_seeds=int(args.num_seeds),
            num_clusters=(int(args.num_clusters) or None),
            cluster_sweep=sweep,
            feature_norm=str(args.feature_norm),
            explained_variance=float(args.explained_variance),
            scaling_factor=float(args.scaling_factor),
            curve_points=int(args.curve_points),
            endpoint_lambda=float(args.endpoint_lambda),
            batch_size=int(args.batch_size),
            device=device,
            label=str(args.label),
            cache_dir=cache_dir,
            seed=int(args.seed),
            model=model,
            converter=converter,
            ref_features=ref_features,
            save_metrics_path=args.save_metrics_path,
        )
        summary_json = write_summary_json(res, samples_dir, str(args.label))
        summaries.append({
            'samples_dir': str(samples_dir),
            'overall_mean': res['overall_mean'],
            'overall_std': res['overall_std'],
            'reference_overall_mean': res['reference_overall_mean'],
            'overall_normalized': res['overall_normalized'],
            'num_biomes': len(res['biome_names']),
            'metrics_path': res['metrics_path'],
            'summary_json': summary_json,
        })

    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
