"""
Voxel-native FID: score generated chunks with the 3D feature extractor from
`train_fid3d.py` instead of rendering them and running Inception.

    python fid3d_eval.py \
      --samples_dir results/md4_run/fid_samples \
      --checkpoint results/fid3d_32_balanced/exp_x/feature_extractor_best.pt \
      --ref_dataset data/balanced_dataset/32_biome_parts_dir_balanced_cc_val_dir

Why this exists.  The 2D pipeline in `calc_fid.py` renders every chunk to a PNG
and feeds it to an ImageNet Inception network, so it only sees what a camera
sees: occluded interiors are invisible, and the metric inherits Inception's
sensitivity to texture and lighting.  This path feeds the voxel grid straight
into a network trained on this data, so the whole volume is scored.

Two sample layouts are accepted: the per-biome files `calc_fid.py` writes
(`generated_<biome>.pt` or `<biome>_samples_*.pt`), and the mixed dumps the
trainers write during training (`samples_step_N.pt`, whose 'labels' entry covers
several biomes in one file and is split automatically).

Values must be *dataset indices* by the time they reach the extractor, because
that is what it was trained on.  Sample dumps hold original Minecraft block IDs,
so the default `samples_format='block_ids'` converts them through the same
mappings file the extractor was trained with; the conversion is a lookup table
rather than `BlockBiomeConverter.convert_to_indices`, which is a per-voxel Python
loop and far too slow here.

Every biome is reported with a real-vs-real floor computed by splitting the
reference chunks, at the same sample count as the generated set.  A distance
below or near that floor means "indistinguishable at this N", not "perfect", and
the two numbers should always be quoted together.

Output mirrors the per-biome format of `calc_fid.py` (`biome_names`, `fid`,
`counts`, `overall_mean` in an .npz plus a JSON summary), tagged with a distinct
label so it never overwrites the 2D results.

`--calibrate` scores no model and instead measures what a distance means on this
reference set: every real biome against every other real biome, and
structure-free volumes against each biome.  Together with the floor those bound
the scale, so a generated distance can be placed on it rather than read raw.
"""

import glob
import json
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from data_utils import BlockBiomeConverter
from fid3d_model import (
    compute_feature_stats,
    extract_features,
    frechet_distance,
    load_feature_extractor,
)

FID3D_LABEL = 'fid3d'
# The reference side defaults to every chunk of that biome in the split, matching
# how calc_fid.py compares against whatever is in the reference directory. Cap it
# only to bound runtime; the cap becomes part of the cache filename because it
# changes the numbers.
DEFAULT_REF_PER_BIOME = None


# ---------------------------------------------------------------------------
# filenames, kept parallel to calc_fid.py's helpers
# ---------------------------------------------------------------------------


def metrics_filename_for_label(label: str) -> str:
    safe = str(label).strip().lower().replace('-', '_')
    return f'per_biome_fid_{safe}.npz'


def summary_filename_for_label(label: str) -> str:
    safe = str(label).strip().lower().replace('-', '_')
    return f'per_biome_fid_{safe}.json'


def infer_biome_name_from_filename(path: str) -> str:
    """Same convention as calc_fid.py: '<biome>_samples_*.pt'."""
    stem = os.path.splitext(os.path.basename(str(path)))[0]
    if stem.startswith('generated_'):
        stem = stem[len('generated_'):]
    if '_samples' in stem:
        return stem.split('_samples', 1)[0]
    return stem


def normalize_biome(name) -> str:
    """Loose key for matching filenames against mapping values.

    Biome names reach us from two directions -- sample filenames and the mappings
    file -- and one side may have dropped a 'minecraft:' prefix or replaced
    punctuation, so compare on a reduced form.
    """
    s = str(name).strip().lower()
    if ':' in s:
        s = s.split(':', 1)[1]
    return ''.join(c if c.isalnum() else '_' for c in s).strip('_')


def biome_name_for_index(converter: BlockBiomeConverter, label: int) -> str:
    mapping = converter.index_to_biome or {}
    for key in (int(label), str(int(label))):
        if key in mapping:
            return str(mapping[key])
    return str(label)


def load_samples_payload(path: str) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return (volumes, biome_labels) from a sample dump.

    Handles both layouts in this repo: the per-biome files calc_fid writes
    ('generated_<biome>.pt', no labels) and the mixed dumps the trainers write
    during training ('samples_step_N.pt', with a 'labels' entry covering several
    biomes in one file).
    """
    payload = torch.load(str(path), map_location='cpu', weights_only=False)
    if isinstance(payload, torch.Tensor):
        return payload, None
    if not isinstance(payload, dict):
        raise ValueError(f'unsupported sample payload {type(payload)} in {path}')

    volumes = None
    for key in ('voxels', 'samples', 'x', 'data'):
        if key in payload and isinstance(payload[key], torch.Tensor):
            volumes = payload[key]
            break
    if volumes is None:
        raise KeyError(f'no sample tensor in {path}; found keys {list(payload.keys())}')

    labels = None
    for key in ('labels', 'classes', 'class_labels', 'biomes'):
        value = payload.get(key)
        if isinstance(value, torch.Tensor) and value.numel() == volumes.shape[0]:
            labels = value.reshape(-1).long()
            break
    return volumes, labels


def load_samples_tensor(path: str) -> torch.Tensor:
    return load_samples_payload(path)[0]


# ---------------------------------------------------------------------------
# block IDs -> dataset indices
# ---------------------------------------------------------------------------


def build_index_lut(converter: BlockBiomeConverter) -> torch.Tensor:
    """Lookup table mapping original block IDs to dataset indices."""
    block_to_index = converter.block_to_index
    if not block_to_index:
        raise ValueError('mappings file has no block_to_index table')
    keys = [int(k) for k in block_to_index.keys()]
    lut = torch.full((max(keys) + 1,), -1, dtype=torch.long)
    for block_id, index in block_to_index.items():
        lut[int(block_id)] = int(index)
    return lut


def to_index_volume(volumes: torch.Tensor, lut: Optional[torch.Tensor]) -> torch.Tensor:
    """Normalize a batch to [B, X, Y, Z] dataset indices."""
    v = volumes
    if v.dim() == 5:                     # one-hot / embedding channels
        v = torch.argmax(v, dim=1)
    if v.dim() == 3:
        v = v.unsqueeze(0)
    if v.dim() != 4:
        raise ValueError(f'expected [N, X, Y, Z] or [N, C, X, Y, Z], got {tuple(volumes.shape)}')
    v = v.long()
    if lut is None:
        return v
    if int(v.min()) < 0 or int(v.max()) >= lut.numel():
        raise ValueError(
            f'block IDs outside the mappings range [0, {lut.numel() - 1}]: '
            f'saw [{int(v.min())}, {int(v.max())}]. Are these already indices? '
            "If so pass samples_format='indices'.")
    out = lut[v]
    if bool((out < 0).any()):
        missing = torch.unique(v[out < 0]).tolist()[:10]
        raise ValueError(
            f'block IDs absent from the mappings file: {missing} (showing up to 10). '
            'The samples were probably generated with a different vocabulary than '
            'the feature extractor was trained on.')
    return out


# ---------------------------------------------------------------------------
# reference chunks
# ---------------------------------------------------------------------------


def _open_reference(ref_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (voxels, biome_label_indices) for a memmap dir or a .pt dataset.

    Voxels stay lazy for the memmap layout so a large split is never fully
    materialized.
    """
    ref_path = str(ref_path)
    manifest_path = os.path.join(ref_path, 'manifest.json')
    if os.path.isdir(ref_path) and os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        voxels = np.load(os.path.join(ref_path, manifest['paths']['voxels']), mmap_mode='r')
        labels = np.load(os.path.join(ref_path, manifest['paths']['biome_labels']), mmap_mode='r')
        labels = np.asarray(labels)
        if labels.ndim == 2:
            labels = labels.argmax(axis=1)
        return voxels, labels.astype(np.int64)

    if not os.path.exists(ref_path):
        raise FileNotFoundError(f'reference dataset not found: {ref_path}')

    payload = torch.load(ref_path, map_location='cpu', weights_only=False)
    if not isinstance(payload, dict) or 'voxels' not in payload:
        raise ValueError(f'{ref_path} is not a dataset dict with a "voxels" entry')
    voxels = payload['voxels']
    labels = payload.get('biomes', payload.get('biome_labels'))
    if labels is None:
        raise ValueError(f'{ref_path} has no biome labels')
    labels = torch.as_tensor(labels)
    if labels.dim() == 2:
        labels = labels.argmax(dim=-1)
    return voxels.numpy() if isinstance(voxels, torch.Tensor) else voxels, \
        labels.long().numpy()


def _batched_index_volumes(voxels, row_indices: np.ndarray, lut, batch_size: int):
    for start in range(0, len(row_indices), batch_size):
        rows = row_indices[start:start + batch_size]
        block = np.asarray(voxels[rows])
        yield to_index_volume(torch.from_numpy(np.ascontiguousarray(block)), lut)


def _validated_index_range(batches, vocab_size: int, tag: str, source: str):
    """Fail loudly if reference indices fall outside the extractor's vocabulary.

    Stored indices are used as-is, and the extractor clamps anything out of range,
    so a split built from different mappings would quietly score the wrong blocks
    instead of erroring.
    """
    for batch in batches:
        lowest = int(batch.min())
        highest = int(batch.max())
        if lowest < 0 or highest >= vocab_size:
            raise ValueError(
                f'{tag}: reference indices [{lowest}, {highest}] in {source} are outside '
                f'the extractor vocabulary [0, {vocab_size - 1}]. That split was built with '
                'different mappings than the extractor was trained on; point the '
                'reference at the val split of the extractor training data.')
        yield batch


def reference_features_by_biome(
    ref_path: str,
    model,
    converter: BlockBiomeConverter,
    *,
    ref_values: str = 'indices',
    per_biome: Optional[int] = DEFAULT_REF_PER_BIOME,
    batch_size: int = 64,
    device: str = 'cuda',
    seed: int = 0,
    cache_path: Optional[str] = None,
    refresh_cache: bool = False,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """Per-biome real features, cached so repeated model comparisons are cheap."""
    if cache_path and os.path.exists(cache_path) and not refresh_cache:
        blob = np.load(cache_path, allow_pickle=False)
        if str(blob['weight_hash'].item()) == model.weight_hash():
            names = [str(n) for n in blob['biome_names'].tolist()]
            feats = {n: blob[f'feat__{n}'] for n in names}
            if verbose:
                print(f'[fid3d] reusing cached reference features from {cache_path}')
            return feats
        if verbose:
            print(f'[fid3d] cache {cache_path} was built with a different extractor; rebuilding')

    voxels, labels = _open_reference(ref_path)
    lut = None if ref_values == 'indices' else build_index_lut(converter)

    features: Dict[str, np.ndarray] = {}
    for label in sorted(set(int(v) for v in labels.tolist())):
        rows = np.flatnonzero(labels == label)
        if rows.size == 0:
            continue
        if per_biome not in (None, 0) and rows.size > int(per_biome):
            rng = random.Random(f'{seed}|{label}')
            rows = np.array(sorted(rng.sample(list(rows), int(per_biome))), dtype=np.int64)
        biome_name = biome_name_for_index(converter, label)
        feats = extract_features(
            model,
            _validated_index_range(
                _batched_index_volumes(voxels, rows, lut, batch_size),
                int(model.vocab_size), biome_name, str(ref_path),
            ),
            device=device,
        )
        features[biome_name] = feats
        if verbose:
            print(f'[fid3d] reference biome {biome_name:<20} {feats.shape[0]} chunks')

    if cache_path:
        try:
            os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
            payload = {f'feat__{n}': v for n, v in features.items()}
            np.savez_compressed(
                cache_path,
                biome_names=np.array(sorted(features.keys())),
                weight_hash=np.array(model.weight_hash()),
                ref_path=np.array(str(ref_path)),
                **payload,
            )
            if verbose:
                print(f'[fid3d] cached reference features to {cache_path}')
        except Exception as e:
            print(f'[fid3d][warn] failed to cache reference features: {e}')

    return features


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def real_vs_real_floor(feats: np.ndarray, source_size: int, *, seed: int = 0,
                       tag: str = '') -> Optional[float]:
    """Frechet distance between two disjoint halves of the real features.

    `source_size` should match the generated sample count so the floor is quoted
    at the same N -- the estimator is biased upward at small N and comparing a
    distance to a floor measured at a different N is meaningless.
    """
    total = int(feats.shape[0])
    if total < source_size + 2:
        return None
    order = list(range(total))
    random.Random(f'{seed}|{tag}').shuffle(order)
    source = np.array(order[:source_size], dtype=np.int64)
    rest = np.array(order[source_size:], dtype=np.int64)
    mu_a, sigma_a = compute_feature_stats(feats[source])
    mu_b, sigma_b = compute_feature_stats(feats[rest])
    return frechet_distance(mu_a, sigma_a, mu_b, sigma_b)


def group_samples_by_biome(
    files: List[str],
    converter: BlockBiomeConverter,
    ref_by_key: Dict[str, str],
) -> Dict[str, torch.Tensor]:
    """Bucket every sample file into biomes, keyed by the reference biome names.

    A file carrying its own labels is split by label; otherwise the biome comes
    from the filename. Files spanning several steps of the same biome are
    concatenated.
    """
    buckets: Dict[str, List[torch.Tensor]] = {}
    unmatched = set()

    def add(name, volumes):
        resolved = ref_by_key.get(normalize_biome(name))
        if resolved is None:
            unmatched.add(str(name))
            return
        buckets.setdefault(resolved, []).append(volumes)

    for path in files:
        volumes, labels = load_samples_payload(path)
        if labels is None:
            add(infer_biome_name_from_filename(path), volumes)
            continue
        for label in sorted(set(int(v) for v in labels.tolist())):
            rows = (labels == label).nonzero(as_tuple=True)[0]
            add(biome_name_for_index(converter, label), volumes[rows])

    if unmatched:
        print(f'[fid3d][warn] no reference chunks for {sorted(unmatched)} - skipped. '
              f'Reference biomes: {sorted(ref_by_key.values())}')
    return {name: (parts[0] if len(parts) == 1 else torch.cat(parts, dim=0))
            for name, parts in buckets.items()}


def load_extractor_and_converter(checkpoint: str, mappings_path: Optional[str] = None,
                                 device: str = 'cuda', verbose: bool = True):
    """Extractor plus the converter for the vocabulary it was trained on."""
    model, payload = load_feature_extractor(str(checkpoint), device=device)
    weight_hash = model.weight_hash()
    if verbose:
        print(f'[fid3d] extractor {checkpoint}')
        print(f'[fid3d] feature_dim={model.feature_dim} vocab={model.vocab_size} '
              f'biomes={model.num_biomes} hash={weight_hash}')
    if payload.get('weight_hash') and payload['weight_hash'] != weight_hash:
        print('[fid3d][warn] weight hash differs from the value stored at save time')

    # The mappings must be the ones the extractor was trained with, otherwise the
    # index space it learned does not line up with the data being scored.
    resolved = mappings_path or payload.get('mappings_file_path')
    if not resolved or not os.path.exists(str(resolved)):
        raise FileNotFoundError(
            f'mappings file not found (got {resolved!r}). Pass --mappings_path; '
            'the checkpoint records the path used during training but it may not '
            'resolve from this working directory.')
    mappings = torch.load(str(resolved), map_location='cpu', weights_only=False)
    converter = BlockBiomeConverter(mappings.get('block_mappings', {}),
                                    mappings.get('biome_mappings', {}))
    if len(converter.block_to_index) != model.vocab_size:
        raise ValueError(
            f'vocabulary mismatch: mappings hold {len(converter.block_to_index)} blocks '
            f'but the extractor was trained on {model.vocab_size}. '
            f'Use the mappings from {payload.get("mappings_file_path")!r}.')
    return model, converter, str(resolved), weight_hash


def reference_cache_path(cache_dir: Optional[str], ref_dataset: str, weight_hash: str,
                         ref_per_biome: Optional[int], seed: int) -> Optional[str]:
    """Cache filename for a reference featurization.

    The extractor and the per-biome cap both change the features, so both belong
    in the name rather than being silently reused.
    """
    if not cache_dir:
        return None
    ref_name = os.path.basename(os.path.normpath(str(ref_dataset)))
    cap = 'all' if ref_per_biome in (None, 0) else int(ref_per_biome)
    # Which chunks get picked only depends on the seed once a cap is active, and
    # seed 0 keeps its existing filename so old caches stay usable.
    seed_tag = '' if cap == 'all' or int(seed) == 0 else f'_s{int(seed)}'
    return os.path.join(str(cache_dir), f'{ref_name}_{weight_hash[:8]}_n{cap}{seed_tag}.npz')


@torch.no_grad()
def compute_per_biome_fid3d(
    samples_dir: str,
    *,
    checkpoint: str,
    ref_dataset: Optional[str] = None,
    mappings_path: Optional[str] = None,
    samples_glob: str = '*.pt',
    samples_format: str = 'block_ids',
    ref_values: str = 'indices',
    gen_out_base_dir: str = 'renders/per_biome_gen',
    run_id: Optional[str] = None,
    samples_per_biome: Optional[int] = None,
    ref_per_biome: Optional[int] = DEFAULT_REF_PER_BIOME,
    batch_size: int = 64,
    device: Optional[str] = None,
    label: str = FID3D_LABEL,
    cache_dir: Optional[str] = 'fid3d_cache',
    refresh_cache: bool = False,
    baseline: bool = True,
    seed: int = 0,
    save_metrics_path: Optional[str] = None,
) -> Dict:
    """Per-biome voxel FID for a directory of '<biome>_samples_*.pt' files."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.isdir(str(samples_dir)):
        raise FileNotFoundError(f'samples_dir not found: {samples_dir}')

    model, converter, resolved_mappings, weight_hash = load_extractor_and_converter(
        checkpoint, mappings_path, device=device)

    if ref_dataset in (None, ''):
        raise ValueError('ref_dataset is required: point it at the val split of the '
                         'dataset the extractor was trained on')

    cache_path = reference_cache_path(cache_dir, str(ref_dataset), weight_hash,
                                      ref_per_biome, seed)

    ref_features = reference_features_by_biome(
        str(ref_dataset), model, converter,
        ref_values=ref_values, per_biome=ref_per_biome, batch_size=batch_size,
        device=device, seed=seed, cache_path=cache_path, refresh_cache=refresh_cache,
    )
    if not ref_features:
        raise RuntimeError(f'no reference features built from {ref_dataset}')

    files = sorted(glob.glob(os.path.join(str(samples_dir), samples_glob or '*.pt')))
    if not files:
        raise FileNotFoundError(
            f'no sample files matched {os.path.join(str(samples_dir), samples_glob)}. '
            'This backend needs the raw voxel dumps, not rendered images.')
    print(f'[fid3d] found {len(files)} sample files in {samples_dir}')

    resolved_run_id = run_id or os.path.basename(os.path.normpath(str(samples_dir)))
    out_root = os.path.join(str(gen_out_base_dir), str(resolved_run_id))
    os.makedirs(out_root, exist_ok=True)

    lut = build_index_lut(converter) if samples_format == 'block_ids' else None
    ref_by_key = {normalize_biome(name): name for name in ref_features}

    grouped = group_samples_by_biome(files, converter, ref_by_key)

    results: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    floors: Dict[str, float] = {}
    ref_counts: Dict[str, int] = {}

    for biome_name in sorted(grouped):
        samples = grouped[biome_name]
        if samples_per_biome not in (None, 0):
            samples = samples[:int(samples_per_biome)]
        n_gen = int(samples.shape[0])
        if n_gen < 2:
            print(f'[fid3d][warn] {biome_name} has {n_gen} samples - skipping')
            continue

        gen_feats = extract_features(
            model,
            (to_index_volume(chunk, lut) for chunk in torch.split(samples, batch_size)),
            device=device,
        )
        real_feats = ref_features[biome_name]
        mu_g, sigma_g = compute_feature_stats(gen_feats)
        mu_r, sigma_r = compute_feature_stats(real_feats)
        score = frechet_distance(mu_g, sigma_g, mu_r, sigma_r)

        results[biome_name] = float(score)
        counts[biome_name] = n_gen
        ref_counts[biome_name] = int(real_feats.shape[0])

        if n_gen < model.feature_dim:
            # A 512-d covariance from fewer than 512 samples is rank-deficient, so
            # the distance is dominated by estimator bias. Still usable for ranking
            # models at a fixed N, but only alongside the floor.
            print(f'[fid3d][warn] {biome_name}: {n_gen} samples for a '
                  f'{model.feature_dim}-d feature space; quote the floor with it')

        floor_txt = ''
        if baseline:
            floor = real_vs_real_floor(real_feats, n_gen, seed=seed, tag=biome_name)
            if floor is not None:
                floors[biome_name] = float(floor)
                floor_txt = f'   (real-vs-real floor {floor:.4f})'
        print(f'[fid3d] {biome_name:<20} {score:.4f}{floor_txt}   '
              f'gen={n_gen} real={real_feats.shape[0]}')

    if not results:
        raise RuntimeError('no biomes were scored')

    biome_names = sorted(results.keys())
    fid_values = np.array([results[b] for b in biome_names], dtype=np.float32)
    counts_arr = np.array([counts.get(b, 0) for b in biome_names], dtype=np.int32)
    ref_counts_arr = np.array([ref_counts.get(b, 0) for b in biome_names], dtype=np.int32)
    floor_arr = np.array([floors.get(b, np.nan) for b in biome_names], dtype=np.float32)
    overall_mean = float(np.mean(fid_values))
    floor_mean = float(np.nanmean(floor_arr)) if floors else None

    metrics_path = save_metrics_path or os.path.join(out_root, metrics_filename_for_label(label))
    try:
        os.makedirs(os.path.dirname(metrics_path) or '.', exist_ok=True)
        np.savez_compressed(
            metrics_path,
            biome_names=np.array(biome_names),
            fid=fid_values,
            counts=counts_arr,
            overall_mean=np.array([overall_mean], dtype=np.float32),
            baseline_fid=floor_arr,
            baseline_overall_mean=np.array([np.nan if floor_mean is None else floor_mean],
                                          dtype=np.float32),
            ref_counts=ref_counts_arr,
            feature_dim=np.array([int(model.feature_dim)], dtype=np.int32),
            weight_hash=np.array(weight_hash),
            checkpoint=np.array(str(checkpoint)),
            ref_dataset=np.array(str(ref_dataset)),
        )
    except Exception as e:
        print(f'[fid3d][warn] failed to write {metrics_path}: {e}')

    print(f'[fid3d] mean over {len(biome_names)} biomes: {overall_mean:.4f}'
          + (f'   (mean floor {floor_mean:.4f})' if floor_mean is not None else ''))

    return {
        'biome_names': biome_names,
        'fid': {b: float(results[b]) for b in biome_names},
        'counts': counts_arr.tolist(),
        'overall_mean': overall_mean,
        'metrics_path': metrics_path,
        'out_root': out_root,
        'ref_dir': str(ref_dataset),
        'fid_mode': str(label),
        'baseline_fid': {b: float(floors[b]) for b in floors},
        'baseline_overall_mean': floor_mean,
        'ref_counts': {b: int(ref_counts.get(b, 0)) for b in biome_names},
        'feature_dim': int(model.feature_dim),
        'weight_hash': weight_hash,
        'checkpoint': str(checkpoint),
        'mappings_path': str(resolved_mappings),
    }


def write_summary_json(res: Dict, samples_dir: str, label: str) -> str:
    """JSON summary alongside the npz, matching calc_fid.py's per-biome format."""
    path = os.path.join(os.path.dirname(res['metrics_path']),
                        summary_filename_for_label(label))
    try:
        with open(path, 'w') as f:
            json.dump({
                'timestamp': int(time.time()),
                'overall_mean': res['overall_mean'],
                'biome_fid': res['fid'],
                'metrics_path': res['metrics_path'],
                'ref_dir': res.get('ref_dir'),
                'samples_dir': str(samples_dir),
                'fid_mode': res.get('fid_mode', label),
                'baseline_overall_mean': res.get('baseline_overall_mean'),
                'biome_baseline_fid': res.get('baseline_fid'),
                'ref_counts': res.get('ref_counts'),
                'counts': res.get('counts'),
                'feature_dim': res.get('feature_dim'),
                'weight_hash': res.get('weight_hash'),
                'checkpoint': res.get('checkpoint'),
                'mappings_path': res.get('mappings_path'),
            }, f, indent=2)
    except Exception as e:
        print(f'[fid3d][warn] failed to write {path}: {e}')
    return path


# ---------------------------------------------------------------------------
# calibration: what does a distance of N actually mean?
# ---------------------------------------------------------------------------

NOISE_KINDS = ('shuffle', 'marginal', 'uniform')


def _sample_rows(total: int, n: int, seed: int, tag: str) -> np.ndarray:
    order = list(range(int(total)))
    random.Random(f'{seed}|{tag}').shuffle(order)
    return np.array(sorted(order[:int(min(n, total))]), dtype=np.int64)


def cross_biome_distances(ref_features: Dict[str, np.ndarray], *, n: int = 100,
                          seed: int = 0, verbose: bool = True):
    """Real-vs-real distance from each biome's chunks to every biome's reference.

    Off-diagonal entries price the failure "generated the wrong biome entirely",
    which is the scale a very large generated distance should be read against.
    The diagonal is the usual floor, so one table spans both ends of the range.
    Query sides use `n` chunks to match how generated sets are scored.
    """
    names = sorted(ref_features)
    query = {}
    reference = {}
    for name in names:
        feats = ref_features[name]
        rows = _sample_rows(feats.shape[0], n, seed, f'cal|{name}')
        query[name] = compute_feature_stats(feats[rows])
        reference[name] = compute_feature_stats(feats)

    matrix = np.full((len(names), len(names)), np.nan, dtype=np.float64)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if a == b:
                floor = real_vs_real_floor(ref_features[a], min(n, ref_features[a].shape[0]),
                                           seed=seed, tag=a)
                matrix[i, j] = np.nan if floor is None else float(floor)
                continue
            mu_a, sigma_a = query[a]
            mu_b, sigma_b = reference[b]
            matrix[i, j] = frechet_distance(mu_a, sigma_a, mu_b, sigma_b)
        if verbose:
            row = matrix[i].copy()
            row[i] = np.nan
            nearest = names[int(np.nanargmin(row))]
            print(f'[fid3d][cal] {a:<20} floor {matrix[i, i]:8.2f}   '
                  f'nearest other biome {nearest} {np.nanmin(row):8.2f}   '
                  f'mean other {np.nanmean(row):8.2f}')
    return names, matrix


def _noise_like(real: torch.Tensor, kind: str, vocab_size: int) -> torch.Tensor:
    """Structure-free volumes derived from a batch of real index volumes.

    'shuffle' permutes the voxels of each chunk, so the per-chunk block histogram
    survives exactly and only the geometry is destroyed; 'marginal' redraws every
    voxel from the biome's block frequencies, matching the histogram in
    expectation; 'uniform' ignores the data entirely.  If the metric were just
    reading block counts, the first two would score near the floor.
    """
    if kind == 'uniform':
        return torch.randint(0, int(vocab_size), real.shape, dtype=torch.long)
    if kind == 'shuffle':
        flat = real.reshape(real.shape[0], -1)
        perm = torch.argsort(torch.rand(flat.shape), dim=1)
        return torch.gather(flat, 1, perm).reshape(real.shape)
    if kind == 'marginal':
        counts = torch.bincount(real.reshape(-1), minlength=int(vocab_size)).double()
        probs = counts / counts.sum()
        flat = torch.empty(real.numel(), dtype=torch.long)
        step = 1 << 22                       # torch.multinomial caps at 2**24 draws
        for start in range(0, flat.numel(), step):
            end = min(start + step, flat.numel())
            flat[start:end] = torch.multinomial(probs, end - start, replacement=True)
        return flat.reshape(real.shape)
    raise ValueError(f'unknown noise kind {kind!r}; expected one of {NOISE_KINDS}')


def noise_distances(
    ref_path: str,
    model,
    converter: BlockBiomeConverter,
    ref_features: Dict[str, np.ndarray],
    *,
    kinds=NOISE_KINDS,
    n: int = 100,
    ref_values: str = 'indices',
    batch_size: int = 64,
    device: str = 'cuda',
    seed: int = 0,
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Distance from structure-free volumes to each biome's real reference."""
    noise_features = structure_free_features(
        ref_path,
        model,
        converter,
        ref_features,
        kinds=kinds,
        n=n,
        ref_values=ref_values,
        batch_size=batch_size,
        device=device,
        seed=seed,
    )
    out: Dict[str, Dict[str, float]] = {kind: {} for kind in kinds}
    for biome_name in sorted(ref_features):
        if not any(biome_name in values for values in noise_features.values()):
            continue
        mu_r, sigma_r = compute_feature_stats(ref_features[biome_name])
        scores = []
        for kind in kinds:
            feats = noise_features.get(kind, {}).get(biome_name)
            if feats is None:
                continue
            mu_f, sigma_f = compute_feature_stats(feats)
            out[kind][biome_name] = float(
                frechet_distance(mu_f, sigma_f, mu_r, sigma_r)
            )
            scores.append(f'{kind} {out[kind][biome_name]:.2f}')
        if verbose:
            print(f'[fid3d][cal] {biome_name:<20} ' + '   '.join(scores))
    return out


def structure_free_features(
    ref_path: str,
    model,
    converter: BlockBiomeConverter,
    ref_features: Dict[str, np.ndarray],
    *,
    kinds=NOISE_KINDS,
    n: int = 100,
    ref_values: str = 'indices',
    batch_size: int = 64,
    device: str = 'cuda',
    seed: int = 0,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Extract 3D-CNN features for geometry-destroying reference corruptions.

    This is shared by FID, KID/MMD, and MAUVE calibration so every metric sees
    exactly the same shuffled, marginal-resampled, and uniform voxel volumes.
    """
    voxels, labels = _open_reference(str(ref_path))
    lut = None if ref_values == 'indices' else build_index_lut(converter)

    out: Dict[str, Dict[str, np.ndarray]] = {kind: {} for kind in kinds}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        for label in sorted(set(int(v) for v in labels.tolist())):
            biome_name = biome_name_for_index(converter, label)
            if biome_name not in ref_features:
                continue
            biome_rows = np.flatnonzero(labels == label)
            rows = biome_rows[_sample_rows(biome_rows.size, n, seed, f'noise|{biome_name}')]
            real = to_index_volume(
                torch.from_numpy(np.ascontiguousarray(np.asarray(voxels[rows]))), lut)

            for kind in kinds:
                fake = _noise_like(real, kind, int(model.vocab_size))
                out[kind][biome_name] = extract_features(
                    model, torch.split(fake, batch_size), device=device)
    return out


@torch.no_grad()
def run_calibration(
    *,
    checkpoint: str,
    ref_dataset: str,
    mappings_path: Optional[str] = None,
    ref_values: str = 'indices',
    ref_per_biome: Optional[int] = DEFAULT_REF_PER_BIOME,
    n: int = 100,
    kinds=NOISE_KINDS,
    cross_biome: bool = True,
    batch_size: int = 64,
    device: Optional[str] = None,
    cache_dir: Optional[str] = 'fid3d_cache',
    refresh_cache: bool = False,
    seed: int = 0,
    out_dir: str = 'analysis/fid3d_calibration',
) -> Dict:
    """Reference points for reading a 3D-FID value, all computed on real data.

    Reuses the same cached reference features as scoring, so the numbers are
    directly comparable to the ones in `per_biome_fid_<label>.npz`.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model, converter, resolved_mappings, weight_hash = load_extractor_and_converter(
        checkpoint, mappings_path, device=device)
    cache_path = reference_cache_path(cache_dir, str(ref_dataset), weight_hash,
                                      ref_per_biome, seed)
    ref_features = reference_features_by_biome(
        str(ref_dataset), model, converter,
        ref_values=ref_values, per_biome=ref_per_biome, batch_size=batch_size,
        device=device, seed=seed, cache_path=cache_path, refresh_cache=refresh_cache,
    )
    if not ref_features:
        raise RuntimeError(f'no reference features built from {ref_dataset}')

    names = sorted(ref_features)
    print(f'[fid3d][cal] {len(names)} biomes, query size n={n}')

    matrix = None
    if cross_biome:
        print(f'[fid3d][cal] --- real biome vs real biome '
              f'({len(names) ** 2} matrix square roots, a few minutes) ---')
        names, matrix = cross_biome_distances(ref_features, n=n, seed=seed)

    noise = {}
    if kinds:
        print('[fid3d][cal] --- structure-free volumes vs real biome ---')
        noise = noise_distances(
            str(ref_dataset), model, converter, ref_features,
            kinds=tuple(kinds), n=n, ref_values=ref_values, batch_size=batch_size,
            device=device, seed=seed)

    floors = {name: float(matrix[i, i]) for i, name in enumerate(names)} if matrix is not None else {}
    summary = {}
    if matrix is not None:
        for i, name in enumerate(names):
            row = matrix[i].copy()
            row[i] = np.nan
            summary[name] = {
                'nearest_biome': names[int(np.nanargmin(row))],
                'nearest': float(np.nanmin(row)),
                'farthest_biome': names[int(np.nanargmax(row))],
                'farthest': float(np.nanmax(row)),
                'mean_other': float(np.nanmean(row)),
            }

    result = {
        'n': int(n),
        'biomes': names,
        'floor': floors,
        'cross_biome': ({a: {b: float(matrix[i, j]) for j, b in enumerate(names)}
                         for i, a in enumerate(names)} if matrix is not None else {}),
        'cross_biome_summary': summary,
        'noise': noise,
        'ref_dataset': str(ref_dataset),
        'ref_counts': {name: int(ref_features[name].shape[0]) for name in names},
        'feature_dim': int(model.feature_dim),
        'weight_hash': weight_hash,
        'checkpoint': str(checkpoint),
        'mappings_path': resolved_mappings,
    }

    tag = f'{os.path.basename(os.path.normpath(str(ref_dataset)))}_{weight_hash[:8]}_n{int(n)}'
    os.makedirs(str(out_dir), exist_ok=True)
    json_path = os.path.join(str(out_dir), f'calibration_{tag}.json')
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    if matrix is not None:
        np.savez_compressed(
            os.path.join(str(out_dir), f'cross_biome_{tag}.npz'),
            biome_names=np.array(names),
            distances=matrix.astype(np.float32),
        )

    if floors:
        print(f'[fid3d][cal] mean floor {np.mean(list(floors.values())):.2f}')
    if summary:
        others = [s['mean_other'] for s in summary.values()]
        print(f'[fid3d][cal] mean distance to a different real biome {np.mean(others):.2f}')
    for kind, per_biome in noise.items():
        if per_biome:
            print(f'[fid3d][cal] mean {kind} noise distance {np.mean(list(per_biome.values())):.2f}')
    print(f'[fid3d][cal] wrote {json_path}')
    return result


def build_argparser():
    import argparse

    p = argparse.ArgumentParser(
        description='Per-biome voxel-native FID using a trained 3D feature extractor.')
    p.add_argument('--samples_dir', default=None,
                   help="Directory of voxel dumps: per-biome files ('generated_<biome>.pt' "
                        "or '<biome>_samples_*.pt') or labelled trainer dumps. "
                        'Not needed with --calibrate.')
    p.add_argument('--checkpoint', required=True,
                   help='feature_extractor_best.pt from train_fid3d.py.')
    p.add_argument('--ref_dataset', required=True,
                   help='Reference chunks: a memmap dataset dir or a .pt dataset. '
                        'Use the val split of the extractor training composition.')
    p.add_argument('--mappings_path', default=None,
                   help='Defaults to the path recorded in the checkpoint.')
    p.add_argument('--samples_glob', default='*.pt')
    p.add_argument('--samples_format', choices=['block_ids', 'indices'], default='block_ids')
    p.add_argument('--ref_values', choices=['indices', 'block_ids'], default='indices',
                   help='Processed datasets store indices; override if yours holds raw IDs.')
    p.add_argument('--out_dir', default='renders/per_biome_gen',
                   help='Results go to <out_dir>/<run_id>/.')
    p.add_argument('--run_id', default=None)
    p.add_argument('--samples_per_biome', type=int, default=0)
    p.add_argument('--ref_per_biome', type=int, default=0,
                   help='Cap real chunks per biome; 0 uses the whole split. '
                        'Keep it identical across models you intend to compare.')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--device', default=None)
    p.add_argument('--label', default=FID3D_LABEL,
                   help='Filename tag: per_biome_fid_<label>.{npz,json}.')
    p.add_argument('--cache_dir', default='fid3d_cache')
    p.add_argument('--refresh_cache', action='store_true')
    p.add_argument('--no_baseline', action='store_true',
                   help='Skip the real-vs-real floor.')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--save_metrics_path', default=None)

    cal = p.add_argument_group('calibration')
    cal.add_argument('--calibrate', action='store_true',
                     help='Score no model; instead measure what a distance means on this '
                          'reference set: every real biome against every other, and '
                          'structure-free volumes against each biome.')
    cal.add_argument('--calibration_n', type=int, default=100,
                     help='Query-side chunk count; match the generated sample count so '
                          'the numbers are comparable (default 100).')
    cal.add_argument('--calibration_kinds', default=','.join(NOISE_KINDS),
                     help="Comma-separated noise types, or 'none'. 'shuffle' permutes each "
                          "real chunk's voxels, 'marginal' redraws from the biome's block "
                          "frequencies, 'uniform' ignores the data.")
    cal.add_argument('--no_cross_biome', action='store_true',
                     help='Skip the real-biome-vs-real-biome matrix.')
    cal.add_argument('--calibration_out', default='analysis/fid3d_calibration')
    return p


def main():
    args = build_argparser().parse_args()

    if args.calibrate:
        kinds = [k.strip() for k in str(args.calibration_kinds).split(',') if k.strip()]
        if kinds == ['none']:
            kinds = []
        run_calibration(
            checkpoint=args.checkpoint,
            ref_dataset=args.ref_dataset,
            mappings_path=args.mappings_path,
            ref_values=args.ref_values,
            ref_per_biome=(int(args.ref_per_biome) or None),
            n=int(args.calibration_n),
            kinds=kinds,
            cross_biome=(not args.no_cross_biome),
            batch_size=int(args.batch_size),
            device=args.device,
            cache_dir=(None if args.cache_dir in (None, '', 'none') else args.cache_dir),
            refresh_cache=bool(args.refresh_cache),
            seed=int(args.seed),
            out_dir=args.calibration_out,
        )
        return

    if not args.samples_dir:
        raise SystemExit('--samples_dir is required unless you pass --calibrate.')

    res = compute_per_biome_fid3d(
        args.samples_dir,
        checkpoint=args.checkpoint,
        ref_dataset=args.ref_dataset,
        mappings_path=args.mappings_path,
        samples_glob=args.samples_glob,
        samples_format=args.samples_format,
        ref_values=args.ref_values,
        gen_out_base_dir=args.out_dir,
        run_id=args.run_id,
        samples_per_biome=(int(args.samples_per_biome) or None),
        ref_per_biome=(int(args.ref_per_biome) or None),
        batch_size=int(args.batch_size),
        device=args.device,
        label=args.label,
        cache_dir=(None if args.cache_dir in (None, '', 'none') else args.cache_dir),
        refresh_cache=bool(args.refresh_cache),
        baseline=(not args.no_baseline),
        seed=int(args.seed),
        save_metrics_path=args.save_metrics_path,
    )
    summary = write_summary_json(res, args.samples_dir, args.label)
    print(json.dumps({
        'overall_mean': res['overall_mean'],
        'baseline_overall_mean': res['baseline_overall_mean'],
        'num_biomes': len(res['fid']),
        'metrics_path': res['metrics_path'],
        'summary_json': summary,
        'fid_mode': res['fid_mode'],
        'weight_hash': res['weight_hash'],
    }, indent=2))


if __name__ == '__main__':
    main()
