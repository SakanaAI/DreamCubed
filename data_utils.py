import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
import os
from pathlib import Path
import json
import ast
from typing import Optional
import glob
import re


def _load_block_types_map(path='assets/block_types.json'):
    with open(path, 'r') as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def _load_name_to_name_mapping(path: str):
    with open(path, 'r') as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected dict in {path}, got {type(raw).__name__}")
    out = {}
    for k, v in raw.items():
        out[str(k)] = str(v)
    return out


def _build_compression_lut_from_name_mapping(
    block_types_path: str,
    name_to_name_path: str,
):
    block_types = _load_block_types_map(block_types_path)
    if len(block_types) == 0:
        raise ValueError(f"No block types loaded from {block_types_path}")

    name_to_name = _load_name_to_name_mapping(name_to_name_path)
    name_to_id = {str(name): int(block_id) for block_id, name in block_types.items()}

    missing_source_names = sorted([name for name in name_to_name.keys() if name not in name_to_id])
    missing_target_names = sorted([name for name in name_to_name.values() if name != "AIR" and name not in name_to_id])
    if len(missing_source_names) > 0:
        raise ValueError(
            f"Compression mapping contains source names not present in {block_types_path}. "
            f"Examples: {missing_source_names[:20]}"
        )
    if len(missing_target_names) > 0:
        raise ValueError(
            f"Compression mapping contains target names not present in {block_types_path}. "
            f"Examples: {missing_target_names[:20]}"
        )

    max_id = int(max(block_types.keys()))
    lut = np.arange(max_id + 1, dtype=np.int32)
    air_id = int(name_to_id.get("AIR", 5))
    covered = set()
    for src_name, dst_name in name_to_name.items():
        src_id = int(name_to_id[src_name])
        dst_id = air_id if dst_name == "AIR" else int(name_to_id[dst_name])
        lut[src_id] = int(dst_id)
        covered.add(src_id)

    return lut, int(max_id), covered

def _normalize_half(val):
    if val is None:
        return 'bottom'
    v = str(val).lower()
    if v in ('bottom', 'lower', 'down'):
        return 'bottom'
    if v in ('top', 'upper', 'up'):
        return 'top'
    return 'bottom'

def _parse_shape_and_turn(shape_val):
    if shape_val is None or shape_val == '' or str(shape_val) == '{}':
        return 'straight', None
    s = str(shape_val)
    if ':' in s:
        s = s.split(':', 1)[1]
    s = s.lower()
    if s == 'straight':
        return 'straight', None
    if s.startswith('outer_'):
        return 'outer', s.split('_', 1)[1]
    if s.startswith('inner_'):
        return 'inner', s.split('_', 1)[1]
    return 'straight', None

def _as_dict_meta(meta):
    if isinstance(meta, dict):
        return meta
    if isinstance(meta, (bytes, bytearray)):
        try:
            meta = meta.decode('utf-8')
        except Exception:
            meta = str(meta)
    if isinstance(meta, str):
        try:
            v = ast.literal_eval(meta)
            if isinstance(v, dict):
                return v
        except Exception:
            return {}
    return {}

# Finalized facing mapping used in visualization: flip E/W only, keep N/S
_FACING_TO_AXIS = {
    'EAST': 'nx',
    'WEST': 'px',
    'SOUTH': 'py',
    'NORTH': 'ny',
}
_AXIS_OFFSET = {'px': 0, 'nx': 1, 'py': 2, 'ny': 3}
_OFFSET_TO_AXIS = {v: k for k, v in _AXIS_OFFSET.items()}

# Renderer target IDs for slabs and stairs
_SLAB_BOTTOM_ID = 3000
_SLAB_TOP_ID = 3001
_STRAIGHT_BOTTOM_BASE = 3010
_STRAIGHT_TOP_BASE = 3020
_OUTER_BOTTOM_BASE = 3030
_OUTER_TOP_BASE = 3040
_INNER_BOTTOM_BASE = 3050
_INNER_TOP_BASE = 3060

def _stair_render_id(half, facing_axis, shape, turn):
    fidx = _AXIS_OFFSET.get(facing_axis, 0)
    if shape == 'straight':
        base = _STRAIGHT_BOTTOM_BASE if half == 'bottom' else _STRAIGHT_TOP_BASE
        return base + fidx
    if shape == 'outer':
        base = _OUTER_BOTTOM_BASE if half == 'bottom' else _OUTER_TOP_BASE
        tidx = 0 if (turn or 'left') == 'left' else 1
        return base + fidx * 2 + tidx
    if shape == 'inner':
        base = _INNER_BOTTOM_BASE if half == 'bottom' else _INNER_TOP_BASE
        tidx = 0 if (turn or 'left') == 'left' else 1
        return base + fidx * 2 + tidx
    base = _STRAIGHT_BOTTOM_BASE if half == 'bottom' else _STRAIGHT_TOP_BASE
    return base + fidx

def remap_slabs_and_stairs_with_metadata(voxels: np.ndarray, metadata: np.ndarray, block_types_path='assets/block_types.json', simplify: bool=False) -> np.ndarray:
    """Return a copy of voxels with slab/stair blocks remapped to renderer IDs using metadata.

    - Works only if metadata is provided; otherwise returns input voxels unchanged
    - Upcasts to int32 to support IDs >= 3000
    """
    # print(f"remapping slabs and stairs with metadata: {metadata}")
    if metadata is None:
        print("no metadata provided, returning input voxels unchanged")
        return voxels

    block_types = _load_block_types_map(block_types_path)
    dataset_ids = set(int(v) for v in np.unique(voxels))
    stair_ids = {bid for bid, name in block_types.items() if name.endswith('_STAIRS')} & dataset_ids
    slab_ids = {bid for bid, name in block_types.items() if ('SLAB' in name and not name.startswith('DOUBLE') and not name.endswith('DOUBLE_SLAB'))} & dataset_ids
    # print(f"stair_ids: {stair_ids}")
    # print(f"slab_ids: {slab_ids}")
    if len(stair_ids) == 0 and len(slab_ids) == 0:
        return voxels

    vox_new = voxels.astype(np.int32, copy=True)
    mask = np.isin(voxels, np.array(sorted(list(stair_ids | slab_ids)), dtype=int))
    coords = np.where(mask)
    for idx in zip(*coords):
        old_id = int(voxels[idx])
        meta = _as_dict_meta(metadata[idx])
        # print(f"old_id: {old_id}")
        # print(f"meta: {meta}")
        if old_id in slab_ids:
            if simplify:
                vox_new[idx] = _SLAB_BOTTOM_ID
            else:
                half = _normalize_half(meta.get('slab_half') or meta.get('half'))
                vox_new[idx] = _SLAB_BOTTOM_ID if half == 'bottom' else _SLAB_TOP_ID
        else:
            if simplify:
                # collapse all stair orientations/types into a single canonical: bottom, NORTH, straight
                vox_new[idx] = _stair_render_id('bottom', 'ny', 'straight', None)
            else:
                facing_card = (meta.get('stair_facing') or meta.get('facing') or '').upper()
                facing_axis = _FACING_TO_AXIS.get(facing_card, 'px')
                half = _normalize_half(meta.get('stair_half') or meta.get('half'))
                shape, turn = _parse_shape_and_turn(meta.get('stair_shape'))
                vox_new[idx] = _stair_render_id(half, facing_axis, shape, turn)
                # print(f"vox_new: {vox_new[idx]}")
    
    return vox_new

def _decode_stair_renderer_id(renderer_id: int):
    """
    Decode a renderer stair ID (3010-3067) into (half, facing_axis, shape, turn).
    Returns None if the ID is not a recognized stair renderer ID.
    """
    # straight bottom/top
    if _STRAIGHT_BOTTOM_BASE <= renderer_id <= _STRAIGHT_BOTTOM_BASE + 3:
        fidx = renderer_id - _STRAIGHT_BOTTOM_BASE
        return 'bottom', _OFFSET_TO_AXIS.get(fidx, 'px'), 'straight', None
    if _STRAIGHT_TOP_BASE <= renderer_id <= _STRAIGHT_TOP_BASE + 3:
        fidx = renderer_id - _STRAIGHT_TOP_BASE
        return 'top', _OFFSET_TO_AXIS.get(fidx, 'px'), 'straight', None

    # outer corners bottom/top
    if _OUTER_BOTTOM_BASE <= renderer_id <= _OUTER_BOTTOM_BASE + 7:
        idx = renderer_id - _OUTER_BOTTOM_BASE
        fidx, tidx = divmod(idx, 2)
        turn = 'left' if tidx == 0 else 'right'
        return 'bottom', _OFFSET_TO_AXIS.get(fidx, 'px'), 'outer', turn
    if _OUTER_TOP_BASE <= renderer_id <= _OUTER_TOP_BASE + 7:
        idx = renderer_id - _OUTER_TOP_BASE
        fidx, tidx = divmod(idx, 2)
        turn = 'left' if tidx == 0 else 'right'
        return 'top', _OFFSET_TO_AXIS.get(fidx, 'px'), 'outer', turn

    # inner corners bottom/top
    if _INNER_BOTTOM_BASE <= renderer_id <= _INNER_BOTTOM_BASE + 7:
        idx = renderer_id - _INNER_BOTTOM_BASE
        fidx, tidx = divmod(idx, 2)
        turn = 'left' if tidx == 0 else 'right'
        return 'bottom', _OFFSET_TO_AXIS.get(fidx, 'px'), 'inner', turn
    if _INNER_TOP_BASE <= renderer_id <= _INNER_TOP_BASE + 7:
        idx = renderer_id - _INNER_TOP_BASE
        fidx, tidx = divmod(idx, 2)
        turn = 'left' if tidx == 0 else 'right'
        return 'top', _OFFSET_TO_AXIS.get(fidx, 'px'), 'inner', turn

    return None

def _rotate_facing_axis_y90(axis: str) -> str:
    """Rotate facing axis by +90° clockwise around Y, matching np.rot90 axes=(0, 2).

    Verified sequence from user test:
    3010(px) -> 3013(ny) -> 3011(nx) -> 3012(py) -> 3010(px)
    Thus: px->ny, ny->nx, nx->py, py->px
    """
    return {
        'px': 'ny',
        'ny': 'nx',
        'nx': 'py',
        'py': 'px',
    }.get(axis, axis)

def _build_stair_rotation_map_y90():
    """Build a dict mapping each stair renderer ID to its +90° rotated counterpart."""
    mapping = {}
    # Covered ID ranges: 3010-3013, 3020-3023, 3030-3037, 3040-3047, 3050-3057, 3060-3067
    ranges = [
        (_STRAIGHT_BOTTOM_BASE, _STRAIGHT_BOTTOM_BASE + 3),
        (_STRAIGHT_TOP_BASE, _STRAIGHT_TOP_BASE + 3),
        (_OUTER_BOTTOM_BASE, _OUTER_BOTTOM_BASE + 7),
        (_OUTER_TOP_BASE, _OUTER_TOP_BASE + 7),
        (_INNER_BOTTOM_BASE, _INNER_BOTTOM_BASE + 7),
        (_INNER_TOP_BASE, _INNER_TOP_BASE + 7),
    ]
    for lo, hi in ranges:
        for rid in range(lo, hi + 1):
            decoded = _decode_stair_renderer_id(rid)
            if decoded is None:
                continue
            half, facing_axis, shape, turn = decoded
            new_facing = _rotate_facing_axis_y90(facing_axis)
            new_id = _stair_render_id(half, new_facing, shape, turn)
            mapping[rid] = new_id
    return mapping

_STAIR_ROTATE_Y90_MAP = _build_stair_rotation_map_y90()
_STAIR_ROTATE_Y90_KEYS_T = torch.tensor(sorted(_STAIR_ROTATE_Y90_MAP.keys()), dtype=torch.long)
_STAIR_ROTATE_Y90_VALS_T = torch.tensor([
    _STAIR_ROTATE_Y90_MAP[int(k)] for k in _STAIR_ROTATE_Y90_KEYS_T.tolist()
], dtype=torch.long)

def rotate_voxels_90_fix_stairs(voxels: np.ndarray, k: int = 1) -> np.ndarray:
    """
    Rotate voxel IDs by k*90° around the Y axis and remap renderer stair IDs (3010–3067)
    to their rotated counterparts so their orientation stays correct.

    Accepts [H, W, D] or [N, H, W, D] arrays of integer block IDs.
    """
    if isinstance(voxels, torch.Tensor):
        return rotate_voxels_90_fix_stairs_torch(voxels, k=k)

    if not isinstance(voxels, np.ndarray):
        raise TypeError("voxels must be a numpy.ndarray of block IDs")

    k = int(k) % 4
    out = voxels.copy()
    if k == 0:
        return out

    for _ in range(k):
        if out.ndim == 3:  # [H, W, D]
            out = np.rot90(out, k=1, axes=(0, 2))
        elif out.ndim == 4:  # [N, H, W, D]
            out = np.rot90(out, k=1, axes=(1, 3))
        else:
            raise ValueError(f"Unsupported voxels ndim {out.ndim}; expected 3 or 4")

        # Remap stair IDs once per 90° rotation step without chaining
        pre = out.copy()
        stair_keys = np.fromiter(_STAIR_ROTATE_Y90_MAP.keys(), dtype=pre.dtype)
        stair_mask = np.isin(pre, stair_keys)
        if np.any(stair_mask):
            vals = pre[stair_mask]
            mapped = np.array([_STAIR_ROTATE_Y90_MAP[int(v)] for v in vals], dtype=pre.dtype)
            out[stair_mask] = mapped

    return out

def rotate_voxels_90(voxels, k=1):
    """
    Rotate voxels around Y axis by k*90 degrees. Scraped voxels have rotated and flipped axes, and will render misoriented.
    Args:
        voxels: tensor of shape [B, C, H, W, D] or [C, H, W, D]
        k: number of 90 degree rotations (1 = 90°, 2 = 180°, 3 = 270°)
    Returns:
        Rotated voxels
    """
    # Handle both batched and unbatched inputs
    if len(voxels.shape) == 5:  # Batched [B, C, H, W, D]
        # Rotate around Y (height) axis by swapping width and depth dimensions
        return torch.rot90(voxels, k=k, dims=(2, 4))
    elif len(voxels.shape) == 4:  # Unbatched [C, H, W, D]
        return torch.rot90(voxels, k=k, dims=(1, 3))
    elif len(voxels.shape) == 3:  # Unbatched [H, W, D]
        return torch.rot90(voxels, k=k, dims=(0, 2))
    else:
        raise ValueError(f"Unexpected voxel shape: {voxels.shape}")


def _remap_stair_ids_torch(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.numel() == 0:
        return tensor
    if tensor.dtype.is_floating_point:
        return tensor

    device = tensor.device
    dtype = tensor.dtype

    keys = _STAIR_ROTATE_Y90_KEYS_T.to(device=device, dtype=dtype)
    vals = _STAIR_ROTATE_Y90_VALS_T.to(device=device, dtype=dtype)

    flat = tensor.view(-1)
    for key, val in zip(keys, vals):
        mask = flat == key
        if mask.any():
            flat[mask] = val
    return tensor


def rotate_voxels_90_fix_stairs_torch(voxels: torch.Tensor, k: int = 1) -> torch.Tensor:
    """Torch equivalent of rotate_voxels_90_fix_stairs supporting 90° multiples."""
    if not isinstance(voxels, torch.Tensor):
        raise TypeError("voxels must be a torch.Tensor")

    k = int(k) % 4
    if k == 0 or voxels.numel() == 0:
        return voxels.clone()

    out = voxels.clone()

    if out.dim() == 3:
        rotate_dims = (0, 2)
    elif out.dim() == 4:
        rotate_dims = (1, 3)
    elif out.dim() == 5:
        rotate_dims = (2, 4)
    else:
        raise ValueError(f"Unsupported voxels ndim {out.dim()}; expected 3, 4, or 5")

    for _ in range(k):
        out = torch.rot90(out, k=1, dims=rotate_dims).contiguous()
        out = _remap_stair_ids_torch(out)

    return out


def condense_biomes(biomes, biome_mapping):
    """
    Combine similar biomes according to mapping.

    Args:
        biomes: numpy array of biome strings
        biome_mapping: dict mapping original biome names to new consolidated names

    Returns:
        biomes (consolidated biomes)
    """
    print("Condensing biomes...")
    # print("Original unique biomes:", np.unique(biomes))

    # Create a copy to modify
    new_biomes = biomes.copy()

    # Apply the mapping
    for old_biome, new_biome in biome_mapping.items():
        new_biomes[biomes == old_biome] = new_biome

    # print("Condensed unique biomes:", np.unique(new_biomes))
    return new_biomes

def process_class_conditional_dataset(
    data_path,
    simplify_metadata=False,
    val_split=0.0,
    val_seed=42,
    store_indices=False,
):
    """
    Process class-conditional dataset using per-sample biome labels.

    Accepts either a single .npz (legacy) or a dataset directory produced by
    build_biome_dataset_from_parts_memmap() containing manifest.json and .npy files.

    No biome condensation, majority voting, or block embedding is performed.

    Args:
        data_path: Path to the .npz file OR dataset directory
        simplify_metadata: Whether to simplify metadata-driven remapping
        val_split: Fraction of data to use for validation split
        val_seed: RNG seed for validation split
        store_indices: If True, save voxels as integer indices [B, H, W, D];
                       otherwise one-hot [B, C, H, W, D]

    Returns:
        None (saves processed data to disk)
    """
    data_path = Path(data_path)

    print(f"Loading data from {data_path}")
    # Resolve inputs
    metadata = None
    metadata_mask = None
    metadata_applied = False
    if data_path.is_dir():
        manifest_path = data_path / 'manifest.json'
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest.json not found in {data_path}")
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        voxels_path = data_path / manifest['paths']['voxels']
        labels_path = data_path / manifest['paths']['biome_labels']
        meta_path = manifest['paths'].get('metadata')
        mask_path = manifest['paths'].get('metadata_mask')

        voxels = np.load(str(voxels_path), mmap_mode='r')
        biome_labels = np.load(str(labels_path), allow_pickle=True)
        metadata = np.load(str(data_path / meta_path), mmap_mode='r') if (meta_path not in (None, '') and (data_path / meta_path).exists()) else None
        metadata_mask = np.load(str(data_path / mask_path)) if (mask_path not in (None, '') and (data_path / mask_path).exists()) else None
        metadata_applied = bool(manifest.get('metadata_applied_to_voxels', False))
    else:
        data = np.load(data_path, allow_pickle=True)
        voxels = data['voxels']
        metadata = data.get('metadata', None)
        metadata_mask = data.get('metadata_mask', None)
        biome_labels = data['biome_labels']
        metadata_applied = False

    print(f"Original dataset size: {int(voxels.shape[0])} samples")

    # Build block and biome mappings without materializing entire voxels array
    def _stream_unique_blocks(vx: np.ndarray, step: int = 2000):
        total = int(vx.shape[0])
        uniq = set()
        for i in range(0, total, step):
            end = min(i + step, total)
            uniq.update(np.unique(vx[i:end]).astype(int).tolist())
        return np.array(sorted(uniq), dtype=int)

    unique_blocks = _stream_unique_blocks(voxels)
    print(f'unique_blocks (count): {len(unique_blocks)}')

    block_to_index = {int(block): idx for idx, block in enumerate(unique_blocks)}
    index_to_block = {idx: int(block) for idx, block in enumerate(unique_blocks)}
    block_to_str = load_block_to_str_mapping()

    # Normalize biome labels to Python strings
    labels_list = biome_labels.tolist() if hasattr(biome_labels, 'tolist') else list(biome_labels)
    unique_biomes = list(sorted(set(map(str, labels_list))))
    biome_to_index = {str(b): idx for idx, b in enumerate(unique_biomes)}
    index_to_biome = {idx: str(b) for idx, b in enumerate(unique_biomes)}

    block_mappings = {
        'index_to_block': index_to_block,
        'block_to_index': block_to_index,
        'block_to_str': block_to_str,
    }
    biome_mappings = {'index_to_biome': index_to_biome, 'biome_to_index': biome_to_index}

    converter = BlockBiomeConverter(block_mappings, biome_mappings)

    mappings_path = (data_path.parent / f"{data_path.name}_cc_oh_mappings.pt") if data_path.is_dir() else (data_path.parent / f"{data_path.stem}_cc_oh_mappings.pt")
    converter.save_mappings(mappings_path)

    num_blocks = len(converter.block_to_index)
    num_classes = len(converter.biome_to_index)

    # Prepare LUT for fast ID -> index mapping
    lut_size = int(max(unique_blocks)) + 1 if len(unique_blocks) > 0 else 1
    lut = np.full((lut_size,), -1, dtype=np.int32)
    for b, idx_b in block_to_index.items():
        if b < lut_size:
            lut[b] = idx_b

    # Process samples in batches to avoid OOM
    num_samples = int(voxels.shape[0])
    batch_size = 100
    num_batches = (num_samples + batch_size - 1) // batch_size

    processed_chunks_list = []
    processed_classes_list = []

    print("Processing batches (index or one-hot)...")
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)

        # Slice memmap
        vx_slice = voxels[start_idx:end_idx]

        # Optionally remap using metadata (if not already applied at build stage)
        if (metadata is not None) and (not metadata_applied):
            batch_voxels = np.array(vx_slice)
            if metadata_mask is None:
                for j in range(batch_voxels.shape[0]):
                    batch_voxels[j] = remap_slabs_and_stairs_with_metadata(batch_voxels[j], metadata[start_idx + j], simplify=simplify_metadata)
            else:
                mask_batch = np.array(metadata_mask[start_idx:end_idx]).astype(bool)
                for j, ok in enumerate(mask_batch.tolist()):
                    if ok:
                        batch_voxels[j] = remap_slabs_and_stairs_with_metadata(batch_voxels[j], metadata[start_idx + j], simplify=simplify_metadata)
        else:
            batch_voxels = np.array(vx_slice)

        batch_labels = [str(lbl) for lbl in (biome_labels[start_idx:end_idx]).tolist()] if hasattr(biome_labels, 'tolist') else [str(x) for x in biome_labels[start_idx:end_idx]]

        # Ensure LUT covers range
        max_id_in_batch = int(np.max(batch_voxels))
        if max_id_in_batch >= lut_size:
            new_size = max_id_in_batch + 1
            ext = np.full((new_size - lut_size,), -1, dtype=np.int32)
            lut = np.concatenate([lut, ext], axis=0)
            for b, idx_b in block_to_index.items():
                if b < new_size:
                    lut[b] = idx_b
            lut_size = new_size

        batch_chunks_np = lut[batch_voxels]
        batch_chunks = torch.from_numpy(batch_chunks_np)

        if store_indices:
            processed_batch_chunks = batch_chunks.long()
        else:
            processed_batch_chunks = F.one_hot(
                batch_chunks.long(),
                num_classes=num_blocks
            ).permute(0, 4, 1, 2, 3).float()

        # Convert class labels to indices (one-hot classes)
        batch_class_indices = torch.tensor(np.array([converter.biome_to_index[label] for label in batch_labels]))
        processed_batch_class_indices = F.one_hot(
            batch_class_indices.long(),
            num_classes=num_classes
        )

        processed_chunks_list.append(processed_batch_chunks)
        processed_classes_list.append(processed_batch_class_indices)

        del batch_chunks, processed_batch_chunks, batch_class_indices, processed_batch_class_indices
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("Concatenating processed batches...")
    processed_chunks = torch.cat(processed_chunks_list, dim=0)
    processed_classes = torch.cat(processed_classes_list, dim=0)

    print(f'processed_chunks shape: {processed_chunks.shape}')
    print(f'processed_classes shape: {processed_classes.shape}')

    print("Saving processed class-conditional data...")
    processed_data_path = (data_path.parent / f"{data_path.name}_cc.pt") if data_path.is_dir() else (data_path.parent / f"{data_path.stem}_cc.pt")
    processed_val_path = (data_path.parent / f"{data_path.name}_cc_val.pt") if data_path.is_dir() else (data_path.parent / f"{data_path.stem}_cc_val.pt")

    N = processed_chunks.shape[0]
    use_val = (val_split is not None) and (float(val_split) > 0.0)
    if use_val and N > 1:
        v = int(round(N * float(val_split)))
        v = max(1, min(N - 1, v))
        rng = np.random.default_rng(int(val_seed))
        perm = rng.permutation(N)
        val_idx = torch.tensor(perm[:v], dtype=torch.long)
        train_idx = torch.tensor(perm[v:], dtype=torch.long)

        train_chunks = processed_chunks.index_select(0, train_idx)
        train_classes = processed_classes.index_select(0, train_idx)
        val_chunks = processed_chunks.index_select(0, val_idx)
        val_classes = processed_classes.index_select(0, val_idx)

        torch.save({'voxels': train_chunks, 'biomes': train_classes}, processed_data_path)
        torch.save({'voxels': val_chunks, 'biomes': val_classes}, processed_val_path)
        print(f"Saved {len(train_chunks)} train samples to {processed_data_path}")
        print(f"Saved {len(val_chunks)} val samples to {processed_val_path}")
    else:
        torch.save({'voxels': processed_chunks, 'biomes': processed_classes}, processed_data_path)
        print(f"Saved {len(processed_chunks)} samples to {processed_data_path}")


def process_class_conditional_dataset_memmap(
    data_path,
    simplify_metadata: bool = False,
    val_split: float = 0.0,
    val_seed: int = 42,
    voxel_dtype_out=np.int32,
    store_class_one_hot: bool = False,
    batch_size: int = 100,
):
    """
    Memory-mapped variant of `process_class_conditional_dataset`.

    - Streams the source dataset (legacy .npz or a memmapped dir built by
      `build_biome_dataset_from_parts_memmap`) and writes a processed dataset
      directory with memory-mapped `.npy` files.
    - Output voxels are class indices [N,H,W,D] (int32 by default).
    - Output labels are class indices [N] (int64). Set `store_class_one_hot=True`
      to store one-hot labels [N,C]; indices are strongly recommended for memory.

    Returns:
        (train_dir: str, val_dir: Optional[str])
    """
    src_path = Path(data_path)

    # Resolve source dataset (supports legacy .npz or memmap dir w/ manifest)
    metadata = None
    metadata_mask = None
    metadata_applied = False
    if src_path.is_dir():
        manifest_path = src_path / 'manifest.json'
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest.json not found in {src_path}")
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        voxels_path = src_path / manifest['paths']['voxels']
        labels_path = src_path / manifest['paths']['biome_labels']
        meta_path = manifest['paths'].get('metadata')
        mask_path = manifest['paths'].get('metadata_mask')

        voxels = np.load(str(voxels_path), mmap_mode='r')
        biome_labels = np.load(str(labels_path), allow_pickle=True)
        metadata = np.load(str(src_path / meta_path), mmap_mode='r') if (meta_path not in (None, '') and (src_path / meta_path).exists()) else None
        metadata_mask = np.load(str(src_path / mask_path)) if (mask_path not in (None, '') and (src_path / mask_path).exists()) else None
        metadata_applied = bool(manifest.get('metadata_applied_to_voxels', False))
    else:
        data = np.load(str(src_path), allow_pickle=True)
        voxels = data['voxels']
        biome_labels = data['biome_labels']
        metadata = data.get('metadata', None)
        metadata_mask = data.get('metadata_mask', None)
        metadata_applied = False

    N = int(voxels.shape[0])
    voxel_shape = tuple(voxels.shape[1:])
    print(f"[memmap] Source samples: {N:,}; voxel_shape: {voxel_shape}")

    # Build block + biome mappings by streaming over the source voxels
    def _stream_unique_blocks(vx: np.ndarray, step: int = 2000):
        total = int(vx.shape[0])
        uniq = set()
        for i in range(0, total, step):
            end = min(i + step, total)
            uniq.update(np.unique(vx[i:end]).astype(int).tolist())
        return np.array(sorted(uniq), dtype=int)

    unique_blocks = _stream_unique_blocks(voxels)
    print(f"[memmap] unique_blocks: {len(unique_blocks)}")

    block_to_index = {int(b): i for i, b in enumerate(unique_blocks)}
    index_to_block = {i: int(b) for i, b in enumerate(unique_blocks)}
    block_to_str = load_block_to_str_mapping()

    # Normalize biome labels and mappings
    labels_list = biome_labels.tolist() if hasattr(biome_labels, 'tolist') else list(biome_labels)
    unique_biomes = list(sorted(set(map(str, labels_list))))
    biome_to_index = {str(b): i for i, b in enumerate(unique_biomes)}
    index_to_biome = {i: str(b) for i, b in enumerate(unique_biomes)}

    block_mappings = {
        'index_to_block': index_to_block,
        'block_to_index': block_to_index,
        'block_to_str': block_to_str,
    }
    biome_mappings = {'index_to_biome': index_to_biome, 'biome_to_index': biome_to_index}
    converter = BlockBiomeConverter(block_mappings, biome_mappings)

    # Save mappings (same naming convention as tensor-based processor)
    mappings_path = (src_path.parent / f"{src_path.name}_cc_oh_mappings.pt") if src_path.is_dir() else (src_path.parent / f"{src_path.stem}_cc_oh_mappings.pt")
    converter.save_mappings(mappings_path)

    num_blocks = len(converter.block_to_index)
    num_classes = len(converter.biome_to_index)

    # LUT for fast block-id -> index mapping
    lut_size = int(max(unique_blocks)) + 1 if len(unique_blocks) > 0 else 1
    lut = np.full((lut_size,), -1, dtype=np.int32)
    for b, idx_b in block_to_index.items():
        if b < lut_size:
            lut[b] = idx_b

    # Output directories
    base = src_path.name if src_path.is_dir() else src_path.stem
    out_train_dir = src_path.parent / f"{base}_cc_dir"
    out_val_dir = None

    use_val = (val_split is not None) and (float(val_split) > 0.0)
    if use_val and N > 1:
        out_val_dir = src_path.parent / f"{base}_cc_val_dir"
        os.makedirs(out_val_dir, exist_ok=True)
    os.makedirs(out_train_dir, exist_ok=True)

    # Precompute split membership (compact boolean mask)
    is_val = None
    val_count = 0
    if use_val and N > 1:
        v = int(round(N * float(val_split)))
        v = max(1, min(N - 1, v))
        rng = np.random.default_rng(int(val_seed))
        perm = rng.permutation(N)
        is_val = np.zeros(N, dtype=bool)
        is_val[perm[:v]] = True
        val_count = int(v)

    train_count = N - val_count

    # Allocate output memmaps
    def _alloc_dataset_dir(dir_path: Path, n_items: int):
        vox_p = dir_path / 'voxels.npy'
        lbl_p = dir_path / 'biome_labels.npy'
        vox_m = np.lib.format.open_memmap(
            str(vox_p), mode='w+', dtype=voxel_dtype_out, shape=(n_items,) + voxel_shape
        )
        if store_class_one_hot:
            lbl_m = np.lib.format.open_memmap(
                str(lbl_p), mode='w+', dtype=np.int8, shape=(n_items, num_classes)
            )
        else:
            lbl_m = np.lib.format.open_memmap(
                str(lbl_p), mode='w+', dtype=np.int64, shape=(n_items,)
            )
        return vox_p, lbl_p, vox_m, lbl_m

    train_vox_p, train_lbl_p, train_vox_m, train_lbl_m = _alloc_dataset_dir(out_train_dir, train_count)
    if out_val_dir is not None:
        val_vox_p, val_lbl_p, val_vox_m, val_lbl_m = _alloc_dataset_dir(out_val_dir, val_count)
    else:
        val_vox_p = val_lbl_p = val_vox_m = val_lbl_m = None

    # Stream and write in batches
    t_write = 0
    v_write = 0
    did_apply_metadata = False
    for start in range(0, N, int(batch_size)):
        end = min(start + int(batch_size), N)

        vx_slice = voxels[start:end]
        # Metadata-driven remap if not already applied at build stage
        if (metadata is not None) and (not metadata_applied):
            batch_voxels = np.array(vx_slice)
            if metadata_mask is None:
                for j in range(batch_voxels.shape[0]):
                    batch_voxels[j] = remap_slabs_and_stairs_with_metadata(batch_voxels[j], metadata[start + j], simplify=simplify_metadata)
            else:
                mask_batch = np.array(metadata_mask[start:end]).astype(bool)
                for j, ok in enumerate(mask_batch.tolist()):
                    if ok:
                        batch_voxels[j] = remap_slabs_and_stairs_with_metadata(batch_voxels[j], metadata[start + j], simplify=simplify_metadata)
            did_apply_metadata = True
        else:
            batch_voxels = np.array(vx_slice)

        # Expand LUT if necessary
        max_id = int(np.max(batch_voxels))
        if max_id >= lut_size:
            new_size = max_id + 1
            ext = np.full((new_size - lut_size,), -1, dtype=np.int32)
            lut = np.concatenate([lut, ext], axis=0)
            for b, idx_b in block_to_index.items():
                if b < new_size:
                    lut[b] = idx_b
            lut_size = new_size

        batch_mapped = lut[batch_voxels]  # [B,H,W,D] int32

        # Labels -> indices (or one-hot)
        raw_labels = biome_labels[start:end]
        if hasattr(raw_labels, 'tolist'):
            raw_labels = raw_labels.tolist()
        class_indices = np.array([converter.biome_to_index[str(lbl)] for lbl in raw_labels], dtype=np.int64)
        if store_class_one_hot:
            one_hot = np.zeros((class_indices.shape[0], num_classes), dtype=np.int8)
            one_hot[np.arange(class_indices.shape[0]), class_indices] = 1

        if is_val is None:
            # All to train
            n_b = batch_mapped.shape[0]
            train_vox_m[t_write:t_write + n_b] = batch_mapped
            if store_class_one_hot:
                train_lbl_m[t_write:t_write + n_b] = one_hot
            else:
                train_lbl_m[t_write:t_write + n_b] = class_indices
            t_write += n_b
        else:
            idx_range = np.arange(start, end)
            mask_val = is_val[idx_range]
            mask_train = ~mask_val

            if mask_train.any():
                sel = batch_mapped[mask_train]
                n_t = sel.shape[0]
                train_vox_m[t_write:t_write + n_t] = sel
                if store_class_one_hot:
                    train_lbl_m[t_write:t_write + n_t] = one_hot[mask_train]
                else:
                    train_lbl_m[t_write:t_write + n_t] = class_indices[mask_train]
                t_write += n_t

            if mask_val.any():
                sel = batch_mapped[mask_val]
                n_v = sel.shape[0]
                val_vox_m[v_write:v_write + n_v] = sel
                if store_class_one_hot:
                    val_lbl_m[v_write:v_write + n_v] = one_hot[mask_val]
                else:
                    val_lbl_m[v_write:v_write + n_v] = class_indices[mask_val]
                v_write += n_v

        # free batch buffers
        del batch_voxels, batch_mapped

    # Write manifests
    def _write_manifest(dir_path: Path, n_items: int):
        man_path = dir_path / 'manifest.json'
        manifest = {
            'format_version': 2,
            'is_processed': True,
            'num_samples': int(n_items),
            'voxel_shape': list(voxel_shape),
            'voxel_dtype': str(voxel_dtype_out),
            'paths': {
                'voxels': 'voxels.npy',
                'biome_labels': 'biome_labels.npy',
            },
            'class_labels_format': ('one_hot' if store_class_one_hot else 'indices'),
            'num_blocks': int(num_blocks),
            'num_classes': int(num_classes),
            'metadata_applied_during_processing': bool(did_apply_metadata),
            'source': {
                'path': str(src_path),
                'had_metadata': bool(metadata is not None),
                'metadata_applied_at_source': bool(metadata_applied),
            }
        }
        with open(man_path, 'w') as f:
            json.dump(manifest, f)
        return man_path

    train_manifest = _write_manifest(out_train_dir, train_count)
    val_manifest = _write_manifest(out_val_dir, val_count) if out_val_dir is not None else None

    print("[memmap] Saved processed dataset:")
    print(f"  train: {out_train_dir}  (N={train_count:,}) -> {train_vox_p}")
    if out_val_dir is not None:
        print(f"  val:   {out_val_dir}  (N={val_count:,}) -> {val_vox_p}")
    print(f"  mappings: {mappings_path}")

    return str(out_train_dir), (str(out_val_dir) if out_val_dir is not None else None)



# Block and biome converted for processed data
# Takes in block and biome mappings loaded from mappings file (created when raw data is processed) and provides easy mapping back to minecraft block ids / block names

def load_block_to_str_mapping(file_path="assets/block_types.json"):
    """
    Loads the block ID to block name mapping from a JSON file.
    The JSON file is expected to be a dictionary mapping string numbers to string names.
    """
    with open(file_path, 'r') as f:
        str_mapping = json.load(f)
    return {int(k): v for k, v in str_mapping.items()}

class BlockBiomeConverter:
    def __init__(self, block_mappings=None, biome_mappings=None):
        """
        Initialize with pre-computed mappings for both blocks and biomes
        
        Args:
            block_mappings: dict containing 'index_to_block', 'block_to_index', 'block_to_str', and 'block_to_emb'
            biome_mappings: dict containing 'index_to_biome' and 'biome_to_index'
        """
        # from evocraft
        self.block_to_str = load_block_to_str_mapping()
        self.block_to_emb = None
        self.index_to_block = None
        self.block_to_index = None
        
        # Cache for embedding-to-block conversion (for efficiency)
        self._embedding_matrix = None
        self._block_ids_list = None

        if block_mappings:
            self.index_to_block = block_mappings.get('index_to_block')
            self.block_to_index = block_mappings.get('block_to_index')
            self.block_to_str = block_mappings.get('block_to_str', self.block_to_str)
            self.block_to_emb = block_mappings.get('block_to_emb')

        self.index_to_biome = biome_mappings['index_to_biome'] if biome_mappings else None
        self.biome_to_index = biome_mappings['biome_to_index'] if biome_mappings else None
    
    @classmethod
    def from_dataset(cls, data_path):
        """Create mappings from a dataset file"""
        data = np.load(data_path, allow_pickle=True)
        voxels = data['voxels']
        biomes = data['biomes']
        
        # Create block mappings (blocks are integers)
        unique_blocks = np.unique(voxels)
        block_to_index = {int(block): idx for idx, block in enumerate(unique_blocks)}
        index_to_block = {idx: int(block) for idx, block in enumerate(unique_blocks)}
        
        # Create biome mappings (biomes are strings)
        unique_biomes = np.unique(biomes)
        biome_to_index = {str(biome): idx for idx, biome in enumerate(unique_biomes)}
        index_to_biome = {idx: str(biome) for idx, biome in enumerate(unique_biomes)}
        
        block_mappings = {'index_to_block': index_to_block, 'block_to_index': block_to_index}
        biome_mappings = {'index_to_biome': index_to_biome, 'biome_to_index': biome_to_index}
        
        return cls(block_mappings, biome_mappings)
    
    @classmethod
    def from_arrays(cls, voxels, biomes, block_embeddings_path="assets/block_embeddings_norm.npy", block_embedding_dim=64, use_normalized_embeddings=True):
        """Create mappings directly from numpy arrays"""
        # Create block mappings (blocks are integers)
        unique_blocks = np.unique(voxels)
        block_to_index = {int(block): idx for idx, block in enumerate(unique_blocks)}
        index_to_block = {idx: int(block) for idx, block in enumerate(unique_blocks)}
        
        # Create biome mappings (biomes are strings)
        unique_biomes = np.unique(biomes)
        biome_to_index = {str(biome): idx for idx, biome in enumerate(unique_biomes)}
        index_to_biome = {idx: str(biome) for idx, biome in enumerate(unique_biomes)}

        # create embedding mappings
        if use_normalized_embeddings:
            # Use pre-normalized embeddings (already trimmed and normalized)
            print(f"Loading normalized embeddings from {block_embeddings_path}")
            block_embeddings = np.load(block_embeddings_path)
            cut_embeddings = block_embeddings  # Already normalized and trimmed
        else:
            # Use original embeddings with trimming and normalization
            print(f"Loading original embeddings from {block_embeddings_path}")
            block_embeddings = np.load(block_embeddings_path)
            cut_embeddings = normalize_openai_l2(block_embeddings, cut_dim=block_embedding_dim)
        
        print(f'size of embeddings: {cut_embeddings.shape}')
        num_emb_rows = cut_embeddings.shape[0]
        # Fallback embedding for renderer-only IDs (e.g., 3000+). Prefer PLANKS (160) else STONE (217) else 0.
        fallback_ids = [160, 217, 10, 0]
        fallback_base = next((fid for fid in fallback_ids if fid < num_emb_rows), 0)
        block_to_emb = {}
        for block in unique_blocks:
            b = int(block)
            if 0 <= b < num_emb_rows:
                block_to_emb[b] = cut_embeddings[b]
            else:
                block_to_emb[b] = cut_embeddings[fallback_base]
        # print(f'block_to_emb: {block_to_emb}')
        block_to_str = load_block_to_str_mapping()
        
        block_mappings = {
            'index_to_block': index_to_block, 
            'block_to_index': block_to_index,
            'block_to_str': block_to_str,
            'block_to_emb': block_to_emb
        }
        biome_mappings = {'index_to_biome': index_to_biome, 'biome_to_index': biome_to_index}
        
        return cls(block_mappings, biome_mappings)
    
    @classmethod
    def from_mappings(cls, path):
        """Load pre-saved mappings"""
        mappings = torch.load(path, weights_only=False)
        return cls(mappings['block_mappings'], mappings['biome_mappings'])
    
    def save_mappings(self, path):
        """Save mappings for later use"""
        torch.save({
            'block_mappings': {
                'index_to_block': self.index_to_block,
                'block_to_index': self.block_to_index,
                'block_to_str': self.block_to_str,
                'block_to_emb': self.block_to_emb
            },
            'biome_mappings': {
                'index_to_biome': self.index_to_biome,
                'biome_to_index': self.biome_to_index
            }
        }, path)
    
    def convert_to_original_blocks(self, data):
        """
        Convert from indices back to original block IDs.
        Handles both one-hot encoded and already-indexed data.
        
        Args:
            data: torch.Tensor of either:
                - one-hot encoded blocks [B, C, H, W, D] or [C, H, W, D]
                - indexed blocks [B, H, W, D] or [H, W, D]
        Returns:
            torch.Tensor of original block IDs with shape [B, H, W, D] or [H, W, D]
        """
        # If one-hot encoded (dim == 5 or first dim == num_blocks), convert to indices first
        if len(data.shape) == 5 or (len(data.shape) == 4 and data.shape[0] == len(self.block_to_index)):
            data = torch.argmax(data, dim=1 if len(data.shape) == 5 else 0)
        
        # Now convert indices to original blocks
        if len(data.shape) == 4:  # Batch dimension present
            return torch.tensor([[[[self.index_to_block[int(b)] 
                                for b in row]
                                for row in layer]
                                for layer in slice_]
                                for slice_ in data])
        else:  # No batch dimension
            return torch.tensor([[[self.index_to_block[int(b)] 
                                for b in row]
                                for row in layer]
                                for layer in data])
        
    def convert_to_indices(self, data):
        """
        Convert from indices back to original block IDs.
        Handles both one-hot encoded and already-indexed data.
        
        Args:
            data: torch.Tensor of either:
                - one-hot encoded blocks [B, C, H, W, D] or [C, H, W, D]
                - indexed blocks [B, H, W, D] or [H, W, D]
        Returns:
            torch.Tensor of original block IDs with shape [B, H, W, D] or [H, W, D]
        """
        # If one-hot encoded (dim == 5 or first dim == num_blocks), convert to indices first
        if len(data.shape) == 5 or (len(data.shape) == 4 and data.shape[0] == len(self.block_to_index)):
            data = torch.argmax(data, dim=1 if len(data.shape) == 5 else 0)
        
        # Now convert indices to original blocks
        if len(data.shape) == 4:  # Batch dimension present
            return torch.tensor([[[[self.block_to_index[int(b)] 
                                for b in row]
                                for row in layer]
                                for layer in slice_]
                                for slice_ in data])
        else:  # No batch dimension
            return torch.tensor([[[self.block_to_index[int(b)] 
                                for b in row]
                                for row in layer]
                                for layer in data])

    def convert_to_original_biomes(self, data):
        """
        Convert from indices back to original biome strings.
        Handles both one-hot encoded and already-indexed data.
        
        Args:
            data: torch.Tensor of either:
                - one-hot encoded biomes [B, C, H, W, D] or [C, H, W, D]
                - indexed biomes [B, H, W, D] or [H, W, D]
        Returns:
            numpy array of original biome strings with shape [B, H, W, D] or [H, W, D]
        """
        # If one-hot encoded (dim == 5 or first dim == num_biomes), convert to indices first
        if len(data.shape) == 5 or (len(data.shape) == 4 and data.shape[0] == len(self.biome_to_index)):
            data = torch.argmax(data, dim=1 if len(data.shape) == 5 else 0)
        
        # Now convert indices to original biomes
        if len(data.shape) == 4:  # Batch dimension present
            return np.array([[[[self.index_to_biome[int(b)] 
                            for b in row]
                            for row in layer]
                            for layer in slice_]
                            for slice_ in data])
        else:  # No batch dimension
            return np.array([[[self.index_to_biome[int(b)] 
                            for b in row]
                            for row in layer]
                            for layer in data])

    def convert_class_to_biomes(self, class_indices):
        """
        Convert class indices to biome strings for class-conditional data.
        
        This method handles single biome class labels (not volumetric biome arrays).
        Use this for data processed with process_class_conditional_dataset().
        
        Args:
            class_indices: torch.Tensor or numpy.ndarray of class indices
                - [B, C] for batch of one-hot encoded class labels
                - [C] for single one-hot encoded class label
                - [B] for batch of integer class indices
                - scalar for single integer class index
        Returns:
            list of biome strings corresponding to the class indices
        """
        if self.index_to_biome is None:
            raise ValueError("Class biome mappings not initialized. Load mappings from a class-conditional dataset.")
        
        # Convert to tensor if needed
        if isinstance(class_indices, np.ndarray):
            class_indices = torch.from_numpy(class_indices)
        
        # Check if we have one-hot encoded data
        if len(class_indices.shape) == 2:  # Batch of one-hot vectors [B, C]
            # Convert from one-hot to indices
            class_indices = torch.argmax(class_indices, dim=1)
            # Convert to list of biome strings
            return [self.index_to_biome[int(idx)] for idx in class_indices]
        elif len(class_indices.shape) == 1 and class_indices.shape[0] > 1 and torch.max(class_indices) <= 1:
            # Single one-hot vector [C] (assuming class_indices.shape[0] is num_classes)
            class_idx = torch.argmax(class_indices)
            return self.index_to_biome[int(class_idx)]
        elif len(class_indices.shape) == 1:
            # Batch of integer indices [B] 
            return [self.index_to_biome[int(idx)] for idx in class_indices]
        else:
            # Single integer index (scalar)
            return self.index_to_biome[int(class_indices)]

    def get_block_name_from_index(self, index):
        """
        Convert a single index to the corresponding block name.
        
        Args:
            index: int - the index of the block type
        Returns:
            str - the name of the block
        """
        if self.index_to_block is None:
            raise ValueError("Block mappings not initialized")
        
        block_id = self.index_to_block[index]
        return self.block_to_str.get(block_id, f"UNKNOWN_BLOCK_{block_id}")
    
    def get_block_id_from_index(self, index):
        """
        Convert a single index to the corresponding block ID.
        
        Args:
            index: int - the index of the block type
        Returns:
            int - the ID of the block
        """
        if self.index_to_block is None:
            raise ValueError("Block mappings not initialized")
        
        return self.index_to_block[index]
    
    def convert_to_block_names(self, data):
        """
        Convert from indices or IDs to block names.
        
        Args:
            data: torch.Tensor of either:
                - one-hot encoded blocks [B, C, H, W, D] or [C, H, W, D]
                - indexed blocks [B, H, W, D] or [H, W, D]
                - block ID blocks [B, H, W, D] or [H, W, D]
        Returns:
            numpy array of block names with shape [B, H, W, D] or [H, W, D]
        """
        # First convert to block IDs if necessary
        if len(data.shape) == 5 or (len(data.shape) == 4 and data.shape[0] == len(self.block_to_index)):
            # One-hot encoded, convert to indices first
            data = torch.argmax(data, dim=1 if len(data.shape) == 5 else 0)
            # Then convert indices to block IDs
            data = self.convert_to_original_blocks(data)
        elif self.index_to_block is not None and (
            (len(data.shape) == 4 and torch.max(data) < len(self.index_to_block)) or
            (len(data.shape) == 3 and torch.max(data) < len(self.index_to_block))
        ):
            # These are indices, convert to block IDs
            data = self.convert_to_original_blocks(data)
        
        # Now data contains block IDs, convert to names
        if data.dim() == 4:  # Batch dimension present
            return np.array([[[[self.block_to_str.get(int(b), f"UNKNOWN_BLOCK_{int(b)}") 
                            for b in row]
                            for row in layer]
                            for layer in slice_]
                            for slice_ in data])
        else:  # No batch dimension
            return np.array([[[self.block_to_str.get(int(b), f"UNKNOWN_BLOCK_{int(b)}") 
                            for b in row]
                            for row in layer]
                            for layer in data])
        
    def get_air_block_index(self):
        """
        Find the one-hot index corresponding to the air block (ID 5).
        Returns:
            int: The index where air blocks are encoded in one-hot format
        """
        # Find the index that maps to block ID 5 (air) in our index_to_block mapping
        for idx, block_id in self.index_to_block.items():
            if block_id == 5:  # Air block ID
                return idx
        raise ValueError("Air block (ID 5) not found in block mappings!")
    
    def get_water_block_index(self):
        """
        Find the one-hot index corresponding to the air block (ID 5).
        Returns:
            int: The index where air blocks are encoded in one-hot format
        """
        # Find the index that maps to block ID 5 (air) in our index_to_block mapping
        for idx, block_id in self.index_to_block.items():
            if block_id == 240 :  # water block ID
                return idx
        raise ValueError("water block (ID 240) not found in block mappings!")
    
    def get_blockid_indices(self, block_ids):
        """
        Find the one-hot index corresponding to the air block (ID 5).
        Returns:
            int: The index where air blocks are encoded in one-hot format
        """
        # Find the index that maps to block ID 5 (air) in our index_to_block mapping
        idxs = []
        for idx, block_id in self.index_to_block.items():
            if block_id in block_ids:  # Air block ID
                idxs.append(idx)
        if len(idxs) == 0:
            raise ValueError("Air block (ID 5) not found in block mappings!")
        return idxs

    def _prepare_embedding_lookup(self):
        """
        Prepare cached matrices for efficient embedding-to-block conversion.
        This creates a matrix of all embeddings and corresponding block IDs.
        """
        if self.block_to_emb is None:
            raise ValueError("Block embeddings not initialized")
        
        if self._embedding_matrix is None or self._block_ids_list is None:
            # Create sorted lists for consistent ordering
            self._block_ids_list = sorted(list(self.block_to_emb.keys()))
            embeddings_list = [self.block_to_emb[block_id] for block_id in self._block_ids_list]
            self._embedding_matrix = torch.tensor(np.array(embeddings_list), dtype=torch.float32)
        
        return self._embedding_matrix, self._block_ids_list

    def convert_emb_to_blocks(self, data, distance_metric='l2'):
        """
        Convert from embedding representation back to original block IDs.
        Uses nearest neighbor lookup to find the closest embedding.
        
        Args:
            data: torch.Tensor of embeddings with shape:
                - [B, E, H, W, D] for batched data
                - [E, H, W, D] for single sample
                where E is the embedding dimension (64)
            distance_metric: str, either 'l2' for L2 distance or 'cosine' for cosine similarity
        Returns:
            torch.Tensor of original block IDs with shape [B, H, W, D] or [H, W, D]
        """
        if self.block_to_emb is None:
            raise ValueError("Block embeddings not initialized")
        
        # Prepare cached embedding matrix and block IDs
        embedding_matrix, block_ids_list = self._prepare_embedding_lookup()
        
        # Ensure data is a torch tensor
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        
        # Determine if we have batch dimension
        has_batch = len(data.shape) == 5
        original_shape = data.shape
        
        if has_batch:
            B, E, H, W, D = data.shape
            data_permuted = data.permute(0, 2, 3, 4, 1)  # [B, E, H, W, D] -> [B, H, W, D, E]
            data_flat = data_permuted.reshape(-1, E)  # [B*H*W*D, E]
        else:
            E, H, W, D = data.shape
            data_permuted = data.permute(1, 2, 3, 0)  # [E, H, W, D] -> [H, W, D, E]
            data_flat = data_permuted.reshape(-1, E)  # [H*W*D, E]
        
        # Compute distances/similarities to all embeddings
        if distance_metric == 'l2':
            # L2 distance: smaller is better
            distances = torch.cdist(data_flat, embedding_matrix)
            nearest_indices = torch.argmin(distances, dim=1)
        elif distance_metric == 'cosine':
            # Cosine similarity: larger is better
            # Normalize vectors for cosine similarity
            data_norm = torch.nn.functional.normalize(data_flat, p=2, dim=1)
            emb_norm = torch.nn.functional.normalize(embedding_matrix, p=2, dim=1)
            similarities = torch.mm(data_norm, emb_norm.t())
            nearest_indices = torch.argmax(similarities, dim=1)
        else:
            raise ValueError("distance_metric must be 'l2' or 'cosine'")
        
        # Convert indices to block IDs
        block_ids_tensor = torch.tensor(block_ids_list, dtype=torch.long)
        result_flat = block_ids_tensor[nearest_indices]
        
        # Reshape back to original spatial dimensions
        if has_batch:
            result = result_flat.reshape(B, H, W, D)
        else:
            result = result_flat.reshape(H, W, D)
        
        return result

    def convert_emb_to_indices(self, data, distance_metric='l2'):
        """
        Convert from embedding representation to block indices (for ML).
        Uses nearest neighbor lookup to find the closest embedding, then converts to indices.
        
        Args:
            data: torch.Tensor of embeddings with shape:
                - [B, H, W, D, E] for batched data  
                - [H, W, D, E] for single sample
            distance_metric: str, either 'l2' or 'cosine'
        Returns:
            torch.Tensor of block indices with shape [B, H, W, D] or [H, W, D]
        """
        # First convert to block IDs
        block_ids = self.convert_emb_to_blocks(data, distance_metric)
        
        # Then convert block IDs to indices using existing method
        return self.convert_to_indices(block_ids)

    def get_embedding_from_block_id(self, block_id):
        """
        Get the embedding vector for a specific block ID.
        
        Args:
            block_id: int - the original block ID
        Returns:
            numpy.ndarray - the embedding vector
        """
        if self.block_to_emb is None:
            raise ValueError("Block embeddings not initialized")
        
        if block_id not in self.block_to_emb:
            raise ValueError(f"Block ID {block_id} not found in embeddings")
        
        return self.block_to_emb[block_id]

    def get_all_embeddings_matrix(self):
        """
        Get the full matrix of embeddings and corresponding block IDs.
        Useful for analysis or external nearest neighbor implementations.
        
        Returns:
            tuple: (embeddings_matrix, block_ids_list) where:
                - embeddings_matrix: torch.Tensor of shape [num_blocks, embedding_dim]
                - block_ids_list: list of block IDs corresponding to each row
        """
        embedding_matrix, block_ids_list = self._prepare_embedding_lookup()
        return embedding_matrix.clone(), block_ids_list.copy()



# List all unique block names in a processed one-hot dataset
def list_unique_block_names(processed_path_str: str, mappings_path_str: str):
    from data_utils import BlockBiomeConverter
    p = Path(processed_path_str)
    data = torch.load(p, map_location="cpu")
    vox = data['voxels']  # expect [B, C, H, W, D] one-hot
    if vox.dim() != 5:
        print("Expected one-hot [B, C, H, W, D]. Aborting.")
        return
    present = (vox.sum(dim=(0,2,3,4)) > 0).nonzero(as_tuple=False).squeeze(1).cpu().numpy().tolist()
    mappings_path = Path(mappings_path_str)
    conv = BlockBiomeConverter.from_mappings(mappings_path)
    entries = []
    for idx in present:
        block_id = conv.index_to_block[idx]
        name = conv.block_to_str.get(block_id, f"UNKNOWN_BLOCK_{block_id}")
        entries.append((block_id, name))
    # Dedupe by block_id and sort by ID
    unique_by_id = {bid: nm for bid, nm in entries}
    items = sorted(unique_by_id.items(), key=lambda x: x[0])
    print(f"Unique blocks ({len(items)}):")
    for bid, nm in items:
        print(f"{bid}: {nm}")

    # Also print unique biomes and counts if available
    if 'biomes' in data:
        biomes = data['biomes']  # [B, C]
        print(f"Unique biomes: {torch.unique(torch.argmax(biomes, dim=1))}")
        print(f"Unique biome labels: {conv.index_to_biome}")
        print(f'converted unique biomes: {conv.convert_class_to_biomes(torch.argmax(biomes, dim=1))}')
        if biomes.dim() == 2:
            counts = biomes.sum(dim=0).cpu().numpy().astype(int)
            present_biomes = (counts > 0).nonzero()[0].tolist()
            # Load mappings to map class index -> label
            conv = BlockBiomeConverter.from_mappings(mappings_path)
            print("Unique biomes with counts:")
            for i in present_biomes:
                label = conv.index_to_biome.get(int(i), str(i))
                print(f"{label}: {int(counts[i])}")


def chop_to_16(
    biome_dir: str,
    biome_names: list,
    file_pattern: str = "{biome}_chunks.npz",
    seed: int = 42,
    village_structural_blocks: Optional[list] = None,
    village_min_structural_count: int = 20,
):
    """
    Chop 32x32x32 chunks into 16x16x16 chunks by extracting the 4 corners in the x-z plane.
    
    For each 32x32x32 chunk:
    1. Define 4 corner windows in the x-z plane (each 16x16)
    2. For each corner, find the highest non-air block (air = block ID 5) at the center
    3. Use that height as the center Y position for the 16x16x16 window
    4. Apply bounds to ensure the window stays within the 32x32x32 chunk
    5. Add a random Y offset between -6 and +6 (also bounded)
    6. Filter out chunks with >90% air or <20% air
    7. For village biomes: filter out chunks without enough structural blocks
    
    Args:
        biome_dir: directory containing biome-specific .npz files (32x32x32 chunks)
        biome_names: list of biome names to process
        file_pattern: filename pattern; must include '{biome}' placeholder
        seed: RNG seed for reproducibility when applying random Y offsets
        village_structural_blocks: list of block IDs considered structural blocks for villages
                                    (only used if biome name contains 'village')
        village_min_structural_count: minimum number of structural blocks required in village chunks
    
    Returns:
        output_dir: path to the new directory containing chopped 16x16x16 chunks
    """
    # Create output directory with _chopped suffix
    output_dir = biome_dir.rstrip('/') + '_chopped'
    os.makedirs(output_dir, exist_ok=True)
    
    rng = np.random.default_rng(seed)
    
    # Define the 4 corner positions in the x-z plane
    # Each corner defines the start of a 16x16x16 region
    corners_xz = [
        (0, 0),    # Corner 0: x in [0, 16), z in [0, 16)
        (16, 0),   # Corner 1: x in [16, 32), z in [0, 16)
        (0, 16),   # Corner 2: x in [0, 16), z in [16, 32)
        (16, 16),  # Corner 3: x in [16, 32), z in [16, 32)
    ]
    
    print(f"Chopping 32x32x32 chunks into 16x16x16 chunks...")
    print(f"Output directory: {output_dir}")
    
    # Process each biome
    for biome in biome_names:
        biome_path = os.path.join(biome_dir, file_pattern.format(biome=biome))
        
        if not os.path.exists(biome_path):
            print(f"[warn] Missing biome file: {biome_path} — skipping")
            continue
        
        # Load the 32x32x32 chunks
        data = np.load(biome_path, allow_pickle=True)
        voxels_32 = data["voxels"]  # Shape: [N, 32, 32, 32]
        biome_labels = data["biome_labels"]
        metadata = data.get("metadata", None)
        
        num_chunks = voxels_32.shape[0]
        print(f"Processing biome '{biome}': {num_chunks} chunks")
        
        # Storage for chopped 16x16x16 chunks
        chopped_voxels = []
        chopped_biomes = []
        chopped_metadata = []
        
        # Track filtering statistics
        total_potential_chunks = num_chunks * 4  # 4 corners per chunk
        filtered_air = 0
        filtered_village_structural = 0
        
        # Process each 32x32x32 chunk
        for chunk_idx in range(num_chunks):
            chunk_32 = voxels_32[chunk_idx]  # Shape: [32, 32, 32]
            chunk_biome = biome_labels[chunk_idx]
            chunk_meta = metadata[chunk_idx] if metadata is not None else None
            
            # Extract 4 corner windows as 16x16x16 chunks
            for corner_idx, (x_start, z_start) in enumerate(corners_xz):
                # Calculate the center position of this 16x16 window in the x-z plane
                x_center = x_start + 8
                z_center = z_start + 8
                
                # Find the highest non-air block at the center (x_center, z_center)
                # Air blocks have block ID 5
                # We search from top to bottom (y from 31 down to 0)
                highest_y = None
                for y in range(31, -1, -1):
                    if chunk_32[y, x_center, z_center] != 5:  # Not air
                        highest_y = y
                        break
                
                # If all blocks at this position are air, default to center
                if highest_y is None:
                    highest_y = 16
                
                # Use the highest non-air block as the center of our Y window
                y_center = highest_y
                
                # Calculate the Y bounds for a 16-block window centered at y_center
                # We want y_start to y_start+16, where y_center is in the middle
                y_start = y_center - 8
                
                # Apply bounds to ensure we stay within [0, 32)
                # The window must fit entirely within the 32x32x32 chunk
                if y_start < 0:
                    y_start = 0
                elif y_start + 16 > 32:
                    y_start = 16  # This ensures y_start + 16 = 32
                
                # Apply random Y offset between -6 and +6
                y_offset = rng.integers(-6, 7)  # 7 is exclusive, so range is [-6, 6]
                y_start_offset = y_start + y_offset
                
                # Apply bounds again after offset
                if y_start_offset < 0:
                    y_start_offset = 0
                elif y_start_offset + 16 > 32:
                    y_start_offset = 16
                
                # Extract the 16x16x16 chunk
                chunk_16 = chunk_32[
                    y_start_offset:y_start_offset+16,
                    x_start:x_start+16,
                    z_start:z_start+16
                ]
                
                # Filter out chunks with too much air (>80% air, or <20% non-air)
                # Air blocks have block ID 5
                total_blocks = 16 * 16 * 16  # 4096 blocks
                air_blocks = np.sum(chunk_16 == 5)
                air_percentage = air_blocks / total_blocks
                
                # Only keep chunks with at most 90% air and at least 20% air
                if air_percentage > 0.9 or air_percentage < 0.2:
                    # Skip this chunk - too much air or too little air
                    filtered_air += 1
                    continue
                
                # Additional filter for village biomes: check for structural blocks
                if 'village' in biome.lower() and village_structural_blocks is not None:
                    # Count how many structural blocks are in this chunk
                    structural_block_count = 0
                    for block_id in village_structural_blocks:
                        structural_block_count += np.sum(chunk_16 == block_id)
                    
                    # If not enough structural blocks, skip this chunk
                    if structural_block_count < village_min_structural_count:
                        filtered_village_structural += 1
                        continue
                
                # Store the chopped chunk
                chopped_voxels.append(chunk_16)
                chopped_biomes.append(chunk_biome)
                if chunk_meta is not None:
                    # Extract corresponding metadata window
                    meta_16 = chunk_meta[
                        y_start_offset:y_start_offset+16,
                        x_start:x_start+16,
                        z_start:z_start+16
                    ]
                    chopped_metadata.append(meta_16)
        
        # Convert lists to arrays
        chopped_voxels = np.array(chopped_voxels)
        chopped_biomes = np.array(chopped_biomes)
        
        # Save the chopped chunks to the output directory
        output_path = os.path.join(output_dir, file_pattern.format(biome=biome))
        
        if chopped_metadata:
            chopped_metadata = np.array(chopped_metadata)
            np.savez_compressed(
                output_path,
                voxels=chopped_voxels,
                biome_labels=chopped_biomes,
                metadata=chopped_metadata
            )
        else:
            np.savez_compressed(
                output_path,
                voxels=chopped_voxels,
                biome_labels=chopped_biomes
            )
        
        kept_chunks = len(chopped_voxels)
        total_filtered = filtered_air + filtered_village_structural
        filter_rate = (total_filtered / total_potential_chunks * 100) if total_potential_chunks > 0 else 0
        print(f"  Saved {kept_chunks} chopped chunks to {output_path}")
        print(f"  Filtered out {total_filtered}/{total_potential_chunks} chunks ({filter_rate:.1f}%):")
        if filtered_air > 0:
            print(f"    - {filtered_air} due to air percentage (<20% or >90% air)")
        if filtered_village_structural > 0:
            print(f"    - {filtered_village_structural} village chunks due to insufficient structural blocks")
    
    print(f"\nChopping complete! Output directory: {output_dir}")
    return output_dir


def build_balanced_biome_dataset(
    biome_dir: str,
    biome_names: list,
    output_path: str,
    file_pattern: str = "{biome}_chunks.npz",
    seed: int = 42,
    shuffle_each: bool = True,
    other_fraction: float = 0.5,
):
    """
    Construct a weighted dataset by taking ALL samples from the first biome in
    biome_names (e.g., 'village') and only a FRACTION of that anchor count from
    each subsequent biome.

    - biome_dir: directory containing biome-specific .npz files
    - biome_names: list of biome names in the order to sample from; first is the anchor
    - output_path: path to write the balanced .npz file
    - file_pattern: filename pattern; must include '{biome}' placeholder
    - seed: RNG seed for reproducibility when sampling
    - shuffle_each: whether to shuffle within each biome before sampling
    - other_fraction: fraction of the anchor count to take from each non-anchor biome
      (e.g., 0.5 means half as many samples from every other biome)
    """

    assert len(biome_names) > 0, "biome_names must contain at least one biome"
    rng = np.random.default_rng(seed)

    def _load_biome(npz_path: str):
        if not os.path.exists(npz_path):
            print(f"[warn] Missing biome file: {npz_path} — skipping")
            return None
        data = np.load(npz_path, allow_pickle=True)
        vox = data["voxels"]
        bio = data["biome_labels"]
        meta = data["metadata"] if "metadata" in data.files else None
        return vox, bio, meta

    # Load anchor biome (the first in the list)
    anchor_biome = biome_names[0]
    anchor_path = os.path.join(biome_dir, file_pattern.format(biome=anchor_biome))
    anchor_loaded = _load_biome(anchor_path)
    if anchor_loaded is None:
        raise FileNotFoundError(f"Anchor biome file not found: {anchor_path}")
    anchor_voxels, anchor_biomes, anchor_meta = anchor_loaded

    # Metadata is mandatory: enforce presence on anchor and record reference shape
    if anchor_meta is None:
        print(f"[error] Anchor biome '{anchor_biome}' at {anchor_path} is missing metadata")
        raise ValueError(f"Metadata is required but missing for anchor biome: {anchor_path}")

    # temp limiter:
    anchor_voxels = anchor_voxels
    anchor_biomes = anchor_biomes
    anchor_meta = anchor_meta
    anchor_count = anchor_voxels.shape[0]
    print(f"Anchor biome '{anchor_biome}': {anchor_count} samples")

    out_voxels = [anchor_voxels]
    out_biomes = [anchor_biomes]
    # Establish reference ndim and per-sample tail shape from the anchor metadata
    ref_ndim = anchor_meta.ndim
    ref_tail_shape = tuple(anchor_meta.shape[1:])
    meta_parts = [anchor_meta]
    meta_mask_parts = [np.ones((anchor_voxels.shape[0],), dtype=bool)]

    print(f"Anchor biome '{anchor_biome}': using all {anchor_count} samples")

    # Process remaining biomes
    for biome in biome_names[1:]:
        biome_path = os.path.join(biome_dir, file_pattern.format(biome=biome))
        loaded = _load_biome(biome_path)
        if loaded is None:
            continue
        vox, bio, meta = loaded
        n = vox.shape[0]
        desired = max(0, int(anchor_count * other_fraction))
        take = min(desired, n)
        if take <= 0:
            print(f"[warn] Biome '{biome}' has no samples — skipping")
            continue

        if shuffle_each:
            idx = rng.permutation(n)[:take]
        else:
            idx = np.arange(take)

        out_voxels.append(vox[idx])
        out_biomes.append(bio[idx])
        # If metadata is missing or not compatible, synthesize placeholder and mask as False
        if (meta is None) or (meta.ndim != ref_ndim) or (tuple(meta.shape[1:]) != ref_tail_shape):
            if meta is None:
                print(f"[info] Biome '{biome}' has no metadata; filling placeholders for {take} samples")
            else:
                print(f"[info] Biome '{biome}' metadata shape {meta.shape} incompatible with anchor tail {ref_tail_shape}; filling placeholders for {take} samples")
            take_shape = (take,) + ref_tail_shape
            if anchor_meta.dtype == object:
                placeholder = np.empty(take_shape, dtype=object)
                placeholder.fill('')
            else:
                placeholder = np.zeros(take_shape, dtype=anchor_meta.dtype)
            meta_parts.append(placeholder)
            meta_mask_parts.append(np.zeros((take,), dtype=bool))
        else:
            meta_parts.append(meta[idx])
            meta_mask_parts.append(np.ones((take,), dtype=bool))

        print(f"Biome '{biome}': adding {take} of {n} samples (fraction={other_fraction})")

    # Concatenate and save
    voxels_balanced = np.concatenate(out_voxels, axis=0)
    biome_labels_balanced = np.concatenate(out_biomes, axis=0)
    # Concatenate per-sample metadata and mask, then save
    metadata_balanced = np.concatenate(meta_parts, axis=0)
    metadata_mask = np.concatenate(meta_mask_parts, axis=0)
    np.savez_compressed(output_path, voxels=voxels_balanced, biome_labels=biome_labels_balanced, metadata=metadata_balanced, metadata_mask=metadata_mask)

    print(
        f"Saved balanced dataset to {output_path} — voxels: {voxels_balanced.shape}, biome_labels: {biome_labels_balanced.shape}"
    )




def build_biome_dataset_from_parts(
    biome_dir: str,
    biome_names: list,
    output_path: str,
    part_pattern: str = "{biome}_chunks*.npz",
    shuffle: bool = False,
    seed: int = 42,
    single_file_pattern: str = "{biome}_chunks.npz",
    single_file_overrides: Optional[dict] = None,
):
    """
    Load multiple part .npz files per biome and concatenate into a single unbalanced dataset.

    Expected filename format for parts: <biome>_chunks_part_seed_<seed>_<part_index>.npz
    Use glob-style wildcards in part_pattern if needed.

    Saves a single .npz at output_path with keys:
      - 'voxels' (int array) [N, H, W, D]
      - 'biome_labels' (object/str array) [N]
      - 'metadata' (optional) [N, ...] if any part provides it
      - 'metadata_mask' (optional) [N] bool mask indicating which samples had real metadata

    Prints counts per biome label at the end.
    """
    assert len(biome_names) > 0, "biome_names must contain at least one biome"

    def _sort_key(path: str):
        base = os.path.basename(path)
        m = re.match(r".*_chunks_part_seed_(-?\d+)_([0-9]+)\.npz$", base)
        if m:
            s = int(m.group(1))
            p = int(m.group(2))
            return (s, p)
        return (0, base)

    def _load_npz(npz_path: str):
        data = np.load(npz_path, allow_pickle=True)
        vox = data["voxels"]
        bio = data["biome_labels"]
        meta = data["metadata"] if "metadata" in data.files else None
        return vox, bio, meta

    rng = np.random.default_rng(seed)

    all_voxels_parts = []
    all_biome_labels_parts = []

    # Handle metadata across heterogeneous parts by recording reference shape when first seen
    has_any_metadata = False
    ref_meta_ndim = None
    ref_meta_tail = None
    ref_meta_dtype = None
    meta_parts = []
    meta_mask_parts = []

    total_found_files = 0

    for biome in biome_names:
        pattern = os.path.join(biome_dir, part_pattern.format(biome=biome))
        part_files = sorted(glob.glob(pattern), key=_sort_key)

        # Fallbacks: per-biome override or single-file convention
        if len(part_files) == 0:
            override_path = None
            if single_file_overrides and biome in single_file_overrides:
                candidate = single_file_overrides[biome]
                candidate_str = str(candidate)
                if os.path.exists(candidate_str):
                    override_path = candidate_str
                else:
                    override_path = candidate_str if os.path.isabs(candidate_str) else os.path.join(biome_dir, candidate_str)
                if not os.path.exists(override_path):
                    print(f"[warn] Override file not found for biome '{biome}': {override_path}")
                    override_path = None

            if override_path is None:
                candidate = os.path.join(biome_dir, single_file_pattern.format(biome=biome))
                if os.path.exists(candidate):
                    override_path = candidate

            if override_path is not None:
                part_files = [override_path]
            else:
                print(f"[warn] No part files found for biome '{biome}' with pattern: {pattern}")
                continue
        print(f"Biome '{biome}': found {len(part_files)} part files")
        total_found_files += len(part_files)

        for pf in part_files:
            vox, bio, meta = _load_npz(pf)
            n = int(vox.shape[0])
            print(f"  - {os.path.basename(pf)}: {n} samples")

            all_voxels_parts.append(vox)
            all_biome_labels_parts.append(bio)

            if meta is not None:
                if not has_any_metadata:
                    has_any_metadata = True
                    ref_meta_ndim = meta.ndim
                    ref_meta_tail = tuple(meta.shape[1:])
                    ref_meta_dtype = meta.dtype

                # Check compatibility; fill placeholders if needed
                if (meta.ndim != ref_meta_ndim) or (tuple(meta.shape[1:]) != ref_meta_tail):
                    take_shape = (n,) + ref_meta_tail
                    if ref_meta_dtype == object:
                        placeholder = np.empty(take_shape, dtype=object)
                        placeholder.fill('')
                    else:
                        placeholder = np.zeros(take_shape, dtype=ref_meta_dtype)
                    meta_parts.append(placeholder)
                    meta_mask_parts.append(np.zeros((n,), dtype=bool))
                else:
                    meta_parts.append(meta)
                    meta_mask_parts.append(np.ones((n,), dtype=bool))
            else:
                # No metadata in this part; if we've already seen metadata elsewhere, create placeholders
                if has_any_metadata:
                    take_shape = (n,) + ref_meta_tail
                    if ref_meta_dtype == object:
                        placeholder = np.empty(take_shape, dtype=object)
                        placeholder.fill('')
                    else:
                        placeholder = np.zeros(take_shape, dtype=ref_meta_dtype)
                    meta_parts.append(placeholder)
                    meta_mask_parts.append(np.zeros((n,), dtype=bool))

    if len(all_voxels_parts) == 0:
        raise FileNotFoundError(f"No part files found in '{biome_dir}' for any of the provided biomes")

    voxels_all = np.concatenate(all_voxels_parts, axis=0)
    biome_labels_all = np.concatenate(all_biome_labels_parts, axis=0)

    # Optional shuffle
    if shuffle and voxels_all.shape[0] > 1:
        idx = rng.permutation(voxels_all.shape[0])
        voxels_all = voxels_all[idx]
        biome_labels_all = biome_labels_all[idx]
        if has_any_metadata:
            meta_all = np.concatenate(meta_parts, axis=0)[idx]
            meta_mask_all = np.concatenate(meta_mask_parts, axis=0)[idx]
        else:
            meta_all = None
            meta_mask_all = None
    else:
        if has_any_metadata:
            meta_all = np.concatenate(meta_parts, axis=0)
            meta_mask_all = np.concatenate(meta_mask_parts, axis=0)
        else:
            meta_all = None
            meta_mask_all = None

    # Save combined dataset
    if has_any_metadata:
        np.savez_compressed(
            output_path,
            voxels=voxels_all,
            biome_labels=biome_labels_all,
            metadata=meta_all,
            metadata_mask=meta_mask_all,
        )
    else:
        np.savez_compressed(
            output_path,
            voxels=voxels_all,
            biome_labels=biome_labels_all,
        )

    # Print summary and counts per biome label
    print(
        f"Saved dataset to {output_path} — voxels: {voxels_all.shape}, biome_labels: {biome_labels_all.shape} (from {total_found_files} part files)"
    )

    labels, counts = np.unique(biome_labels_all, return_counts=True)
    print("Counts per biome label:")
    for label, count in sorted(zip(labels, counts), key=lambda x: str(x[0])):
        print(f"  {label}: {int(count)}")


def build_biome_dataset_from_parts_memmap(
    biome_dir: str,
    biome_names: list,
    output_path: str,
    part_pattern: str = "{biome}_chunks*.npz",
    shuffle: bool = False,
    seed: int = 42,
    single_file_pattern: str = "{biome}_chunks.npz",
    single_file_overrides: Optional[dict] = None,
    apply_metadata_remap: bool = True,
    store_metadata_payload: bool = False,
):
    """
    Memory-efficient dataset builder using a two-pass, streaming approach.

    Important behavior changes:
      - Writes a persistent dataset directory containing memory-mapped `.npy` files
        instead of producing a monolithic `.npz` file.
      - Outputs a `manifest.json` describing shapes, dtypes, paths, and counts.
      - Skips aggregating object-dtype metadata payloads to avoid RAM blowups; a
        boolean `metadata_mask.npy` is written if metadata presence is detected.

    Two-pass flow:
      1) Count total samples and determine shapes/dtypes.
      2) Preallocate `voxels.npy` (memmap) and fill it slice-by-slice. Labels are
         kept in RAM (object dtype) and saved at the end. Numeric metadata is
         memmapped; object metadata values are not stored (mask only).

    Args:
        output_path: Path-like hint used to derive the dataset directory. If it
                     ends with an extension (e.g., .npz), the extension is removed
                     and a `_dir` suffix is appended to form the directory. If it
                     has no extension, the path itself is treated as the dataset
                     directory.

    Returns:
        str: path to the dataset directory containing `voxels.npy`,
             `biome_labels.npy`, optional `metadata.npy` and `metadata_mask.npy`,
             and a `manifest.json`.
    """
    import psutil
    
    def _log_memory(stage):
        """Log memory usage at different stages."""
        process = psutil.Process(os.getpid())
        mem_gb = process.memory_info().rss / (1024 ** 3)
        vm = psutil.virtual_memory()
        print(f"  [MEM] {stage}: Process={mem_gb:.2f}GB, System={vm.percent:.1f}% ({vm.used/(1024**3):.1f}/{vm.total/(1024**3):.1f}GB)")
    
    assert len(biome_names) > 0, "biome_names must contain at least one biome"
    
    def _sort_key(path: str):
        base = os.path.basename(path)
        m = re.match(r".*_chunks_part_seed_(-?\d+)_([0-9]+)\.npz$", base)
        if m:
            s = int(m.group(1))
            p = int(m.group(2))
            return (s, p)
        return (0, base)
    
    def _discover_files(biome):
        """Discover files for a given biome."""
        pattern = os.path.join(biome_dir, part_pattern.format(biome=biome))
        part_files = sorted(glob.glob(pattern), key=_sort_key)
        
        # Fallbacks: per-biome override or single-file convention
        if len(part_files) == 0:
            override_path = None
            if single_file_overrides and biome in single_file_overrides:
                candidate = single_file_overrides[biome]
                candidate_str = str(candidate)
                if os.path.exists(candidate_str):
                    override_path = candidate_str
                else:
                    override_path = candidate_str if os.path.isabs(candidate_str) else os.path.join(biome_dir, candidate_str)
                if not os.path.exists(override_path):
                    print(f"[warn] Override file not found for biome '{biome}': {override_path}")
                    override_path = None
            
            if override_path is None:
                candidate = os.path.join(biome_dir, single_file_pattern.format(biome=biome))
                if os.path.exists(candidate):
                    override_path = candidate
            
            if override_path is not None:
                part_files = [override_path]
            else:
                print(f"[warn] No part files found for biome '{biome}' with pattern: {pattern}")
        
        return part_files
    
    rng = np.random.default_rng(seed)
    
    # ========== FIRST PASS: Count samples and determine shapes ==========
    print("=" * 60)
    print("PASS 1: Counting samples and determining shapes...")
    print("=" * 60)
    _log_memory("pass1_start")
    
    total_samples = 0
    voxel_shape = None
    voxel_dtype = None
    has_any_metadata = False
    ref_meta_shape = None
    ref_meta_dtype = None
    
    all_file_info = []  # List of (biome, file_path, num_samples, has_metadata)
    
    for biome in biome_names:
        part_files = _discover_files(biome)
        if len(part_files) == 0:
            continue
        
        print(f"Biome '{biome}': found {len(part_files)} part files")
        
        for pf in part_files:
            with np.load(pf, allow_pickle=True) as data:
                n = int(data["voxels"].shape[0])
                print(f"  - {os.path.basename(pf)}: {n} samples")
                
                # Store shapes from first file
                if voxel_shape is None:
                    voxel_shape = tuple(data["voxels"].shape[1:])
                    voxel_dtype = data["voxels"].dtype
                    print(f"  Voxel shape: {voxel_shape}, dtype: {voxel_dtype}")
                
                # Check for metadata
                has_meta = "metadata" in data.files
                if has_meta and not has_any_metadata:
                    has_any_metadata = True
                    ref_meta_shape = tuple(data["metadata"].shape[1:])
                    ref_meta_dtype = data["metadata"].dtype
                    print(f"  Metadata shape: {ref_meta_shape}, dtype: {ref_meta_dtype}")
                
                all_file_info.append((biome, pf, n, has_meta))
                total_samples += n
    
    if len(all_file_info) == 0:
        raise FileNotFoundError(f"No part files found in '{biome_dir}' for any of the provided biomes")
    
    print(f"\nTotal samples to process: {total_samples:,}")
    print(f"Voxel shape per sample: {voxel_shape}")
    if has_any_metadata:
        print(f"Metadata shape per sample: {ref_meta_shape}")
    print("=" * 60)
    _log_memory("pass1_end")
    
    # ========== Create dataset directory and memory-mapped arrays ==========
    print("\nAllocating memory-mapped arrays...")
    _log_memory("before_memmap_allocation")

    # Derive dataset directory from output_path
    out_p = Path(output_path)
    if out_p.suffix:
        dataset_dir = Path(str(out_p.with_suffix('')) + "_dir")
    else:
        dataset_dir = out_p
    os.makedirs(dataset_dir, exist_ok=True)

    voxels_path = dataset_dir / "voxels.npy"
    labels_path = dataset_dir / "biome_labels.npy"
    metadata_path = dataset_dir / "metadata.npy"
    metadata_mask_path = dataset_dir / "metadata_mask.npy"
    manifest_path = dataset_dir / "manifest.json"

    # Choose output voxel dtype (upcast if applying metadata remap to allow 3000+ IDs)
    voxel_out_dtype = np.int32 if (apply_metadata_remap and has_any_metadata) else voxel_dtype

    # Create memory-mapped array for voxels (persistent .npy)
    voxels_mmap = np.lib.format.open_memmap(
        str(voxels_path),
        mode='w+',
        dtype=voxel_out_dtype,
        shape=(total_samples,) + voxel_shape
    )
    print(f"Voxels memmap: {voxels_mmap.shape}, {voxels_mmap.dtype}, ~{voxels_mmap.nbytes / (1024**3):.2f} GB")

    # Labels: small object array in RAM, saved at end as .npy
    biome_labels_all = np.empty(total_samples, dtype=object)

    # Metadata handling: memmap numeric, skip object payloads (mask only)
    meta_mmap = None
    meta_all = None
    meta_mask_all = None
    if has_any_metadata:
        meta_mask_all = np.zeros(total_samples, dtype=bool)
        if ref_meta_dtype != object and store_metadata_payload:
            meta_mmap = np.lib.format.open_memmap(
                str(metadata_path),
                mode='w+',
                dtype=ref_meta_dtype,
                shape=(total_samples,) + ref_meta_shape
            )
            print(f"Metadata memmap: {meta_mmap.shape}, {meta_mmap.dtype}")
        else:
            print("[info] Metadata payload will not be aggregated (either object dtype or disabled); writing mask only")

    # ========== SECOND PASS: Fill the arrays ==========
    print("\n" + "=" * 60)
    print("PASS 2: Loading data into memory-mapped arrays...")
    print("=" * 60)
    _log_memory("pass2_start")

    idx = 0
    for file_idx, (biome, pf, n, has_meta) in enumerate(all_file_info):
        print(f"Loading {os.path.basename(pf)} ({n} samples)...")

        with np.load(pf, allow_pickle=True) as data:
            vox_batch = data["voxels"]
            biome_labels_all[idx:idx+n] = data["biome_labels"]

            # Apply metadata-driven remap if requested and available
            if apply_metadata_remap and has_meta and ("metadata" in data.files):
                meta_data = data["metadata"]
                if tuple(meta_data.shape[1:]) == voxel_shape:
                    # remap per-sample to avoid huge temporary arrays
                    remapped = np.empty_like(vox_batch, dtype=voxel_out_dtype)
                    for j in range(n):
                        remapped[j] = remap_slabs_and_stairs_with_metadata(vox_batch[j], meta_data[j], simplify=False)
                    vox_batch = remapped
                else:
                    print("  [warn] Metadata shape mismatch for remap; skipping remap for this file")

            # Write voxels to memmap (will cast to voxel_out_dtype if needed)
            voxels_mmap[idx:idx+n] = vox_batch

            # Handle metadata payload saving / mask
            if has_any_metadata:
                if has_meta and ("metadata" in data.files):
                    if meta_mmap is not None:
                        meta_data = data["metadata"]
                        if tuple(meta_data.shape[1:]) == ref_meta_shape:
                            meta_mmap[idx:idx+n] = meta_data
                            meta_mask_all[idx:idx+n] = True
                        else:
                            print("  [warn] Metadata shape mismatch, marking mask False for this file")
                            meta_mask_all[idx:idx+n] = False
                    else:
                        meta_mask_all[idx:idx+n] = True
                else:
                    meta_mask_all[idx:idx+n] = False

            idx += n

            # Log memory periodically
            if (file_idx + 1) % 10 == 0:
                _log_memory(f"pass2_file_{file_idx+1}/{len(all_file_info)}")

    print(f"\nLoaded all {total_samples:,} samples")
    _log_memory("pass2_end")

    # ========== Optional shuffle (in-place) ==========
    if shuffle and total_samples > 1:
        print("\nShuffling data...")
        _log_memory("before_shuffle")
        perm = rng.permutation(total_samples)

        # Shuffle voxels in chunks
        chunk_size = min(10000, total_samples)
        temp_voxel_chunk = np.empty((chunk_size,) + voxel_shape, dtype=voxel_dtype)

        print("Shuffling voxels (in chunks)...")
        for i in range(0, total_samples, chunk_size):
            end = min(i + chunk_size, total_samples)
            current_chunk_size = end - i
            temp_voxel_chunk[:current_chunk_size] = voxels_mmap[perm[i:end]]
            voxels_mmap[i:end] = temp_voxel_chunk[:current_chunk_size]

        print("Shuffling biome labels...")
        biome_labels_all[:] = biome_labels_all[perm]

        if has_any_metadata and meta_mask_all is not None:
            print("Shuffling metadata mask...")
            meta_mask_all[:] = meta_mask_all[perm]

        if has_any_metadata and meta_mmap is not None:
            print("Shuffling numeric metadata (in chunks)...")
            temp_meta_chunk = np.empty((chunk_size,) + ref_meta_shape, dtype=ref_meta_dtype)
            for i in range(0, total_samples, chunk_size):
                end = min(i + chunk_size, total_samples)
                current_chunk_size = end - i
                temp_meta_chunk[:current_chunk_size] = meta_mmap[perm[i:end]]
                meta_mmap[i:end] = temp_meta_chunk[:current_chunk_size]

        _log_memory("after_shuffle")

    # ========== Persist labels, masks, and manifest ==========
    print("\nSaving dataset shards and manifest...")

    # Save labels (object array)
    np.save(str(labels_path), biome_labels_all)
    if has_any_metadata and meta_mask_all is not None:
        np.save(str(metadata_mask_path), meta_mask_all)

    # Manifest
    labels, counts = np.unique(biome_labels_all, return_counts=True)
    biome_counts = {str(l): int(c) for l, c in zip(labels.tolist(), counts.tolist())}
    manifest = {
        'format_version': 1,
        'num_samples': int(total_samples),
        'voxel_shape': list(voxel_shape),
        'voxel_dtype': str(voxel_out_dtype),
        'paths': {
            'voxels': os.path.basename(str(voxels_path)),
            'biome_labels': os.path.basename(str(labels_path)),
            'metadata': (os.path.basename(str(metadata_path)) if (meta_mmap is not None) else None),
            'metadata_mask': (os.path.basename(str(metadata_mask_path)) if (has_any_metadata and meta_mask_all is not None) else None),
        },
        'has_metadata': bool(has_any_metadata),
        'metadata_dtype': (str(ref_meta_dtype) if has_any_metadata else None),
        'metadata_applied_to_voxels': bool(apply_metadata_remap and has_any_metadata),
        'stored_metadata_payload': bool(meta_mmap is not None),
        'biome_counts': biome_counts,
        'source': {
            'biome_dir': str(biome_dir),
            'biomes': list(biome_names),
            'part_pattern': str(part_pattern),
            'single_file_overrides': single_file_overrides or {},
        }
    }
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f)

    # Print summary
    print(f"\nSaved dataset directory to {dataset_dir}")
    print(f"  Voxels: {voxels_mmap.shape}")
    print(f"  Labels: {biome_labels_all.shape} -> {labels_path}")
    if has_any_metadata:
        print(f"  Metadata: {'numeric' if meta_mmap is not None else 'skipped (object)'}")
    print(f"  Manifest: {manifest_path}")
    _log_memory("after_save")

    print("\nCounts per biome label:")
    for label, count in sorted(biome_counts.items(), key=lambda x: str(x[0])):
        print(f"  {label}: {count:,}")

    return str(dataset_dir)


def build_biome_dataset_from_parts_memmap_balanced(
    biome_dir: str,
    biome_names: list,
    output_path: str,
    total_size: int = 1_000_000,
    boosted_biome: Optional[str] = None,
    boost_factor: float = 1.0,
    part_pattern: str = "{biome}_chunks*.npz",
    shuffle: bool = False,
    seed: int = 42,
    single_file_pattern: str = "{biome}_chunks.npz",
    single_file_overrides: Optional[dict] = None,
    apply_metadata_remap: bool = True,
    store_metadata_payload: bool = False,
    apply_block_compression: bool = False,
    compression_mapping_path: str = "assets/block_compression_mapping.json",
    compression_mapping_by_name_path: str = "assets/block_compression_mapping_by_name.json",
    compression_block_types_path: str = "assets/block_types_updated.json",
    compression_strict: bool = True,
):
    """
    Balanced, memory-efficient dataset builder using the same on-disk format as
    `build_biome_dataset_from_parts_memmap`, but with controlled class counts.

    Key behavior:
      - Targets `total_size` samples total (default 1,000,000).
      - Default mode (balanced): equal samples per biome.
      - Optional "boosted" mode: one biome is upsampled relative to the rest
        while keeping the same total dataset size. Example: boost_factor=3 means
        boosted biome gets ~3x the samples of a non-boosted biome, and the other
        biomes are reduced evenly to compensate.
      - Efficient: only reads as many part files per biome as needed to hit each
        biome's target count.
      - If any biome cannot reach its target, this will *not* error; it logs the
        shortfall and reduces targets as needed (dataset may end up smaller than
        requested if the boosted biome is also insufficient).

    Output format is intentionally identical to `build_biome_dataset_from_parts_memmap`:
      - dataset_dir/voxels.npy (memmap)
      - dataset_dir/biome_labels.npy
      - optional dataset_dir/metadata.npy (if numeric + enabled)
      - optional dataset_dir/metadata_mask.npy
      - dataset_dir/manifest.json

    Returns:
        str: dataset directory path
    """
    import psutil

    def _log_memory(stage):
        process = psutil.Process(os.getpid())
        mem_gb = process.memory_info().rss / (1024 ** 3)
        vm = psutil.virtual_memory()
        print(
            f"  [MEM] {stage}: Process={mem_gb:.2f}GB, System={vm.percent:.1f}% ({vm.used/(1024**3):.1f}/{vm.total/(1024**3):.1f}GB)"
        )

    assert len(biome_names) > 0, "biome_names must contain at least one biome"
    total_size = int(total_size)
    if total_size <= 0:
        raise ValueError(f"total_size must be > 0, got {total_size}")

    def _sort_key(path: str):
        base = os.path.basename(path)
        m = re.match(r".*_chunks_part_seed_(-?\d+)_([0-9]+)\.npz$", base)
        if m:
            s = int(m.group(1))
            p = int(m.group(2))
            return (s, p)
        return (0, base)

    def _discover_files(biome):
        pattern = os.path.join(biome_dir, part_pattern.format(biome=biome))
        part_files = sorted(glob.glob(pattern), key=_sort_key)

        # Fallbacks: per-biome override or single-file convention
        if len(part_files) == 0:
            override_path = None
            if single_file_overrides and biome in single_file_overrides:
                candidate = single_file_overrides[biome]
                candidate_str = str(candidate)
                if os.path.exists(candidate_str):
                    override_path = candidate_str
                else:
                    override_path = (
                        candidate_str if os.path.isabs(candidate_str) else os.path.join(biome_dir, candidate_str)
                    )
                if not os.path.exists(override_path):
                    print(f"[warn] Override file not found for biome '{biome}': {override_path}")
                    override_path = None

            if override_path is None:
                candidate = os.path.join(biome_dir, single_file_pattern.format(biome=biome))
                if os.path.exists(candidate):
                    override_path = candidate

            if override_path is None:
                candidate_dir = os.path.join(biome_dir, str(biome))
                if os.path.isdir(candidate_dir):
                    override_path = candidate_dir

            if override_path is not None:
                if os.path.isdir(override_path):
                    part_files = sorted([
                        *glob.glob(os.path.join(override_path, "**", "*.npz"), recursive=True),
                        *glob.glob(os.path.join(override_path, "**", "*.npy"), recursive=True),
                    ], key=lambda p: str(p).lower())
                else:
                    part_files = [override_path]
            else:
                print(f"[warn] No part files found for biome '{biome}' with pattern: {pattern}")

        return part_files

    def _inspect_source_file(path: str):
        suffix = str(path).lower()
        if suffix.endswith(".npz"):
            with np.load(path, allow_pickle=True) as data:
                vox = data["voxels"]
                has_meta = "metadata" in data.files
                meta_shape = tuple(data["metadata"].shape[1:]) if has_meta else None
                meta_dtype = data["metadata"].dtype if has_meta else None
                return {
                    "n_file": int(vox.shape[0]),
                    "voxel_shape": tuple(vox.shape[1:]),
                    "voxel_dtype": vox.dtype,
                    "has_meta": bool(has_meta),
                    "meta_shape": meta_shape,
                    "meta_dtype": meta_dtype,
                }
        if suffix.endswith(".npy"):
            vox = np.load(path, mmap_mode="r")
            if vox.ndim == 3:
                return {
                    "n_file": 1,
                    "voxel_shape": tuple(vox.shape),
                    "voxel_dtype": vox.dtype,
                    "has_meta": False,
                    "meta_shape": None,
                    "meta_dtype": None,
                }
            if vox.ndim == 4:
                return {
                    "n_file": int(vox.shape[0]),
                    "voxel_shape": tuple(vox.shape[1:]),
                    "voxel_dtype": vox.dtype,
                    "has_meta": False,
                    "meta_shape": None,
                    "meta_dtype": None,
                }
            raise ValueError(f"Unsupported .npy voxel shape in {path}: {tuple(vox.shape)}")
        raise ValueError(f"Unsupported source file type: {path}")

    def _load_source_batch(path: str, biome: str, take: int):
        suffix = str(path).lower()
        if suffix.endswith(".npz"):
            with np.load(path, allow_pickle=True) as data:
                vox_batch = data["voxels"][:take]
                if "biome_labels" in data.files:
                    batch_labels = np.array(data["biome_labels"][:take], dtype=object, copy=True)
                else:
                    batch_labels = np.full(int(take), str(biome), dtype=object)
                meta_data = data["metadata"][:take] if "metadata" in data.files else None
            return vox_batch, batch_labels, meta_data
        if suffix.endswith(".npy"):
            raw = np.load(path, mmap_mode="r")
            if raw.ndim == 3:
                if int(take) > 1:
                    raise ValueError(f"Requested {take} samples from single-chunk file {path}")
                vox_batch = np.asarray(raw)[np.newaxis, ...]
            elif raw.ndim == 4:
                vox_batch = np.asarray(raw[:take])
            else:
                raise ValueError(f"Unsupported .npy voxel shape in {path}: {tuple(raw.shape)}")
            batch_labels = np.full(int(vox_batch.shape[0]), str(biome), dtype=object)
            return vox_batch, batch_labels, None
        raise ValueError(f"Unsupported source file type: {path}")

    rng = np.random.default_rng(seed)
    compression_lut = None
    compression_lut_max = None
    metadata_block_types_path = (
        str(compression_block_types_path) if apply_block_compression else "assets/block_types.json"
    )
    if apply_block_compression:
        try:
            compression_lut, compression_lut_max, _covered = _build_compression_lut_from_name_mapping(
                block_types_path=str(compression_block_types_path),
                name_to_name_path=str(compression_mapping_by_name_path),
            )
            print(
                f"[balanced] block compression enabled: LUT size={int(compression_lut_max) + 1} "
                f"from {compression_mapping_by_name_path}"
            )
        except Exception as e:
            raise ValueError(
                f"Failed to build compression LUT from {compression_mapping_by_name_path} "
                f"against {compression_block_types_path}: {e}"
            )

    def _apply_block_compression_batch(vox: np.ndarray) -> np.ndarray:
        if compression_lut is None or compression_lut_max is None:
            return vox
        if vox.size == 0:
            return vox.astype(np.int32, copy=False)

        a = vox.astype(np.int64, copy=False)
        if np.any(a >= 3000):
            bad = np.unique(a[a >= 3000])[:20].tolist()
            raise ValueError(
                f"Input voxels already contain renderer IDs (>=3000) before metadata remap. Examples: {bad}. "
                "This suggests metadata remap already happened upstream."
            )

        if compression_strict:
            present = np.unique(a)
            unknown = [
                int(x)
                for x in present.tolist()
                if int(x) < 0 or int(x) > int(compression_lut_max)
            ]
            if len(unknown) > 0:
                raise ValueError(
                    f"Compression LUT does not cover {len(unknown)} block IDs present in data. "
                    f"Examples: {unknown[:50]}"
                )

        out = a.astype(np.int32, copy=True)
        mask = (a >= 0) & (a <= int(compression_lut_max))
        if np.any(mask):
            out[mask] = compression_lut[a[mask]]
        return out

    # ========== PASS 0: Discover which biomes actually have data ==========
    biome_to_files = {b: _discover_files(b) for b in biome_names}
    used_biomes = [b for b in biome_names if len(biome_to_files.get(b, [])) > 0]
    dropped_biomes = [b for b in biome_names if b not in used_biomes]
    for b in dropped_biomes:
        print(f"[warn] Dropping biome '{b}' from balanced build (no files found)")

    if len(used_biomes) == 0:
        raise FileNotFoundError(
            f"No part files found in '{biome_dir}' for any of the provided biomes"
        )

    # Targets per biome: either perfectly balanced, or boosted one-vs-rest while keeping total_size fixed.
    boost_biome = str(boosted_biome) if boosted_biome is not None else None
    boost_factor = float(boost_factor)
    use_boost = (boost_biome is not None) and (boost_factor is not None) and (boost_factor > 1.0)
    if use_boost and boost_biome not in used_biomes:
        print(f"[warn] boosted_biome '{boost_biome}' not found among discovered biomes; ignoring boost")
        use_boost = False
        boost_biome = None

    if not use_boost:
        per_biome_target = total_size // len(used_biomes)
        if per_biome_target <= 0:
            raise ValueError(
                f"total_size={total_size} too small for num_biomes={len(used_biomes)} (per_biome_target={per_biome_target})"
            )
        if total_size % len(used_biomes) != 0:
            print(
                f"[warn] total_size={total_size:,} not divisible by num_biomes={len(used_biomes)}; "
                f"will build a perfectly balanced dataset of size {per_biome_target * len(used_biomes):,} "
                f"({per_biome_target:,} per biome)"
            )
        target_by_biome = {b: int(per_biome_target) for b in used_biomes}
    else:
        others = [b for b in used_biomes if b != boost_biome]
        denom = float(len(others)) + float(boost_factor)
        other_target = int(total_size // denom)
        if other_target <= 0:
            raise ValueError(
                f"total_size={total_size} too small for boost_factor={boost_factor} over {len(used_biomes)} biomes"
            )
        boosted_target = int(total_size - other_target * len(others))
        target_by_biome = {b: int(other_target) for b in others}
        target_by_biome[str(boost_biome)] = int(boosted_target)
        eff = float(boosted_target) / float(other_target) if other_target > 0 else float("inf")
        print(
            f"[boosted] boosted_biome='{boost_biome}', boost_factor={boost_factor:g} "
            f"-> targets: boosted={boosted_target:,}, others={other_target:,} each "
            f"(effective factor={eff:.3f}), total={sum(target_by_biome.values()):,}"
        )

    # ========== PASS 1: Select minimal part files per biome, determine shapes ==========
    print("=" * 60)
    print("PASS 1 (balanced): Selecting minimal files per biome & determining shapes...")
    print("=" * 60)
    _log_memory("pass1_start")

    voxel_shape = None
    voxel_dtype = None
    has_any_metadata = False
    ref_meta_shape = None
    ref_meta_dtype = None

    # biome -> list[[file_path, take, has_meta, n_file]]
    # (list for in-place take adjustment if we later need to top up the boosted biome)
    selected_by_biome = {b: [] for b in used_biomes}
    discovered_by_biome = {b: 0 for b in used_biomes}

    for biome in used_biomes:
        needed = int(target_by_biome.get(biome, 0))
        part_files = biome_to_files[biome]
        print(f"Biome '{biome}': found {len(part_files)} part files; need {needed:,} samples")

        for pf in part_files:
            if needed <= 0:
                break
            info = _inspect_source_file(pf)
            n_file = int(info["n_file"])
            take = min(n_file, needed)
            if take <= 0:
                continue

            if voxel_shape is None:
                voxel_shape = tuple(info["voxel_shape"])
                voxel_dtype = info["voxel_dtype"]
                print(f"  Voxel shape: {voxel_shape}, dtype: {voxel_dtype}")
            else:
                this_shape = tuple(info["voxel_shape"])
                if this_shape != voxel_shape:
                    raise ValueError(
                        f"Voxel shape mismatch in {pf}: expected {voxel_shape}, got {this_shape}"
                    )

            has_meta = bool(info["has_meta"])
            if has_meta and not has_any_metadata:
                has_any_metadata = True
                ref_meta_shape = tuple(info["meta_shape"])
                ref_meta_dtype = info["meta_dtype"]
                print(f"  Metadata shape: {ref_meta_shape}, dtype: {ref_meta_dtype}")

            selected_by_biome[biome].append([pf, int(take), bool(has_meta), int(n_file)])
            discovered_by_biome[biome] += int(take)
            needed -= int(take)

        got = discovered_by_biome[biome]
        req = int(target_by_biome.get(biome, 0))
        if got >= req:
            print(f"  [ok] Biome '{biome}': collected {got:,}/{req:,} samples")
        else:
            print(
                f"  [warn] Biome '{biome}': only collected {got:,}/{req:,} samples "
                f"(short by {req - got:,})"
            )

    if voxel_shape is None:
        raise FileNotFoundError("Could not determine voxel shape: no readable part files found")

    # Decide final per-biome targets based on discovery (may shrink if some biomes are short).
    if not use_boost:
        per_biome_target = int(next(iter(target_by_biome.values()))) if len(target_by_biome) else 0
        per_biome_final = min(discovered_by_biome.values()) if len(discovered_by_biome) else 0

        if per_biome_final < per_biome_target:
            print(
                f"[warn] Not enough samples for at least one biome to reach target {per_biome_target:,}. "
                f"To keep perfect balance, reducing to {per_biome_final:,} samples per biome "
                f"(final total {per_biome_final * len(used_biomes):,})."
            )
        else:
            print(
                f"[ok] All biomes met target. Building perfectly balanced dataset: "
                f"{per_biome_final * len(used_biomes):,} total ({per_biome_final:,} per biome)."
            )
        final_target_by_biome = {b: int(per_biome_final) for b in used_biomes}
    else:
        # Boosted mode: keep other biomes even, then top-up boosted biome as needed to keep total_size.
        boost_biome = str(boost_biome)
        others = [b for b in used_biomes if b != boost_biome]
        other_target = int(target_by_biome[others[0]]) if len(others) else 0
        boost_target = int(target_by_biome[boost_biome])

        other_final = min(discovered_by_biome[b] for b in others) if len(others) else 0
        required_boost = int(max(0, total_size - other_final * len(others)))

        # If some non-boosted biome was short, we may need MORE boosted samples to keep total_size.
        extra_needed = int(max(0, required_boost - int(discovered_by_biome.get(boost_biome, 0))))
        if extra_needed > 0:
            print(
                f"[boosted][warn] Other-biome shortfall forces boosted target to increase: "
                f"required_boost={required_boost:,} (initial boost target {boost_target:,}). "
                f"Attempting to collect an extra {extra_needed:,} samples for boosted biome '{boost_biome}'."
            )

            # Try to extend selection within the last used file first, then consume more files in order.
            part_files = biome_to_files[boost_biome]
            sel = selected_by_biome.get(boost_biome, [])

            # Extend within last file if we didn't take all of it.
            if len(sel) > 0 and extra_needed > 0:
                last = sel[-1]
                last_take = int(last[1])
                last_n = int(last[3])
                if last_take < last_n:
                    inc = min(last_n - last_take, extra_needed)
                    last[1] = last_take + int(inc)
                    discovered_by_biome[boost_biome] += int(inc)
                    extra_needed -= int(inc)

            # Continue with additional files after the last selected file (avoid duplicates).
            used_paths = set(x[0] for x in sel)
            for pf in part_files:
                if extra_needed <= 0:
                    break
                if pf in used_paths:
                    continue
                info = _inspect_source_file(pf)
                n_file = int(info["n_file"])
                take = min(n_file, extra_needed)
                if take <= 0:
                    continue

                this_shape = tuple(info["voxel_shape"])
                if this_shape != voxel_shape:
                    raise ValueError(
                        f"Voxel shape mismatch in {pf}: expected {voxel_shape}, got {this_shape}"
                    )

                has_meta = bool(info["has_meta"])
                if has_meta and not has_any_metadata:
                    has_any_metadata = True
                    ref_meta_shape = tuple(info["meta_shape"])
                    ref_meta_dtype = info["meta_dtype"]
                    print(f"  Metadata shape: {ref_meta_shape}, dtype: {ref_meta_dtype}")

                sel.append([pf, int(take), bool(has_meta), int(n_file)])
                discovered_by_biome[boost_biome] += int(take)
                extra_needed -= int(take)
                used_paths.add(pf)

            if extra_needed > 0:
                print(
                    f"[boosted][warn] Still short {extra_needed:,} samples for boosted biome '{boost_biome}'. "
                    f"Dataset will be smaller than requested total_size={total_size:,}."
                )

        boost_final = min(int(discovered_by_biome.get(boost_biome, 0)), int(required_boost))
        final_total = int(boost_final + other_final * len(others))
        eff = float(boost_final) / float(other_final) if other_final > 0 else float("inf")

        print(
            f"[boosted] Final targets: boosted={boost_final:,}, others={other_final:,} each "
            f"(effective factor={eff:.3f}), total={final_total:,} (requested {total_size:,})"
        )

        final_target_by_biome = {b: int(other_final) for b in others}
        final_target_by_biome[boost_biome] = int(boost_final)
        per_biome_target = int(other_target)
        per_biome_final = int(other_final)

    # Build final file plan, trimming any extra (if we had to reduce targets)
    all_file_info = []  # list[(biome, file_path, take, has_meta)]
    for biome in used_biomes:
        remaining = int(final_target_by_biome.get(biome, 0))
        for pf, take, has_meta, _n_file in selected_by_biome[biome]:
            if remaining <= 0:
                break
            take2 = min(int(take), int(remaining))
            all_file_info.append((biome, pf, int(take2), bool(has_meta)))
            remaining -= int(take2)
        if remaining != 0:
            print(
                f"[warn] Biome '{biome}' did not reach target={int(final_target_by_biome.get(biome, 0)):,} during trimming "
                f"(missing {remaining:,}). Dataset will still be written with fewer samples."
            )

    total_samples = int(sum(t for _, _, t, _ in all_file_info))
    print(f"\nTotal samples to process ({'boosted' if use_boost else 'balanced'}): {total_samples:,}")
    print(f"Voxel shape per sample: {voxel_shape}")
    if has_any_metadata:
        print(f"Metadata shape per sample: {ref_meta_shape}")
    print("=" * 60)
    _log_memory("pass1_end")

    # ========== Create dataset directory and memory-mapped arrays ==========
    print("\nAllocating memory-mapped arrays...")
    _log_memory("before_memmap_allocation")

    out_p = Path(output_path)
    if out_p.suffix:
        dataset_dir = Path(str(out_p.with_suffix("")) + "_dir")
    else:
        dataset_dir = out_p
    os.makedirs(dataset_dir, exist_ok=True)

    voxels_path = dataset_dir / "voxels.npy"
    labels_path = dataset_dir / "biome_labels.npy"
    metadata_path = dataset_dir / "metadata.npy"
    metadata_mask_path = dataset_dir / "metadata_mask.npy"
    manifest_path = dataset_dir / "manifest.json"

    voxel_out_dtype = np.int32 if (apply_block_compression or (apply_metadata_remap and has_any_metadata)) else voxel_dtype

    voxels_mmap = np.lib.format.open_memmap(
        str(voxels_path),
        mode="w+",
        dtype=voxel_out_dtype,
        shape=(total_samples,) + voxel_shape,
    )
    print(
        f"Voxels memmap: {voxels_mmap.shape}, {voxels_mmap.dtype}, ~{voxels_mmap.nbytes / (1024**3):.2f} GB"
    )

    biome_labels_all = np.empty(total_samples, dtype=object)

    meta_mmap = None
    meta_mask_all = None
    if has_any_metadata:
        meta_mask_all = np.zeros(total_samples, dtype=bool)
        if ref_meta_dtype != object and store_metadata_payload:
            meta_mmap = np.lib.format.open_memmap(
                str(metadata_path),
                mode="w+",
                dtype=ref_meta_dtype,
                shape=(total_samples,) + ref_meta_shape,
            )
            print(f"Metadata memmap: {meta_mmap.shape}, {meta_mmap.dtype}")
        else:
            print(
                "[info] Metadata payload will not be aggregated (either object dtype or disabled); writing mask only"
            )

    # ========== PASS 2: Fill the arrays ==========
    print("\n" + "=" * 60)
    print("PASS 2 (balanced): Loading data into memory-mapped arrays...")
    print("=" * 60)
    _log_memory("pass2_start")

    idx = 0
    for file_idx, (biome, pf, take, _) in enumerate(all_file_info):
        print(f"Loading {os.path.basename(pf)} ({take} samples for biome '{biome}')...")
        vox_batch, batch_labels, meta_data = _load_source_batch(pf, biome, take)
        biome_labels_all[idx : idx + take] = batch_labels

        has_meta = meta_data is not None

        if apply_block_compression:
            vox_batch = _apply_block_compression_batch(vox_batch)

        if apply_metadata_remap and has_meta:
            if tuple(meta_data.shape[1:]) == voxel_shape:
                remapped = np.empty_like(vox_batch, dtype=voxel_out_dtype)
                for j in range(take):
                    remapped[j] = remap_slabs_and_stairs_with_metadata(
                        vox_batch[j],
                        meta_data[j],
                        block_types_path=metadata_block_types_path,
                        simplify=False,
                    )
                vox_batch = remapped
            else:
                print("  [warn] Metadata shape mismatch for remap; skipping remap for this file")

        voxels_mmap[idx : idx + take] = vox_batch

        if has_any_metadata and meta_mask_all is not None:
            if has_meta:
                if meta_mmap is not None:
                    if tuple(meta_data.shape[1:]) == ref_meta_shape:
                        meta_mmap[idx : idx + take] = meta_data
                        meta_mask_all[idx : idx + take] = True
                    else:
                        print("  [warn] Metadata shape mismatch, marking mask False for this file")
                        meta_mask_all[idx : idx + take] = False
                else:
                    meta_mask_all[idx : idx + take] = True
            else:
                meta_mask_all[idx : idx + take] = False

        idx += take

        if (file_idx + 1) % 10 == 0:
            _log_memory(f"pass2_file_{file_idx+1}/{len(all_file_info)}")

    if idx != total_samples:
        raise RuntimeError(
            f"Internal error: filled {idx:,} samples but expected {total_samples:,}. "
            f"Refusing to write an inconsistent memmap dataset."
        )

    print(f"\nLoaded all {total_samples:,} samples")
    _log_memory("pass2_end")

    # ========== Optional shuffle (in-place) ==========
    if shuffle and total_samples > 1:
        print("\nShuffling data...")
        _log_memory("before_shuffle")
        perm = rng.permutation(total_samples)

        chunk_size = min(10000, total_samples)
        temp_voxel_chunk = np.empty((chunk_size,) + voxel_shape, dtype=voxel_dtype)

        print("Shuffling voxels (in chunks)...")
        for i in range(0, total_samples, chunk_size):
            end = min(i + chunk_size, total_samples)
            current_chunk_size = end - i
            temp_voxel_chunk[:current_chunk_size] = voxels_mmap[perm[i:end]]
            voxels_mmap[i:end] = temp_voxel_chunk[:current_chunk_size]

        print("Shuffling biome labels...")
        biome_labels_all[:total_samples] = biome_labels_all[:total_samples][perm]

        if has_any_metadata and meta_mask_all is not None:
            print("Shuffling metadata mask...")
            meta_mask_all[:total_samples] = meta_mask_all[:total_samples][perm]

        if has_any_metadata and meta_mmap is not None:
            print("Shuffling numeric metadata (in chunks)...")
            temp_meta_chunk = np.empty((chunk_size,) + ref_meta_shape, dtype=ref_meta_dtype)
            for i in range(0, total_samples, chunk_size):
                end = min(i + chunk_size, total_samples)
                current_chunk_size = end - i
                temp_meta_chunk[:current_chunk_size] = meta_mmap[perm[i:end]]
                meta_mmap[i:end] = temp_meta_chunk[:current_chunk_size]

        _log_memory("after_shuffle")

    # ========== Persist labels, masks, and manifest ==========
    print("\nSaving dataset shards and manifest...")

    np.save(str(labels_path), biome_labels_all[:total_samples])
    if has_any_metadata and meta_mask_all is not None:
        np.save(str(metadata_mask_path), meta_mask_all[:total_samples])

    labels, counts = np.unique(biome_labels_all[:total_samples], return_counts=True)
    biome_counts = {str(l): int(c) for l, c in zip(labels.tolist(), counts.tolist())}
    biome_fractions = {
        str(label): (float(count) / float(total_samples) if total_samples > 0 else 0.0)
        for label, count in biome_counts.items()
    }
    manifest = {
        "format_version": 1,
        "num_samples": int(total_samples),
        "voxel_shape": list(voxel_shape),
        "voxel_dtype": str(voxel_out_dtype),
        "paths": {
            "voxels": os.path.basename(str(voxels_path)),
            "biome_labels": os.path.basename(str(labels_path)),
            "metadata": (os.path.basename(str(metadata_path)) if (meta_mmap is not None) else None),
            "metadata_mask": (
                os.path.basename(str(metadata_mask_path))
                if (has_any_metadata and meta_mask_all is not None)
                else None
            ),
        },
        "has_metadata": bool(has_any_metadata),
        "metadata_dtype": (str(ref_meta_dtype) if has_any_metadata else None),
        "metadata_applied_to_voxels": bool(apply_metadata_remap and has_any_metadata),
        "stored_metadata_payload": bool(meta_mmap is not None),
        "biome_counts": biome_counts,
        "source": {
            "biome_dir": str(biome_dir),
            "biomes": list(used_biomes),
            "part_pattern": str(part_pattern),
            "single_file_overrides": single_file_overrides or {},
            "block_compression": {
                "applied": bool(apply_block_compression),
                "mapping_path": str(compression_mapping_path) if apply_block_compression else None,
                "mapping_by_name_path": str(compression_mapping_by_name_path) if apply_block_compression else None,
                "block_types_path": str(compression_block_types_path) if apply_block_compression else None,
                "strict": bool(compression_strict) if apply_block_compression else None,
            },
            "balanced": {
                "requested_total_size": int(total_size),
                "mode": ("boosted" if use_boost else "balanced"),
                "per_biome_target": int(per_biome_target),
                "per_biome_final": int(per_biome_final),
                "boosted_biome": (str(boost_biome) if use_boost else None),
                "boost_factor": (float(boost_factor) if use_boost else None),
                "target_by_biome_requested": {str(k): int(v) for k, v in target_by_biome.items()},
                "target_by_biome_final": {str(k): int(v) for k, v in final_target_by_biome.items()},
                "dropped_biomes_no_files": list(dropped_biomes),
            },
        },
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    print(f"\nSaved dataset directory to {dataset_dir}")
    print(f"  Voxels: {voxels_mmap.shape}")
    print(f"  Labels: {(total_samples,)} -> {labels_path}")
    if has_any_metadata:
        print(f"  Metadata: {'numeric' if meta_mmap is not None else 'skipped (object)'}")
    print(f"  Manifest: {manifest_path}")
    _log_memory("after_save")

    print("\nCounts per biome label:")
    for label, count in sorted(biome_counts.items(), key=lambda x: str(x[0])):
        print(f"  {label}: {count:,}")

    # Log discovery status per biome
    print("\nBalanced discovery summary (requested vs discovered):")
    for biome in used_biomes:
        got = int(discovered_by_biome.get(biome, 0))
        req = int(target_by_biome.get(biome, 0))
        status = "ok" if got >= req else "short"
        print(f"  {biome}: {got:,}/{req:,} ({status})")

    return str(dataset_dir)


def build_biome_dataset_from_parts_memmap_distributed(
    biome_dir: str,
    biome_names: list,
    output_path: str,
    biome_distribution: dict,
    total_size: int = 1_000_000,
    part_pattern: str = "{biome}_chunks*.npz",
    shuffle: bool = False,
    seed: int = 42,
    single_file_pattern: str = "{biome}_chunks.npz",
    single_file_overrides: Optional[dict] = None,
    village_relabel_blocks: Optional[list] = None,
    village_relabel_threshold: int = 60,
    village_relabel_label: str = "village",
    village_relabel_source_biomes: Optional[list] = None,
    match_cave_to_village: bool = False,
    cave_biome_name: str = "cave",
    cave_label: str = "cave",
    apply_metadata_remap: bool = True,
    store_metadata_payload: bool = False,
):
    """
    Distribution-driven, memory-efficient dataset builder using the same on-disk
    format as `build_biome_dataset_from_parts_memmap`.

    Key behavior:
      - Targets `total_size` samples total (default 1,000,000).
      - Accepts an arbitrary positive weight/probability distribution over biomes.
      - Normalizes the provided weights and converts them into integer per-biome
        sample targets whose sum is exactly `total_size`.
      - Optionally scans selected chunks from natural-biome files and relabels
        chunks to `village` if they contain more than `village_relabel_threshold`
        village structural blocks.
      - Optionally reserves a `cave` class after village relabeling by matching
        the cave count to the realized village count, then shrinking the remaining
        non-village natural chunks so the final dataset still has `total_size`
        samples.
      - Efficient: only reads as many part files per biome as needed to hit each
        biome's target count.
      - If any biome cannot reach its target, this preserves the requested
        distribution as closely as possible by scaling all targets down together.

    Notes:
      - `biome_distribution` does not need to sum to 1.0; it is normalized.
      - Biomes with zero or missing weight are ignored.
      - Biomes present in the distribution but missing on disk are dropped and the
        remaining discovered biomes are renormalized.

    Returns:
        str: dataset directory path
    """
    import psutil

    def _log_memory(stage):
        process = psutil.Process(os.getpid())
        mem_gb = process.memory_info().rss / (1024 ** 3)
        vm = psutil.virtual_memory()
        print(
            f"  [MEM] {stage}: Process={mem_gb:.2f}GB, System={vm.percent:.1f}% ({vm.used/(1024**3):.1f}/{vm.total/(1024**3):.1f}GB)"
        )

    def _sort_key(path: str):
        base = os.path.basename(path)
        m = re.match(r".*_chunks_part_seed_(-?\d+)_([0-9]+)\.npz$", base)
        if m:
            s = int(m.group(1))
            p = int(m.group(2))
            return (s, p)
        return (0, base)

    def _discover_files(biome):
        pattern = os.path.join(biome_dir, part_pattern.format(biome=biome))
        part_files = sorted(glob.glob(pattern), key=_sort_key)

        if len(part_files) == 0:
            override_path = None
            if single_file_overrides and biome in single_file_overrides:
                candidate = single_file_overrides[biome]
                candidate_str = str(candidate)
                if os.path.exists(candidate_str):
                    override_path = candidate_str
                else:
                    override_path = (
                        candidate_str if os.path.isabs(candidate_str) else os.path.join(biome_dir, candidate_str)
                    )
                if not os.path.exists(override_path):
                    print(f"[warn] Override file not found for biome '{biome}': {override_path}")
                    override_path = None

            if override_path is None:
                candidate = os.path.join(biome_dir, single_file_pattern.format(biome=biome))
                if os.path.exists(candidate):
                    override_path = candidate

            if override_path is not None:
                part_files = [override_path]
            else:
                print(f"[warn] No part files found for biome '{biome}' with pattern: {pattern}")

        return part_files

    def _largest_remainder_targets(total, normalized_weights, ordered_biomes, caps=None):
        raw_targets = {
            biome: float(total) * float(normalized_weights[biome])
            for biome in ordered_biomes
        }
        base_targets = {}
        fractions = []
        assigned = 0

        for idx, biome in enumerate(ordered_biomes):
            cap = None if caps is None else int(caps.get(biome, 0))
            base = int(np.floor(raw_targets[biome]))
            if cap is not None:
                base = min(base, cap)
            base_targets[biome] = base
            assigned += base
            fractions.append((raw_targets[biome] - np.floor(raw_targets[biome]), idx, biome))

        remaining = int(total - assigned)
        if remaining < 0:
            raise ValueError(f"Internal error: assigned={assigned} exceeds requested total={total}")

        fractions.sort(key=lambda x: (-x[0], x[1]))
        while remaining > 0:
            progress = False
            for _frac, _idx, biome in fractions:
                cap = None if caps is None else int(caps.get(biome, 0))
                if cap is not None and base_targets[biome] >= cap:
                    continue
                base_targets[biome] += 1
                remaining -= 1
                progress = True
                if remaining == 0:
                    break
            if not progress:
                break

        return base_targets

    def _compute_village_relabel_mask(vox_batch):
        if (not use_village_relabel) or (vox_batch.shape[0] == 0):
            return np.zeros(vox_batch.shape[0], dtype=bool)
        structural_counts = np.zeros(vox_batch.shape[0], dtype=np.int32)
        for block_id in village_relabel_blocks:
            structural_counts += np.count_nonzero(vox_batch == block_id, axis=(1, 2, 3))
        return structural_counts > village_relabel_threshold

    assert len(biome_names) > 0, "biome_names must contain at least one biome"

    total_size = int(total_size)
    if total_size <= 0:
        raise ValueError(f"total_size must be > 0, got {total_size}")

    if not isinstance(biome_distribution, dict) or len(biome_distribution) == 0:
        raise ValueError("biome_distribution must be a non-empty dict of biome -> weight")

    distribution_clean = {}
    for biome, weight in biome_distribution.items():
        weight = float(weight)
        if weight < 0:
            raise ValueError(f"Distribution weight for biome '{biome}' must be >= 0, got {weight}")
        if weight > 0:
            distribution_clean[str(biome)] = weight

    if len(distribution_clean) == 0:
        raise ValueError("biome_distribution must contain at least one positive weight")

    use_village_relabel = village_relabel_blocks is not None and len(village_relabel_blocks) > 0
    village_relabel_blocks = (
        [int(block_id) for block_id in village_relabel_blocks] if use_village_relabel else []
    )
    village_relabel_threshold = int(village_relabel_threshold)
    if use_village_relabel:
        if village_relabel_source_biomes is None:
            village_relabel_source_biomes = list(biome_names)
        village_relabel_source_set = {str(biome) for biome in village_relabel_source_biomes}
        village_relabel_label = str(village_relabel_label)
        print(
            f"[village_relabel] enabled for {len(village_relabel_source_set)} source biomes; "
            f"relabel to '{village_relabel_label}' when structural count > {village_relabel_threshold}"
        )
    else:
        village_relabel_source_set = set()

    use_cave_match = bool(match_cave_to_village)
    cave_biome_name = str(cave_biome_name)
    cave_label = str(cave_label)
    if use_cave_match and not use_village_relabel:
        raise ValueError("match_cave_to_village=True requires village relabeling to be enabled")
    if use_cave_match:
        print(
            f"[cave_match] enabled; cave biome '{cave_biome_name}' will be matched to realized "
            f"village count after relabeling"
        )

    rng = np.random.default_rng(seed)

    biome_to_files = {b: _discover_files(b) for b in biome_names}
    discovered_biomes = [b for b in biome_names if len(biome_to_files.get(b, [])) > 0]
    dropped_biomes = [b for b in biome_names if b not in discovered_biomes]
    for biome in dropped_biomes:
        print(f"[warn] Dropping biome '{biome}' from distributed build (no files found)")

    if len(discovered_biomes) == 0:
        raise FileNotFoundError(
            f"No part files found in '{biome_dir}' for any of the provided biomes"
        )

    weighted_biomes = [b for b in discovered_biomes if distribution_clean.get(b, 0.0) > 0]
    ignored_biomes = [b for b in discovered_biomes if b not in weighted_biomes]
    for biome in ignored_biomes:
        print(f"[info] Ignoring biome '{biome}' because it has no positive requested weight")

    missing_distribution_biomes = [
        biome for biome in distribution_clean.keys() if biome not in discovered_biomes
    ]
    for biome in missing_distribution_biomes:
        print(
            f"[warn] Requested biome '{biome}' is missing or undiscoverable on disk; "
            "renormalizing across discovered weighted biomes"
        )

    if len(weighted_biomes) == 0:
        raise ValueError(
            "None of the discovered biomes have a positive requested weight in biome_distribution"
        )

    weight_sum = float(sum(distribution_clean[b] for b in weighted_biomes))
    normalized_distribution = {
        biome: float(distribution_clean[biome]) / weight_sum for biome in weighted_biomes
    }

    target_by_biome = _largest_remainder_targets(
        total=total_size,
        normalized_weights=normalized_distribution,
        ordered_biomes=weighted_biomes,
    )
    target_total = int(sum(target_by_biome.values()))

    print("[distribution] Requested target counts:")
    for biome in weighted_biomes:
        pct = 100.0 * normalized_distribution[biome]
        print(f"  {biome}: {target_by_biome[biome]:,} samples ({pct:.3f}%)")
    print(f"[distribution] Requested total after rounding: {target_total:,}")

    print("=" * 60)
    print("PASS 1 (distributed): Selecting minimal files per biome & determining shapes...")
    print("=" * 60)
    _log_memory("pass1_start")

    voxel_shape = None
    voxel_dtype = None
    has_any_metadata = False
    ref_meta_shape = None
    ref_meta_dtype = None

    selected_by_biome = {b: [] for b in weighted_biomes}
    discovered_by_biome = {b: 0 for b in weighted_biomes}

    for biome in weighted_biomes:
        needed = int(target_by_biome.get(biome, 0))
        part_files = biome_to_files[biome]
        print(f"Biome '{biome}': found {len(part_files)} part files; need {needed:,} samples")

        for pf in part_files:
            if needed <= 0:
                break
            with np.load(pf, allow_pickle=True) as data:
                n_file = int(data["voxels"].shape[0])
                take = min(n_file, needed)
                if take <= 0:
                    continue

                if voxel_shape is None:
                    voxel_shape = tuple(data["voxels"].shape[1:])
                    voxel_dtype = data["voxels"].dtype
                    print(f"  Voxel shape: {voxel_shape}, dtype: {voxel_dtype}")
                else:
                    this_shape = tuple(data["voxels"].shape[1:])
                    if this_shape != voxel_shape:
                        raise ValueError(
                            f"Voxel shape mismatch in {pf}: expected {voxel_shape}, got {this_shape}"
                        )

                has_meta = "metadata" in data.files
                if has_meta and not has_any_metadata:
                    has_any_metadata = True
                    ref_meta_shape = tuple(data["metadata"].shape[1:])
                    ref_meta_dtype = data["metadata"].dtype
                    print(f"  Metadata shape: {ref_meta_shape}, dtype: {ref_meta_dtype}")

                selected_by_biome[biome].append([pf, int(take), bool(has_meta), int(n_file)])
                discovered_by_biome[biome] += int(take)
                needed -= int(take)

        got = discovered_by_biome[biome]
        req = int(target_by_biome.get(biome, 0))
        if got >= req:
            print(f"  [ok] Biome '{biome}': collected {got:,}/{req:,} samples")
        else:
            print(
                f"  [warn] Biome '{biome}': only collected {got:,}/{req:,} samples "
                f"(short by {req - got:,})"
            )

    if voxel_shape is None:
        raise FileNotFoundError("Could not determine voxel shape: no readable part files found")

    limiting_scale = 1.0
    final_total_target = int(target_total)
    final_target_by_biome = {str(k): int(v) for k, v in target_by_biome.items()}
    final_total = int(sum(final_target_by_biome.values()))

    print("[distribution] Final target counts:")
    for biome in weighted_biomes:
        pct = (100.0 * final_target_by_biome[biome] / final_total) if final_total > 0 else 0.0
        print(f"  {biome}: {final_target_by_biome[biome]:,} samples ({pct:.3f}%)")
    print(f"[distribution] Final total: {final_total:,} (requested {total_size:,})")
    for biome in weighted_biomes:
        got = int(discovered_by_biome[biome])
        req = int(final_target_by_biome[biome])
        if got < req:
            if got <= 0:
                raise ValueError(
                    f"Biome '{biome}' has target={req:,} but no source samples available for bootstrapping"
                )
            print(
                f"[bootstrap] Biome '{biome}' is short by {req - got:,} samples before relabeling; "
                f"will duplicate existing samples as needed."
            )

    natural_file_plans = []
    for biome in weighted_biomes:
        remaining = int(final_target_by_biome.get(biome, 0))
        for pf, take, has_meta, _n_file in selected_by_biome[biome]:
            if remaining <= 0:
                break
            take2 = min(int(take), int(remaining))
            natural_file_plans.append({
                "source_biome": str(biome),
                "output_label": None,
                "file_path": pf,
                "has_meta": bool(has_meta),
                "selected_indices": np.arange(int(take2), dtype=np.int64),
                "relabel_mask": None,
            })
            remaining -= int(take2)
        if remaining != 0:
            print(
                f"[bootstrap] Biome '{biome}' has {remaining:,} samples of post-selection shortfall "
                f"that may be filled by duplication after relabeling."
            )

    requested_village_by_source = {b: 0 for b in weighted_biomes}
    cave_target_requested = 0
    cave_target_final = 0

    if use_village_relabel and len(natural_file_plans) > 0:
        print("\nPre-scanning natural chunks for village relabeling...")
        for plan in natural_file_plans:
            biome = str(plan["source_biome"])
            idxs = plan["selected_indices"]
            take = int(len(idxs))
            if biome in village_relabel_source_set and take > 0:
                with np.load(plan["file_path"], allow_pickle=True) as data:
                    vox_batch = data["voxels"][idxs]
                    relabel_mask = _compute_village_relabel_mask(vox_batch)
                plan["relabel_mask"] = relabel_mask
                requested_village_by_source[biome] += int(np.count_nonzero(relabel_mask))
            else:
                plan["relabel_mask"] = np.zeros(take, dtype=bool)

    if use_cave_match:
        cave_target_requested = int(sum(requested_village_by_source.values()))
        cave_part_files = _discover_files(cave_biome_name)
        if cave_target_requested > 0 and len(cave_part_files) == 0:
            raise FileNotFoundError(
                f"match_cave_to_village requested {cave_target_requested:,} cave samples, "
                f"but no files were found for cave biome '{cave_biome_name}'"
            )

        cave_remaining = int(cave_target_requested)
        cave_file_plans = []
        cave_discovered = 0
        for pf in cave_part_files:
            if cave_remaining <= 0:
                break
            with np.load(pf, allow_pickle=True) as data:
                n_file = int(data["voxels"].shape[0])
                take = min(n_file, cave_remaining)
                if take <= 0:
                    continue

                this_shape = tuple(data["voxels"].shape[1:])
                if this_shape != voxel_shape:
                    raise ValueError(
                        f"Voxel shape mismatch in cave file {pf}: expected {voxel_shape}, got {this_shape}"
                    )

                has_meta = "metadata" in data.files
                if has_meta and not has_any_metadata:
                    has_any_metadata = True
                    ref_meta_shape = tuple(data["metadata"].shape[1:])
                    ref_meta_dtype = data["metadata"].dtype
                    print(f"  Metadata shape: {ref_meta_shape}, dtype: {ref_meta_dtype}")

                cave_file_plans.append({
                    "source_biome": str(cave_biome_name),
                    "output_label": str(cave_label),
                    "file_path": pf,
                    "has_meta": bool(has_meta),
                    "selected_indices": np.arange(int(take), dtype=np.int64),
                    "relabel_mask": np.zeros(int(take), dtype=bool),
                })
                cave_discovered += int(take)
                cave_remaining -= int(take)

        cave_target_final = int(cave_target_requested)
        if cave_target_requested > 0 and cave_discovered <= 0:
            raise ValueError(
                f"Cave matching requested {cave_target_requested:,} cave samples, but none were available "
                f"for biome '{cave_biome_name}'"
            )
        if cave_discovered < cave_target_requested:
            print(
                f"[cave_match][warn] Requested {cave_target_requested:,} cave samples to match village, "
                f"but only found {cave_discovered:,}. Will duplicate cave samples to fill the shortfall."
            )
    else:
        cave_file_plans = []

    relabeled_village_target = int(sum(requested_village_by_source.values()))
    if use_cave_match and (relabeled_village_target + cave_target_final > final_total):
        raise ValueError(
            f"Too many special-class samples: village={relabeled_village_target:,}, "
            f"cave={cave_target_final:,}, natural_total={final_total:,}"
        )

    non_village_keep_total = int(final_total - relabeled_village_target - cave_target_final)
    non_village_available_by_source = {
        b: 0 for b in weighted_biomes
    }
    for plan in natural_file_plans:
        biome = str(plan["source_biome"])
        relabel_mask = plan["relabel_mask"]
        if relabel_mask is None:
            relabel_mask = np.zeros(len(plan["selected_indices"]), dtype=bool)
            plan["relabel_mask"] = relabel_mask
        non_village_available_by_source[biome] += int(np.count_nonzero(~relabel_mask))

    if use_cave_match:
        non_village_keep_by_source = _largest_remainder_targets(
            total=non_village_keep_total,
            normalized_weights=normalized_distribution,
            ordered_biomes=weighted_biomes,
        )
    else:
        non_village_keep_by_source = {
            str(k): int(v) for k, v in non_village_available_by_source.items()
        }

    write_file_plans = []
    non_village_kept_by_source = {b: 0 for b in weighted_biomes}
    non_village_candidates_by_source = {b: [] for b in weighted_biomes}
    for plan in natural_file_plans:
        biome = str(plan["source_biome"])
        relabel_mask = plan["relabel_mask"]
        relabel_idx = np.flatnonzero(relabel_mask)
        non_village_idx = np.flatnonzero(~relabel_mask)
        if len(non_village_idx) > 0:
            non_village_candidates_by_source[biome].append({
                "file_path": str(plan["file_path"]),
                "has_meta": bool(plan["has_meta"]),
                "selected_indices": np.asarray(plan["selected_indices"][non_village_idx], dtype=np.int64),
            })
        remaining_non_village = int(
            non_village_keep_by_source[biome] - non_village_kept_by_source[biome]
        )
        keep_non_village = max(0, min(len(non_village_idx), remaining_non_village))
        kept_non_village_idx = non_village_idx[:keep_non_village]
        kept_idx = np.concatenate([relabel_idx, kept_non_village_idx]).astype(np.int64, copy=False)
        kept_idx.sort()
        kept_mask = relabel_mask[kept_idx]
        write_file_plans.append({
            "source_biome": biome,
            "output_label": None,
            "file_path": plan["file_path"],
            "has_meta": bool(plan["has_meta"]),
            "selected_indices": kept_idx,
            "relabel_mask": kept_mask,
        })
        non_village_kept_by_source[biome] += int(keep_non_village)

    bootstrapped_by_label = {}
    for biome in weighted_biomes:
        shortfall = int(non_village_keep_by_source[biome] - non_village_kept_by_source[biome])
        if shortfall <= 0:
            continue
        candidates = non_village_candidates_by_source.get(biome, [])
        if len(candidates) == 0:
            raise ValueError(
                f"Biome '{biome}' needs {shortfall:,} bootstrapped samples, but no non-village chunks remained after relabeling"
            )
        print(f"[bootstrap] Filling {shortfall:,} samples for biome '{biome}' by duplication.")
        cand_idx = 0
        while shortfall > 0:
            cand = candidates[cand_idx % len(candidates)]
            cand_indices = np.asarray(cand["selected_indices"], dtype=np.int64)
            if cand_indices.size == 0:
                cand_idx += 1
                continue
            take = min(shortfall, int(cand_indices.size))
            dup_indices = np.array(cand_indices[:take], dtype=np.int64, copy=True)
            write_file_plans.append({
                "source_biome": str(biome),
                "output_label": None,
                "file_path": str(cand["file_path"]),
                "has_meta": bool(cand["has_meta"]),
                "selected_indices": dup_indices,
                "relabel_mask": np.zeros(take, dtype=bool),
            })
            non_village_kept_by_source[biome] += int(take)
            bootstrapped_by_label[biome] = bootstrapped_by_label.get(biome, 0) + int(take)
            shortfall -= int(take)
            cand_idx += 1

    write_file_plans.extend(cave_file_plans)

    if use_cave_match and cave_target_final > 0:
        cave_unique = int(sum(len(plan["selected_indices"]) for plan in cave_file_plans))
        cave_shortfall = int(cave_target_final - cave_unique)
        if cave_shortfall > 0:
            if len(cave_file_plans) == 0:
                raise ValueError(
                    f"Cave matching needs {cave_shortfall:,} duplicated cave samples, but no cave source plans were created"
                )
            print(f"[bootstrap] Filling {cave_shortfall:,} cave samples by duplication.")
            cand_idx = 0
            while cave_shortfall > 0:
                cand = cave_file_plans[cand_idx % len(cave_file_plans)]
                cand_indices = np.asarray(cand["selected_indices"], dtype=np.int64)
                if cand_indices.size == 0:
                    cand_idx += 1
                    continue
                take = min(cave_shortfall, int(cand_indices.size))
                dup_indices = np.array(cand_indices[:take], dtype=np.int64, copy=True)
                write_file_plans.append({
                    "source_biome": str(cave_biome_name),
                    "output_label": str(cave_label),
                    "file_path": str(cand["file_path"]),
                    "has_meta": bool(cand["has_meta"]),
                    "selected_indices": dup_indices,
                    "relabel_mask": np.zeros(take, dtype=bool),
                })
                bootstrapped_by_label[cave_label] = bootstrapped_by_label.get(cave_label, 0) + int(take)
                cave_shortfall -= int(take)
                cand_idx += 1

    total_samples = int(sum(len(plan["selected_indices"]) for plan in write_file_plans))
    print(f"\nTotal samples to process (distributed): {total_samples:,}")
    print(f"Voxel shape per sample: {voxel_shape}")
    if has_any_metadata:
        print(f"Metadata shape per sample: {ref_meta_shape}")
    if use_cave_match:
        print(
            f"[cave_match] village target={relabeled_village_target:,}, "
            f"cave target={cave_target_final:,}, remaining non-village natural={non_village_keep_total:,}"
        )
    print("=" * 60)
    _log_memory("pass1_end")

    print("\nAllocating memory-mapped arrays...")
    _log_memory("before_memmap_allocation")

    out_p = Path(output_path)
    if out_p.suffix:
        dataset_dir = Path(str(out_p.with_suffix("")) + "_dir")
    else:
        dataset_dir = out_p
    os.makedirs(dataset_dir, exist_ok=True)

    voxels_path = dataset_dir / "voxels.npy"
    labels_path = dataset_dir / "biome_labels.npy"
    metadata_path = dataset_dir / "metadata.npy"
    metadata_mask_path = dataset_dir / "metadata_mask.npy"
    manifest_path = dataset_dir / "manifest.json"

    voxel_out_dtype = np.int32 if (apply_metadata_remap and has_any_metadata) else voxel_dtype

    voxels_mmap = np.lib.format.open_memmap(
        str(voxels_path),
        mode="w+",
        dtype=voxel_out_dtype,
        shape=(total_samples,) + voxel_shape,
    )
    print(
        f"Voxels memmap: {voxels_mmap.shape}, {voxels_mmap.dtype}, ~{voxels_mmap.nbytes / (1024**3):.2f} GB"
    )

    biome_labels_all = np.empty(total_samples, dtype=object)

    meta_mmap = None
    meta_mask_all = None
    if has_any_metadata:
        meta_mask_all = np.zeros(total_samples, dtype=bool)
        if ref_meta_dtype != object and store_metadata_payload:
            meta_mmap = np.lib.format.open_memmap(
                str(metadata_path),
                mode="w+",
                dtype=ref_meta_dtype,
                shape=(total_samples,) + ref_meta_shape,
            )
            print(f"Metadata memmap: {meta_mmap.shape}, {meta_mmap.dtype}")
        else:
            print(
                "[info] Metadata payload will not be aggregated (either object dtype or disabled); writing mask only"
            )

    print("\n" + "=" * 60)
    print("PASS 2 (distributed): Loading data into memory-mapped arrays...")
    print("=" * 60)
    _log_memory("pass2_start")

    idx = 0
    relabeled_village_total = 0
    relabeled_village_by_source = {}
    for file_idx, plan in enumerate(write_file_plans):
        biome = str(plan["source_biome"])
        pf = str(plan["file_path"])
        selected_indices = np.asarray(plan["selected_indices"], dtype=np.int64)
        take = int(len(selected_indices))
        output_label = plan["output_label"]
        relabel_mask = np.asarray(plan["relabel_mask"], dtype=bool)
        print(f"Loading {os.path.basename(pf)} ({take} samples for biome '{biome}')...")
        with np.load(pf, allow_pickle=True) as data:
            raw_vox_batch = data["voxels"][selected_indices]
            vox_batch = raw_vox_batch
            batch_labels = np.array(data["biome_labels"][selected_indices], dtype=object, copy=True)

            has_meta = "metadata" in data.files

            if output_label is not None:
                batch_labels[:] = output_label
            elif use_village_relabel and biome in village_relabel_source_set:
                num_relabeled = int(np.count_nonzero(relabel_mask))
                if num_relabeled > 0:
                    batch_labels[relabel_mask] = village_relabel_label
                    relabeled_village_total += num_relabeled
                    relabeled_village_by_source[biome] = (
                        relabeled_village_by_source.get(biome, 0) + num_relabeled
                    )

            if apply_metadata_remap and has_meta and ("metadata" in data.files):
                meta_data = data["metadata"][selected_indices]
                if tuple(meta_data.shape[1:]) == voxel_shape:
                    remapped = np.empty_like(vox_batch, dtype=voxel_out_dtype)
                    for j in range(take):
                        remapped[j] = remap_slabs_and_stairs_with_metadata(
                            vox_batch[j], meta_data[j], simplify=False
                        )
                    vox_batch = remapped
                else:
                    print("  [warn] Metadata shape mismatch for remap; skipping remap for this file")

            biome_labels_all[idx : idx + take] = batch_labels
            voxels_mmap[idx : idx + take] = vox_batch

            if has_any_metadata and meta_mask_all is not None:
                if has_meta and ("metadata" in data.files):
                    if meta_mmap is not None:
                        meta_data = data["metadata"][selected_indices]
                        if tuple(meta_data.shape[1:]) == ref_meta_shape:
                            meta_mmap[idx : idx + take] = meta_data
                            meta_mask_all[idx : idx + take] = True
                        else:
                            print("  [warn] Metadata shape mismatch, marking mask False for this file")
                            meta_mask_all[idx : idx + take] = False
                    else:
                        meta_mask_all[idx : idx + take] = True
                else:
                    meta_mask_all[idx : idx + take] = False

            idx += take

            if (file_idx + 1) % 10 == 0:
                _log_memory(f"pass2_file_{file_idx+1}/{len(write_file_plans)}")

    if idx != total_samples:
        raise RuntimeError(
            f"Internal error: filled {idx:,} samples but expected {total_samples:,}. "
            f"Refusing to write an inconsistent memmap dataset."
        )

    print(f"\nLoaded all {total_samples:,} samples")
    _log_memory("pass2_end")

    if shuffle and total_samples > 1:
        print("\nShuffling data...")
        _log_memory("before_shuffle")
        perm = rng.permutation(total_samples)

        chunk_size = min(10000, total_samples)
        temp_voxel_chunk = np.empty((chunk_size,) + voxel_shape, dtype=voxel_dtype)

        print("Shuffling voxels (in chunks)...")
        for i in range(0, total_samples, chunk_size):
            end = min(i + chunk_size, total_samples)
            current_chunk_size = end - i
            temp_voxel_chunk[:current_chunk_size] = voxels_mmap[perm[i:end]]
            voxels_mmap[i:end] = temp_voxel_chunk[:current_chunk_size]

        print("Shuffling biome labels...")
        biome_labels_all[:total_samples] = biome_labels_all[:total_samples][perm]

        if has_any_metadata and meta_mask_all is not None:
            print("Shuffling metadata mask...")
            meta_mask_all[:total_samples] = meta_mask_all[:total_samples][perm]

        if has_any_metadata and meta_mmap is not None:
            print("Shuffling numeric metadata (in chunks)...")
            temp_meta_chunk = np.empty((chunk_size,) + ref_meta_shape, dtype=ref_meta_dtype)
            for i in range(0, total_samples, chunk_size):
                end = min(i + chunk_size, total_samples)
                current_chunk_size = end - i
                temp_meta_chunk[:current_chunk_size] = meta_mmap[perm[i:end]]
                meta_mmap[i:end] = temp_meta_chunk[:current_chunk_size]

        _log_memory("after_shuffle")

    print("\nSaving dataset shards and manifest...")

    np.save(str(labels_path), biome_labels_all[:total_samples])
    if has_any_metadata and meta_mask_all is not None:
        np.save(str(metadata_mask_path), meta_mask_all[:total_samples])

    labels, counts = np.unique(biome_labels_all[:total_samples], return_counts=True)
    biome_counts = {str(l): int(c) for l, c in zip(labels.tolist(), counts.tolist())}
    biome_fractions = {
        str(label): (float(count) / float(total_samples) if total_samples > 0 else 0.0)
        for label, count in biome_counts.items()
    }
    manifest = {
        "format_version": 1,
        "num_samples": int(total_samples),
        "voxel_shape": list(voxel_shape),
        "voxel_dtype": str(voxel_out_dtype),
        "paths": {
            "voxels": os.path.basename(str(voxels_path)),
            "biome_labels": os.path.basename(str(labels_path)),
            "metadata": (os.path.basename(str(metadata_path)) if (meta_mmap is not None) else None),
            "metadata_mask": (
                os.path.basename(str(metadata_mask_path))
                if (has_any_metadata and meta_mask_all is not None)
                else None
            ),
        },
        "has_metadata": bool(has_any_metadata),
        "metadata_dtype": (str(ref_meta_dtype) if has_any_metadata else None),
        "metadata_applied_to_voxels": bool(apply_metadata_remap and has_any_metadata),
        "stored_metadata_payload": bool(meta_mmap is not None),
        "biome_counts": biome_counts,
        "biome_fractions": biome_fractions,
        "source": {
            "biome_dir": str(biome_dir),
            "biomes": list(weighted_biomes) + ([str(cave_biome_name)] if use_cave_match else []),
            "part_pattern": str(part_pattern),
            "single_file_overrides": single_file_overrides or {},
            "distribution": {
                "requested_total_size": int(total_size),
                "requested_weights": {str(k): float(v) for k, v in biome_distribution.items()},
                "normalized_weights_used": {
                    str(k): float(v) for k, v in normalized_distribution.items()
                },
                "target_by_biome_requested": {str(k): int(v) for k, v in target_by_biome.items()},
                "target_by_biome_final": {str(k): int(v) for k, v in final_target_by_biome.items()},
                "final_total_size": int(total_samples),
                "limiting_scale": float(limiting_scale),
                "dropped_biomes_no_files": list(dropped_biomes),
                "ignored_biomes_no_weight": list(ignored_biomes),
                "missing_distribution_biomes": list(missing_distribution_biomes),
            },
            "village_relabel": {
                "enabled": bool(use_village_relabel),
                "label": (str(village_relabel_label) if use_village_relabel else None),
                "threshold_exclusive": (
                    int(village_relabel_threshold) if use_village_relabel else None
                ),
                "blocks": (list(village_relabel_blocks) if use_village_relabel else []),
                "source_biomes": (
                    sorted(village_relabel_source_set) if use_village_relabel else []
                ),
                "total_relabeled": int(relabeled_village_total),
                "relabeled_by_source_biome": {
                    str(k): int(v) for k, v in sorted(relabeled_village_by_source.items())
                },
            },
            "cave_match": {
                "enabled": bool(use_cave_match),
                "cave_biome_name": (str(cave_biome_name) if use_cave_match else None),
                "cave_label": (str(cave_label) if use_cave_match else None),
                "requested_cave_count": int(cave_target_requested),
                "final_cave_count": int(cave_target_final),
                "non_village_keep_total": int(non_village_keep_total),
                "non_village_keep_by_source_biome": {
                    str(k): int(v) for k, v in sorted(non_village_keep_by_source.items())
                },
            },
            "bootstrap": {
                "enabled": True,
                "bootstrapped_by_label": {
                    str(k): int(v) for k, v in sorted(bootstrapped_by_label.items())
                },
            },
        },
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    print(f"\nSaved dataset directory to {dataset_dir}")
    print(f"  Voxels: {voxels_mmap.shape}")
    print(f"  Labels: {(total_samples,)} -> {labels_path}")
    if has_any_metadata:
        print(f"  Metadata: {'numeric' if meta_mmap is not None else 'skipped (object)'}")
    print(f"  Manifest: {manifest_path}")
    _log_memory("after_save")

    print("\nCounts per biome label:")
    for label, count in sorted(biome_counts.items(), key=lambda x: str(x[0])):
        print(f"  {label}: {count:,}")

    print("\nFinal biome distribution:")
    for label, count in sorted(biome_counts.items(), key=lambda x: str(x[0])):
        frac = (float(count) / float(total_samples)) if total_samples > 0 else 0.0
        print(f"  {label}: {frac:.6f}")

    print("\nFinal biome distribution dict:")
    print({label: biome_fractions[label] for label in sorted(biome_fractions)})

    if use_village_relabel:
        print(
            f"\nVillage relabel summary: {relabeled_village_total:,} chunks relabeled to "
            f"'{village_relabel_label}'"
        )
        for source_biome, count in sorted(relabeled_village_by_source.items()):
            print(f"  {source_biome} -> {village_relabel_label}: {count:,}")

    if use_cave_match:
        print(
            f"\nCave match summary: requested={cave_target_requested:,}, "
            f"final={cave_target_final:,}, label='{cave_label}'"
        )

    if len(bootstrapped_by_label) > 0:
        print("\nBootstrap summary:")
        for label, count in sorted(bootstrapped_by_label.items()):
            print(f"  {label}: {count:,}")

    print("\nDistributed discovery summary (requested vs discovered):")
    for biome in weighted_biomes:
        got = int(discovered_by_biome.get(biome, 0))
        req = int(target_by_biome.get(biome, 0))
        status = "ok" if got >= req else "short"
        print(f"  {biome}: {got:,}/{req:,} ({status})")

    return str(dataset_dir)
