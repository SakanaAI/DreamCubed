#!/usr/bin/env python3
"""
Compute per-biome block-distribution statistics directly in voxel space.

This script is designed to compare:
1) a processed or raw memmapped dataset directory containing a manifest, and
2) a directory with generated samples named `generated_<biome>.pt`.

It caches the expensive dataset-side counts so repeated comparisons against
different models do not need to rescan the dataset voxels.

Main metrics:
  - vocabulary coverage: fraction of real block names that also appear in generated
  - KL divergence: D_KL(real || generated) over block-name frequencies

Additional cheap complements included here:
  - support precision: fraction of generated block names that also appear in real
  - Jensen-Shannon divergence: symmetric, bounded complement to KL
  - total variation distance: intuitive L1-style distribution gap
  - non-air variants of the above metrics, since AIR can dominate voxel counts
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from data_utils import BlockBiomeConverter, load_block_to_str_mapping


CACHE_FORMAT_VERSION = 1
DEFAULT_SAMPLE_GLOB = "generated_*.pt"
DEFAULT_SANITY_BLOCKS = [
    "AIR",
    "WATER",
    "STONE",
    "DIRT",
    "GRASS",
    "SAND",
    "SNOW",
    "LOG",
    "LEAVES",
]


def _dataset_name_from_path(path: str) -> str:
    base = os.path.basename(str(path).rstrip("/\\"))
    return os.path.splitext(base)[0]


def _infer_biome_name_from_filename(path: str) -> str:
    base = os.path.basename(str(path))
    stem, _ = os.path.splitext(base)
    if stem.startswith("generated_"):
        stem = stem[len("generated_") :]
    if "_samples" in stem:
        return stem.split("_samples", 1)[0]
    return stem


def _load_samples_tensor_from_file(path: str) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, torch.Tensor):
        return payload
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported sample payload type in {path}: {type(payload)}")
    for key in ("voxels", "samples", "x", "data"):
        value = payload.get(key)
        if isinstance(value, torch.Tensor):
            return value
    raise KeyError(
        f"No tensor found in {path}. Expected one of keys "
        f"'voxels', 'samples', 'x', or 'data'; found {list(payload.keys())}."
    )


def _stat_signature(path: Path) -> dict[str, Any]:
    st = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_int_keys(d: dict[Any, Any] | None) -> dict[int, Any]:
    if not d:
        return {}
    return {int(k): v for k, v in d.items()}


def _normalize_index_to_biome(d: dict[Any, Any] | None) -> dict[int, str]:
    if not d:
        return {}
    return {int(k): str(v) for k, v in d.items()}


def _normalize_block_to_str(d: dict[Any, Any] | None) -> dict[int, str]:
    if not d:
        return {}
    return {int(k): str(v) for k, v in d.items()}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _resolve_output_dir(dataset_dir: Path, samples_dir: Path | None, output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir)
    dataset_name = _dataset_name_from_path(str(dataset_dir))
    model_name = _dataset_name_from_path(str(samples_dir)) if samples_dir is not None else "dataset_only"
    return Path("analysis") / "voxel_block_stats" / dataset_name / model_name


def _resolve_cache_path(dataset_dir: Path, cache_path: str | None) -> Path:
    if cache_path:
        return Path(cache_path)
    return dataset_dir / "analysis_cache" / "voxel_block_stats_dataset_cache.pt"


def _resolve_dataset_mappings_path(dataset_dir: Path, manifest: dict[str, Any], explicit_path: str | None) -> Path | None:
    if explicit_path:
        p = Path(explicit_path)
        if not p.exists():
            raise FileNotFoundError(f"--dataset-mappings-path does not exist: {p}")
        return p

    source_path = manifest.get("source", {}).get("path", "")
    candidates = []
    if source_path:
        candidates.append(dataset_dir.parent / f"{Path(source_path).name}_cc_oh_mappings.pt")

    name = dataset_dir.name
    base_name = (
        name.removesuffix("_cc_val_dir")
        .removesuffix("_cc_train_dir")
        .removesuffix("_cc_dir")
        .removesuffix("_val_dir")
        .removesuffix("_train_dir")
        .removesuffix("_dir")
    )
    candidates.extend(
        [
            dataset_dir.parent / f"{base_name}_cc_oh_mappings.pt",
            dataset_dir.parent / f"{name}_cc_oh_mappings.pt",
        ]
    )

    seen = set()
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return None


def _load_dataset_context(dataset_dir: Path, dataset_mappings_path: str | None) -> dict[str, Any]:
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {dataset_dir}")

    manifest = _load_json(manifest_path)
    paths = manifest.get("paths", {})
    voxels_path = dataset_dir / paths["voxels"]
    labels_path = dataset_dir / paths["biome_labels"]

    if not voxels_path.exists():
        raise FileNotFoundError(f"voxels file not found: {voxels_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"biome labels file not found: {labels_path}")

    is_processed = bool(manifest.get("is_processed", False))
    mappings_path = _resolve_dataset_mappings_path(dataset_dir, manifest, dataset_mappings_path) if is_processed else None
    if is_processed and mappings_path is None:
        raise FileNotFoundError(
            "Could not locate the processed dataset mappings file. "
            "Pass --dataset-mappings-path explicitly."
        )

    converter = BlockBiomeConverter.from_mappings(str(mappings_path)) if mappings_path is not None else None

    return {
        "dataset_dir": dataset_dir,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "voxels_path": voxels_path,
        "labels_path": labels_path,
        "is_processed": is_processed,
        "mappings_path": mappings_path,
        "converter": converter,
    }


class BlockNameIndexer:
    def __init__(self, block_to_str: dict[int, str], extra_block_ids: Iterable[int] = ()) -> None:
        self.block_to_str = _normalize_block_to_str(block_to_str)
        self.block_names: list[str] = []
        self.name_to_index: dict[str, int] = {}
        self.block_id_to_name_index = np.full(1, -1, dtype=np.int64)
        self._torch_luts: dict[str, torch.Tensor] = {}

        all_block_ids = sorted({int(b) for b in self.block_to_str.keys()} | {int(b) for b in extra_block_ids})
        for block_id in all_block_ids:
            self._register_block_id(block_id)

    @property
    def num_names(self) -> int:
        return len(self.block_names)

    def _canonical_name(self, block_id: int) -> str:
        return self.block_to_str.get(int(block_id), f"UNKNOWN_BLOCK_{int(block_id)}")

    def _register_name(self, name: str) -> int:
        idx = self.name_to_index.get(name)
        if idx is not None:
            return idx
        idx = len(self.block_names)
        self.name_to_index[name] = idx
        self.block_names.append(name)
        return idx

    def _register_block_id(self, block_id: int) -> None:
        block_id = int(block_id)
        if block_id >= self.block_id_to_name_index.shape[0]:
            pad = block_id + 1 - self.block_id_to_name_index.shape[0]
            self.block_id_to_name_index = np.pad(
                self.block_id_to_name_index,
                (0, pad),
                mode="constant",
                constant_values=-1,
            )
        if self.block_id_to_name_index[block_id] >= 0:
            return
        name_idx = self._register_name(self._canonical_name(block_id))
        self.block_id_to_name_index[block_id] = name_idx
        self._torch_luts.clear()

    def ensure_block_ids(self, block_ids: Iterable[int]) -> bool:
        before = self.num_names
        for block_id in block_ids:
            self._register_block_id(int(block_id))
        return self.num_names != before

    def pad_vector(self, vector: np.ndarray) -> np.ndarray:
        if vector.shape[0] >= self.num_names:
            return vector
        return np.pad(vector, (0, self.num_names - vector.shape[0]), mode="constant")

    def get_lut_torch(self, device: torch.device) -> torch.Tensor:
        key = str(device)
        cached = self._torch_luts.get(key)
        if cached is not None and cached.shape[0] == self.block_id_to_name_index.shape[0]:
            return cached
        lut = torch.as_tensor(self.block_id_to_name_index, dtype=torch.long, device=device)
        self._torch_luts[key] = lut
        return lut


def _pad_counts_dict(counts_by_biome: dict[str, np.ndarray], width: int) -> None:
    for biome, vec in list(counts_by_biome.items()):
        if vec.shape[0] < width:
            counts_by_biome[biome] = np.pad(vec, (0, width - vec.shape[0]), mode="constant")


def _labels_to_indices(labels: np.ndarray) -> np.ndarray:
    if labels.ndim == 2:
        return np.asarray(np.argmax(labels, axis=1), dtype=np.int64)
    return np.asarray(labels, dtype=np.int64).reshape(-1)


def _dict_to_dense_lut(mapping: dict[int, int], fill_value: int = -1) -> np.ndarray:
    if not mapping:
        return np.full(1, fill_value, dtype=np.int64)
    max_key = max(int(k) for k in mapping.keys())
    lut = np.full(max_key + 1, fill_value, dtype=np.int64)
    for key, value in mapping.items():
        lut[int(key)] = int(value)
    return lut


def _count_block_name_occurrences(
    values: np.ndarray | torch.Tensor,
    *,
    value_kind: str,
    indexer: BlockNameIndexer,
    index_to_block_lut: np.ndarray | None = None,
    device: torch.device,
) -> np.ndarray:
    if isinstance(values, np.ndarray):
        arr = values
    else:
        arr = values.detach().cpu().numpy()

    if arr.size == 0:
        return np.zeros(indexer.num_names, dtype=np.int64)

    if device.type == "cuda":
        tensor = torch.as_tensor(arr, dtype=torch.long, device=device).reshape(-1)
        if value_kind == "indices":
            if index_to_block_lut is None:
                raise ValueError("index_to_block_lut is required for index-valued inputs.")
            lut = torch.as_tensor(index_to_block_lut, dtype=torch.long, device=device)
            tensor = lut[tensor]

        unique_ids = torch.unique(tensor).detach().cpu().tolist()
        indexer.ensure_block_ids(unique_ids)
        name_lut = indexer.get_lut_torch(device)
        name_indices = name_lut[tensor]
        counts = torch.bincount(name_indices, minlength=indexer.num_names)
        return counts.detach().cpu().numpy().astype(np.int64, copy=False)

    arr = np.asarray(arr, dtype=np.int64).reshape(-1)
    if value_kind == "indices":
        if index_to_block_lut is None:
            raise ValueError("index_to_block_lut is required for index-valued inputs.")
        arr = index_to_block_lut[arr]

    indexer.ensure_block_ids(np.unique(arr).tolist())
    name_indices = indexer.block_id_to_name_index[arr]
    return np.bincount(name_indices, minlength=indexer.num_names).astype(np.int64, copy=False)


def _prepare_generated_tensor(
    tensor: torch.Tensor,
    *,
    generated_format: str,
    generated_converter: BlockBiomeConverter | None,
    source_path: Path,
) -> tuple[torch.Tensor, str]:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor from {source_path}, got {type(tensor)}")

    if generated_format == "auto":
        if tensor.ndim == 5:
            generated_format = "one_hot"
        elif tensor.is_floating_point():
            if tensor.ndim == 4 and generated_converter is not None and generated_converter.index_to_block is not None:
                num_blocks = len(_normalize_int_keys(generated_converter.index_to_block))
                if tensor.shape[0] == num_blocks:
                    generated_format = "one_hot"
                else:
                    raise ValueError(
                        f"Ambiguous floating-point tensor in {source_path}. "
                        "Pass --generated-format explicitly."
                    )
            else:
                raise ValueError(
                    f"Ambiguous floating-point tensor in {source_path}. "
                    "Pass --generated-format explicitly."
                )
        else:
            generated_format = "block_ids"

    if generated_format == "one_hot":
        if generated_converter is None or generated_converter.index_to_block is None:
            raise ValueError("--generated-format=one_hot requires --generated-mappings-path.")
        if tensor.ndim == 5:
            tensor = torch.argmax(tensor, dim=1)
        elif tensor.ndim == 4:
            tensor = torch.argmax(tensor, dim=0).unsqueeze(0)
        else:
            raise ValueError(f"Expected 4D or 5D one-hot tensor in {source_path}, got shape {tuple(tensor.shape)}")
        return tensor.long().cpu(), "indices"

    if generated_format == "indices":
        if generated_converter is None or generated_converter.index_to_block is None:
            raise ValueError("--generated-format=indices requires --generated-mappings-path.")
        if tensor.ndim == 5:
            tensor = torch.argmax(tensor, dim=1)
        elif tensor.ndim == 4 and tensor.is_floating_point():
            tensor = torch.argmax(tensor, dim=0).unsqueeze(0)
        elif tensor.ndim not in (3, 4):
            raise ValueError(f"Unsupported indexed tensor shape in {source_path}: {tuple(tensor.shape)}")
        return tensor.long().cpu(), "indices"

    if generated_format == "block_ids":
        if tensor.ndim not in (3, 4):
            raise ValueError(
                f"Expected a 3D or 4D block-ID tensor in {source_path}, got shape {tuple(tensor.shape)}"
            )
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        return tensor.long().cpu(), "block_ids"

    raise ValueError(f"Unsupported generated format: {generated_format}")


def _summarize_distribution_metrics(
    real_counts: np.ndarray,
    gen_counts: np.ndarray,
    *,
    exclude_index: int | None = None,
    eps: float = 1e-12,
) -> dict[str, float]:
    real = np.asarray(real_counts, dtype=np.float64)
    gen = np.asarray(gen_counts, dtype=np.float64)

    if exclude_index is not None and 0 <= exclude_index < real.shape[0]:
        real = real.copy()
        gen = gen.copy()
        real[exclude_index] = 0.0
        gen[exclude_index] = 0.0

    real_support = real > 0
    gen_support = gen > 0
    real_vocab = int(real_support.sum())
    gen_vocab = int(gen_support.sum())
    overlap = int(np.logical_and(real_support, gen_support).sum())

    coverage = float(overlap / real_vocab) if real_vocab > 0 else math.nan
    precision = float(overlap / gen_vocab) if gen_vocab > 0 else math.nan

    real_total = float(real.sum())
    gen_total = float(gen.sum())
    if real_total <= 0.0 or gen_total <= 0.0:
        return {
            "coverage": coverage,
            "precision": precision,
            "kl_real_to_gen": math.nan,
            "js_divergence": math.nan,
            "tv_distance": math.nan,
            "real_unique_blocks": float(real_vocab),
            "gen_unique_blocks": float(gen_vocab),
        }

    p = real / real_total
    q = gen / gen_total
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    p = p / p.sum()
    q = q / q.sum()

    m = 0.5 * (p + q)
    kl_real_to_gen = float(np.sum(p * np.log(p / q)))
    js_divergence = float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))
    tv_distance = float(0.5 * np.abs(p - q).sum())

    return {
        "coverage": coverage,
        "precision": precision,
        "kl_real_to_gen": kl_real_to_gen,
        "js_divergence": js_divergence,
        "tv_distance": tv_distance,
        "real_unique_blocks": float(real_vocab),
        "gen_unique_blocks": float(gen_vocab),
    }


def _compute_dataset_cache_signature(ctx: dict[str, Any]) -> dict[str, Any]:
    signature = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "dataset_dir": str(Path(ctx["dataset_dir"]).resolve()),
        "manifest": _stat_signature(Path(ctx["manifest_path"])),
        "voxels": _stat_signature(Path(ctx["voxels_path"])),
        "labels": _stat_signature(Path(ctx["labels_path"])),
        "is_processed": bool(ctx["is_processed"]),
    }
    if ctx["mappings_path"] is not None:
        signature["mappings"] = _stat_signature(Path(ctx["mappings_path"]))
    return signature


def _save_dataset_cache(cache_path: Path, payload: dict[str, Any]) -> None:
    _ensure_dir(cache_path.parent)
    torch.save(payload, str(cache_path))


def _load_dataset_cache_if_valid(cache_path: Path, signature: dict[str, Any]) -> dict[str, Any] | None:
    if not cache_path.exists():
        return None
    try:
        payload = torch.load(str(cache_path), map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"[cache] Failed to load dataset cache ({cache_path}): {exc}. Recomputing.")
        return None
    if payload.get("signature") != signature:
        return None
    return payload


def _compute_dataset_stats(
    ctx: dict[str, Any],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    dataset_dir = Path(ctx["dataset_dir"])
    manifest = ctx["manifest"]
    voxels = np.load(str(ctx["voxels_path"]), mmap_mode="r")
    labels = np.load(str(ctx["labels_path"]), mmap_mode="r", allow_pickle=True)
    print(f"[dataset] Loaded voxels {voxels.shape} from {ctx['voxels_path']}")
    print(f"[dataset] Loaded labels {labels.shape} from {ctx['labels_path']}")

    is_processed = bool(ctx["is_processed"])
    converter: BlockBiomeConverter | None = ctx["converter"]

    if is_processed:
        if converter is None or converter.index_to_block is None or converter.index_to_biome is None:
            raise ValueError("Processed datasets require block and biome mappings.")
        index_to_block = _normalize_int_keys(converter.index_to_block)
        index_to_biome = _normalize_index_to_biome(converter.index_to_biome)
        block_to_str = _normalize_block_to_str(converter.block_to_str)
        labels_indices = _labels_to_indices(labels)
        index_to_block_lut = _dict_to_dense_lut({int(k): int(v) for k, v in index_to_block.items()})
        indexer = BlockNameIndexer(block_to_str, extra_block_ids=index_to_block.values())
        biome_names = [index_to_biome[idx] for idx in sorted(index_to_biome.keys())]
        dataset_value_kind = "indices"
    else:
        block_to_str = load_block_to_str_mapping()
        indexer = BlockNameIndexer(block_to_str)
        labels_indices = None
        biome_names = sorted({str(x) for x in np.asarray(labels).reshape(-1).tolist()})
        index_to_block_lut = None
        dataset_value_kind = "block_ids"

    counts_by_biome = {biome: np.zeros(indexer.num_names, dtype=np.int64) for biome in biome_names}
    chunk_counts_by_biome = {biome: 0 for biome in biome_names}
    voxel_counts_by_biome = {biome: 0 for biome in biome_names}

    num_samples = int(voxels.shape[0])
    num_batches = max(1, math.ceil(num_samples / max(1, int(batch_size))))
    progress_stride = max(1, num_batches // 10)

    print(f"[dataset] Counting per-biome block names across {num_samples:,} samples ...")
    for batch_idx, start in enumerate(range(0, num_samples, int(batch_size)), start=1):
        end = min(start + int(batch_size), num_samples)
        batch_vox = np.asarray(voxels[start:end])
        if batch_vox.ndim == 5:
            batch_vox = np.argmax(batch_vox, axis=1)

        if is_processed:
            batch_labels = labels_indices[start:end]
            unique_labels = np.unique(batch_labels)
            for label_idx in unique_labels.tolist():
                biome = index_to_biome[int(label_idx)]
                sample_mask = batch_labels == int(label_idx)
                selected = batch_vox[sample_mask]
                prev_width = indexer.num_names
                counts = _count_block_name_occurrences(
                    selected,
                    value_kind=dataset_value_kind,
                    indexer=indexer,
                    index_to_block_lut=index_to_block_lut,
                    device=device,
                )
                if indexer.num_names != prev_width:
                    _pad_counts_dict(counts_by_biome, indexer.num_names)
                counts_by_biome[biome] = indexer.pad_vector(counts_by_biome[biome])
                counts_by_biome[biome] += counts
                chunk_counts_by_biome[biome] += int(selected.shape[0])
                voxel_counts_by_biome[biome] += int(selected.size)
        else:
            batch_labels = np.asarray(labels[start:end]).reshape(-1)
            unique_labels = sorted({str(x) for x in batch_labels.tolist()})
            for biome in unique_labels:
                sample_mask = np.asarray(batch_labels == biome)
                selected = batch_vox[sample_mask]
                prev_width = indexer.num_names
                counts = _count_block_name_occurrences(
                    selected,
                    value_kind=dataset_value_kind,
                    indexer=indexer,
                    index_to_block_lut=None,
                    device=device,
                )
                if indexer.num_names != prev_width:
                    _pad_counts_dict(counts_by_biome, indexer.num_names)
                counts_by_biome[biome] = indexer.pad_vector(counts_by_biome[biome])
                counts_by_biome[biome] += counts
                chunk_counts_by_biome[biome] += int(selected.shape[0])
                voxel_counts_by_biome[biome] += int(selected.size)

        if batch_idx == 1 or batch_idx % progress_stride == 0 or end == num_samples:
            print(f"[dataset] processed batch {batch_idx:,}/{num_batches:,} ({end:,}/{num_samples:,} samples)")

    return {
        "dataset_dir": str(dataset_dir),
        "dataset_name": _dataset_name_from_path(str(dataset_dir)),
        "is_processed": is_processed,
        "block_names": list(indexer.block_names),
        "counts_by_biome": counts_by_biome,
        "chunk_counts_by_biome": chunk_counts_by_biome,
        "voxel_counts_by_biome": voxel_counts_by_biome,
        "manifest": manifest,
    }


def _get_or_compute_dataset_stats(
    ctx: dict[str, Any],
    *,
    cache_path: Path,
    batch_size: int,
    device: torch.device,
    force_recompute: bool,
) -> dict[str, Any]:
    signature = _compute_dataset_cache_signature(ctx)
    if not force_recompute:
        cached = _load_dataset_cache_if_valid(cache_path, signature)
        if cached is not None:
            print(f"[cache] Using cached dataset statistics: {cache_path}")
            return cached["dataset_stats"]

    dataset_stats = _compute_dataset_stats(ctx, batch_size=batch_size, device=device)
    _save_dataset_cache(
        cache_path,
        {
            "signature": signature,
            "dataset_stats": dataset_stats,
        },
    )
    print(f"[cache] Saved dataset statistics cache to {cache_path}")
    return dataset_stats


def _build_indexer_from_dataset_stats(
    dataset_stats: dict[str, Any],
    *,
    converter: BlockBiomeConverter | None,
) -> BlockNameIndexer:
    if converter is not None:
        block_to_str = _normalize_block_to_str(converter.block_to_str)
    else:
        block_to_str = load_block_to_str_mapping()
    indexer = BlockNameIndexer(block_to_str)
    if dataset_stats.get("block_names"):
        for name in dataset_stats["block_names"]:
            indexer._register_name(str(name))
    return indexer


def _load_generated_converter(generated_mappings_path: str | None) -> BlockBiomeConverter | None:
    if not generated_mappings_path:
        return None
    p = Path(generated_mappings_path)
    if not p.exists():
        raise FileNotFoundError(f"--generated-mappings-path does not exist: {p}")
    return BlockBiomeConverter.from_mappings(str(p))


def _compute_generated_stats(
    samples_dir: Path,
    *,
    sample_glob: str,
    generated_format: str,
    generated_converter: BlockBiomeConverter | None,
    indexer: BlockNameIndexer,
    device: torch.device,
) -> dict[str, Any]:
    sample_files = sorted(samples_dir.glob(sample_glob))
    if not sample_files:
        raise FileNotFoundError(f"No sample files matching '{sample_glob}' found in {samples_dir}")

    generated_index_to_block = None
    if generated_converter is not None and generated_converter.index_to_block is not None:
        generated_index_to_block = _normalize_int_keys(generated_converter.index_to_block)
    index_to_block_lut = (
        _dict_to_dense_lut({int(k): int(v) for k, v in generated_index_to_block.items()})
        if generated_index_to_block
        else None
    )

    counts_by_biome: dict[str, np.ndarray] = {}
    chunk_counts_by_biome: dict[str, int] = {}
    voxel_counts_by_biome: dict[str, int] = {}

    print(f"[generated] Counting block names from {len(sample_files)} sample files in {samples_dir}")
    for idx, sample_path in enumerate(sample_files, start=1):
        biome = _infer_biome_name_from_filename(str(sample_path))
        raw_tensor = _load_samples_tensor_from_file(str(sample_path))
        prepared, value_kind = _prepare_generated_tensor(
            raw_tensor,
            generated_format=generated_format,
            generated_converter=generated_converter,
            source_path=sample_path,
        )

        prev_width = indexer.num_names
        counts = _count_block_name_occurrences(
            prepared,
            value_kind=value_kind,
            indexer=indexer,
            index_to_block_lut=index_to_block_lut,
            device=device,
        )
        if indexer.num_names != prev_width:
            _pad_counts_dict(counts_by_biome, indexer.num_names)

        if biome not in counts_by_biome:
            counts_by_biome[biome] = np.zeros(indexer.num_names, dtype=np.int64)
        counts_by_biome[biome] = indexer.pad_vector(counts_by_biome[biome])
        counts_by_biome[biome] += counts
        chunk_counts_by_biome[biome] = chunk_counts_by_biome.get(biome, 0) + int(prepared.shape[0])
        voxel_counts_by_biome[biome] = voxel_counts_by_biome.get(biome, 0) + int(prepared.numel())

        print(
            f"[generated] processed {idx:,}/{len(sample_files):,}: {sample_path.name} "
            f"-> biome={biome}, chunks={prepared.shape[0]:,}"
        )

    return {
        "samples_dir": str(samples_dir),
        "model_name": _dataset_name_from_path(str(samples_dir)),
        "block_names": list(indexer.block_names),
        "counts_by_biome": counts_by_biome,
        "chunk_counts_by_biome": chunk_counts_by_biome,
        "voxel_counts_by_biome": voxel_counts_by_biome,
    }


def _align_counts_to_block_names(stats: dict[str, Any], target_block_names: list[str]) -> None:
    current_names = list(stats.get("block_names", []))
    if current_names == target_block_names:
        return
    current_name_to_index = {name: i for i, name in enumerate(current_names)}
    new_counts: dict[str, np.ndarray] = {}
    for biome, vec in stats["counts_by_biome"].items():
        out = np.zeros(len(target_block_names), dtype=np.int64)
        for new_idx, name in enumerate(target_block_names):
            old_idx = current_name_to_index.get(name)
            if old_idx is not None and old_idx < len(vec):
                out[new_idx] = int(vec[old_idx])
        new_counts[biome] = out
    stats["counts_by_biome"] = new_counts
    stats["block_names"] = list(target_block_names)


def _compute_comparison_rows(
    dataset_stats: dict[str, Any],
    generated_stats: dict[str, Any] | None,
    *,
    block_names: list[str],
) -> list[dict[str, Any]]:
    air_idx = block_names.index("AIR") if "AIR" in block_names else None
    all_biomes = set(dataset_stats["counts_by_biome"].keys())
    if generated_stats is not None:
        all_biomes |= set(generated_stats["counts_by_biome"].keys())

    rows: list[dict[str, Any]] = []
    for biome in sorted(all_biomes):
        real_counts = dataset_stats["counts_by_biome"].get(biome)
        gen_counts = generated_stats["counts_by_biome"].get(biome) if generated_stats is not None else None
        real_chunks = int(dataset_stats["chunk_counts_by_biome"].get(biome, 0))
        gen_chunks = int(generated_stats["chunk_counts_by_biome"].get(biome, 0)) if generated_stats is not None else 0
        real_voxels = int(dataset_stats["voxel_counts_by_biome"].get(biome, 0))
        gen_voxels = int(generated_stats["voxel_counts_by_biome"].get(biome, 0)) if generated_stats is not None else 0

        row: dict[str, Any] = {
            "biome": biome,
            "real_present": bool(real_counts is not None),
            "generated_present": bool(gen_counts is not None),
            "real_chunks": real_chunks,
            "generated_chunks": gen_chunks,
            "real_voxels": real_voxels,
            "generated_voxels": gen_voxels,
        }

        if real_counts is not None:
            row["real_air_fraction"] = (
                float(real_counts[air_idx] / max(1, real_counts.sum())) if air_idx is not None else math.nan
            )
        else:
            row["real_air_fraction"] = math.nan

        if gen_counts is not None:
            row["generated_air_fraction"] = (
                float(gen_counts[air_idx] / max(1, gen_counts.sum())) if air_idx is not None else math.nan
            )
        else:
            row["generated_air_fraction"] = math.nan

        if real_counts is not None and gen_counts is not None:
            all_metrics = _summarize_distribution_metrics(real_counts, gen_counts, exclude_index=None)
            non_air_metrics = _summarize_distribution_metrics(real_counts, gen_counts, exclude_index=air_idx)
            for key, value in all_metrics.items():
                row[f"{key}_all"] = value
            for key, value in non_air_metrics.items():
                row[f"{key}_non_air"] = value
        else:
            for prefix in ("all", "non_air"):
                for key in (
                    "coverage",
                    "precision",
                    "kl_real_to_gen",
                    "js_divergence",
                    "tv_distance",
                    "real_unique_blocks",
                    "gen_unique_blocks",
                ):
                    row[f"{key}_{prefix}"] = math.nan

        rows.append(row)

    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    _ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_frequency_table(path: Path, stats: dict[str, Any]) -> None:
    block_names = list(stats["block_names"])
    rows: list[dict[str, Any]] = []

    air_idx = block_names.index("AIR") if "AIR" in block_names else None
    for biome in sorted(stats["counts_by_biome"].keys()):
        counts = np.asarray(stats["counts_by_biome"][biome], dtype=np.int64)
        total_all = int(counts.sum())
        total_non_air = int(total_all - (counts[air_idx] if air_idx is not None and air_idx < len(counts) else 0))
        present = np.where(counts > 0)[0]
        order = sorted(present.tolist(), key=lambda i: (-int(counts[i]), block_names[i]))
        for rank, idx in enumerate(order, start=1):
            count = int(counts[idx])
            is_air = bool(air_idx is not None and idx == air_idx)
            rows.append(
                {
                    "biome": biome,
                    "rank": rank,
                    "block_name": block_names[idx],
                    "count": count,
                    "frequency_all": float(count / total_all) if total_all > 0 else math.nan,
                    "frequency_non_air": (
                        float(count / total_non_air) if (total_non_air > 0 and not is_air) else math.nan
                    ),
                    "is_air": is_air,
                }
            )

    _write_csv(
        path,
        rows,
        ["biome", "rank", "block_name", "count", "frequency_all", "frequency_non_air", "is_air"],
    )


def _write_paper_markdown_table(path: Path, rows: list[dict[str, Any]]) -> None:
    header = [
        "Biome",
        "Coverage (non-air)",
        "KL real->gen (non-air)",
        "JS (non-air)",
        "TV (non-air)",
        "Real air frac",
        "Gen air frac",
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["biome"]),
                    _format_float(row.get("coverage_non_air")),
                    _format_float(row.get("kl_real_to_gen_non_air")),
                    _format_float(row.get("js_divergence_non_air")),
                    _format_float(row.get("tv_distance_non_air")),
                    _format_float(row.get("real_air_fraction")),
                    _format_float(row.get("generated_air_fraction")),
                ]
            )
            + " |"
        )
    _ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_float(value: Any, digits: int = 4) -> str:
    try:
        v = float(value)
    except Exception:
        return "nan"
    if math.isnan(v):
        return "nan"
    return f"{v:.{digits}f}"


def _save_bundle(
    path: Path,
    *,
    dataset_stats: dict[str, Any],
    generated_stats: dict[str, Any] | None,
    comparison_rows: list[dict[str, Any]],
) -> None:
    _ensure_dir(path.parent)
    torch.save(
        {
            "dataset_stats": dataset_stats,
            "generated_stats": generated_stats,
            "comparison_rows": comparison_rows,
        },
        str(path),
    )


def _import_matplotlib():
    import matplotlib.pyplot as plt

    return plt


def _plot_metric_summary(rows: list[dict[str, Any]], out_path: Path) -> None:
    usable = [r for r in rows if r.get("generated_present") and not math.isnan(float(r.get("coverage_non_air", math.nan)))]
    if not usable:
        print("[plot] Skipping metric summary plot: no overlapping real/generated biomes.")
        return

    plt = _import_matplotlib()
    biomes = [r["biome"] for r in usable]
    coverage = [float(r["coverage_non_air"]) for r in usable]
    kl_vals = [float(r["kl_real_to_gen_non_air"]) for r in usable]
    js_vals = [float(r["js_divergence_non_air"]) for r in usable]
    tv_vals = [float(r["tv_distance_non_air"]) for r in usable]

    fig, axes = plt.subplots(2, 2, figsize=(max(12, len(biomes) * 0.65), 9), constrained_layout=True)
    plots = [
        (coverage, "Coverage (non-air, higher is better)", axes[0, 0], "#4c78a8"),
        (kl_vals, "KL real->gen (non-air, lower is better)", axes[0, 1], "#f58518"),
        (js_vals, "JS divergence (non-air, lower is better)", axes[1, 0], "#54a24b"),
        (tv_vals, "TV distance (non-air, lower is better)", axes[1, 1], "#e45756"),
    ]
    x = np.arange(len(biomes))
    for values, title, ax, color in plots:
        ax.bar(x, values, color=color)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(biomes, rotation=70, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle("Per-biome voxel-space block distribution metrics", fontsize=14)
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_sanity_heatmap(
    dataset_stats: dict[str, Any],
    generated_stats: dict[str, Any] | None,
    out_path: Path,
    sanity_blocks: list[str],
) -> None:
    plt = _import_matplotlib()
    block_names = list(dataset_stats["block_names"])
    available_blocks = [b for b in sanity_blocks if b in block_names]
    if not available_blocks:
        print("[plot] Skipping sanity heatmap: none of the requested block names are present.")
        return

    biomes = sorted(dataset_stats["counts_by_biome"].keys())
    rows = 2 if generated_stats is not None else 1
    fig, axes = plt.subplots(rows, 1, figsize=(max(10, len(available_blocks) * 1.2), max(5, len(biomes) * 0.45 * rows)))
    if rows == 1:
        axes = [axes]

    def _matrix_from_stats(stats: dict[str, Any]) -> np.ndarray:
        mat = np.zeros((len(biomes), len(available_blocks)), dtype=np.float64)
        for biome_idx, biome in enumerate(biomes):
            counts = np.asarray(stats["counts_by_biome"].get(biome, np.zeros(len(block_names), dtype=np.int64)), dtype=np.float64)
            total = max(1.0, counts.sum())
            for block_idx, block_name in enumerate(available_blocks):
                name_idx = block_names.index(block_name)
                mat[biome_idx, block_idx] = counts[name_idx] / total
        return mat

    matrices = [(_matrix_from_stats(dataset_stats), "Real dataset")]
    if generated_stats is not None:
        matrices.append((_matrix_from_stats(generated_stats), "Generated samples"))

    for ax, (matrix, title) in zip(axes, matrices):
        im = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(available_blocks)))
        ax.set_xticklabels(available_blocks, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(biomes)))
        ax.set_yticklabels(biomes)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Voxel frequency")

    fig.suptitle("Sanity-check block frequencies by biome", fontsize=14)
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _select_top_block_indices(
    real_counts: np.ndarray,
    gen_counts: np.ndarray,
    *,
    air_idx: int | None,
    top_k: int,
) -> list[int]:
    real = np.asarray(real_counts, dtype=np.int64).copy()
    gen = np.asarray(gen_counts, dtype=np.int64).copy()
    if air_idx is not None and air_idx < len(real):
        real[air_idx] = 0
        gen[air_idx] = 0

    real_present = np.where(real > 0)[0]
    gen_present = np.where(gen > 0)[0]
    real_order = sorted(real_present.tolist(), key=lambda i: (-int(real[i]), i))
    gen_order = sorted(gen_present.tolist(), key=lambda i: (-int(gen[i]), i))

    selected: list[int] = []
    for idx in real_order:
        if idx not in selected:
            selected.append(idx)
        if len(selected) >= top_k:
            break
    for idx in gen_order:
        if idx not in selected:
            selected.append(idx)
        if len(selected) >= top_k:
            break
    return selected


def _plot_per_biome_top_blocks(
    dataset_stats: dict[str, Any],
    generated_stats: dict[str, Any] | None,
    comparison_rows: list[dict[str, Any]],
    *,
    out_dir: Path,
    top_k: int,
) -> None:
    if generated_stats is None:
        return

    plt = _import_matplotlib()
    block_names = list(dataset_stats["block_names"])
    air_idx = block_names.index("AIR") if "AIR" in block_names else None
    metrics_by_biome = {row["biome"]: row for row in comparison_rows}

    _ensure_dir(out_dir)
    for biome in sorted(set(dataset_stats["counts_by_biome"].keys()) & set(generated_stats["counts_by_biome"].keys())):
        real_counts = np.asarray(dataset_stats["counts_by_biome"][biome], dtype=np.int64)
        gen_counts = np.asarray(generated_stats["counts_by_biome"][biome], dtype=np.int64)
        selected = _select_top_block_indices(real_counts, gen_counts, air_idx=air_idx, top_k=top_k)
        if not selected:
            continue

        real_non_air_total = float(real_counts.sum() - (real_counts[air_idx] if air_idx is not None else 0))
        gen_non_air_total = float(gen_counts.sum() - (gen_counts[air_idx] if air_idx is not None else 0))
        if real_non_air_total <= 0 or gen_non_air_total <= 0:
            continue

        order = sorted(selected, key=lambda i: (-int(real_counts[i]), -int(gen_counts[i]), block_names[i]))
        labels = [block_names[i] for i in order]
        real_freq = [float(real_counts[i] / real_non_air_total) for i in order]
        gen_freq = [float(gen_counts[i] / gen_non_air_total) for i in order]
        y = np.arange(len(order))

        fig, ax = plt.subplots(figsize=(10, max(4.5, len(order) * 0.45)))
        ax.barh(y + 0.18, real_freq, height=0.35, label="Real", color="#4c78a8")
        ax.barh(y - 0.18, gen_freq, height=0.35, label="Generated", color="#f58518")
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("Non-air block frequency")
        ax.set_title(
            f"{biome}: top non-air block-name frequencies\n"
            f"coverage={_format_float(metrics_by_biome[biome].get('coverage_non_air'))}, "
            f"KL={_format_float(metrics_by_biome[biome].get('kl_real_to_gen_non_air'))}, "
            f"JS={_format_float(metrics_by_biome[biome].get('js_divergence_non_air'))}"
        )
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{biome}_top_blocks.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device=cuda but CUDA is not available.")
    return torch.device(device_arg)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute per-biome voxel-space block statistics for a dataset and generated samples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--dataset-dir", required=True, help="Processed or raw memmapped dataset directory with manifest.json")
    ap.add_argument("--samples-dir", default=None, help="Directory containing generated_<biome>.pt files")
    ap.add_argument("--dataset-mappings-path", default=None, help="Optional explicit *_cc_oh_mappings.pt for the dataset")
    ap.add_argument("--generated-mappings-path", default=None, help="Mappings file for generated samples if they are stored as indices or one-hot")
    ap.add_argument(
        "--generated-format",
        default="auto",
        choices=["auto", "block_ids", "indices", "one_hot"],
        help=(
            "How generated tensors are encoded. 'auto' assumes integer 3D/4D tensors are block IDs, "
            "matching inference.py and extract_real_samples.py."
        ),
    )
    ap.add_argument("--samples-glob", default=DEFAULT_SAMPLE_GLOB, help="Glob under --samples-dir for generated sample files")
    ap.add_argument("--output-dir", default=None, help="Where to write CSV/JSON/plots")
    ap.add_argument("--cache-path", default=None, help="Where to cache dataset-side statistics")
    ap.add_argument("--batch-size", type=int, default=256, help="Dataset batch size when streaming memmaps")
    ap.add_argument("--top-k", type=int, default=15, help="How many block names to show in each biome comparison plot")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Counting device")
    ap.add_argument("--force-recompute-dataset", action="store_true", help="Ignore any cached dataset statistics")
    ap.add_argument("--no-plots", action="store_true", help="Skip plot generation and only write raw tables")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"--dataset-dir does not exist: {dataset_dir}")
    samples_dir = Path(args.samples_dir) if args.samples_dir else None
    if samples_dir is not None and not samples_dir.exists():
        raise FileNotFoundError(f"--samples-dir does not exist: {samples_dir}")

    device = _resolve_device(args.device)
    print(f"[setup] Using device: {device}")

    dataset_ctx = _load_dataset_context(dataset_dir, args.dataset_mappings_path)
    cache_path = _resolve_cache_path(dataset_dir, args.cache_path)
    output_dir = _resolve_output_dir(dataset_dir, samples_dir, args.output_dir)
    _ensure_dir(output_dir)

    dataset_stats = _get_or_compute_dataset_stats(
        dataset_ctx,
        cache_path=cache_path,
        batch_size=args.batch_size,
        device=device,
        force_recompute=bool(args.force_recompute_dataset),
    )

    dataset_converter: BlockBiomeConverter | None = dataset_ctx["converter"]
    indexer = _build_indexer_from_dataset_stats(dataset_stats, converter=dataset_converter)
    _align_counts_to_block_names(dataset_stats, indexer.block_names)

    generated_stats = None
    if samples_dir is not None:
        generated_converter = _load_generated_converter(args.generated_mappings_path)
        generated_stats = _compute_generated_stats(
            samples_dir,
            sample_glob=args.samples_glob,
            generated_format=args.generated_format,
            generated_converter=generated_converter,
            indexer=indexer,
            device=device,
        )
        _align_counts_to_block_names(dataset_stats, indexer.block_names)
        _align_counts_to_block_names(generated_stats, indexer.block_names)

    comparison_rows = _compute_comparison_rows(
        dataset_stats,
        generated_stats,
        block_names=indexer.block_names,
    )

    comparison_fieldnames = [
        "biome",
        "real_present",
        "generated_present",
        "real_chunks",
        "generated_chunks",
        "real_voxels",
        "generated_voxels",
        "real_air_fraction",
        "generated_air_fraction",
        "coverage_all",
        "precision_all",
        "kl_real_to_gen_all",
        "js_divergence_all",
        "tv_distance_all",
        "real_unique_blocks_all",
        "gen_unique_blocks_all",
        "coverage_non_air",
        "precision_non_air",
        "kl_real_to_gen_non_air",
        "js_divergence_non_air",
        "tv_distance_non_air",
        "real_unique_blocks_non_air",
        "gen_unique_blocks_non_air",
    ]

    _write_csv(output_dir / "per_biome_metrics.csv", comparison_rows, comparison_fieldnames)
    _write_paper_markdown_table(output_dir / "per_biome_metrics.md", comparison_rows)
    _write_frequency_table(output_dir / "real_block_frequencies.csv", dataset_stats)
    if generated_stats is not None:
        _write_frequency_table(output_dir / "generated_block_frequencies.csv", generated_stats)

    config_payload = {
        "dataset_dir": str(dataset_dir.resolve()),
        "samples_dir": str(samples_dir.resolve()) if samples_dir is not None else None,
        "dataset_mappings_path": str(dataset_ctx["mappings_path"].resolve()) if dataset_ctx["mappings_path"] is not None else None,
        "generated_mappings_path": str(Path(args.generated_mappings_path).resolve()) if args.generated_mappings_path else None,
        "generated_format": args.generated_format,
        "samples_glob": args.samples_glob,
        "batch_size": int(args.batch_size),
        "top_k": int(args.top_k),
        "device": str(device),
        "cache_path": str(cache_path.resolve()),
        "block_names": indexer.block_names,
    }
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)

    _save_bundle(
        output_dir / "voxel_block_stats_bundle.pt",
        dataset_stats=dataset_stats,
        generated_stats=generated_stats,
        comparison_rows=comparison_rows,
    )

    if not args.no_plots:
        _plot_metric_summary(comparison_rows, output_dir / "summary_metrics.png")
        _plot_sanity_heatmap(dataset_stats, generated_stats, output_dir / "sanity_block_heatmap.png", DEFAULT_SANITY_BLOCKS)
        _plot_per_biome_top_blocks(
            dataset_stats,
            generated_stats,
            comparison_rows,
            out_dir=output_dir / "per_biome_top_blocks",
            top_k=int(args.top_k),
        )

    print(f"[done] Wrote outputs to {output_dir}")
    print(f"[done] Dataset cache path: {cache_path}")


if __name__ == "__main__":
    main()
