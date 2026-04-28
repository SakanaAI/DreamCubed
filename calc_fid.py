import os
import argparse
import json
import time
from typing import List, Tuple, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from scipy import linalg
from accelerate import Accelerator

try:
    from cleanfid.features import build_feature_extractor
    from cleanfid.fid import get_folder_features
    from cleanfid.resize import build_resizer
except ImportError:  # pragma: no cover - dependency is validated at runtime
    build_feature_extractor = None
    get_folder_features = None
    build_resizer = None

from render_dataset import render_dataset_to_images
from visualization_utils import MinecraftVisualizerPyVista
from data_utils import BlockBiomeConverter


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
DEFAULT_FID_MODE = "clean"


def _is_distributed(accelerator: Accelerator) -> bool:
    try:
        return int(getattr(accelerator, "num_processes", 1)) > 1
    except Exception:
        return False


def _require_cleanfid() -> None:
    if build_feature_extractor is None or get_folder_features is None or build_resizer is None:
        raise ImportError(
            "calc_fid.py now uses clean-fid for canonical FID features. "
            "Install it with `pip install clean-fid`."
        )


def _gather_image_files(folder: str) -> List[str]:
    all_files = []
    for root, _, filenames in os.walk(folder):
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                all_files.append(os.path.join(root, name))
    all_files.sort()
    return all_files


def _stats_filename_for_mode(fid_mode: str) -> str:
    safe_mode = str(fid_mode).strip().lower().replace("-", "_")
    return f"fid_stats_{safe_mode}.npz"


def _per_biome_metrics_filename_for_mode(fid_mode: str) -> str:
    safe_mode = str(fid_mode).strip().lower().replace("-", "_")
    return f"per_biome_fid_{safe_mode}.npz"


def _per_biome_summary_filename_for_mode(fid_mode: str) -> str:
    safe_mode = str(fid_mode).strip().lower().replace("-", "_")
    return f"per_biome_fid_{safe_mode}.json"


def _to_uint8_image(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    return np.clip(np.rint(arr), 0, 255).astype(np.uint8)


def save_preprocessed_preview(
    image_dir: str,
    preview_dir: str,
    *,
    fid_mode: str,
    preview_label: str,
) -> Optional[str]:
    """
    Save a single original/preprocessed image pair outside any FID-scanned folder.

    This previews the image after clean-fid's folder preprocessing step. For
    legacy_tensorflow mode, clean-fid performs no external resize; the remaining
    resize happens inside the TorchScript Inception model, so the saved
    "preprocessed" image is identical to the original input image.
    """
    _require_cleanfid()
    paths = _gather_image_files(image_dir)
    if len(paths) == 0:
        return None

    src_path = paths[0]
    with Image.open(src_path) as img:
        img_np = np.array(img.convert("RGB"))

    resized_np = build_resizer(fid_mode)(img_np)
    original_uint8 = _to_uint8_image(img_np)
    resized_uint8 = _to_uint8_image(np.asarray(resized_np))

    os.makedirs(preview_dir, exist_ok=True)
    safe_label = str(preview_label).strip().replace(" ", "_")
    original_path = os.path.join(preview_dir, f"{safe_label}_original.png")
    processed_path = os.path.join(preview_dir, f"{safe_label}_preprocessed.png")
    metadata_path = os.path.join(preview_dir, f"{safe_label}_metadata.json")

    Image.fromarray(original_uint8).save(original_path)
    Image.fromarray(resized_uint8).save(processed_path)

    note = None
    if fid_mode == "legacy_tensorflow":
        note = (
            "clean-fid legacy_tensorflow mode does not resize images in the folder "
            "preprocessing step. Any remaining resize occurs inside the TorchScript "
            "Inception model, so this preview reflects the pre-model image."
        )

    with open(metadata_path, "w") as f:
        json.dump(
            {
                "fid_mode": str(fid_mode),
                "source_image": src_path,
                "preview_dir": preview_dir,
                "original_path": original_path,
                "preprocessed_path": processed_path,
                "note": note,
            },
            f,
            indent=2,
        )
    return processed_path


@torch.no_grad()
def extract_inception_features(
    image_dir: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    fid_mode: str = DEFAULT_FID_MODE,
) -> Tuple[np.ndarray, List[str]]:
    _require_cleanfid()
    paths = _gather_image_files(image_dir)
    if len(paths) == 0:
        return np.empty((0, 2048), dtype=np.float64), paths

    feat_model = build_feature_extractor(
        mode=fid_mode,
        device=device,
        use_dataparallel=False,
    )
    all_features = get_folder_features(
        fdir=image_dir,
        model=feat_model,
        num_workers=num_workers,
        batch_size=batch_size,
        device=device,
        mode=fid_mode,
        description=f"Extracting features from {os.path.basename(image_dir) or image_dir}",
        verbose=True,
    )
    return all_features.astype(np.float64), paths


def compute_statistics(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if features.shape[0] == 0:
        raise ValueError("No features to compute statistics from.")
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray, eps: float = 1e-6) -> float:
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean
    return float(fid)


def per_image_fid_approx(features: np.ndarray, ref_mu: np.ndarray, ref_sigma: np.ndarray) -> np.ndarray:
    # Treat each image feature as a degenerate Gaussian with zero covariance.
    # FID(x, N(mu,Sigma)) = ||x - mu||^2 + trace(Sigma)
    diffs = features - ref_mu[None, :]
    sq_norms = np.sum(diffs * diffs, axis=1)
    trace_sigma = float(np.trace(ref_sigma))
    return sq_norms + trace_sigma


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FID utility: either compute FID between two image directories or run end-to-end render+generate flow.")
    # Direct directory-to-directory FID mode (optional)
    parser.add_argument("--images_dir", default=None, help="Target images directory to evaluate.")
    parser.add_argument("--ref_dir", default=None, help="Reference images directory to compare against.")
    parser.add_argument("--output", default=None, help="Path to save JSON results. Defaults to <images_dir>/fid_results.json or renders/gen/fid_results.json")
    parser.add_argument("--ref_stats", default=None, help="Optional path to precomputed reference stats .npz (mu/sigma).")
    parser.add_argument("--save_ref_stats", default=None, help="Optional path to save computed reference stats .npz.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default=None, help="cpu or cuda. Defaults to cuda if available else cpu.")
    parser.add_argument(
        "--fid_mode",
        choices=["legacy_tensorflow", "legacy_pytorch", "clean"],
        default=DEFAULT_FID_MODE,
        help=(
            "Feature pipeline for FID. "
            "'legacy_tensorflow' matches the original TTUR TensorFlow FID implementation most closely; "
            "'legacy_pytorch' matches pytorch-fid; 'clean' uses clean-fid's corrected resize pipeline."
        ),
    )

    # End-to-end flow (render dataset, generate, render, compute FID)
    parser.add_argument("--dataset_path", default=None, help="Path to processed dataset .pt (contains 'voxels').")
    parser.add_argument("--mappings_path", default=None, help="Path to mappings .pt for BlockBiomeConverter.")
    parser.add_argument("--checkpoint_path", default=None, help="Path to trained model checkpoint .pt.")
    parser.add_argument("--config_path", default=None, help="Optional path to config.json corresponding to checkpoint.")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to render/generate for FID.")
    parser.add_argument("--render_ref_dir", default="renders/ref", help="Base directory to render reference dataset images.")
    parser.add_argument("--render_gen_dir", default="renders/gen", help="Base directory to render generated images.")
    parser.add_argument("--textures_dir", default="block_textures/", help="Textures directory for textured rendering.")
    parser.add_argument("--image_px", type=int, default=256, help="Rendered image size in pixels (square).")
    parser.add_argument("--show_axis", action="store_true", help="Show axes in renders.")
    parser.add_argument("--zoom", type=float, default=1.0, help="Camera zoom factor (>1 zooms in).")
    parser.add_argument("--dataset_start", type=int, default=0, help="Start index in dataset for rendering subset.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip rendering images that already exist.")
    parser.add_argument("--cond_scale", type=float, default=None, help="Optional override for classifier-free guidance scale.")
    # Per-biome FID mode
    parser.add_argument("--per_biome_fid_model", default=None, help="Path to model directory or checkpoint for per-biome FID. If directory, will try to locate model_best.*")
    # Per-biome FID from pre-generated samples (one file per biome)
    parser.add_argument(
        "--per_biome_fid_samples_dir",
        default=None,
        help=(
            "Directory containing either one samples file per biome, named like '<biome>_samples_...', "
            "or an already-rendered biome image tree like '<run>/<biome>/*.png'. "
            "This mode will render samples if needed, or reuse existing rendered images, and compute per-biome FID vs --per_biome_ref_dir (or auto-rendered refs)."
        ),
    )
    parser.add_argument(
        "--per_biome_samples_glob",
        default="*.pt",
        help="Glob relative to --per_biome_fid_samples_dir to find sample files (default: *.pt).",
    )
    parser.add_argument(
        "--per_biome_samples_format",
        choices=["block_ids", "indices"],
        default="block_ids",
        help=(
            "Interpretation of voxel values inside each biome samples file. "
            "'block_ids' means values are original Minecraft block IDs (no conversion). "
            "'indices' means values are model/dataset indices and require --mappings_path to convert for rendering."
        ),
    )
    parser.add_argument(
        "--per_biome_samples_id",
        default=None,
        help="Optional run id used to name the output folder under --per_biome_gen_dir. Defaults to basename of --per_biome_fid_samples_dir.",
    )
    parser.add_argument("--per_biome_ref_dir", default=None, help="(Optional) Reference images dir containing biome subfolders. If omitted, will auto-render val split per-biome refs from the model config.")
    parser.add_argument("--per_biome_ref_split", default="val", choices=["val", "train"], help="Which dataset split to use as reference when auto-rendering per-biome refs.")
    parser.add_argument("--per_biome_ref_renders_dir", default="renders/per_biome_ref", help="Base directory where per-biome reference renders live (or will be created).")
    parser.add_argument("--ref_samples_per_biome", type=int, default=None, help="When auto-rendering refs, max images per biome (default: match --samples_per_biome).")
    parser.add_argument("--force_render_ref", action="store_true", help="Force re-render of per-biome reference images even if they already exist.")
    parser.add_argument("--per_biome_gen_dir", default="renders/per_biome_gen", help="Output base directory for generated per-biome images.")
    parser.add_argument("--samples_per_biome", type=int, default=100, help="Number of samples to generate per biome.")
    parser.add_argument(
        "--sampling_timesteps",
        type=int,
        default=None,
        help="Optional override for the number of denoising steps used during per-biome generation. Defaults to the loaded model config.",
    )
    parser.add_argument("--sampling_chunk_size", type=int, default=None, help="Sampling batch size per diffusion pass; overrides config fid_batch_size/sampling_batch_size.")
    parser.add_argument("--save_metrics_path", default=None, help="Optional path to save per-biome FID metrics npz. Defaults to <per_biome_gen_dir>/per_biome_fid.npz")
    return parser

def _count_images(directory: str) -> int:
    count = 0
    for root, _, files in os.walk(directory):
        for name in files:
            if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                count += 1
    return count


# ----------------------------
# Helper APIs for programmatic use
# ----------------------------

@torch.no_grad()
def compute_fid_between_dirs(
    images_dir: str,
    ref_dir: str,
    *,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 2,
    ref_stats_path: Optional[str] = None,
    save_ref_stats_path: Optional[str] = None,
    fid_mode: str = DEFAULT_FID_MODE,
):
    outer = tqdm(total=3, desc='FID', disable=False)
    tgt_features, _ = extract_inception_features(
        image_dir=images_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        fid_mode=fid_mode,
    )
    outer.update(1)
    if tgt_features.shape[0] == 0:
        raise ValueError(f"No images found in {images_dir}")
    num_target_images = int(tgt_features.shape[0])
    # Prefer stats saved inside ref_dir unless an explicit path was provided
    auto_ref_stats = os.path.join(ref_dir, _stats_filename_for_mode(fid_mode))
    chosen_ref_stats = ref_stats_path if ref_stats_path else (auto_ref_stats if os.path.exists(auto_ref_stats) else None)
    num_ref_images = _count_images(ref_dir)
    if chosen_ref_stats and os.path.exists(chosen_ref_stats):
        print(f"Using precomputed reference stats from {chosen_ref_stats}")
        stats = np.load(chosen_ref_stats)
        ref_mu = stats["mu"]
        ref_sigma = stats["sigma"]
        outer.update(1)
    else:
        ref_features, _ = extract_inception_features(
            image_dir=ref_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            fid_mode=fid_mode,
        )
        outer.update(1)
        if ref_features.shape[0] == 0:
            raise ValueError(f"No images found in {ref_dir}")
        ref_mu, ref_sigma = compute_statistics(ref_features)
        # Auto-save reference stats into ref_dir/fid_stats.npz unless an explicit save path is given
        target_stats_path = save_ref_stats_path or auto_ref_stats
        try:
            os.makedirs(os.path.dirname(target_stats_path) or ".", exist_ok=True)
            np.savez_compressed(target_stats_path, mu=ref_mu, sigma=ref_sigma)
        except Exception:
            pass
        num_ref_images = int(ref_features.shape[0])
    print(
        f"[fid] mode={fid_mode} generated_images={num_target_images} "
        f"reference_images={num_ref_images} images_dir='{images_dir}' ref_dir='{ref_dir}'"
    )
    tgt_mu, tgt_sigma = compute_statistics(tgt_features)
    fid_overall = frechet_distance(tgt_mu, tgt_sigma, ref_mu, ref_sigma)
    per_image_fids = per_image_fid_approx(tgt_features, ref_mu, ref_sigma)
    outer.update(1)
    outer.close()
    # Auto-save target (images_dir) stats if missing
    try:
        auto_tgt_stats = os.path.join(images_dir, _stats_filename_for_mode(fid_mode))
        print(f"Saving target stats to {auto_tgt_stats}")
        if not os.path.exists(auto_tgt_stats):
            os.makedirs(os.path.dirname(auto_tgt_stats) or ".", exist_ok=True)
            np.savez_compressed(auto_tgt_stats, mu=tgt_mu, sigma=tgt_sigma)
    except Exception:
        pass
    return {
        "fid_overall": float(fid_overall),
        "mean_per_image_fid": float(np.mean(per_image_fids)),
        "std_per_image_fid": float(np.std(per_image_fids)),
        "num_images": num_target_images,
        "num_ref_images": int(num_ref_images),
        "fid_mode": str(fid_mode),
    }


def _resolve_best_checkpoint(model_path: str) -> str:
    """
    If model_path is a directory, attempt to locate the best checkpoint inside it.
    Accepted filenames (priority order): model_best.pt, model-best.pt, best.pt, model_final.pt, last.pt, any *.pt
    If model_path is a file, return it as-is.
    """
    if os.path.isfile(model_path):
        return model_path
    # Directory case
    candidates = [
        "model_best.pt",
        "model-best.pt",
        "best.pt",
        "model_final.pt",
        "model-final.pt",
        "last.pt",
    ]
    for name in candidates:
        p = os.path.join(model_path, name)
        if os.path.exists(p):
            return p
    # Fallback: any .pt in directory, prefer ones with 'best' in name
    pts = [os.path.join(model_path, f) for f in os.listdir(model_path) if f.endswith(".pt")]
    if len(pts) == 0:
        raise FileNotFoundError(f"No checkpoint (.pt) files found in directory: {model_path}")
    pts_sorted = sorted(pts, key=lambda x: (0 if "best" in os.path.basename(x).lower() else 1, os.path.getmtime(x)))
    return pts_sorted[0]


def _resolve_path_from_config(value: Optional[str], *, config_dir: Optional[str]) -> Optional[str]:
    """
    Resolve a potentially relative path coming from a saved config.json.
    Strategy:
      - If absolute or exists as-is, use it.
      - Else, try relative to config_dir.
      - Else, return the original string (best effort).
    """
    if value in (None, ""):
        return None
    p = os.path.expanduser(str(value))
    if os.path.isabs(p) and os.path.exists(p):
        return p
    if os.path.exists(p):
        return p
    if config_dir not in (None, ""):
        cand = os.path.join(str(config_dir), p)
        if os.path.exists(cand):
            return cand
    return p


def _dataset_name_from_path(p: str) -> str:
    # Matches render_dataset.py behavior: os.path.splitext(os.path.basename(data_path))[0]
    base = os.path.basename(str(p).rstrip("/"))
    return os.path.splitext(base)[0]


def _has_any_images(directory: str) -> bool:
    for root, _, files in os.walk(directory):
        for name in files:
            if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                return True
    return False


def _infer_biome_name_from_filename(path: str) -> str:
    """
    Extract biome name from a filename like '<biome>_samples_...'.
    Supports biome names containing underscores (e.g. 'birch_forest_samples_...').
    Fallback: uses the file stem.
    """
    base = os.path.basename(str(path))
    stem, _ = os.path.splitext(base)
    if stem.startswith("generated_"):
        stem = stem[len("generated_"):]
    if "_samples" in stem:
        return stem.split("_samples", 1)[0]
    return stem


def _load_samples_tensor_from_file(path: str) -> torch.Tensor:
    """
    Load a tensor of samples from a torch file. Supports:
      - torch.save(tensor)
      - torch.save({'voxels': tensor, ...})
      - torch.save({'samples': tensor, ...})
      - torch.save({'x': tensor, ...}) [best-effort]
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, torch.Tensor):
        return payload
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported sample file payload type: {type(payload)} in {path}")
    for key in ("voxels", "samples", "x", "data"):
        if key in payload and isinstance(payload[key], torch.Tensor):
            return payload[key]
    raise KeyError(f"No tensor samples found in {path}. Expected keys like 'voxels' or 'samples'. Found: {list(payload.keys())}")


def _ensure_per_biome_reference_images(
    *,
    ref_biome_images_dir: Optional[str],
    dataset_path: Optional[str],
    mappings_path: Optional[str],
    ref_split: str,
    ref_renders_base_dir: str,
    ref_samples_per_biome: Optional[int],
    force_render_ref: bool,
    skip_existing_ref: bool,
    textures_dir: Optional[str],
    image_px: int,
    show_axis: bool,
    zoom: float,
):
    """
    Ensure we have a directory layout:
      <ref_biome_images_dir>/<biome_name>/*.png

    If ref_biome_images_dir is provided, use it as-is.
    Else, auto-render from dataset_path + mappings_path into:
      <ref_renders_base_dir>/<dataset_name>/<biome_name>/*.png
    """
    if ref_biome_images_dir not in (None, ""):
        return str(ref_biome_images_dir)

    # Auto-render case (no model/config available) requires dataset + mappings.
    if dataset_path in (None, ""):
        raise ValueError("Missing --per_biome_ref_dir; to auto-render per-biome refs, provide --dataset_path.")
    if mappings_path in (None, ""):
        raise ValueError("Missing --per_biome_ref_dir; to auto-render per-biome refs, provide --mappings_path.")
    if not os.path.exists(str(dataset_path)):
        raise FileNotFoundError(f"--dataset_path not found: {dataset_path}")
    if not os.path.exists(str(mappings_path)):
        raise FileNotFoundError(f"--mappings_path not found: {mappings_path}")

    dataset_name = _dataset_name_from_path(str(dataset_path))
    inferred_ref_dir = os.path.join(str(ref_renders_base_dir), dataset_name)

    # Default ref sample count: if not specified, render all (or caller can pass a value).
    need_render = force_render_ref or (not os.path.isdir(inferred_ref_dir)) or (not _has_any_images(inferred_ref_dir))
    if need_render:
        split = str(ref_split or "val").lower()
        if split not in ("val", "train"):
            raise ValueError(f"ref_split must be 'val' or 'train', got: {ref_split}")
        # NOTE: without a model config, we can't resolve "val_dataset_path" automatically.
        # So here, dataset_path is the caller's chosen reference dataset (train or val).
        print(f"Rendering per-biome reference images from dataset -> {inferred_ref_dir}")
        from render_dataset import render_dataset_to_images_by_biome
        render_dataset_to_images_by_biome(
            data_path=str(dataset_path),
            mappings_path=str(mappings_path),
            output_dir=str(ref_renders_base_dir),
            textures_dir=textures_dir,
            image_size=int(image_px),
            show_axis=bool(show_axis),
            per_biome=(int(ref_samples_per_biome) if ref_samples_per_biome is not None else None),
            skip_existing=bool(skip_existing_ref) and (not force_render_ref),
            zoom=float(zoom),
        )
    else:
        print(f"Using existing per-biome reference images at {inferred_ref_dir}")

    return inferred_ref_dir


@torch.no_grad()
def compute_per_biome_fid_for_samples_dir(
    samples_dir: str,
    *,
    samples_glob: str = "*.pt",
    samples_format: str = "block_ids",
    mappings_path: Optional[str] = None,
    ref_biome_images_dir: Optional[str] = None,
    gen_out_base_dir: str,
    run_id: Optional[str] = None,
    ref_split: str = "val",
    ref_renders_base_dir: str = "renders/per_biome_ref",
    dataset_path: Optional[str] = None,
    ref_samples_per_biome: Optional[int] = None,
    force_render_ref: bool = False,
    skip_existing_ref: bool = True,
    textures_dir: Optional[str] = None,
    image_px: int = 256,
    show_axis: bool = False,
    zoom: float = 1.0,
    batch_size: int = 32,
    num_workers: int = 2,
    device: Optional[str] = None,
    fid_mode: str = DEFAULT_FID_MODE,
    samples_per_biome: Optional[int] = None,
    save_metrics_path: Optional[str] = None,
):
    """
    Compute per-biome FID from either:
      1) pre-generated sample files in a directory, or
      2) an already-rendered directory tree with biome subfolders of images.

    Sample-file input layout:
      <samples_dir>/<biome>_samples_*.pt    (one file per biome)

    Rendered-image input layout:
      <samples_dir>/<biome_name>/*.png

    Output layout (matching the diffusion per-biome mode for sample-file input):
      <gen_out_base_dir>/<run_id>/<biome_name>/*.png
      <gen_out_base_dir>/<run_id>/per_biome_fid_<mode>.npz
      <gen_out_base_dir>/<run_id>/per_biome_fid_<mode>.json

    For rendered-image input, results are written directly into <samples_dir>.
    """
    device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device_str)

    if samples_dir in (None, ""):
        raise ValueError("samples_dir is required")
    if not os.path.isdir(samples_dir):
        raise FileNotFoundError(f"samples_dir not found or not a directory: {samples_dir}")

    # Resolve reference directory (either passed explicitly or auto-rendered from dataset+mappings).
    resolved_ref_dir = _ensure_per_biome_reference_images(
        ref_biome_images_dir=ref_biome_images_dir,
        dataset_path=dataset_path,
        mappings_path=mappings_path,
        ref_split=ref_split,
        ref_renders_base_dir=ref_renders_base_dir,
        ref_samples_per_biome=ref_samples_per_biome,
        force_render_ref=force_render_ref,
        skip_existing_ref=skip_existing_ref,
        textures_dir=textures_dir,
        image_px=image_px,
        show_axis=show_axis,
        zoom=zoom,
    )

    # Optional converter if samples are indices.
    converter = None
    if str(samples_format) == "indices":
        if mappings_path in (None, ""):
            raise ValueError("samples_format='indices' requires --mappings_path so we can convert indices -> original block IDs for rendering.")
        mappings = torch.load(str(mappings_path), map_location="cpu", weights_only=False)
        converter = BlockBiomeConverter(mappings.get("block_mappings", {}), mappings.get("biome_mappings", {}))

    # Gather files / already-rendered biome directories
    import glob
    pattern = os.path.join(samples_dir, samples_glob or "*.pt")
    files = sorted(glob.glob(pattern))
    rendered_biome_dirs = sorted(
        [
            os.path.join(samples_dir, name)
            for name in os.listdir(samples_dir)
            if os.path.isdir(os.path.join(samples_dir, name))
            and (not str(name).startswith("_"))
            and _has_any_images(os.path.join(samples_dir, name))
        ]
    )
    if len(files) == 0 and len(rendered_biome_dirs) == 0:
        raise FileNotFoundError(
            f"No sample files found for glob: {pattern}, and no rendered biome image directories were found in {samples_dir}"
        )

    using_rendered_image_dirs = len(files) == 0
    rendered_dirs_by_biome = {
        os.path.basename(os.path.normpath(p)): p for p in rendered_biome_dirs
    }
    if len(files) > 0:
        print(f"Found {len(files)} per-biome sample files in {samples_dir}")
    if len(rendered_biome_dirs) > 0:
        print(f"Found {len(rendered_biome_dirs)} rendered biome directories in {samples_dir}")

    # Output root
    if using_rendered_image_dirs or len(rendered_biome_dirs) > 0:
        out_root = str(samples_dir)
    else:
        resolved_run_id = run_id or os.path.basename(os.path.normpath(samples_dir))
        out_root = os.path.join(str(gen_out_base_dir), str(resolved_run_id))
        os.makedirs(out_root, exist_ok=True)
    print(f"Output root: {out_root}")
    preview_dir = os.path.join(out_root, f"_fid_preprocess_preview_{str(fid_mode).strip().lower().replace('-', '_')}")
    preview_saved = False

    results_per_biome = {}
    counts = []

    biome_items = rendered_biome_dirs if using_rendered_image_dirs else files
    for item in biome_items:
        if using_rendered_image_dirs:
            biome_name = os.path.basename(os.path.normpath(item))
            gen_dir = str(item)
            existing_image_count = _count_images(gen_dir)
            if samples_per_biome not in (None, 0) and existing_image_count < int(samples_per_biome):
                raise ValueError(
                    f"Rendered biome directory '{gen_dir}' only contains {existing_image_count} images, "
                    f"which is fewer than requested samples_per_biome={int(samples_per_biome)}."
                )
            print(f"Using existing rendered images for biome '{biome_name}' from {gen_dir}")
        else:
            fp = str(item)
            biome_name = _infer_biome_name_from_filename(fp)
            samples = _load_samples_tensor_from_file(fp)
            if samples_per_biome not in (None, 0):
                samples = samples[: int(samples_per_biome)]

            gen_dir = rendered_dirs_by_biome.get(biome_name, os.path.join(out_root, biome_name))
            expected_images = int(samples.shape[0])
            existing_image_count = _count_images(gen_dir) if os.path.isdir(gen_dir) else 0
            reuse_rendered_images = os.path.isdir(gen_dir) and existing_image_count >= expected_images
            print(
                f"[debug] biome='{biome_name}' sample_file='{fp}' expected_images={expected_images} "
                f"candidate_gen_dir='{gen_dir}' dir_exists={os.path.isdir(gen_dir)} existing_image_count={existing_image_count}"
            )
            if reuse_rendered_images:
                print(f"Using existing rendered images for biome '{biome_name}' from {gen_dir}")
            else:
                print(
                    f"[debug] Rendering required for biome '{biome_name}' because "
                    f"dir_exists={os.path.isdir(gen_dir)} and existing_image_count={existing_image_count} < expected_images={expected_images}"
                )
                # Render generated samples for this biome into the standard folder name.
                render_samples_to_images(
                    samples,
                    gen_dir,
                    textures_dir=textures_dir,
                    image_px=image_px,
                    show_axis=show_axis,
                    zoom=zoom,
                    converter=converter,
                )

        if (not preview_saved) and os.path.isdir(gen_dir) and _count_images(gen_dir) > 0:
            preview_path = save_preprocessed_preview(
                gen_dir,
                preview_dir,
                fid_mode=fid_mode,
                preview_label=f"{biome_name}_generated",
            )
            if preview_path:
                print(f"Saved FID preprocessing preview to {preview_path}")
                preview_saved = True

        # Compute FID vs reference for this biome
        ref_dir = os.path.join(str(resolved_ref_dir), biome_name)
        print(f"[debug] Looking for reference biome directory at '{ref_dir}'")
        if not os.path.isdir(ref_dir):
            print(f"[warn] Reference biome directory not found: {ref_dir} - skipping FID for this biome")
            continue

        res = compute_fid_between_dirs(
            images_dir=gen_dir,
            ref_dir=ref_dir,
            device=dev,
            batch_size=batch_size,
            num_workers=num_workers,
            fid_mode=fid_mode,
        )
        fid = float(res["fid_overall"])
        nimgs = int(res.get("num_images", 0))
        results_per_biome[biome_name] = fid
        counts.append(nimgs)

    fid_values_arr = np.array(list(results_per_biome.values()), dtype=np.float32)
    counts_arr = np.array(counts, dtype=np.int32) if len(counts) > 0 else np.array([], dtype=np.int32)
    overall_mean = float(np.mean(fid_values_arr)) if fid_values_arr.size > 0 else float("nan")

    metrics_path = save_metrics_path or os.path.join(out_root, _per_biome_metrics_filename_for_mode(fid_mode))
    try:
        np.savez_compressed(
            metrics_path,
            biome_names=np.array(list(results_per_biome.keys())),
            fid=np.array(list(results_per_biome.values()), dtype=np.float32),
            counts=counts_arr,
            overall_mean=np.array([overall_mean], dtype=np.float32),
        )
    except Exception as e:
        print(f"[warn] Failed to save metrics npz at {metrics_path}: {e}")

    return {
        "biome_names": list(results_per_biome.keys()),
        "fid": results_per_biome,
        "counts": counts_arr.tolist(),
        "overall_mean": overall_mean,
        "metrics_path": metrics_path,
        "out_root": out_root,
        "ref_dir": resolved_ref_dir,
        "fid_mode": str(fid_mode),
    }


@torch.no_grad()
def compute_per_biome_fid_for_model(
    model_path: str,
    mappings_path: Optional[str] = None,
    ref_biome_images_dir: Optional[str] = None,
    *,
    accelerator: Optional[Accelerator] = None,
    gen_out_base_dir: str,
    ref_split: str = "val",
    ref_renders_base_dir: str = "renders/per_biome_ref",
    ref_samples_per_biome: Optional[int] = None,
    force_render_ref: bool = False,
    skip_existing_ref: bool = True,
    samples_per_biome: int = 100,
    sampling_timesteps: Optional[int] = None,
    sampling_chunk_size: Optional[int] = None,
    textures_dir: Optional[str] = None,
    image_px: int = 256,
    show_axis: bool = False,
    zoom: float = 1.0,
    batch_size: int = 32,
    num_workers: int = 2,
    device: Optional[str] = None,
    fid_mode: str = DEFAULT_FID_MODE,
    cond_scale: Optional[float] = None,
    save_metrics_path: Optional[str] = None,
):
    """
    Load a model (best checkpoint), generate N samples per biome, render them into biome-specific directories,
    and compute FID per biome versus reference renders for the same biome.

    Returns:
        dict with keys: 'biome_names', 'fid', 'counts', 'overall_mean', 'results_per_biome'
    """
    # Lazy imports to avoid circular import when this module is imported by training code
    from inference import load_model_from_file, generate_samples_for_class
    from render_dataset import render_dataset_to_images_by_biome

    accel = accelerator or Accelerator()

    device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device_str)

    ckpt = _resolve_best_checkpoint(model_path)
    loaded = load_model_from_file(
        checkpoint_path=ckpt,
        config_path=None,
        mappings_path=mappings_path,
        device=device_str,
        use_ema_if_available=True,
    )
    cfg = loaded.get("config", {}) or {}
    cfg_dir = os.path.dirname(str(loaded.get("config_path", ""))) or None

    if sampling_timesteps not in (None, 0):
        override_steps = int(sampling_timesteps)
        loaded["sampling_timesteps"] = override_steps

        # Some sampler implementations cache DDIM step settings on a live diffusion object
        # instead of consulting the loaded metadata dict during generation.
        for key in ("diffusion", "ema_model", "model"):
            sampler = loaded.get(key)
            if sampler is None or not hasattr(sampler, "sampling_timesteps"):
                continue

            try:
                sampler.sampling_timesteps = override_steps
            except Exception:
                continue

            total_steps = None
            for total_attr in ("num_timesteps", "reverse_steps"):
                try:
                    val = getattr(sampler, total_attr, None)
                    if val not in (None, 0):
                        total_steps = int(val)
                        break
                except Exception:
                    pass

            if total_steps is not None and hasattr(sampler, "is_ddim_sampling"):
                try:
                    sampler.is_ddim_sampling = override_steps < total_steps
                except Exception:
                    pass

    # Infer defaults from training config if caller didn't pass explicit overrides
    if textures_dir is None:
        textures_dir = cfg.get("fid_textures_dir", None) or cfg.get("textures_dir", None) or "block_textures/"
        textures_dir = textures_dir if (textures_dir and os.path.exists(str(textures_dir))) else None
    if (batch_size in (None, 0)) or (batch_size == 32 and "fid_batch_size" in cfg):
        try:
            batch_size = int(cfg.get("fid_batch_size", batch_size))
        except Exception:
            pass
    if (num_workers in (None, 0)) or (num_workers == 2 and "fid_num_workers" in cfg):
        try:
            num_workers = int(cfg.get("fid_num_workers", num_workers))
        except Exception:
            pass

    # Auto-render per-biome reference directory from model config (default: validation split)
    if ref_biome_images_dir in (None, ""):
        split = str(ref_split or "val").lower()
        if split not in ("val", "train"):
            raise ValueError(f"ref_split must be 'val' or 'train', got: {ref_split}")
        if split == "val":
            ref_dataset_path = cfg.get("val_dataset_path", None)
            ref_mappings_path = cfg.get("val_mappings_file_path", None) or cfg.get("mappings_file_path", None)
        else:
            ref_dataset_path = cfg.get("dataset_path", None)
            ref_mappings_path = cfg.get("mappings_file_path", None)

        ref_dataset_path = _resolve_path_from_config(ref_dataset_path, config_dir=cfg_dir)
        ref_mappings_path = _resolve_path_from_config(ref_mappings_path, config_dir=cfg_dir)
        if ref_dataset_path in (None, "") or not os.path.exists(str(ref_dataset_path)):
            raise FileNotFoundError(
                f"Could not resolve reference dataset path from config (split={split}). "
                f"Got val_dataset_path={cfg.get('val_dataset_path')} dataset_path={cfg.get('dataset_path')} "
                f"(resolved={ref_dataset_path})"
            )
        if ref_mappings_path in (None, "") or not os.path.exists(str(ref_mappings_path)):
            raise FileNotFoundError(
                f"Could not resolve reference mappings path from config (split={split}). "
                f"Got val_mappings_file_path={cfg.get('val_mappings_file_path')} mappings_file_path={cfg.get('mappings_file_path')} "
                f"(resolved={ref_mappings_path})"
            )

        dataset_name = _dataset_name_from_path(str(ref_dataset_path))
        inferred_ref_dir = os.path.join(ref_renders_base_dir, dataset_name)
        ref_biome_images_dir = inferred_ref_dir

        need_render = force_render_ref or (not os.path.isdir(inferred_ref_dir)) or (not _has_any_images(inferred_ref_dir))
        if need_render:
            if accel.is_main_process:
                print(f"Rendering per-biome reference images from {split} set -> {inferred_ref_dir}")
                render_dataset_to_images_by_biome(
                    data_path=str(ref_dataset_path),
                    mappings_path=str(ref_mappings_path),
                    output_dir=str(ref_renders_base_dir),
                    textures_dir=textures_dir,
                    image_size=int(image_px),
                    show_axis=bool(show_axis),
                    per_biome=(int(ref_samples_per_biome) if ref_samples_per_biome is not None else None),
                    skip_existing=bool(skip_existing_ref) and (not force_render_ref),
                    zoom=float(zoom),
                )
            accel.wait_for_everyone()
        else:
            if accel.is_main_process:
                print(f"Using existing per-biome reference images at {inferred_ref_dir}")
            accel.wait_for_everyone()

    converter: BlockBiomeConverter = loaded["converter"]
    if converter.index_to_biome is None or converter.biome_to_index is None:
        raise ValueError("Loaded mappings do not include biome mappings; per-biome FID requires conditional model.")
    num_classes = int(loaded.get("num_classes") or 0)
    if num_classes in (None, 0):
        raise ValueError("Model appears to be unconditional; per-biome FID requires class-conditional model.")

    # Resolve model-named output root.
    # We want it to be stable + unique across runs, and avoid collisions when different models
    # share the same experiment folder name (layout: <model_name>/<experiment_name>/...).
    if os.path.isdir(model_path):
        exp_name = os.path.basename(model_path.rstrip("/"))
        model_name = os.path.basename(os.path.dirname(model_path.rstrip("/")))
        model_id = f"{model_name}_{exp_name}" if model_name not in ("", ".", "/") else exp_name
    else:
        # If a checkpoint file is passed, try to capture both <model>/<experiment> from its path.
        ckpt_path = os.path.normpath(model_path.rstrip("/"))
        exp_name = os.path.basename(os.path.dirname(ckpt_path))
        model_name = os.path.basename(os.path.dirname(os.path.dirname(ckpt_path)))
        ckpt_stem = os.path.splitext(os.path.basename(ckpt))[0]
        if model_name not in ("", ".", "/") and exp_name not in ("", ".", "/"):
            model_id = f"{model_name}_{exp_name}_{ckpt_stem}"
        else:
            # Fallback to old behavior
            parent_dir = os.path.basename(os.path.dirname(model_path.rstrip("/")))
            model_id = f"{parent_dir}_{ckpt_stem}" if parent_dir else ckpt_stem
    out_root = os.path.join(gen_out_base_dir, model_id)
    os.makedirs(out_root, exist_ok=True)
    preview_dir = os.path.join(out_root, f"_fid_preprocess_preview_{str(fid_mode).strip().lower().replace('-', '_')}")
    preview_saved = False

    if accel.is_main_process:
        print(f"Output root: {out_root}")

    # Prepare output directories and iterate per biome
    biome_names = [converter.index_to_biome[i] for i in sorted(converter.index_to_biome.keys())]

    fid_values = []
    counts = []
    results_per_biome = {}

    for biome_name in biome_names:
        class_idx = int(converter.biome_to_index[biome_name])
        gen_dir = os.path.join(out_root, biome_name)
        reuse_rendered_images = os.path.isdir(gen_dir) and _count_images(gen_dir) >= int(samples_per_biome)

        if reuse_rendered_images:
            if accel.is_main_process:
                print(f"Using existing rendered images for biome '{biome_name}' from {gen_dir}")
        else:
        # Reuse previously generated samples so a restarted job can resume per-biome FID.
            payload_path = os.path.join(out_root, f"generated_{biome_name}.pt")
            reuse_existing_samples = os.path.exists(payload_path)
            if reuse_existing_samples:
                if accel.is_main_process:
                    existing_samples = _load_samples_tensor_from_file(payload_path)
                    if int(existing_samples.shape[0]) < int(samples_per_biome):
                        print(
                            f"Existing samples for biome '{biome_name}' only contain "
                            f"{int(existing_samples.shape[0])} < {int(samples_per_biome)} samples; regenerating."
                        )
                        reuse_existing_samples = False
                        this_samples = None
                    else:
                        print(f"Using existing generated samples for biome '{biome_name}' from {payload_path}")
                        this_samples = existing_samples[: int(samples_per_biome)]
                else:
                    this_samples = None
                accel.wait_for_everyone()
            else:
                this_samples = None

            if not reuse_existing_samples:
                # Streamed generation per class to keep memory bounded.
                this_samples = generate_samples_for_class(
                    loaded,
                    num_samples=int(samples_per_biome),
                    class_index=class_idx,
                    output_path=payload_path,
                    cond_scale=cond_scale,
                    progress=True,
                    chunk_size=sampling_chunk_size,
                )
                # Ensure all ranks have completed shard writes/merge before anyone renders.
                accel.wait_for_everyone()

        if accel.is_main_process:
            if not reuse_rendered_images:
                # Render generated samples for this biome
                render_samples_to_images(
                    this_samples,
                    gen_dir,
                    textures_dir=textures_dir,
                    image_px=image_px,
                    show_axis=show_axis,
                    zoom=zoom,
                    converter=None,  # already original block IDs
                )

            if (not preview_saved) and os.path.isdir(gen_dir) and _count_images(gen_dir) > 0:
                preview_path = save_preprocessed_preview(
                    gen_dir,
                    preview_dir,
                    fid_mode=fid_mode,
                    preview_label=f"{biome_name}_generated",
                )
                if preview_path:
                    print(f"Saved FID preprocessing preview to {preview_path}")
                    preview_saved = True

            # Reference directory for this biome
            ref_dir = os.path.join(str(ref_biome_images_dir), biome_name)
            if not os.path.isdir(ref_dir):
                print(f"[warn] Reference biome directory not found: {ref_dir} - skipping FID for this biome")
            else:
                # Compute FID for this biome; will auto-cache ref stats in ref_dir/fid_stats.npz
                res = compute_fid_between_dirs(
                    images_dir=gen_dir,
                    ref_dir=ref_dir,
                    device=dev,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    fid_mode=fid_mode,
                )
                fid = float(res["fid_overall"])
                nimgs = int(res.get("num_images", 0))
                fid_values.append(fid)
                counts.append(nimgs)
                results_per_biome[biome_name] = fid

        # Keep ranks in lockstep biome-by-biome (avoids non-main ranks entering next sampling early).
        accel.wait_for_everyone()

    fid_values_arr = np.array(fid_values, dtype=np.float32)
    counts_arr = np.array(counts, dtype=np.int32)
    overall_mean = float(np.mean(fid_values_arr)) if fid_values_arr.size > 0 else float("nan")

    # Save metrics (main process only)
    metrics_path = save_metrics_path or os.path.join(out_root, _per_biome_metrics_filename_for_mode(fid_mode))
    if accel.is_main_process:
        try:
            np.savez_compressed(
                metrics_path,
                biome_names=np.array(list(results_per_biome.keys())),
                fid=np.array(list(results_per_biome.values()), dtype=np.float32),
                counts=counts_arr,
                overall_mean=np.array([overall_mean], dtype=np.float32),
            )
        except Exception as e:
            print(f"[warn] Failed to save metrics npz at {metrics_path}: {e}")
    accel.wait_for_everyone()

    if not accel.is_main_process:
        return None

    return {
        "biome_names": list(results_per_biome.keys()),
        "fid": results_per_biome,
        "counts": counts_arr.tolist(),
        "overall_mean": overall_mean,
        "metrics_path": metrics_path,
        "fid_mode": str(fid_mode),
    }


@torch.no_grad()
def render_samples_to_images(
    samples: torch.Tensor,
    out_dir: str,
    *,
    textures_dir: Optional[str] = None,
    image_px: int = 256,
    show_axis: bool = False,
    zoom: float = 1.0,
    converter: Optional[BlockBiomeConverter] = None,
):
    os.makedirs(out_dir, exist_ok=True)
    # Normalize to [N,H,W,D] indices or block IDs on CPU
    t = samples
    if isinstance(t, np.ndarray):
        t = torch.from_numpy(t)
    if not isinstance(t, torch.Tensor):
        raise ValueError("samples must be a torch.Tensor or numpy.ndarray")
    if t.dim() == 5:  # [N,C,H,W,D] -> indices
        t = torch.argmax(t, dim=1)
    if t.dim() != 4:
        raise ValueError(f"Expected samples of shape [N,H,W,D] or [N,C,H,W,D], got {tuple(t.shape)}")

    # If converter is provided, map indices -> original block IDs; else assume already block IDs
    if converter is not None:
        block_ids = converter.convert_to_original_blocks(t)
    else:
        block_ids = t

    # Setup visualizer
    if textures_dir:
        visualizer = MinecraftVisualizerPyVista(textures_dir=textures_dir, build_textures=True)
        render_fn = visualizer.visualize_chunk_textured
    else:
        visualizer = MinecraftVisualizerPyVista()
        render_fn = visualizer.visualize_chunk

    N = block_ids.shape[0]
    for i in tqdm(range(N), desc="Rendering samples", unit="img"):
        out_path = os.path.join(out_dir, f"chunk_{i:06d}.png")
        sample_ids = block_ids[i]
        plotter = render_fn(sample_ids, interactive=False, show_axis=bool(show_axis))
        if zoom != 1.0 and hasattr(plotter, "camera") and hasattr(plotter.camera, "zoom"):
            plotter.camera.zoom(zoom)
        plotter.screenshot(filename=out_path, window_size=(image_px, image_px), transparent_background=False)
        plotter.close()


@torch.no_grad()
def compute_fid_for_samples(
    samples: torch.Tensor,
    ref_dir: str,
    *,
    out_dir: str,
    device: torch.device,
    textures_dir: Optional[str] = None,
    image_px: int = 256,
    show_axis: bool = False,
    zoom: float = 1.0,
    converter: Optional[BlockBiomeConverter] = None,
    batch_size: int = 32,
    num_workers: int = 2,
    ref_stats_path: Optional[str] = None,
    save_ref_stats_path: Optional[str] = None,
    fid_mode: str = DEFAULT_FID_MODE,
):
    render_samples_to_images(
        samples,
        out_dir,
        textures_dir=textures_dir,
        image_px=image_px,
        show_axis=show_axis,
        zoom=zoom,
        converter=converter,
    )
    return compute_fid_between_dirs(
        images_dir=out_dir,
        ref_dir=ref_dir,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        ref_stats_path=ref_stats_path,
        save_ref_stats_path=save_ref_stats_path,
        fid_mode=fid_mode,
    )


def main():
    parser = build_argparser()
    args = parser.parse_args()

    accelerator = Accelerator()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    # Mode -1: Per-biome FID from pre-generated samples directory
    if args.per_biome_fid_samples_dir:
        # Avoid duplicating heavy compute in multi-process launch.
        if _is_distributed(accelerator) and (not accelerator.is_main_process):
            accelerator.wait_for_everyone()
            return

        res = compute_per_biome_fid_for_samples_dir(
            samples_dir=str(args.per_biome_fid_samples_dir),
            samples_glob=str(args.per_biome_samples_glob or "*.pt"),
            samples_format=str(args.per_biome_samples_format or "block_ids"),
            mappings_path=args.mappings_path,
            ref_biome_images_dir=args.per_biome_ref_dir,
            gen_out_base_dir=args.per_biome_gen_dir,
            run_id=args.per_biome_samples_id,
            ref_split=args.per_biome_ref_split,
            ref_renders_base_dir=args.per_biome_ref_renders_dir,
            dataset_path=args.dataset_path,
            ref_samples_per_biome=(int(args.ref_samples_per_biome) if args.ref_samples_per_biome not in (None, 0) else None),
            force_render_ref=bool(args.force_render_ref),
            textures_dir=args.textures_dir,
            image_px=args.image_px,
            show_axis=bool(args.show_axis),
            zoom=float(args.zoom),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            device=device_str,
            fid_mode=str(args.fid_mode),
            samples_per_biome=(int(args.samples_per_biome) if args.samples_per_biome not in (None, 0) else None),
            save_metrics_path=args.save_metrics_path,
        )

        # Write a JSON summary next to npz (match per_biome_fid_model mode format)
        summary_json = os.path.join(
            os.path.dirname(res["metrics_path"]),
            _per_biome_summary_filename_for_mode(args.fid_mode),
        )
        try:
            with open(summary_json, "w") as f:
                json.dump(
                    {
                        "timestamp": int(time.time()),
                        "overall_mean": res["overall_mean"],
                        "biome_fid": res["fid"],
                        "metrics_path": res["metrics_path"],
                        "ref_dir": res.get("ref_dir"),
                        "samples_dir": str(args.per_biome_fid_samples_dir),
                        "fid_mode": res.get("fid_mode", str(args.fid_mode)),
                    },
                    f,
                    indent=2,
                )
        except Exception:
            pass

        print(
            json.dumps(
                {
                    "overall_mean": res["overall_mean"],
                    "num_biomes": len(res["fid"]),
                    "metrics_path": res["metrics_path"],
                    "summary_json": summary_json,
                    "fid_mode": res.get("fid_mode", str(args.fid_mode)),
                },
                indent=2,
            )
        )
        accelerator.wait_for_everyone()
        return

    # Mode 0: Per-biome FID for a given model
    if args.per_biome_fid_model:
        res = compute_per_biome_fid_for_model(
            model_path=args.per_biome_fid_model,
            mappings_path=args.mappings_path,
            ref_biome_images_dir=args.per_biome_ref_dir,
            accelerator=accelerator,
            gen_out_base_dir=args.per_biome_gen_dir,
            ref_split=args.per_biome_ref_split,
            ref_renders_base_dir=args.per_biome_ref_renders_dir,
            ref_samples_per_biome=(int(args.ref_samples_per_biome) if args.ref_samples_per_biome not in (None, 0) else None),
            force_render_ref=bool(args.force_render_ref),
            samples_per_biome=int(args.samples_per_biome),
            sampling_timesteps=(int(args.sampling_timesteps) if args.sampling_timesteps not in (None, 0) else None),
            sampling_chunk_size=(int(args.sampling_chunk_size) if args.sampling_chunk_size not in (None, 0) else None),
            textures_dir=args.textures_dir,
            image_px=args.image_px,
            show_axis=bool(args.show_axis),
            zoom=float(args.zoom),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            device=device_str,
            fid_mode=str(args.fid_mode),
            cond_scale=args.cond_scale,
            save_metrics_path=args.save_metrics_path,
        )
        if not accelerator.is_main_process or res is None:
            accelerator.wait_for_everyone()
            return
        # Also write a small JSON summary next to npz
        summary_json = os.path.join(
            os.path.dirname(res["metrics_path"]),
            _per_biome_summary_filename_for_mode(args.fid_mode),
        )
        try:
            with open(summary_json, "w") as f:
                json.dump({
                    "timestamp": int(time.time()),
                    "overall_mean": res["overall_mean"],
                    "biome_fid": res["fid"],
                    "metrics_path": res["metrics_path"],
                    "fid_mode": res.get("fid_mode", str(args.fid_mode)),
                }, f, indent=2)
        except Exception:
            pass
        print(json.dumps({
            "overall_mean": res["overall_mean"],
            "num_biomes": len(res["fid"]),
            "metrics_path": res["metrics_path"],
            "summary_json": summary_json,
            "fid_mode": res.get("fid_mode", str(args.fid_mode)),
        }, indent=2))
        accelerator.wait_for_everyone()
        return

    # Mode 1: Direct FID between two directories
    if args.images_dir and args.ref_dir:
        # Avoid duplicating heavy feature extraction in multi-process launch.
        if _is_distributed(accelerator) and (not accelerator.is_main_process):
            accelerator.wait_for_everyone()
            return
        res = compute_fid_between_dirs(
            images_dir=args.images_dir,
            ref_dir=args.ref_dir,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            ref_stats_path=args.ref_stats,
            save_ref_stats_path=args.save_ref_stats,
            fid_mode=str(args.fid_mode),
        )
        results = {
            "timestamp": int(time.time()),
            "device": str(device),
            "mode": "dir_to_dir",
            "images_dir": args.images_dir,
            "ref_dir": args.ref_dir,
            **res,
        }
        out_path = args.output or os.path.join(args.images_dir, "fid_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(json.dumps({
            "fid_overall": results["fid_overall"],
            "mean_per_image_fid": results["mean_per_image_fid"],
            "std_per_image_fid": results["std_per_image_fid"],
            "num_images": results["num_images"],
            "output": out_path,
        }, indent=2))
        accelerator.wait_for_everyone()
        return

    # 1) Render subset of training dataset (skip if sufficient images already present)
    # Only main process should render reference images to avoid clobbering / duplicate work.
    os.makedirs(args.render_ref_dir, exist_ok=True)
    dataset_name = os.path.splitext(os.path.basename(args.dataset_path))[0]
    ref_images_dir = os.path.join(args.render_ref_dir, dataset_name)
    need_render_ref = True
    if os.path.isdir(ref_images_dir):
        if _count_images(ref_images_dir) >= int(args.num_samples):
            need_render_ref = False
    if need_render_ref:
        if accelerator.is_main_process:
            render_dataset_to_images(
                data_path=args.dataset_path,
                mappings_path=args.mappings_path,
                output_dir=args.render_ref_dir,
                textures_dir=args.textures_dir,
                image_size=args.image_px,
                show_axis=args.show_axis,
                start_index=args.dataset_start,
                end_index=(args.dataset_start + args.num_samples),
                skip_existing=bool(args.skip_existing),
                zoom=args.zoom,
                )
    accelerator.wait_for_everyone()

    # 2) Generate samples and compute FID using helpers
    os.makedirs(args.render_gen_dir, exist_ok=True)
    gen_dir = os.path.join(args.render_gen_dir, "generated")
    need_render_gen = True
    if os.path.isdir(gen_dir):
        if _count_images(gen_dir) >= int(args.num_samples):
            need_render_gen = False

    if need_render_gen:
        # Lazy import to avoid circular import when calc_fid is imported by the trainer
        from inference import load_model_from_file, generate_random_samples
        loaded = load_model_from_file(
            checkpoint_path=args.checkpoint_path,
            config_path=args.config_path,
            mappings_path=args.mappings_path,
            device=device_str,
            use_ema_if_available=True,
        )
        gen_payload_path = os.path.join(args.render_gen_dir, "generated_samples.pt")
        block_ids, _ = generate_random_samples(
            loaded,
            num_samples=int(args.num_samples),
            output_path=gen_payload_path,
            accelerator=accelerator,
            cond_scale=args.cond_scale,
            progress=True,
        )
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            res = compute_fid_for_samples(
                samples=block_ids,
                ref_dir=ref_images_dir,
                out_dir=gen_dir,
                device=device,
                textures_dir=args.textures_dir,
                image_px=args.image_px,
                show_axis=args.show_axis,
                zoom=args.zoom,
                # samples are already original block IDs; skip converter to avoid double-conversion
                converter=None,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                fid_mode=str(args.fid_mode),
            )
        else:
            res = None
    else:
        accelerator.wait_for_everyone()
        # Images are already present; compute FID directly between directories (main only)
        if accelerator.is_main_process:
            res = compute_fid_between_dirs(
                images_dir=gen_dir,
                ref_dir=ref_images_dir,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                fid_mode=str(args.fid_mode),
            )
        else:
            res = None

    if not accelerator.is_main_process:
        accelerator.wait_for_everyone()
        return
    if res is None:
        raise RuntimeError("FID result is missing on main process")
    output_path = args.output or os.path.join(args.render_gen_dir, "fid_results.json")
    results = {
        "timestamp": int(time.time()),
        "device": str(device),
        "mode": "end_to_end_fid",
        "dataset_path": args.dataset_path,
        "mappings_path": args.mappings_path,
        "checkpoint_path": args.checkpoint_path,
        "num_samples": int(args.num_samples),
        "ref_images_dir": ref_images_dir,
        "gen_images_dir": gen_dir,
        **res,
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps({
        "fid_overall": results["fid_overall"],
        "mean_per_image_fid": results["mean_per_image_fid"],
        "std_per_image_fid": results["std_per_image_fid"],
        "num_images": results.get("num_images", None),
        "output": output_path,
    }, indent=2))
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()

