import os
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from accelerate import Accelerator
from accelerate.state import AcceleratorState

from data_utils import BlockBiomeConverter
from denoising_diffusion_pytorch_3d import GaussianDiffusion3D_CFG
from dit import DiT3D
from discrete_diffusion_md4 import MD4Discrete3D


def _default_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _try_torch_load(path: str, map_location: torch.device) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Older PyTorch versions do not support weights_only
        return torch.load(path, map_location=map_location)


def _find_config_near_checkpoint(checkpoint_path: Path) -> Optional[Path]:
    # Typically saved next to checkpoint during training
    candidate = checkpoint_path.parent / "config.json"
    if candidate.exists():
        return candidate
    # Fallback: search one level up
    parent_candidate = checkpoint_path.parent.parent / "config.json"
    if parent_candidate.exists():
        return parent_candidate
    return None


def _resolve_path_from_config(path_like: Optional[str], config_dir: Optional[Path]) -> Optional[Path]:
    if path_like in (None, ""):
        return None
    p = Path(str(path_like))
    if p.is_absolute():
        return p
    if config_dir is not None:
        candidate = config_dir / p
        if candidate.exists():
            return candidate
    return p


def _infer_model_family(config: Dict[str, Any]) -> str:
    explicit = str(config.get("model_family", "")).strip().lower()
    if explicit in ("md4", "ddpm"):
        return explicit
    if ("timesteps" in config) or ("objective" in config) or ("beta_schedule" in config):
        return "ddpm"
    return "md4"


def _is_distributed(accelerator: Accelerator) -> bool:
    try:
        return int(getattr(accelerator, "num_processes", 1)) > 1
    except Exception:
        return False


def _contiguous_shard_bounds(total: int, rank: int, world: int) -> Tuple[int, int]:
    """
    Deterministically split range(total) into `world` contiguous shards.
    Returns [start, end) for this rank.
    """
    total = int(total)
    rank = int(rank)
    world = max(1, int(world))
    base = total // world
    rem = total % world
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    start = max(0, min(total, start))
    end = max(0, min(total, end))
    return start, end


def _safe_rmtree_dir(path: Path) -> None:
    try:
        if not path.exists() or not path.is_dir():
            return
        for p in path.iterdir():
            try:
                if p.is_file():
                    p.unlink()
            except Exception:
                pass
        try:
            path.rmdir()
        except Exception:
            pass
    except Exception:
        pass


def _make_accelerator(*, split_batches: bool = False, desired_mixed_precision: Optional[str] = None) -> Accelerator:
    """
    Create an Accelerator in a way that is compatible with an already-initialized
    global AcceleratorState (common when the caller already constructed an Accelerator).
    """
    try:
        state = AcceleratorState()
        if bool(getattr(state, "initialized", False)):
            # Must match the already-initialized state, otherwise Accelerate raises.
            mp = getattr(state, "mixed_precision", "no")
            cpu = bool(getattr(state, "cpu", False))
            return Accelerator(split_batches=bool(split_batches), mixed_precision=mp, cpu=cpu)
    except Exception:
        pass

    # Try with the requested precision first; if the global state is in a
    # broken/half-reset condition the Accelerator constructor may throw an
    # AttributeError — reset the singleton and retry with plain defaults.
    def _try_create(mp: Optional[str] = None):
        kw = dict(split_batches=bool(split_batches))
        if mp not in (None, "", "no"):
            kw["mixed_precision"] = str(mp)
        return Accelerator(**kw)

    try:
        return _try_create(desired_mixed_precision)
    except (AttributeError, RuntimeError):
        try:
            AcceleratorState._reset_state()
        except Exception:
            pass
        return _try_create(desired_mixed_precision)


def _strip_prefix_from_state_dict(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    """Return a copy of `sd` with `prefix` removed from keys that start with it."""
    if not prefix:
        return sd
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
        else:
            out[k] = v
    return out


def _maybe_extract_ema_weights(raw_ema: Any) -> Optional[Dict[str, torch.Tensor]]:
    """
    Try to extract an EMA *model* state_dict from whatever `ema_pytorch.EMA.state_dict()`
    stored in the checkpoint. Returns None if we can't confidently extract weights.
    """
    if raw_ema is None:
        return None
    if not isinstance(raw_ema, dict):
        return None

    # Common patterns:
    # - flat keys like "ema_model.xxx"
    # - nested dict under "ema_model"
    if "ema_model" in raw_ema and isinstance(raw_ema["ema_model"], dict):
        return raw_ema["ema_model"]

    # If keys are prefixed with "ema_model.", strip it.
    has_ema_prefix = any(isinstance(k, str) and k.startswith("ema_model.") for k in raw_ema.keys())
    if has_ema_prefix:
        # IMPORTANT: Filter to *only* EMA model weights. EMA state_dicts often include
        # extra bookkeeping keys like "step", "initted", and "online_model.*".
        return {k[len("ema_model."):]: v for k, v in raw_ema.items() if isinstance(k, str) and k.startswith("ema_model.")}

    # Some versions may store as "ema.ema_model." (rare, but easy to support).
    has_double_prefix = any(isinstance(k, str) and k.startswith("ema.ema_model.") for k in raw_ema.keys())
    if has_double_prefix:
        return {k[len("ema.ema_model."):]: v for k, v in raw_ema.items() if isinstance(k, str) and k.startswith("ema.ema_model.")}

    # Another pattern seen in some EMA implementations: "online_model.*" and "ema_model.*"
    # together, without nesting. If we only find online_model (and not ema_model), we can
    # still fall back to online weights, but that defeats EMA. So we refuse here.

    return None


def _choose_best_key_transform(
    sd: Dict[str, torch.Tensor],
    model_keys: set[str],
) -> Tuple[Dict[str, torch.Tensor], str]:
    """
    Heuristically fix systematic key mismatches (e.g., torch.compile adds "_orig_mod.",
    DDP adds "module."). Returns (transformed_state_dict, description).
    """
    # Candidate transforms. Order matters: we try simplest first.
    candidates: list[Tuple[str, Dict[str, torch.Tensor]]] = []
    candidates.append(("as-is", sd))
    candidates.append(("strip '_orig_mod.'", _strip_prefix_from_state_dict(sd, "_orig_mod.")))
    candidates.append(("strip 'module.'", _strip_prefix_from_state_dict(sd, "module.")))
    candidates.append(("strip 'model.'", _strip_prefix_from_state_dict(sd, "model.")))
    candidates.append(("strip 'ema_model.'", _strip_prefix_from_state_dict(sd, "ema_model.")))
    # Common composition: compiled+ddp
    candidates.append(("strip '_orig_mod.' then 'module.'", _strip_prefix_from_state_dict(_strip_prefix_from_state_dict(sd, "_orig_mod."), "module.")))
    candidates.append(("strip 'module.' then '_orig_mod.'", _strip_prefix_from_state_dict(_strip_prefix_from_state_dict(sd, "module."), "_orig_mod.")))

    best_desc = "as-is"
    best_sd = sd
    best_matches = -1

    for desc, cand in candidates:
        cand_keys = set(cand.keys())
        matches = len(model_keys.intersection(cand_keys))
        if matches > best_matches:
            best_matches = matches
            best_desc = desc
            best_sd = cand

    return best_sd, best_desc


def _build_model_from_config(config: Dict[str, Any], converter: BlockBiomeConverter, device: torch.device) -> DiT3D:
    num_blocks = len(converter.block_to_index)
    in_dim = int(num_blocks) + 1  # mask channel
    out_dim = int(num_blocks)

    # Model hyperparameters (DiT3D)
    model = DiT3D(
        in_dim=in_dim,
        out_dim=out_dim,
        hidden_channels=int(config.get("hidden_channels", 768)),
        image_size=int(config.get("image_size", 32)),
        patch_size=int(config.get("patch_size", 1)),
        time_dim=int(config.get("time_dim", 256)),
        depth=int(config.get("depth", 6)),
        num_heads=int(config.get("num_heads", 8)),
        mlp_ratio=float(config.get("mlp_ratio", 4.0)),
        attn_drop=float(config.get("attn_drop", 0.0)),
        proj_drop=float(config.get("proj_drop", 0.0)),
        class_conditional=bool(config.get("class_conditional", config.get("mode", "unconditional") == "conditional")),
        num_classes=(int(config.get("num_classes")) if config.get("class_conditional", config.get("mode", "unconditional") == "conditional") else None),
        comb_method=str(config.get("comb_method", "add")),
        cond_drop_prob=float(config.get("cond_drop_prob", 0.2)),
    ).to(device)

    return model


def _build_ddpm_model_from_config(config: Dict[str, Any], device: torch.device) -> GaussianDiffusion3D_CFG:
    image_size = int(config.get("image_size", 32))
    data_channels = int(config.get("data_channels", 64))
    class_conditional = bool(config.get("class_conditional", True))
    num_classes = config.get("num_classes", None)
    if class_conditional and num_classes in (None, 0):
        raise ValueError("DDPM config is class-conditional but num_classes is missing/zero.")

    dit = DiT3D(
        in_dim=data_channels,
        out_dim=data_channels,
        hidden_channels=int(config.get("hidden_channels", 768)),
        image_size=image_size,
        patch_size=int(config.get("patch_size", 1)),
        time_dim=int(config.get("time_dim", 256)),
        depth=int(config.get("depth", 6)),
        num_heads=int(config.get("num_heads", 8)),
        mlp_ratio=float(config.get("mlp_ratio", 4.0)),
        attn_drop=float(config.get("attn_drop", 0.0)),
        proj_drop=float(config.get("proj_drop", 0.0)),
        class_conditional=class_conditional,
        num_classes=(int(num_classes) if class_conditional else None),
        comb_method=str(config.get("comb_method", "add")),
        cond_drop_prob=float(config.get("cond_drop_prob", 0.2)),
    ).to(device)

    diffusion = GaussianDiffusion3D_CFG(
        dit,
        image_size=image_size,
        timesteps=int(config.get("timesteps", 1000)),
        objective=str(config.get("objective", "pred_v")),
        beta_schedule=str(config.get("beta_schedule", "cosine")),
        ddim_sampling_eta=float(config.get("ddim_sampling_eta", 0.0)),
        offset_noise_strength=float(config.get("offset_noise_strength", 0.0)),
        min_snr_loss_weight=bool(config.get("min_snr_loss_weight", False)),
        min_snr_gamma=float(config.get("min_snr_gamma", 5.0)),
        sampling_timesteps=(
            None
            if config.get("sampling_timesteps", None) in (None, 0)
            else int(config.get("sampling_timesteps"))
        ),
        auto_normalize=bool(config.get("auto_normalize", False)),
    ).to(device)
    return diffusion


def load_model_from_file(
    checkpoint_path: str,
    *,
    config_path: Optional[str] = None,
    mappings_path: Optional[str] = None,
    device: Optional[str] = None,
    use_ema_if_available: bool = True,
) -> Dict[str, Any]:
    """
    Load a trained DiT-based 3D discrete diffusion model along with its config and mappings.

    Args:
        checkpoint_path: Path to a saved checkpoint (e.g., results/.../model-final.pt).
        config_path: Optional path to the resolved training config.json. If not provided,
                     this will attempt to find config.json next to the checkpoint.
        mappings_path: Optional path to the mappings file (.pt). If not provided, will use
                       the value from the loaded config (mappings_file_path).
        device: Target device string (e.g., "cuda", "cpu"). Defaults to CUDA if available.
        use_ema_if_available: If the checkpoint contains EMA weights, prefer those when present.

    Returns:
        A dictionary containing:
            - model: The instantiated DiT3D model with loaded weights
            - converter: BlockBiomeConverter for block/biome mappings
            - config: The resolved configuration dictionary
            - device: The torch.device used
            - image_size, reverse_steps, sampling_timesteps, default_cond_scale
            - num_blocks, num_classes
    """
    ckpt_path = Path(checkpoint_path)
    assert ckpt_path.exists(), f"Checkpoint not found: {checkpoint_path}"

    dev = _default_device(device)

    # Load or locate config
    cfg_path = Path(config_path) if config_path else _find_config_near_checkpoint(ckpt_path)
    if cfg_path is None or not cfg_path.exists():
        raise FileNotFoundError(
            f"Could not find config.json near checkpoint; specify config_path explicitly. Checked: {cfg_path}"
        )
    with open(cfg_path, "r") as f:
        config = json.load(f)

    cfg_dir = cfg_path.parent if cfg_path is not None else None

    # Mappings for model dimensionality and class labels
    mappings_cfg = mappings_path if mappings_path else config.get("mappings_file_path", "")
    mappings_fp = _resolve_path_from_config(str(mappings_cfg), cfg_dir) if mappings_cfg not in (None, "") else None
    if mappings_fp is None:
        raise FileNotFoundError(
            "Mappings file path is missing. Provide mappings_path or set config['mappings_file_path']."
        )
    if not mappings_fp.exists():
        raise FileNotFoundError(
            f"Mappings file not found. Provide mappings_path or ensure config has 'mappings_file_path'. Got: {mappings_fp}"
        )
    mappings = _try_torch_load(str(mappings_fp), map_location="cpu")
    converter = BlockBiomeConverter(mappings.get("block_mappings"), mappings.get("biome_mappings"))

    model_family = _infer_model_family(config)
    if model_family == "ddpm":
        model = _build_ddpm_model_from_config(config, dev)
        emb_path_cfg = config.get("embeddings_path", None)
        emb_path = _resolve_path_from_config(str(emb_path_cfg), cfg_dir) if emb_path_cfg not in (None, "") else None
        if emb_path is None or (not emb_path.exists()):
            raise FileNotFoundError(
                "DDPM model requires block embeddings for decoding samples. "
                f"Could not resolve embeddings_path from config: {emb_path_cfg}"
            )
        converter.load_embeddings_pt(
            embeddings_path=str(emb_path),
            embeddings_dict_key=config.get("embeddings_dict_key", None),
            strict=True,
        )
    else:
        model = _build_model_from_config(config, converter, dev)
    raw = _try_torch_load(str(ckpt_path), map_location=dev)

    # Checkpoint may be either a dict with 'model'/'ema' or a raw state_dict
    state_dict: Dict[str, torch.Tensor]
    if isinstance(raw, dict) and "model" in raw:
        if use_ema_if_available and (raw.get("ema") is not None):
            ema_sd = _maybe_extract_ema_weights(raw.get("ema"))
            if ema_sd is not None:
                state_dict = ema_sd
            else:
                # Fall back if we can't extract EMA weights.
                state_dict = raw["model"]
        else:
            state_dict = raw["model"]
    elif isinstance(raw, dict):
        # Best-effort: assume entire dict is state_dict
        state_dict = raw
    else:
        state_dict = raw

    # Fix common systematic key mismatches (torch.compile / DDP wrappers).
    model_keys = set(model.state_dict().keys())
    state_dict, transform_desc = _choose_best_key_transform(state_dict, model_keys)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if len(missing) > 0 or len(unexpected) > 0:
        # High-signal debug: counts + transform used + a few example keys.
        print(f"[warn] Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)} (key transform: {transform_desc})")
        try:
            if len(missing) > 0:
                print("  [warn] Missing key examples:", missing[:5])
            if len(unexpected) > 0:
                print("  [warn] Unexpected key examples:", unexpected[:5])
        except Exception:
            pass

    model.eval()

    # Common sampling knobs
    reverse_steps = int(config.get("reverse_steps", config.get("timesteps", 1000)))
    sampling_timesteps = config.get("sampling_timesteps", None)
    sampling_timesteps = None if sampling_timesteps in (None, 0) else int(sampling_timesteps)
    default_cond_scale = config.get("default_cond_scale", None)
    default_cond_scale = None if default_cond_scale in (None, 0) else float(default_cond_scale)

    num_blocks = len(converter.block_to_index)
    num_classes = len(converter.biome_to_index) if converter.biome_to_index is not None else None

    return {
        "model": model,
        "converter": converter,
        "config": config,
        "device": dev,
        # Resolved file paths used for loading (handy for downstream evaluation scripts)
        "checkpoint_path": str(ckpt_path),
        "config_path": str(cfg_path),
        "mappings_path": str(mappings_fp),
        "model_family": model_family,
        "image_size": int(config.get("image_size", 32)),
        "reverse_steps": reverse_steps,
        "sampling_timesteps": sampling_timesteps,
        "default_cond_scale": default_cond_scale,
        "num_blocks": int(num_blocks),
        "num_classes": (int(num_classes) if num_classes is not None else None),
    }


@torch.inference_mode()
def generate_random_samples(
    loaded: Dict[str, Any],
    num_samples: int,
    *,
    output_path: str,
    accelerator: Optional[Accelerator] = None,
    cond_scale: Optional[float] = None,
    progress: bool = True,
    chunk_size: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Generate N voxel samples, evenly sampling across all available classes (if conditional),
    and write the combined batch to disk.

    Args:
        loaded: The dictionary returned by load_model_from_file.
        num_samples: Total number of samples to generate.
        output_path: File path to write the combined batch (torch.save).
        cond_scale: Optional override for classifier-free guidance scale. Defaults to
                    loaded["default_cond_scale"] if available.
        progress: Whether to show a progress bar during sampling.

    Returns:
        (block_ids, class_labels)
            - block_ids: [N, H, W, D] tensor of original block IDs
            - class_labels: [N] tensor of class indices (or None if unconditional)
    """
    assert num_samples > 0, "num_samples must be > 0"

    model = loaded["model"]
    converter: BlockBiomeConverter = loaded["converter"]
    device: torch.device = loaded["device"]
    model_family = str(loaded.get("model_family", "md4")).lower()

    image_size = int(loaded.get("image_size", 32))
    reverse_steps = int(loaded.get("reverse_steps", 1000))
    sampling_timesteps = loaded.get("sampling_timesteps", None)
    default_scale = loaded.get("default_cond_scale", None)
    effective_scale = float(cond_scale if cond_scale is not None else (default_scale if default_scale is not None else 4.0))

    # Setup accelerator context for autocast / distributed sharding.
    # IMPORTANT: do not create a second Accelerator with a different mixed_precision once the
    # global AcceleratorState is initialized (Accelerate will error).
    amp = bool(loaded["config"].get("amp", False))
    mp_type = loaded["config"].get("mixed_precision_type", "no") if amp else "no"
    accelerator = accelerator or _make_accelerator(split_batches=False, desired_mixed_precision=(mp_type if amp else None))

    model_is_conditional = bool(getattr(model, "class_conditional", False) or hasattr(model, "has_class_conditioning"))

    # Determine per-sample class labels if conditional
    num_classes = loaded.get("num_classes", None)
    class_labels_tensor: Optional[torch.Tensor] = None
    if model_is_conditional and num_classes not in (None, 0):
        C = int(num_classes)
        base = int(num_samples) // C
        rem = int(num_samples) % C
        labels: list[int] = []
        for c in range(C):
            take = base + (1 if c < rem else 0)
            labels.extend([c] * take)
        # In case of any rounding mismatch, trim or pad
        if len(labels) > num_samples:
            labels = labels[:num_samples]
        elif len(labels) < num_samples:
            labels.extend([0] * (num_samples - len(labels)))
        # Keep labels on CPU; we'll move chunk-wise to device
        class_labels_tensor = torch.tensor(labels, dtype=torch.long)
    else:
        class_labels_tensor = None

    if model_family == "ddpm" and class_labels_tensor is None:
        raise ValueError("DDPM sampling currently requires class labels (class-conditional model).")

    md4 = None
    if model_family != "ddpm":
        num_blocks = int(loaded["num_blocks"])  # K
        md4 = MD4Discrete3D(accelerator=accelerator, num_classes=num_blocks, device=device)

    # Compute air fallback index for any remaining masked voxels
    try:
        air_idx = converter.get_air_block_index()
    except Exception:
        air_idx = 0

    # Memory-safe micro-batched sampling
    if accelerator.is_main_process:
        print(f"Sampling {num_samples} samples with {sampling_timesteps} timesteps (micro-batched)")
    spatial_size = (image_size, image_size, image_size)
    chunk_size = int(chunk_size or loaded["config"].get("fid_batch_size", loaded["config"].get("sampling_batch_size", 8)))
    chunk_size = max(1, min(int(num_samples), chunk_size))

    # If launched with `accelerate launch` (multi-process), shard sampling across ranks.
    # We avoid all-gathering gigantic tensors (which can OOM) by writing per-rank shards
    # to disk and merging on the main process.
    if _is_distributed(accelerator):
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shard_dir = out_path.parent / f"{out_path.name}.dist"
        if accelerator.is_main_process:
            shard_dir.mkdir(parents=True, exist_ok=True)
        accelerator.wait_for_everyone()

        rank = int(getattr(accelerator, "process_index", 0))
        world = int(getattr(accelerator, "num_processes", 1))
        start, end = _contiguous_shard_bounds(int(num_samples), rank, world)
        local_n = max(0, int(end - start))

        local_blocks: list[torch.Tensor] = []
        local_labels_cpu: Optional[torch.Tensor] = None
        if local_n > 0:
            local_labels = (class_labels_tensor[start:end].to(device) if class_labels_tensor is not None else None)
            if local_labels is not None:
                local_labels_cpu = local_labels.detach().to("cpu")

            remaining = int(local_n)
            cursor = 0
            while remaining > 0:
                bsz = min(remaining, int(chunk_size))
                cls_chunk = (local_labels[cursor:cursor + bsz] if local_labels is not None else None)
                if model_family == "ddpm":
                    samples_emb = model.sample(
                        cls_chunk,
                        cond_scale=(effective_scale if cls_chunk is not None else 1.0),
                    )
                    emb_for_decode = samples_emb.detach().to("cpu").permute(0, 1, 3, 4, 2).contiguous()
                    block_ids_chunk = converter.convert_emb_to_blocks(emb_for_decode)
                else:
                    if (sampling_timesteps is not None) and (sampling_timesteps > 0) and (sampling_timesteps < reverse_steps):
                        samples_indices = md4.ddim_sample(
                            model,
                            batch_size=int(bsz),
                            spatial_size=spatial_size,
                            sampling_timesteps=int(sampling_timesteps),
                            reverse_steps=int(reverse_steps),
                            progress=bool(progress) and accelerator.is_main_process,
                            air_index_fallback=int(air_idx),
                            class_cond=cls_chunk,
                            cond_scale=effective_scale if cls_chunk is not None else 1.0,
                        )
                    else:
                        samples_indices = md4.sample(
                            model,
                            batch_size=int(bsz),
                            spatial_size=spatial_size,
                            reverse_steps=int(reverse_steps),
                            progress=bool(progress) and accelerator.is_main_process,
                            air_index_fallback=int(air_idx),
                            class_cond=cls_chunk,
                            cond_scale=effective_scale if cls_chunk is not None else 1.0,
                        )
                    block_ids_chunk = converter.convert_to_original_blocks(samples_indices.detach().to("cpu"))
                local_blocks.append(block_ids_chunk)
                remaining -= bsz
                cursor += bsz

        local_block_ids = torch.cat(local_blocks, dim=0) if len(local_blocks) > 0 else torch.empty((0, *spatial_size), dtype=torch.long)
        shard_path = shard_dir / f"shard_rank_{rank:03d}.pt"
        torch.save(
            {
                "rank": rank,
                "start": start,
                "end": end,
                "voxels": local_block_ids,
                "classes": local_labels_cpu,
            },
            str(shard_path),
        )

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            merged_blocks: list[torch.Tensor] = []
            merged_labels: list[torch.Tensor] = []
            for r in range(world):
                sp = shard_dir / f"shard_rank_{r:03d}.pt"
                if not sp.exists():
                    continue
                shard = torch.load(str(sp), map_location="cpu", weights_only=False)
                bx = shard.get("voxels", None)
                if isinstance(bx, torch.Tensor) and bx.numel() > 0:
                    merged_blocks.append(bx)
                cx = shard.get("classes", None)
                if isinstance(cx, torch.Tensor) and cx.numel() > 0:
                    merged_labels.append(cx)

            block_ids = torch.cat(merged_blocks, dim=0) if len(merged_blocks) > 0 else torch.empty((0, *spatial_size), dtype=torch.long)
            class_labels_tensor = torch.cat(merged_labels, dim=0) if len(merged_labels) > 0 else None

            # Best-effort cleanup to avoid leaving lots of shard files around.
            _safe_rmtree_dir(shard_dir)
        else:
            block_ids = torch.empty((0, *spatial_size), dtype=torch.long)
            class_labels_tensor = None

        accelerator.wait_for_everyone()
    else:
        blocks_list: list[torch.Tensor] = []
        labels_list: list[torch.Tensor] = []
        remaining = int(num_samples)
        cursor = 0
        while remaining > 0:
            bsz = min(remaining, chunk_size)
            if class_labels_tensor is not None:
                cls_chunk = class_labels_tensor[cursor:cursor + bsz].to(device)
            else:
                cls_chunk = None
            if model_family == "ddpm":
                samples_emb = model.sample(
                    cls_chunk,
                    cond_scale=(effective_scale if cls_chunk is not None else 1.0),
                )
                emb_for_decode = samples_emb.detach().to("cpu").permute(0, 1, 3, 4, 2).contiguous()
                block_ids_chunk = converter.convert_emb_to_blocks(emb_for_decode)
            else:
                if (sampling_timesteps is not None) and (sampling_timesteps > 0) and (sampling_timesteps < reverse_steps):
                    samples_indices = md4.ddim_sample(
                        model,
                        batch_size=int(bsz),
                        spatial_size=spatial_size,
                        sampling_timesteps=int(sampling_timesteps),
                        reverse_steps=int(reverse_steps),
                        progress=bool(progress),
                        air_index_fallback=int(air_idx),
                        class_cond=cls_chunk,
                        cond_scale=effective_scale if cls_chunk is not None else 1.0,
                    )
                else:
                    samples_indices = md4.sample(
                        model,
                        batch_size=int(bsz),
                        spatial_size=spatial_size,
                        reverse_steps=int(reverse_steps),
                        progress=bool(progress),
                        air_index_fallback=int(air_idx),
                        class_cond=cls_chunk,
                        cond_scale=effective_scale if cls_chunk is not None else 1.0,
                    )
                block_ids_chunk = converter.convert_to_original_blocks(samples_indices.detach().to("cpu"))
            blocks_list.append(block_ids_chunk)
            if cls_chunk is not None:
                labels_list.append(cls_chunk.detach().to("cpu"))
            remaining -= bsz
            cursor += bsz

        block_ids = torch.cat(blocks_list, dim=0) if len(blocks_list) > 1 else blocks_list[0]
        class_labels_tensor = (torch.cat(labels_list, dim=0) if len(labels_list) > 0 else None)

    # Prepare optional class name strings for convenience
    class_names = None
    if class_labels_tensor is not None and converter.index_to_biome is not None:
        try:
            class_names = converter.convert_class_to_biomes(class_labels_tensor.detach().to("cpu"))
        except Exception:
            class_names = [str(int(c)) for c in class_labels_tensor.detach().to("cpu").tolist()]

    # Ensure output directory exists
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "voxels": block_ids,  # [N,H,W,D] original block IDs
        "classes": (class_labels_tensor.detach().to("cpu") if class_labels_tensor is not None else None),
        "class_names": class_names,
        "image_size": image_size,
    }
    # In distributed mode only main process writes the combined file (others wrote shards).
    if (not _is_distributed(accelerator)) or accelerator.is_main_process:
        torch.save(payload, str(out_path))
    accelerator.wait_for_everyone()

    return block_ids, (class_labels_tensor.detach().to("cpu") if class_labels_tensor is not None else None)


@torch.inference_mode()
def generate_samples_for_class(
    loaded: Dict[str, Any],
    num_samples: int,
    *,
    class_index: int,
    output_path: str,
    accelerator: Optional[Accelerator] = None,
    cond_scale: Optional[float] = None,
    progress: bool = True,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Generate N voxel samples for a specific class index and write them to disk.
    Returns block IDs [N,H,W,D] (original block IDs) for rendering.
    """
    assert num_samples > 0, "num_samples must be > 0"

    model = loaded["model"]
    converter: BlockBiomeConverter = loaded["converter"]
    device: torch.device = loaded["device"]
    model_family = str(loaded.get("model_family", "md4")).lower()

    model_is_conditional = bool(getattr(model, "class_conditional", False) or hasattr(model, "has_class_conditioning"))
    if not model_is_conditional:
        raise ValueError("Model is not class-conditional; cannot generate samples for a specific class.")

    num_classes = loaded.get("num_classes", None)
    if num_classes in (None, 0):
        raise ValueError("Loaded configuration is missing num_classes; cannot generate class-conditional samples.")
    if int(class_index) < 0 or int(class_index) >= int(num_classes):
        raise ValueError(f"class_index {class_index} out of range [0, {int(num_classes) - 1}]")

    image_size = int(loaded.get("image_size", 32))
    reverse_steps = int(loaded.get("reverse_steps", 1000))
    sampling_timesteps = loaded.get("sampling_timesteps", None)
    default_scale = loaded.get("default_cond_scale", None)
    effective_scale = float(cond_scale if cond_scale is not None else (default_scale if default_scale is not None else 4.0))

    # Setup accelerator (see generate_random_samples for why this is careful)
    amp = bool(loaded["config"].get("amp", False))
    mp_type = loaded["config"].get("mixed_precision_type", "no") if amp else "no"
    accelerator = accelerator or _make_accelerator(split_batches=False, desired_mixed_precision=(mp_type if amp else None))

    md4 = None
    if model_family != "ddpm":
        num_blocks = int(loaded["num_blocks"])  # K
        md4 = MD4Discrete3D(accelerator=accelerator, num_classes=num_blocks, device=device)

    # Fixed class labels for this batch
    class_labels_tensor = torch.full((int(num_samples),), int(class_index), dtype=torch.long, device=device)

    # Compute air fallback index
    try:
        air_idx = converter.get_air_block_index()
    except Exception:
        air_idx = 0

    # Memory-safe micro-batched sampling
    spatial_size = (image_size, image_size, image_size)
    chunk_size = int(chunk_size or loaded["config"].get("fid_batch_size", loaded["config"].get("sampling_batch_size", 8)))
    chunk_size = max(1, min(int(num_samples), chunk_size))
    if accelerator.is_main_process:
        print(f"Sampling {num_samples} samples for class {int(class_index)} with {sampling_timesteps} timesteps. Batch size: {chunk_size} on device {device}")

    if _is_distributed(accelerator):
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shard_dir = out_path.parent / f"{out_path.name}.dist"
        if accelerator.is_main_process:
            shard_dir.mkdir(parents=True, exist_ok=True)
        accelerator.wait_for_everyone()

        rank = int(getattr(accelerator, "process_index", 0))
        world = int(getattr(accelerator, "num_processes", 1))
        start, end = _contiguous_shard_bounds(int(num_samples), rank, world)
        local_n = max(0, int(end - start))

        local_blocks: list[torch.Tensor] = []
        if local_n > 0:
            # contiguous slice of constant-label tensor
            local_labels = class_labels_tensor[start:end]
            remaining = int(local_n)
            cursor = 0
            while remaining > 0:
                bsz = min(remaining, int(chunk_size))
                cls_chunk = local_labels[cursor:cursor + bsz]
                if model_family == "ddpm":
                    samples_emb = model.sample(cls_chunk, cond_scale=effective_scale)
                    emb_for_decode = samples_emb.detach().to("cpu").permute(0, 1, 3, 4, 2).contiguous()
                    block_ids_chunk = converter.convert_emb_to_blocks(emb_for_decode)
                else:
                    if (sampling_timesteps is not None) and (sampling_timesteps > 0) and (sampling_timesteps < reverse_steps):
                        samples_indices = md4.ddim_sample(
                            model,
                            batch_size=int(bsz),
                            spatial_size=spatial_size,
                            sampling_timesteps=int(sampling_timesteps),
                            reverse_steps=int(reverse_steps),
                            progress=bool(progress) and accelerator.is_main_process,
                            air_index_fallback=int(air_idx),
                            class_cond=cls_chunk,
                            cond_scale=effective_scale,
                        )
                    else:
                        samples_indices = md4.sample(
                            model,
                            batch_size=int(bsz),
                            spatial_size=spatial_size,
                            reverse_steps=int(reverse_steps),
                            progress=bool(progress) and accelerator.is_main_process,
                            air_index_fallback=int(air_idx),
                            class_cond=cls_chunk,
                            cond_scale=effective_scale,
                        )
                    block_ids_chunk = converter.convert_to_original_blocks(samples_indices.detach().to("cpu"))
                local_blocks.append(block_ids_chunk)
                remaining -= bsz
                cursor += bsz

        local_block_ids = torch.cat(local_blocks, dim=0) if len(local_blocks) > 0 else torch.empty((0, *spatial_size), dtype=torch.long)
        shard_path = shard_dir / f"shard_rank_{rank:03d}.pt"
        torch.save(
            {
                "rank": rank,
                "start": start,
                "end": end,
                "voxels": local_block_ids,
            },
            str(shard_path),
        )

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            merged_blocks: list[torch.Tensor] = []
            for r in range(world):
                sp = shard_dir / f"shard_rank_{r:03d}.pt"
                if not sp.exists():
                    continue
                shard = torch.load(str(sp), map_location="cpu", weights_only=False)
                bx = shard.get("voxels", None)
                if isinstance(bx, torch.Tensor) and bx.numel() > 0:
                    merged_blocks.append(bx)
            block_ids = torch.cat(merged_blocks, dim=0) if len(merged_blocks) > 0 else torch.empty((0, *spatial_size), dtype=torch.long)
            _safe_rmtree_dir(shard_dir)
        else:
            block_ids = torch.empty((0, *spatial_size), dtype=torch.long)

        accelerator.wait_for_everyone()
    else:
        blocks_list: list[torch.Tensor] = []
        remaining = int(num_samples)
        cursor = 0
        while remaining > 0:
            bsz = min(remaining, chunk_size)
            cls_chunk = class_labels_tensor[cursor:cursor + bsz]
            if model_family == "ddpm":
                samples_emb = model.sample(cls_chunk, cond_scale=effective_scale)
                emb_for_decode = samples_emb.detach().to("cpu").permute(0, 1, 3, 4, 2).contiguous()
                block_ids_chunk = converter.convert_emb_to_blocks(emb_for_decode)
            else:
                if (sampling_timesteps is not None) and (sampling_timesteps > 0) and (sampling_timesteps < reverse_steps):
                    samples_indices = md4.ddim_sample(
                        model,
                        batch_size=int(bsz),
                        spatial_size=spatial_size,
                        sampling_timesteps=int(sampling_timesteps),
                        reverse_steps=int(reverse_steps),
                        progress=bool(progress),
                        air_index_fallback=int(air_idx),
                        class_cond=cls_chunk,
                        cond_scale=effective_scale,
                    )
                else:
                    samples_indices = md4.sample(
                        model,
                        batch_size=int(bsz),
                        spatial_size=spatial_size,
                        reverse_steps=int(reverse_steps),
                        progress=bool(progress),
                        air_index_fallback=int(air_idx),
                        class_cond=cls_chunk,
                        cond_scale=effective_scale,
                    )
                block_ids_chunk = converter.convert_to_original_blocks(samples_indices.detach().to("cpu"))
            blocks_list.append(block_ids_chunk)
            remaining -= bsz
            cursor += bsz

        block_ids = torch.cat(blocks_list, dim=0) if len(blocks_list) > 1 else blocks_list[0]

    # Save payload
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "voxels": block_ids,  # [N,H,W,D] original block IDs
        "classes": torch.full((int(num_samples),), int(class_index), dtype=torch.long),
        "class_names": [converter.index_to_biome[int(class_index)]] * int(num_samples) if converter.index_to_biome is not None else None,
        "image_size": image_size,
    }
    # In distributed mode only main process writes the combined file (others wrote shards).
    if (not _is_distributed(accelerator)) or accelerator.is_main_process:
        torch.save(payload, str(out_path))
    accelerator.wait_for_everyone()

    return block_ids


@torch.inference_mode()
def generate_sample_with_intermediates(
    loaded: Dict[str, Any],
    *,
    class_index: Optional[int] = None,
    num_intermediate_frames: int = 20,
    cond_scale: Optional[float] = None,
    sampling_timesteps: Optional[int] = None,
    reverse_steps: Optional[int] = None,
    progress: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a single sample and return subsampled intermediate denoising states.

    Uses the MD4 sampler's built-in ``return_intermediates`` support, then evenly
    subsamples the full sequence down to *num_intermediate_frames* snapshots.

    No changes to :mod:`discrete_diffusion_md4` are required — both ``sample()``
    and ``ddim_sample()`` already capture per-step states when the flag is set.

    Args:
        loaded: Dict returned by :func:`load_model_from_file`.
        class_index: Biome class index for conditional generation.
            If *None* and the model is class-conditional, defaults to 0.
        num_intermediate_frames: How many intermediate snapshots to keep
            (evenly subsampled from the full denoising trajectory).
        cond_scale: Override for classifier-free guidance scale.
        sampling_timesteps: If provided, use DDIM-style coarse schedule with
            this many steps (recommended — fewer steps = less memory + faster).
        reverse_steps: Override for the total reverse-process grid length.
        progress: Display a ``tqdm`` progress bar during sampling.

    Returns:
        ``(final_blocks, intermediates)``

        * **final_blocks** — ``[H, W, D]`` tensor of original Minecraft block IDs.
        * **intermediates** — ``[num_frames, H, W, D]`` tensor of *model indices*
          (including mask tokens) at evenly-spaced denoising steps.  The mask
          token is ``num_blocks`` (one past the last valid block index).
    """
    if str(loaded.get("model_family", "md4")).lower() == "ddpm":
        raise NotImplementedError("Intermediate MD4 denoising trajectories are not implemented for DDPM models.")

    model: DiT3D = loaded["model"]
    converter: BlockBiomeConverter = loaded["converter"]
    device: torch.device = loaded["device"]

    image_size = int(loaded.get("image_size", 32))
    _reverse_steps = int(reverse_steps or loaded.get("reverse_steps", 1000))
    _sampling_ts = sampling_timesteps or loaded.get("sampling_timesteps", None)
    _sampling_ts = None if _sampling_ts in (None, 0) else int(_sampling_ts)
    default_scale = loaded.get("default_cond_scale", None)
    effective_scale = float(
        cond_scale if cond_scale is not None
        else (default_scale if default_scale is not None else 4.0)
    )

    amp = bool(loaded["config"].get("amp", False))
    mp_type = loaded["config"].get("mixed_precision_type", "no") if amp else "no"
    accelerator = _make_accelerator(
        split_batches=False,
        desired_mixed_precision=(mp_type if amp else None),
    )

    num_blocks = int(loaded["num_blocks"])
    md4 = MD4Discrete3D(accelerator=accelerator, num_classes=num_blocks, device=device)

    try:
        air_idx = converter.get_air_block_index()
    except Exception:
        air_idx = 0

    class_cond = None
    if model.class_conditional:
        ci = int(class_index) if class_index is not None else 0
        class_cond = torch.tensor([ci], device=device, dtype=torch.long)

    spatial = (image_size, image_size, image_size)

    use_ddim = _sampling_ts is not None and _sampling_ts < _reverse_steps
    if use_ddim:
        final, intermediates = md4.ddim_sample(
            model, batch_size=1, spatial_size=spatial,
            sampling_timesteps=_sampling_ts,
            reverse_steps=_reverse_steps,
            progress=progress, air_index_fallback=air_idx,
            class_cond=class_cond, cond_scale=effective_scale,
            return_intermediates=True,
        )
    else:
        final, intermediates = md4.sample(
            model, batch_size=1, spatial_size=spatial,
            reverse_steps=_reverse_steps,
            progress=progress, air_index_fallback=air_idx,
            class_cond=class_cond, cond_scale=effective_scale,
            return_intermediates=True,
        )

    # intermediates: [1, T, H, W, D] — subsample to requested frame count
    T = intermediates.shape[1]
    if num_intermediate_frames < T:
        idx = torch.linspace(0, T - 1, steps=num_intermediate_frames).round().long()
        idx[-1] = T - 1
        intermediates = intermediates[:, idx]

    final_blocks = converter.convert_to_original_blocks(final)
    return final_blocks[0], intermediates[0]


@torch.inference_mode()
def generate_sample_with_logit_intermediates(
    loaded: Dict[str, Any],
    *,
    class_index: Optional[int] = None,
    save_stride: int = 10,
    cond_scale: Optional[float] = None,
    sampling_timesteps: Optional[int] = None,
    reverse_steps: Optional[int] = None,
    progress: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a single sample while capturing logit-sampled composites at
    strided intervals.

    This reimplements the MD4 sampling loop (calling the same helper methods)
    so that at every *save_stride*-th step we build a **composite** frame:

    * **Committed positions** (already unmasked) keep their locked-in token.
    * **Masked positions** are filled with a fresh sample from the model's
      predicted probability distribution at that step.

    The result is a sequence of full-volume "drafts" that start noisy and
    progressively sharpen — visually similar to a DDPM denoising trajectory.

    Only the composites at stride boundaries are stored (never the raw logits),
    so memory overhead is modest even for long schedules.

    Args:
        loaded: Dict returned by :func:`load_model_from_file`.
        class_index: Biome class index for conditional generation (default: 0).
        save_stride: Save a composite every *save_stride* denoising steps.
        cond_scale: Override for classifier-free guidance scale.
        sampling_timesteps: If provided, use DDIM-style coarse schedule with
            this many steps (fewer steps = faster).
        reverse_steps: Override for the total reverse-process grid length.
        progress: Display a ``tqdm`` progress bar during sampling.

    Returns:
        ``(final_blocks, intermediates)``

        * **final_blocks** — ``[H, W, D]`` original Minecraft block IDs.
        * **intermediates** — ``[N, H, W, D]`` model indices (no mask tokens)
          where *N* is the number of snapshots taken at stride boundaries plus
          the final fully-denoised state.
    """
    if str(loaded.get("model_family", "md4")).lower() == "ddpm":
        raise NotImplementedError("Logit intermediate sampling is currently only supported for MD4 models.")

    from tqdm import tqdm as _tqdm

    model: DiT3D = loaded["model"]
    converter: BlockBiomeConverter = loaded["converter"]
    device: torch.device = loaded["device"]

    image_size = int(loaded.get("image_size", 32))
    _reverse_steps = int(reverse_steps or loaded.get("reverse_steps", 1000))
    _sampling_ts = sampling_timesteps or loaded.get("sampling_timesteps", None)
    _sampling_ts = None if _sampling_ts in (None, 0) else int(_sampling_ts)
    default_scale = loaded.get("default_cond_scale", None)
    effective_scale = float(
        cond_scale if cond_scale is not None
        else (default_scale if default_scale is not None else 4.0)
    )

    amp = bool(loaded["config"].get("amp", False))
    mp_type = loaded["config"].get("mixed_precision_type", "no") if amp else "no"
    accelerator = _make_accelerator(
        split_batches=False,
        desired_mixed_precision=(mp_type if amp else None),
    )

    num_blocks = int(loaded["num_blocks"])
    md4 = MD4Discrete3D(accelerator=accelerator, num_classes=num_blocks, device=device)
    mask_token_id = md4.mask_token_id

    try:
        air_idx = converter.get_air_block_index()
    except Exception:
        air_idx = 0

    class_cond = None
    if model.class_conditional:
        ci = int(class_index) if class_index is not None else 0
        class_cond = torch.tensor([ci], device=device, dtype=torch.long)

    H = W = D = image_size

    # ------------------------------------------------------------------
    # Build the time-step schedule (full or DDIM-coarse)
    # ------------------------------------------------------------------
    tgrid = torch.linspace(0, 1, _reverse_steps + 1, device=device)

    use_ddim = _sampling_ts is not None and _sampling_ts < _reverse_steps
    if use_ddim:
        idxs = torch.linspace(0, _reverse_steps, steps=_sampling_ts + 1,
                              device=device).round().long().tolist()
        idxs = sorted(set(idxs))
        if idxs[-1] != _reverse_steps:
            idxs.append(_reverse_steps)
        if idxs[0] == 0:
            start = len(idxs) - 1
        else:
            idxs = [0] + idxs
            start = len(idxs) - 1
        step_pairs = [(idxs[k], idxs[k - 1]) for k in range(start, 1, -1)]
    else:
        step_pairs = [(i, i - 1) for i in range(_reverse_steps, 1, -1)]

    total_steps = len(step_pairs)

    # Which loop iterations to snapshot (stride-based, always include last)
    save_set = set(range(0, total_steps, max(1, int(save_stride))))
    save_set.add(total_steps - 1)

    # ------------------------------------------------------------------
    # Sampling loop (functionally identical to MD4Discrete3D.sample /
    # ddim_sample, but captures logit-sampled composites at stride)
    # ------------------------------------------------------------------
    xt = torch.ones((1, H, W, D), device=device, dtype=torch.long) * mask_token_id
    snapshots = []

    rng = _tqdm(range(total_steps), desc="sampling") if progress else range(total_steps)
    for step_num in rng:
        ti_idx, si_idx = step_pairs[step_num]
        ti = torch.ones((1,), device=device) * tgrid[ti_idx]
        si = torch.ones((1,), device=device) * tgrid[si_idx]

        mask_prob = md4._compute_mask_prob(ti, si, (1, H, W, D))

        x_in = F.one_hot(
            torch.clamp(xt, 0, mask_token_id),
            num_classes=mask_token_id + 1,
        )
        x_in = x_in.permute(0, 4, 1, 2, 3).float()   # [B, K+1, H, W, D]
        x_in = x_in.permute(0, 1, 4, 2, 3)            # [B, K+1, D, H, W]

        with accelerator.autocast():
            if class_cond is not None and hasattr(model, 'forward_with_cond_scale'):
                logits, _ = model.forward_with_cond_scale(
                    x_in, ti, class_cond, cond_scale=effective_scale,
                )
            elif class_cond is not None:
                logits = model(x_in, ti, class_cond)
            else:
                logits = model(x_in, ti)

        logits = logits.permute(0, 1, 3, 4, 2).contiguous()   # [B, K, H, W, D]
        probs = md4._probs_from_logits(logits)
        sampled = md4._safe_multinomial(probs, 1, H, W, D)

        # ---- build & store logit-sampled composite at stride --------
        if step_num in save_set:
            composite = xt.clone()
            still_masked = (composite == mask_token_id)
            composite[still_masked] = sampled[still_masked]
            snapshots.append(composite.detach().cpu())

        # ---- apply the actual unmasking (identical to MD4) ----------
        is_last = (step_num == total_steps - 1)
        mask_t = (xt == mask_token_id)
        if is_last:
            update_mask = mask_t
        else:
            to_unmask = torch.rand((1, H, W, D), device=device) < mask_prob
            update_mask = mask_t & to_unmask
        xt[update_mask] = sampled[update_mask]

        del x_in, logits, probs, sampled, mask_prob, ti, si, mask_t, update_mask
        if not is_last:
            del to_unmask

    # ------------------------------------------------------------------
    # Cleanup (same as MD4)
    # ------------------------------------------------------------------
    remaining_mask = (xt == mask_token_id)
    if remaining_mask.any():
        xt, num_forced = md4._force_unmask_remaining(
            model, xt, class_cond=class_cond, cond_scale=effective_scale,
        )
        if progress and num_forced > 0:
            print(f"Forced final unmask for {num_forced} residual masked voxels.")

    remaining_mask = (xt == mask_token_id)
    if remaining_mask.any():
        xt[remaining_mask] = int(air_idx)

    # Always include the fully-denoised final as the last snapshot
    snapshots.append(xt.clone().detach().cpu())

    final_blocks = converter.convert_to_original_blocks(xt)
    intermediates = torch.stack(snapshots, dim=1)  # [1, N, H, W, D]
    return final_blocks[0], intermediates[0]
