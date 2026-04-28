"""
Inpainting inference script for DDPM (Gaussian diffusion) 3D models.

Implements time-aligned inpainting for continuous diffusion models.

Key difference from MD4: DDPM uses Gaussian noise rather than discrete masking,
so inpainting requires re-injecting the known region at each timestep as a noised
version of the original (replacement inpainting).

Usage:
    python inpaint_ddpm.py --checkpoint path/to/model.pt --mappings path/to/mappings.pt --source_folder path/to/folder
    python inpaint_ddpm.py --checkpoint path/to/model.pt --mappings path/to/mappings.pt --source_folder path/to/folder --experiment_mode biome_swap
"""

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from accelerate import Accelerator

from data_utils import BlockBiomeConverter
from visualization_utils import MinecraftVisualizerPyVista

UNCONDITIONAL_BIOME = "__unconditional__"
UNCONDITIONAL_BIOME_ALIASES = {
    UNCONDITIONAL_BIOME,
    "unconditional",
    "none",
    "null",
}

STANDARD_SAMPLES_PER_BIOME = 2
STANDARD_CONTEXT_CORNER_SIZE = 8
STANDARD_CONTEXT_HEIGHT = 32


def find_start_timestep_for_fraction(
    sqrt_alphas_cumprod: torch.Tensor,
    fraction_known: float,
) -> int:
    """
    Find the discrete timestep where sqrt_alphas_cumprod is closest to fraction_known.
    
    In DDPM, sqrt_alphas_cumprod[t] represents how much of the original signal
    is preserved at timestep t. Higher values mean less noise.
    
    For time-aligned inpainting: if we're keeping fraction_known of the image,
    we want to start at a timestep where the model expects that level of signal.
    
    Args:
        sqrt_alphas_cumprod: [T] tensor of sqrt(alpha_cumprod) values
        fraction_known: Fraction of image being kept as context (0 to 1)
        
    Returns:
        Integer timestep t* to start from
    """
    # Find timestep where sqrt_alphas_cumprod is closest to fraction_known
    # Higher fraction_known -> lower t (less noisy)
    # Lower fraction_known -> higher t (more noisy)
    
    diffs = (sqrt_alphas_cumprod - fraction_known).abs()
    t_star = diffs.argmin().item()
    return int(t_star)


class DDPMInpainter:
    """
    Inpainting helper for DDPM/Gaussian diffusion models.
    
    Uses replacement inpainting: at each step, the known region is replaced
    with the appropriately noised version of the original.
    """
    
    def __init__(
        self,
        diffusion_model,
        device: torch.device,
    ):
        self.diffusion_model = diffusion_model
        self.device = device
        
        # Extract schedule parameters from the diffusion model
        self.num_timesteps = diffusion_model.num_timesteps
        self.sqrt_alphas_cumprod = diffusion_model.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = diffusion_model.sqrt_one_minus_alphas_cumprod.to(device)
        self.alphas_cumprod = diffusion_model.alphas_cumprod.to(device)
        
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward diffusion process: add noise to x_start at timestep t.
        x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise
    
    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        """Extract values from a at indices t and reshape for broadcasting."""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    @torch.inference_mode()
    def inpaint_time_aligned(
        self,
        x_known: torch.Tensor,
        inpaint_mask: torch.Tensor,
        class_cond: torch.Tensor,
        cond_scale: float = 6.0,
        rescaled_phi: float = 0.7,
        progress: bool = True,
    ) -> torch.Tensor:
        """
        Time-aligned inpainting for DDPM.
        
        Args:
            x_known: [B, C, D, H, W] tensor with known voxel embeddings (normalized to [-1, 1])
            inpaint_mask: [B, 1, D, H, W] boolean tensor, True = generate here, False = keep
            class_cond: [B] tensor of class labels for conditioning
            cond_scale: Classifier-free guidance scale
            rescaled_phi: Rescaled CFG parameter
            progress: Whether to show progress bar
            
        Returns:
            [B, C, D, H, W] inpainted result
        """
        B = x_known.shape[0]
        device = x_known.device
        
        # Calculate fraction known
        total_voxels = inpaint_mask[0].numel()
        known_voxels = (~inpaint_mask[0]).sum().item()
        fraction_known = known_voxels / total_voxels
        
        # Find starting timestep
        t_start = find_start_timestep_for_fraction(self.sqrt_alphas_cumprod, fraction_known)
        
        if progress:
            print(f"  Time-aligned: {fraction_known*100:.1f}% known -> starting at t={t_start} (of {self.num_timesteps})")
        
        # Initialize with noise, then blend in known region at t_start noise level
        noise = torch.randn_like(x_known)
        t_tensor = torch.full((B,), t_start, device=device, dtype=torch.long)
        
        # Create initial noisy image
        x_t = noise.clone()
        
        # Replace known region with appropriately noised version
        x_known_noisy = self.q_sample(x_known, t_tensor, noise)
        x_t = torch.where(inpaint_mask, x_t, x_known_noisy)
        
        # Denoising loop from t_start down to 0
        timesteps = list(range(t_start, -1, -1))
        iterator = tqdm(timesteps, desc="Time-aligned inpaint") if progress else timesteps
        
        for t in iterator:
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            
            # Get model prediction
            x_t, _ = self.diffusion_model.p_sample(
                x_t, t, class_cond, 
                cond_scale=cond_scale, 
                rescaled_phi=rescaled_phi
            )
            
            # Replace known region with appropriately noised original (if not final step)
            if t > 0:
                t_prev = torch.full((B,), t - 1, device=device, dtype=torch.long)
                x_known_noisy = self.q_sample(x_known, t_prev)
                x_t = torch.where(inpaint_mask, x_t, x_known_noisy)
            else:
                # Final step: use clean known region
                x_t = torch.where(inpaint_mask, x_t, x_known)
        
        return x_t


def create_inpaint_mask_fraction(
    shape: Tuple[int, int, int, int, int],
    fraction_to_inpaint: float,
    mode: str = "half_x",
    ring_thickness: int = 1,
) -> torch.Tensor:
    """
    Create an inpainting mask where True = generate, False = keep.
    
    Args:
        shape: (B, C, D, H, W) - note this is channel-first for DDPM
        fraction_to_inpaint: Fraction of volume to inpaint (0 to 1)
        mode: How to partition the space
        ring_thickness: Thickness of ring in blocks (for ring mode)
            
    Returns:
        Boolean tensor [B, 1, D, H, W] where True = inpaint this position
    """
    B, C, D, H, W = shape
    # Create spatial mask [B, 1, D, H, W]
    mask = torch.zeros((B, 1, D, H, W), dtype=torch.bool)
    
    if mode == "single_voxel":
        mask[:] = True
        cd, ch, cw = D // 2, H // 2, W // 2
        mask[:, :, cd, ch, cw] = False
        
    elif mode == "half_x":
        # Inpaint the first portion along depth (front in isometric view)
        split = int(D * fraction_to_inpaint)
        mask[:, :, :split, :, :] = True
        
    elif mode == "half_y":
        split = int(H * fraction_to_inpaint)
        mask[:, :, :, :split, :] = True
        
    elif mode == "half_z":
        split = int(W * fraction_to_inpaint)
        mask[:, :, :, :, :split] = True
        
    elif mode == "ring":
        t = ring_thickness
        mask[:] = True
        if t < D // 2 and t < H // 2 and t < W // 2:
            mask[:, :, :t, :, :] = False
            mask[:, :, -t:, :, :] = False
            mask[:, :, :, :t, :] = False
            mask[:, :, :, -t:, :] = False
            mask[:, :, :, :, :t] = False
            mask[:, :, :, :, -t:] = False
        else:
            mask[:] = False
    else:
        raise ValueError(f"Unknown inpaint mode: {mode}")
    
    return mask


def _load_voxels_from_pt_payload(data: Any, source_name: str) -> torch.Tensor:
    """Normalize a saved tensor/dict payload into a voxel tensor."""
    if isinstance(data, torch.Tensor):
        voxels = data
    elif isinstance(data, dict):
        voxels = data.get("voxels", data.get("samples", None))
        if voxels is None:
            raise ValueError(
                f"Source file missing 'voxels'/'samples': {source_name}. "
                f"Keys: {list(data.keys())}"
            )
    else:
        raise ValueError(f"Unexpected source format in {source_name}: {type(data)}")

    if not torch.is_tensor(voxels):
        raise ValueError(f"Loaded voxels are not a tensor in {source_name}")
    return voxels


def _load_standard_biome_samples(
    folder_path: str,
    converter: BlockBiomeConverter,
    device: torch.device,
    pattern: str = "generated_*.pt",
    samples_per_biome: int = STANDARD_SAMPLES_PER_BIOME,
    random_sample: bool = False,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Load a fixed number of source samples from each generated_<biome>.pt file.

    Returns a flat sample list so standard mode can run per-sample variant batches
    while keeping output directories grouped by biome.
    """
    if samples_per_biome < 1:
        raise ValueError(f"samples_per_biome must be >= 1, got {samples_per_biome}")

    import glob

    folder = Path(folder_path)
    files = sorted(glob.glob(str(folder / pattern)))
    if len(files) == 0:
        raise ValueError(f"No files matching '{pattern}' found in {folder_path}")

    samples: List[Dict[str, Any]] = []
    biome_names: List[str] = []
    num_blocks = len(converter.block_to_index)

    for fpath in files:
        path = Path(fpath)
        fname = path.stem
        if fname.startswith("generated_"):
            biome_name = fname[len("generated_"):]
        else:
            biome_name = fname

        if converter.biome_to_index is not None and biome_name in converter.biome_to_index:
            biome_idx = int(converter.biome_to_index[biome_name])
        else:
            print(f"  Warning: Unknown biome '{biome_name}', skipping {fpath}")
            continue

        data = torch.load(path, map_location="cpu", weights_only=False)
        voxels = _load_voxels_from_pt_payload(data, str(path))

        if voxels.dim() == 3:
            voxels = voxels.unsqueeze(0)
        elif voxels.dim() != 4:
            raise ValueError(
                f"Expected [N, D, H, W] or [D, H, W] voxels in {path}, got {tuple(voxels.shape)}"
            )

        num_samples = int(voxels.shape[0])
        if num_samples < samples_per_biome:
            raise ValueError(
                f"{path.name} only contains {num_samples} samples, but standard mode "
                f"requires {samples_per_biome} samples per biome."
            )

        if voxels.max().item() >= num_blocks:
            voxels = converter.convert_to_indices(voxels.long())
        else:
            voxels = voxels.long()

        if random_sample and num_samples > samples_per_biome:
            selected_indices = sorted(random.sample(range(num_samples), samples_per_biome))
        else:
            selected_indices = list(range(samples_per_biome))

        biome_names.append(biome_name)
        for local_rank, sample_idx in enumerate(selected_indices, start=1):
            sample_name = f"sample_{local_rank:02d}_src_{sample_idx:03d}"
            samples.append(
                {
                    "biome_name": biome_name,
                    "biome_index": biome_idx,
                    "source_file": str(path),
                    "source_sample_index": int(sample_idx),
                    "sample_name": sample_name,
                    "source_indices": voxels[sample_idx].to(device),
                }
            )

    if len(samples) == 0:
        raise ValueError(f"No valid biome samples found in {folder_path}")

    print(
        f"  Loaded {len(samples)} standard-mode source samples "
        f"across {len(biome_names)} biomes: {biome_names}"
    )
    return samples, biome_names


def create_standard_corner_inpaint_mask(
    shape: Tuple[int, int, int, int],
    corner_size: int = STANDARD_CONTEXT_CORNER_SIZE,
    vertical_size: int = STANDARD_CONTEXT_HEIGHT,
) -> torch.Tensor:
    """
    Keep a fixed corner prism as context.

    Tensor layout is [B, D, H, W], where H is the vertical axis. Standard mode
    preserves the low-D / low-W corner across the bottom-to-top vertical extent.
    """
    if len(shape) != 4:
        raise ValueError(f"Expected 4D shape [B, D, H, W], got {shape}")

    B, D, H, W = shape
    if corner_size > D or corner_size > W:
        raise ValueError(
            f"Corner size {corner_size} does not fit inside spatial shape {(D, H, W)}"
        )
    if vertical_size > H:
        raise ValueError(
            f"Vertical context size {vertical_size} exceeds chunk height {H}"
        )

    mask = torch.ones((B, 1, D, H, W), dtype=torch.bool)
    mask[:, :, :corner_size, :vertical_size, :corner_size] = False
    return mask


def _resolve_biome_condition(
    converter: BlockBiomeConverter,
    biome_name: str,
) -> Tuple[str, int]:
    """Resolve a biome name case-insensitively to the canonical mapping entry."""
    if converter.biome_to_index is None:
        raise ValueError("Converter does not have biome mappings")

    for candidate, biome_idx in converter.biome_to_index.items():
        if str(candidate).lower() == str(biome_name).lower():
            return str(candidate), int(biome_idx)

    available = sorted(converter.biome_to_index.keys())
    raise ValueError(
        f"Unknown biome '{biome_name}'. "
        f"Available biomes: {available[:10]}{'...' if len(available) > 10 else ''}"
    )


def _parse_biome_list_arg(raw_value: Optional[str]) -> Optional[List[str]]:
    """Parse a comma-separated CLI biome list."""
    if raw_value is None:
        return None
    values = [item.strip() for item in str(raw_value).split(",") if item.strip()]
    if not values:
        raise ValueError("Biome list arguments must contain at least one non-empty biome name")
    return values


def _resolve_biome_list(
    converter: BlockBiomeConverter,
    raw_value: Optional[str],
) -> Optional[List[str]]:
    """Resolve a comma-separated biome list to canonical converter names."""
    requested = _parse_biome_list_arg(raw_value)
    if requested is None:
        return None

    resolved: List[str] = []
    seen = set()
    for biome_name in requested:
        canonical_name, _ = _resolve_biome_condition(converter, biome_name)
        if canonical_name not in seen:
            resolved.append(canonical_name)
            seen.add(canonical_name)
    return resolved


def _is_unconditional_biome_request(biome_name: Optional[str]) -> bool:
    if biome_name is None:
        return True
    return str(biome_name).strip().lower() in UNCONDITIONAL_BIOME_ALIASES


def _unconditional_class_label(
    model,
    converter: BlockBiomeConverter,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    Return a dummy class label tensor for CFG-null unconditional sampling.

    Like the MD4 path, keep passing a valid class label but force cond_scale=0 so
    forward_with_cond_scale() uses the learned null-conditioning branch.
    """
    if not getattr(model, "class_conditional", False):
        return None
    if not hasattr(model, "forward_with_cond_scale"):
        raise ValueError(
            "This class-conditional model does not expose forward_with_cond_scale(), "
            "so unconditional sampling via the learned null-conditioning branch "
            "is not supported by inpaint_ddpm.py."
        )
    if converter.biome_to_index:
        dummy_index = min(converter.biome_to_index.values())
    else:
        num_classes = getattr(model, "num_classes", None)
        if num_classes in (None, 0):
            raise ValueError(
                "Class-conditional model is missing num_classes / biome mappings; "
                "cannot construct a dummy class label for unconditional sampling."
            )
        dummy_index = 0
    return torch.tensor([int(dummy_index)], dtype=torch.long, device=device)


def _resolve_seed_biome_requests(
    biome_names: Optional[List[str]],
    num_contexts: int,
) -> List[str]:
    """Broadcast seeded-inpaint biome requests, defaulting to unconditional."""
    if biome_names is None or len(biome_names) == 0:
        return [UNCONDITIONAL_BIOME] * num_contexts
    if len(biome_names) == 1 and num_contexts > 1:
        return [biome_names[0]] * num_contexts
    if len(biome_names) != num_contexts:
        raise ValueError(
            f"--seed_biomes must provide exactly one entry per context file "
            f"({num_contexts} files) or a single value to broadcast. "
            f"Got {len(biome_names)} entries."
        )
    return list(biome_names)


def _sanitize_output_name(name: str) -> str:
    """Make a file-system friendly experiment/sample name."""
    return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in str(name))


def _resolve_seed_context_files(
    context_files: Optional[List[str]],
    context_dir: str,
) -> List[Path]:
    """Resolve explicit context files, or default to all .pt files in a directory."""
    if context_files:
        resolved = [Path(path) for path in context_files]
    else:
        root = Path(context_dir)
        if not root.exists():
            raise ValueError(
                f"No seed context files were provided and default directory does not exist: {root}"
            )
        resolved = sorted(root.glob("*.pt"))

    if not resolved:
        raise ValueError("No seed context .pt files found for seeded_inpaint mode")

    missing = [str(path) for path in resolved if not path.exists()]
    if missing:
        raise ValueError(f"Seed context files not found: {missing}")

    return resolved


def _load_seed_context_payload(
    context_path: Path,
    converter: BlockBiomeConverter,
) -> Dict[str, Any]:
    """
    Load a notebook-authored seed context file and normalize it for inpainting.

    Expected payload fields:
    - voxels or source_voxels: [D, H, W] tensor in model index space or original block IDs
    - context_mask or known_mask: [D, H, W] bool tensor marking fixed voxels
    - inpaint_mask: optional [D, H, W] bool tensor marking generated voxels
    """
    data = torch.load(context_path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise ValueError(f"Seed context file must contain a dict payload: {context_path}")

    voxels = data.get("source_voxels", data.get("voxels", None))
    if voxels is None:
        raise ValueError(
            f"Seed context file missing 'source_voxels'/'voxels': {context_path}. "
            f"Keys: {list(data.keys())}"
        )

    if not torch.is_tensor(voxels):
        raise ValueError(f"Seed context voxels must be a tensor: {context_path}")
    if voxels.dim() != 3:
        raise ValueError(
            f"Seed context voxels must be 3D [D, H, W], got shape {tuple(voxels.shape)} "
            f"from {context_path}"
        )

    context_mask = data.get("context_mask", data.get("known_mask", None))
    inpaint_mask = data.get("inpaint_mask", None)

    if context_mask is None and inpaint_mask is None:
        raise ValueError(
            f"Seed context file must contain 'context_mask'/'known_mask' or 'inpaint_mask': {context_path}"
        )

    if context_mask is None:
        if not torch.is_tensor(inpaint_mask):
            raise ValueError(f"inpaint_mask must be a tensor in {context_path}")
        inpaint_mask = inpaint_mask.bool().cpu()
        context_mask = ~inpaint_mask
    else:
        if not torch.is_tensor(context_mask):
            raise ValueError(f"context_mask/known_mask must be a tensor in {context_path}")
        context_mask = context_mask.bool().cpu()
        if inpaint_mask is None:
            inpaint_mask = ~context_mask
        else:
            if not torch.is_tensor(inpaint_mask):
                raise ValueError(f"inpaint_mask must be a tensor in {context_path}")
            inpaint_mask = inpaint_mask.bool().cpu()

    if context_mask.shape != voxels.shape or inpaint_mask.shape != voxels.shape:
        raise ValueError(
            f"Seed context tensors must share shape in {context_path}: "
            f"voxels={tuple(voxels.shape)}, context_mask={tuple(context_mask.shape)}, "
            f"inpaint_mask={tuple(inpaint_mask.shape)}"
        )

    num_blocks = len(converter.block_to_index)
    voxels = voxels.long().cpu()
    if voxels.max().item() >= num_blocks:
        print(f"  Converting original block IDs to model indices for {context_path.name}...")
        voxels = converter.convert_to_indices(voxels.unsqueeze(0))[0].cpu()

    metadata = data.get("metadata", {})
    if metadata is None:
        metadata = {}

    return {
        "context_path": context_path,
        "format": data.get("format", "unknown"),
        "source_indices": voxels,
        "context_mask": context_mask,
        "inpaint_mask": inpaint_mask,
        "metadata": metadata,
        "notes": data.get("notes", {}),
    }


def _build_context_preview_blocks(
    source_blocks: torch.Tensor,
    inpaint_mask: torch.Tensor,
    air_block_id: int = 5,
) -> torch.Tensor:
    """Build a visualization tensor that shows only the kept context."""
    context_preview = source_blocks.clone()
    preview_mask = inpaint_mask
    if preview_mask.dim() == source_blocks.dim() + 1 and preview_mask.shape[1] == 1:
        preview_mask = preview_mask[:, 0]
    context_preview[preview_mask.cpu()] = int(air_block_id)
    return context_preview


def _native_mask_to_ddpm_layout(mask: torch.Tensor) -> torch.Tensor:
    """
    Convert a native chunk-layout mask to the DDPM training layout.

    Native chunk tensors are [B, H, W, D] (or [B, 1, H, W, D]); DDPM sampling
    tensors are [B, E, D, H, W], so masks must become [B, 1, D, H, W].
    """
    if mask.dim() == 4:
        return mask.permute(0, 3, 1, 2).contiguous()
    if mask.dim() == 5 and mask.shape[1] == 1:
        return mask.permute(0, 1, 4, 2, 3).contiguous()
    raise ValueError(f"Expected native mask [B,H,W,D] or [B,1,H,W,D], got {tuple(mask.shape)}")


def _build_seed_variants(
    context_payload: Dict[str, Any],
    num_variants: int,
) -> List[Dict[str, Any]]:
    """Duplicate one loaded seed context into a batched set of stochastic variants."""
    if num_variants < 1:
        raise ValueError(f"num_variants must be >= 1, got {num_variants}")

    context_path = Path(context_payload["context_path"])
    source = context_payload["source_indices"]
    inpaint_mask = context_payload["inpaint_mask"]
    metadata = dict(context_payload.get("metadata", {}))
    base_name = _sanitize_output_name(context_path.stem)

    samples: List[Dict[str, Any]] = []
    for variant_idx in range(num_variants):
        samples.append(
            {
                "name": f"{base_name}_variant_{variant_idx + 1:02d}",
                "description": f"Loaded authored seed context from {context_path.name}",
                "context_label": f"Seed Context ({variant_idx + 1:02d})",
                "metadata": {
                    **metadata,
                    "context_file": str(context_path),
                    "context_format": context_payload.get("format"),
                    "variant_idx": variant_idx + 1,
                },
                "source_indices": source.clone(),
                "inpaint_mask": inpaint_mask.clone(),
            }
        )
    return samples


def load_ddpm_model(
    checkpoint_path: str,
    config_path: Optional[str],
    mappings_path: str,
    embeddings_path: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load a trained DDPM model (uses DiT3D backbone with embeddings).
    
    The checkpoint contains decode_embedding_matrix and decode_block_ids for
    converting embeddings back to block indices. If embeddings_path is provided,
    it will be used for voxel->embedding conversion; otherwise falls back to
    the decode_embedding_matrix from the checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to config.json (or None to auto-find)
        mappings_path: Path to mappings file
        embeddings_path: Optional path to block embeddings file (.npy or .pt)
        device: Target device
        
    Returns:
        Dictionary with model, converter, config, embeddings, etc.
    """
    from denoising_diffusion_pytorch_3d import GaussianDiffusion3D_CFG
    from dit import DiT3D
    import numpy as np
    
    ckpt_path = Path(checkpoint_path)
    assert ckpt_path.exists(), f"Checkpoint not found: {checkpoint_path}"
    
    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    
    # Load config
    if config_path is None:
        # Try to find config near checkpoint
        for candidate in [ckpt_path.parent / "config.json", ckpt_path.parent.parent / "config.json"]:
            if candidate.exists():
                config_path = str(candidate)
                break
    
    if config_path is None or not Path(config_path).exists():
        raise FileNotFoundError(f"Could not find config.json. Specify with --config")
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Load mappings
    mappings = torch.load(mappings_path, map_location="cpu", weights_only=False)
    converter = BlockBiomeConverter(mappings.get("block_mappings"), mappings.get("biome_mappings"))
    
    num_blocks = len(converter.block_to_index)
    num_classes = len(converter.biome_to_index) if converter.biome_to_index else None
    
    # Load checkpoint first to get embedding info
    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    
    # Get decode lookup from checkpoint (for converting embeddings back to blocks)
    decode_embedding_matrix = ckpt.get("decode_embedding_matrix", None)
    decode_block_ids = ckpt.get("decode_block_ids", None)
    
    # Helper function to load embeddings from a path (matches DDPMoxelDatasetMemmapConditional logic)
    def load_embeddings_from_path(emb_path: str, dict_key: Optional[str] = None) -> torch.Tensor:
        if emb_path.endswith('.npy'):
            emb_np = np.load(emb_path)
            return torch.from_numpy(np.asarray(emb_np)).float()
        else:
            emb_obj = torch.load(emb_path, map_location="cpu", weights_only=False)
            
            # Handle nested dict key if specified
            if dict_key is not None and isinstance(emb_obj, dict) and dict_key in emb_obj:
                emb_obj = emb_obj[dict_key]
            
            if isinstance(emb_obj, dict):
                # Check if it's a dict of block_id -> embedding vectors
                # Convert dict to dense table
                keys = [int(k) for k in emb_obj.keys()]
                if len(keys) == 0:
                    raise ValueError(f"Empty embeddings dict in {emb_path}")
                M = max(keys) + 1
                
                # Infer E from first value
                first = next(iter(emb_obj.values()))
                if hasattr(first, 'numel'):
                    E = int(first.numel())
                elif hasattr(first, 'shape'):
                    E = int(np.prod(first.shape))
                else:
                    E = len(first)
                
                # Create dense table
                emb_table = torch.zeros((M, E), dtype=torch.float32)
                for k, v in emb_obj.items():
                    tv = torch.as_tensor(v).float().view(-1)
                    emb_table[int(k)] = tv
                return emb_table
            else:
                # Assume it's a tensor-like object
                if isinstance(emb_obj, torch.Tensor):
                    return emb_obj.float()
                else:
                    return torch.as_tensor(emb_obj).float()
    
    # Load embeddings - try multiple sources
    embeddings = None
    embeddings_dict_key = config.get("embeddings_dict_key", None)
    
    # Priority 1: Explicit embeddings_path argument
    if embeddings_path is not None and Path(embeddings_path).exists():
        embeddings = load_embeddings_from_path(embeddings_path, embeddings_dict_key).to(dev)
        print(f"  Loaded embeddings from --embeddings: {embeddings.shape}")
    
    # Priority 2: Config's embeddings_path
    if embeddings is None:
        config_emb_path = config.get("embeddings_path", None)
        if config_emb_path and Path(config_emb_path).exists():
            embeddings = load_embeddings_from_path(config_emb_path, embeddings_dict_key).to(dev)
            print(f"  Loaded embeddings from config path: {embeddings.shape}")
    
    # Priority 3: Checkpoint's decode matrix (for newer checkpoints)
    if embeddings is None and decode_embedding_matrix is not None:
        decode_embedding_matrix = decode_embedding_matrix.float().to(dev)
        if decode_block_ids is not None:
            decode_block_ids = decode_block_ids.to(dev)
            max_block_id = decode_block_ids.max().item() + 1
            embeddings = torch.zeros((max_block_id, decode_embedding_matrix.shape[1]), device=dev)
            embeddings[decode_block_ids] = decode_embedding_matrix
            print(f"  Reconstructed embeddings from checkpoint decode matrix: {embeddings.shape}")
        else:
            embeddings = decode_embedding_matrix
            print(f"  Using checkpoint decode matrix as embeddings: {embeddings.shape}")
    
    # If still no embeddings, error out
    if embeddings is None:
        raise ValueError(
            "Could not load embeddings. This checkpoint does not contain 'decode_embedding_matrix' "
            "(older training run). Please provide --embeddings pointing to the block embeddings file "
            "(e.g., assets/block_embeddings_norm.npy) or ensure the config's embeddings_path is valid."
        )
    
    data_channels = embeddings.shape[1]

    # Build a compact model-indexed embedding table so inference matches the
    # training path: dataset indices -> block IDs -> embedding vectors.
    index_embedding_table = torch.zeros((num_blocks, data_channels), device=dev, dtype=embeddings.dtype)
    missing_index_embeddings: List[Tuple[int, int]] = []
    for model_idx in range(num_blocks):
        block_id = int(converter.index_to_block[model_idx])
        if 0 <= block_id < embeddings.shape[0]:
            index_embedding_table[model_idx] = embeddings[block_id]
        else:
            missing_index_embeddings.append((model_idx, block_id))
    if missing_index_embeddings:
        examples = missing_index_embeddings[:10]
        raise ValueError(
            "Could not build model-indexed DDPM embedding table because some mapped "
            f"block IDs are outside the loaded embedding table range. Examples: {examples}"
        )
    
    # Set up decode matrix - use checkpoint's if available, otherwise use full embeddings
    if decode_embedding_matrix is not None:
        decode_embedding_matrix = decode_embedding_matrix.float().to(dev)
        decode_block_ids = decode_block_ids.to(dev) if decode_block_ids is not None else None
        print(f"  Decode matrix from checkpoint: {decode_embedding_matrix.shape[0]} candidates")
    else:
        # Fallback for older checkpoints: restrict decode candidates to block IDs that
        # actually exist in the model's block mapping, and map them back to model indices.
        decode_pairs = sorted(
            (int(block_id), int(model_idx))
            for block_id, model_idx in converter.block_to_index.items()
            if 0 <= int(block_id) < embeddings.shape[0]
        )
        if not decode_pairs:
            raise ValueError(
                "Could not build DDPM decode candidates from the embedding table and "
                "converter block mappings."
            )

        decode_source_block_ids = torch.tensor(
            [block_id for block_id, _ in decode_pairs],
            dtype=torch.long,
            device=dev,
        )
        decode_block_ids = torch.tensor(
            [model_idx for _, model_idx in decode_pairs],
            dtype=torch.long,
            device=dev,
        )
        decode_embedding_matrix = embeddings[decode_source_block_ids]
        print(
            "  Using mapped decode candidates from full embeddings: "
            f"{decode_embedding_matrix.shape[0]} / {embeddings.shape[0]}"
        )
    
    # Build model (DiT3D backbone)
    image_size = config.get("image_size", 32)
    
    model = DiT3D(
        in_dim=data_channels,
        out_dim=data_channels,
        hidden_channels=config.get("hidden_channels", 768),
        image_size=image_size,
        patch_size=config.get("patch_size", 2),
        time_dim=config.get("time_dim", 256),
        depth=config.get("depth", 6),
        num_heads=config.get("num_heads", 8),
        mlp_ratio=config.get("mlp_ratio", 4.0),
        attn_drop=config.get("attn_drop", 0.0),
        proj_drop=config.get("proj_drop", 0.0),
        comb_method=config.get("comb_method", "add"),
        class_conditional=True,
        num_classes=num_classes,
        cond_drop_prob=config.get("cond_drop_prob", 0.2),
    ).to(dev)
    
    # Build diffusion wrapper
    diffusion_kwargs = {
        'image_size': image_size,
        'timesteps': config.get("timesteps", 1000),
        'objective': config.get("objective", "pred_v"),
        'beta_schedule': config.get("beta_schedule", "cosine"),
        'sampling_timesteps': config.get("sampling_timesteps", None),
        'auto_normalize': config.get("auto_normalize", False),
    }
    
    diffusion = GaussianDiffusion3D_CFG(model, **diffusion_kwargs).to(dev)
    
    # Handle different checkpoint formats - try to get EMA weights
    state_dict = None
    if isinstance(ckpt, dict):
        if "ema" in ckpt and ckpt["ema"] is not None:
            ema_state = ckpt["ema"]
            if isinstance(ema_state, dict):
                if "ema_model" in ema_state:
                    state_dict = ema_state["ema_model"]
                elif "shadow_params" in ema_state:
                    state_dict = ckpt.get("model", ckpt)
                else:
                    state_dict = ckpt.get("model", ckpt)
            else:
                state_dict = ckpt.get("model", ckpt)
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt
    
    # Load weights
    missing, unexpected = diffusion.load_state_dict(state_dict, strict=False)
    if len(missing) > 0 or len(unexpected) > 0:
        print(f"[warn] Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        if len(missing) > 0:
            print(f"  Missing examples: {missing[:3]}")
        if len(unexpected) > 0:
            print(f"  Unexpected examples: {unexpected[:3]}")
    
    diffusion.eval()
    
    return {
        "diffusion": diffusion,
        "model": model,
        "converter": converter,
        "config": config,
        "device": dev,
        "image_size": image_size,
        "num_blocks": num_blocks,
        "num_classes": num_classes,
        "data_channels": data_channels,
        "embeddings": embeddings,
        "index_embeddings": index_embedding_table,
        "decode_embedding_matrix": decode_embedding_matrix,
        "decode_block_ids": decode_block_ids,
    }


def voxels_to_embeddings(
    voxels: torch.Tensor,
    embedding_table: torch.Tensor,
) -> torch.Tensor:
    """
    Convert discrete voxel indices to continuous embeddings using lookup table.

    This matches the conditional DDPM training path in
    `VoxelTrainerClassConditional._embed_batch()`:
    dataset indices are first embedded in native chunk order [B, H, W, D], then
    permuted to Conv3D layout [B, E, D, H, W].

    Args:
        voxels: [B, H, W, D] tensor of model-space block indices
        embedding_table: [num_blocks, E] embedding lookup table

    Returns:
        [B, E, D, H, W] embedding tensor
    """
    B, H, W, D = voxels.shape
    E = embedding_table.shape[1]

    # Clamp indices to valid range
    voxels_clamped = voxels.clamp(0, embedding_table.shape[0] - 1).long()

    # Lookup embeddings: [B, H, W, D] -> [B, H, W, D, E]
    embeddings = embedding_table[voxels_clamped.view(-1)].view(B, H, W, D, E)

    # Match conditional DDPM training layout: [B, X, Y, Z, E] -> [B, E, Z, X, Y]
    embeddings = embeddings.permute(0, 4, 3, 1, 2).contiguous()

    return embeddings


def embeddings_to_voxels(
    embeddings: torch.Tensor,
    decode_embedding_matrix: torch.Tensor,
    decode_block_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Convert continuous embeddings back to discrete voxel indices via nearest neighbor lookup.

    Inverse of `voxels_to_embeddings()`: DDPM tensors are laid out as
    [B, E, D, H, W], but decoded voxel grids should be returned in the native
    chunk order [B, H, W, D] expected by the converter and renderers.

    Args:
        embeddings: [B, E, D, H, W] tensor
        decode_embedding_matrix: [num_candidates, E] embedding lookup table for NN search
        decode_block_ids: [num_candidates] tensor mapping matrix indices to block IDs
                         If None, indices are returned directly.

    Returns:
        [B, H, W, D] tensor of block indices (model indices, not original block IDs)
    """
    B, E, D, H, W = embeddings.shape
    device = embeddings.device

    # Invert training layout: [B, E, D, H, W] -> [B, H, W, D, E]
    flat_emb = embeddings.permute(0, 3, 4, 2, 1).contiguous().view(-1, E)

    # Compute distances to all embedding vectors in chunks to save memory
    chunk_size = 32768
    nearest_indices = torch.empty((flat_emb.shape[0],), dtype=torch.long, device=device)

    for start in range(0, flat_emb.shape[0], chunk_size):
        end = min(start + chunk_size, flat_emb.shape[0])
        chunk_data = flat_emb[start:end]
        dists = torch.cdist(chunk_data.float(), decode_embedding_matrix.float())
        nearest_indices[start:end] = dists.argmin(dim=1)

    # Map matrix indices to block IDs if decode_block_ids is provided
    if decode_block_ids is not None:
        block_indices = decode_block_ids[nearest_indices]
    else:
        block_indices = nearest_indices

    # Reshape back to native chunk order [B, H, W, D]
    return block_indices.view(B, H, W, D)


def load_biomes_from_folder(
    folder_path: str,
    converter: BlockBiomeConverter,
    device: torch.device,
    pattern: str = "generated_*.pt",
    random_sample: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Load one sample from each biome file in a folder.
    """
    import glob
    
    folder = Path(folder_path)
    files = sorted(glob.glob(str(folder / pattern)))
    
    if len(files) == 0:
        raise ValueError(f"No files matching '{pattern}' found in {folder_path}")
    
    all_voxels = []
    all_labels = []
    biome_names = []
    
    for fpath in files:
        fname = Path(fpath).stem
        if fname.startswith("generated_"):
            biome_name = fname[len("generated_"):]
        else:
            biome_name = fname
        
        if converter.biome_to_index is not None and biome_name in converter.biome_to_index:
            biome_idx = converter.biome_to_index[biome_name]
        else:
            print(f"  Warning: Unknown biome '{biome_name}', skipping {fpath}")
            continue
        
        data = torch.load(fpath, map_location="cpu", weights_only=False)
        
        if isinstance(data, torch.Tensor):
            voxels = data
        elif isinstance(data, dict):
            voxels = data.get("voxels", data.get("samples", None))
            if voxels is None:
                continue
        else:
            continue
        
        if voxels.dim() == 4:
            num_samples = voxels.shape[0]
            if random_sample and num_samples > 1:
                sample_idx = random.randint(0, num_samples - 1)
                voxels = voxels[sample_idx]
            else:
                voxels = voxels[0]
        elif voxels.dim() != 3:
            continue
        
        # Convert to model indices if needed
        num_blocks = len(converter.block_to_index)
        if voxels.max().item() >= num_blocks:
            voxels = converter.convert_to_indices(voxels.unsqueeze(0))[0]
        
        all_voxels.append(voxels)
        all_labels.append(biome_idx)
        biome_names.append(biome_name)
    
    if len(all_voxels) == 0:
        raise ValueError(f"No valid biome samples found in {folder_path}")
    
    voxel_batch = torch.stack(all_voxels, dim=0).to(device)
    label_batch = torch.tensor(all_labels, dtype=torch.long, device=device)
    
    return voxel_batch, label_batch, biome_names


def render_chunk_to_file(
    chunk: torch.Tensor,
    output_path: str,
    visualizer: MinecraftVisualizerPyVista,
    textured: bool = True,
    image_size: int = 512,
    zoom: float = 1.0,
) -> None:
    """Render a single chunk to an image file."""
    if textured:
        plotter = visualizer.visualize_chunk_textured(chunk, interactive=False, show_axis=False)
    else:
        plotter = visualizer.visualize_chunk(chunk, interactive=False, show_axis=False)

    if zoom != 1.0 and hasattr(plotter, "camera") and hasattr(plotter.camera, "zoom"):
        plotter.camera.zoom(zoom)

    plotter.screenshot(
        filename=output_path,
        window_size=(image_size, image_size),
        transparent_background=False,
    )
    plotter.close()


def render_side_by_side(
    chunks: list,
    labels: list,
    output_path: str,
    visualizer: MinecraftVisualizerPyVista,
    textured: bool = True,
    image_size: int = 512,
) -> None:
    """Render multiple chunks side by side into a single image with labels."""
    from PIL import Image, ImageDraw, ImageFont
    import tempfile
    
    temp_files = []
    images = []
    
    for chunk in chunks:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name
            temp_files.append(temp_path)
        
        if textured:
            plotter = visualizer.visualize_chunk_textured(chunk, interactive=False, show_axis=False)
        else:
            plotter = visualizer.visualize_chunk(chunk, interactive=False, show_axis=False)
        
        plotter.screenshot(filename=temp_path, window_size=(image_size, image_size), transparent_background=False)
        plotter.close()
        images.append(Image.open(temp_path))
    
    label_height = 40
    total_width = image_size * len(chunks)
    total_height = image_size + label_height
    
    combined = Image.new('RGB', (total_width, total_height), color=(255, 255, 255))
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
    
    draw = ImageDraw.Draw(combined)
    
    for i, (img, label) in enumerate(zip(images, labels)):
        x_offset = i * image_size
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = x_offset + (image_size - text_width) // 2
        draw.text((text_x, 8), label, fill=(0, 0, 0), font=font)
        combined.paste(img, (x_offset, label_height))
    
    combined.save(output_path)
    
    for temp_path in temp_files:
        try:
            os.remove(temp_path)
        except:
            pass
    
    for img in images:
        img.close()


def run_batched_inpainting_experiments(
    model,
    inpainter: DDPMInpainter,
    samples: List[Dict[str, Any]],
    converter: BlockBiomeConverter,
    embedding_table: torch.Tensor,
    decode_embedding_matrix: torch.Tensor,
    decode_block_ids: Optional[torch.Tensor],
    output_dir: Path,
    num_variants: int = 1,
    cond_scale: float = 6.0,
    textures_dir: str = "block_textures/",
    image_size: int = 512,
    render: bool = True,
) -> Dict[str, Any]:
    """
    Run standard mode: N stochastic inpaint variants for 2 samples per biome.

    Each source sample is expanded into a small batch of identical contexts so the
    model can generate multiple inpainted variants in one call.
    """
    if not samples:
        raise ValueError("Standard mode requires at least one loaded source sample")
    if num_variants < 1:
        raise ValueError(f"num_variants must be >= 1, got {num_variants}")

    results: Dict[str, Dict[str, Any]] = {}
    root_dir = output_dir / "standard"
    root_dir.mkdir(parents=True, exist_ok=True)

    visualizer = None
    textured = False
    if render and textures_dir and os.path.exists(textures_dir):
        visualizer = MinecraftVisualizerPyVista(textures_dir=textures_dir, build_textures=True)
        textured = True
    elif render:
        visualizer = MinecraftVisualizerPyVista()

    for sample in samples:
        biome_name = sample["biome_name"]
        biome_dir = root_dir / _sanitize_output_name(biome_name)
        sample_dir = biome_dir / _sanitize_output_name(sample["sample_name"])
        sample_dir.mkdir(parents=True, exist_ok=True)

        source_single = sample["source_indices"].unsqueeze(0).to(inpainter.device)
        source_batch = source_single.repeat(num_variants, 1, 1, 1)
        source_embeddings = voxels_to_embeddings(source_batch, embedding_table)
        native_inpaint_mask = create_standard_corner_inpaint_mask(tuple(source_batch.shape)).to(inpainter.device)
        ddpm_inpaint_mask = _native_mask_to_ddpm_layout(native_inpaint_mask)

        if getattr(model, "class_conditional", False):
            class_labels = torch.full(
                (num_variants,),
                int(sample["biome_index"]),
                dtype=torch.long,
                device=inpainter.device,
            )
        else:
            class_labels = None

        num_context = int((~native_inpaint_mask[0]).sum().item())
        total = int(source_batch.shape[1] * source_batch.shape[2] * source_batch.shape[3])
        num_inpaint = total - num_context

        print("\n" + "=" * 60)
        print(f"Standard variants: {biome_name} / {sample['sample_name']}")
        print(f"  Source file: {sample['source_file']}")
        print(f"  Source sample index: {sample['source_sample_index']}")
        print(
            f"  Context voxels: {num_context} ({num_context/total*100:.2f}%), "
            f"Inpaint voxels: {num_inpaint} ({num_inpaint/total*100:.2f}%)"
        )
        print(f"  Variants: {num_variants}")
        print("=" * 60)

        inpainted_embeddings = inpainter.inpaint_time_aligned(
            source_embeddings,
            ddpm_inpaint_mask,
            class_labels,
            cond_scale=cond_scale,
        )
        inpainted_result = embeddings_to_voxels(
            inpainted_embeddings,
            decode_embedding_matrix,
            decode_block_ids,
        )

        source_blocks = converter.convert_to_original_blocks(source_batch.cpu())
        inpainted_blocks = converter.convert_to_original_blocks(inpainted_result.cpu())
        context_preview = _build_context_preview_blocks(source_blocks, native_inpaint_mask.cpu())

        sample_summary = {
            "mode": "standard_biome_variants",
            "biome_name": biome_name,
            "biome_index": int(sample["biome_index"]),
            "source_file": sample["source_file"],
            "source_sample_index": int(sample["source_sample_index"]),
            "sample_name": sample["sample_name"],
            "num_variants": int(num_variants),
            "context_shape": [
                STANDARD_CONTEXT_CORNER_SIZE,
                STANDARD_CONTEXT_HEIGHT,
                STANDARD_CONTEXT_CORNER_SIZE,
            ],
            "source_indices": source_single.cpu(),
            "source_blocks": source_blocks[0:1].cpu(),
            "inpaint_mask": native_inpaint_mask[0:1, 0].cpu(),
            "source_render": str(sample_dir / "source_chunk.png"),
            "variant_files": [],
        }

        sample_result_summary = {
            "sample_name": sample["sample_name"],
            "source_file": sample["source_file"],
            "source_sample_index": int(sample["source_sample_index"]),
            "source_render": str(sample_dir / "source_chunk.png"),
            "variant_files": [],
        }

        if render and visualizer is not None:
            try:
                render_chunk_to_file(
                    source_blocks[0],
                    str(sample_dir / "source_chunk.png"),
                    visualizer=visualizer,
                    textured=textured,
                    image_size=image_size,
                )
            except Exception as e:
                print(f"  Warning: Could not render source chunk for {sample['sample_name']}: {e}")

        for variant_idx in range(num_variants):
            variant_name = f"{sample['sample_name']}_variant_{variant_idx + 1:02d}"
            variant_stem = _sanitize_output_name(variant_name)
            variant_file = sample_dir / f"{variant_stem}.pt"
            comparison_file = sample_dir / f"{variant_stem}_comparison.png"

            variant_payload = {
                "mode": "standard_biome_variants",
                "biome_name": biome_name,
                "biome_index": int(sample["biome_index"]),
                "source_file": sample["source_file"],
                "source_sample_index": int(sample["source_sample_index"]),
                "sample_name": sample["sample_name"],
                "variant_idx": variant_idx + 1,
                "source_indices": source_single[0].cpu(),
                "source_blocks": source_blocks[variant_idx].cpu(),
                "inpaint_mask": native_inpaint_mask[variant_idx, 0].cpu(),
                "context_preview_blocks": context_preview[variant_idx].cpu(),
                "inpainted": inpainted_result[variant_idx].cpu(),
                "inpainted_blocks": inpainted_blocks[variant_idx].cpu(),
            }
            torch.save(variant_payload, variant_file)

            sample_summary["variant_files"].append(str(variant_file))
            sample_result_summary["variant_files"].append(str(variant_file))

            if render and visualizer is not None:
                try:
                    render_side_by_side(
                        chunks=[context_preview[variant_idx], inpainted_blocks[variant_idx]],
                        labels=[
                            f"Context {STANDARD_CONTEXT_CORNER_SIZE}x{STANDARD_CONTEXT_HEIGHT}x{STANDARD_CONTEXT_CORNER_SIZE}",
                            f"Variant {variant_idx + 1:02d}",
                        ],
                        output_path=str(comparison_file),
                        visualizer=visualizer,
                        textured=textured,
                        image_size=image_size,
                    )
                except Exception as e:
                    print(f"  Warning: Could not render {variant_name}: {e}")

        torch.save(sample_summary, sample_dir / "results.pt")
        results.setdefault(biome_name, {})[sample["sample_name"]] = sample_result_summary

    return results


def _run_seeded_batch(
    model,
    inpainter: DDPMInpainter,
    batch_name: str,
    samples: List[Dict[str, Any]],
    biome_name: str,
    converter: BlockBiomeConverter,
    embedding_table: torch.Tensor,
    decode_embedding_matrix: torch.Tensor,
    decode_block_ids: Optional[torch.Tensor],
    output_dir: Path,
    cond_scale: float = 6.0,
    textures_dir: str = "block_textures/",
    image_size: int = 512,
    render: bool = True,
) -> Dict[str, Any]:
    """
    Run one batched seeded-inpainting experiment family.

    All samples in a batch must share the same number of context voxels so the
    time-aligned start time is computed consistently for the whole batch.
    """
    if not samples:
        raise ValueError(f"Seeded batch '{batch_name}' has no samples")

    source_indices = torch.stack(
        [sample["source_indices"].clone() for sample in samples],
        dim=0,
    ).to(inpainter.device)
    source_embeddings = voxels_to_embeddings(source_indices, embedding_table)
    native_inpaint_mask = torch.stack(
        [sample["inpaint_mask"].clone() for sample in samples],
        dim=0,
    ).unsqueeze(1).to(inpainter.device)
    ddpm_inpaint_mask = _native_mask_to_ddpm_layout(native_inpaint_mask)

    unconditional = _is_unconditional_biome_request(biome_name)
    effective_cond_scale = cond_scale
    biome_idx: Optional[int] = None
    if unconditional:
        biome_name = UNCONDITIONAL_BIOME
        label = _unconditional_class_label(model, converter, inpainter.device)
        class_labels = label.repeat(len(samples)) if label is not None else None
        if class_labels is not None:
            effective_cond_scale = 0.0
    else:
        biome_name, resolved_biome_idx = _resolve_biome_condition(converter, biome_name)
        biome_idx = int(resolved_biome_idx)
        class_labels = torch.full(
            (len(samples),),
            biome_idx,
            dtype=torch.long,
            device=inpainter.device,
        )

    known_counts = (~native_inpaint_mask).sum(dim=(1, 2, 3, 4))
    if not torch.all(known_counts == known_counts[0]):
        raise ValueError(
            f"Seeded batch '{batch_name}' mixes different context counts, which "
            "would invalidate the shared time-aligned schedule"
        )

    visualizer = None
    textured = False
    if render and textures_dir and os.path.exists(textures_dir):
        visualizer = MinecraftVisualizerPyVista(textures_dir=textures_dir, build_textures=True)
        textured = True
    elif render:
        visualizer = MinecraftVisualizerPyVista()

    total = int(source_indices.shape[1] * source_indices.shape[2] * source_indices.shape[3])
    num_context = int(known_counts[0].item())
    num_inpaint = total - num_context

    print("\n" + "=" * 60)
    print(f"Seeded experiment: {batch_name}")
    if unconditional:
        print("  Conditioning: unconditional (CFG null branch)")
    else:
        print(f"  Biome: {biome_name} (index {biome_idx})")
    print(f"  Batch size: {len(samples)}")
    print(
        f"  Context voxels per sample: {num_context} ({num_context/total*100:.2f}%), "
        f"Inpaint voxels: {num_inpaint} ({num_inpaint/total*100:.2f}%)"
    )
    print("=" * 60)
    for sample in samples:
        print(f"  {sample['name']}: {sample.get('description', sample['name'])}")

    print("\nRunning batched time-aligned inpainting...")
    inpainted_embeddings = inpainter.inpaint_time_aligned(
        source_embeddings,
        ddpm_inpaint_mask,
        class_labels,
        cond_scale=effective_cond_scale,
    )
    inpainted_result = embeddings_to_voxels(
        inpainted_embeddings,
        decode_embedding_matrix,
        decode_block_ids,
    )

    result_dir = output_dir / _sanitize_output_name(batch_name)
    result_dir.mkdir(parents=True, exist_ok=True)

    source_blocks = converter.convert_to_original_blocks(source_indices.cpu())
    inpainted_blocks = converter.convert_to_original_blocks(inpainted_result.cpu())
    context_preview = _build_context_preview_blocks(source_blocks, native_inpaint_mask)

    summary = {
        "mode": "seeded_inpaint_batch",
        "batch_name": batch_name,
        "biome_name": biome_name,
        "biome_index": biome_idx,
        "unconditional": unconditional,
        "sample_metadata": [
            {
                "name": sample["name"],
                "description": sample.get("description"),
                "metadata": sample.get("metadata", {}),
            }
            for sample in samples
        ],
        "source_indices": source_indices.cpu(),
        "inpaint_mask": native_inpaint_mask[:, 0].cpu(),
        "inpainted": inpainted_result.cpu(),
    }
    torch.save(summary, result_dir / "results.pt")

    batch_results = {}
    for batch_idx, sample in enumerate(samples):
        stem = _sanitize_output_name(sample["name"])
        sample_result = {
            "name": sample["name"],
            "description": sample.get("description"),
            "biome_name": biome_name,
            "biome_index": biome_idx,
            "unconditional": unconditional,
            "metadata": sample.get("metadata", {}),
            "source_indices": source_indices[batch_idx].cpu(),
            "inpaint_mask": native_inpaint_mask[batch_idx, 0].cpu(),
            "inpainted": inpainted_result[batch_idx].cpu(),
            "source_blocks": source_blocks[batch_idx].cpu(),
            "context_preview_blocks": context_preview[batch_idx].cpu(),
            "inpainted_blocks": inpainted_blocks[batch_idx].cpu(),
        }
        batch_results[stem] = sample_result
        torch.save(sample_result, result_dir / f"{stem}.pt")

        if render and visualizer is not None:
            try:
                render_side_by_side(
                    chunks=[context_preview[batch_idx], inpainted_blocks[batch_idx]],
                    labels=[
                        sample.get("context_label", "Seed Context"),
                        f"Infilled ({biome_name})",
                    ],
                    output_path=str(result_dir / f"{stem}_comparison.png"),
                    visualizer=visualizer,
                    textured=textured,
                    image_size=image_size,
                )
                print(f"  Saved comparison to {result_dir / f'{stem}_comparison.png'}")
            except Exception as e:
                print(f"  Warning: Could not render {sample['name']}: {e}")
            try:
                render_chunk_to_file(
                    context_preview[batch_idx],
                    str(result_dir / f"{stem}_manual_context.png"),
                    visualizer=visualizer,
                    textured=textured,
                    image_size=image_size,
                )
            except Exception as e:
                print(f"  Warning: Could not save standalone context render for {sample['name']}: {e}")

    return batch_results


def run_seeded_inpaint_experiments(
    model,
    inpainter: DDPMInpainter,
    converter: BlockBiomeConverter,
    embedding_table: torch.Tensor,
    decode_embedding_matrix: torch.Tensor,
    decode_block_ids: Optional[torch.Tensor],
    output_dir: Path,
    context_files: Optional[List[str]] = None,
    context_dir: str = "seed_context_outputs",
    biome_names: Optional[List[str]] = None,
    num_variants: int = 1,
    cond_scale: float = 6.0,
    textures_dir: str = "block_textures/",
    image_size: int = 512,
    render: bool = True,
) -> Dict[str, Any]:
    """Run seeded inpainting from notebook-authored seed context files."""
    root_dir = output_dir / "seeded_inpaint"
    root_dir.mkdir(parents=True, exist_ok=True)

    resolved_files = _resolve_seed_context_files(context_files, context_dir)
    resolved_biomes = _resolve_seed_biome_requests(biome_names, len(resolved_files))

    results = {}
    for context_path, biome_name in zip(resolved_files, resolved_biomes):
        payload = _load_seed_context_payload(context_path, converter)
        samples = _build_seed_variants(payload, num_variants=num_variants)
        batch_name = f"{_sanitize_output_name(context_path.stem)}_{_sanitize_output_name(biome_name)}"
        results[batch_name] = _run_seeded_batch(
            model=model,
            inpainter=inpainter,
            batch_name=batch_name,
            samples=samples,
            biome_name=biome_name,
            converter=converter,
            embedding_table=embedding_table,
            decode_embedding_matrix=decode_embedding_matrix,
            decode_block_ids=decode_block_ids,
            output_dir=root_dir,
            cond_scale=cond_scale,
            textures_dir=textures_dir,
            image_size=image_size,
            render=render,
        )

    return results


def run_biome_swap_experiments(
    inpainter: DDPMInpainter,
    source_embeddings: torch.Tensor,
    source_voxels: torch.Tensor,
    source_labels: torch.Tensor,
    biome_names: List[str],
    converter: BlockBiomeConverter,
    decode_embedding_matrix: torch.Tensor,
    decode_block_ids: Optional[torch.Tensor],
    output_dir: Path,
    cond_scale: float = 6.0,
    textures_dir: str = "block_textures/",
    image_size: int = 512,
) -> Dict[str, Any]:
    """Run biome swap inpainting experiments."""
    B, C, D, H, W = source_embeddings.shape
    results = {}
    
    if textures_dir and os.path.exists(textures_dir):
        visualizer = MinecraftVisualizerPyVista(textures_dir=textures_dir, build_textures=True)
        textured = True
    else:
        visualizer = MinecraftVisualizerPyVista()
        textured = False
    
    # Select target biomes once
    all_biome_indices = list(range(len(biome_names)))
    swapped_labels = []
    swapped_biome_names = []
    for i in range(B):
        original_idx = source_labels[i].item()
        available = [idx for idx in all_biome_indices if idx != original_idx]
        swapped_idx = random.choice(available) if available else original_idx
        swapped_labels.append(swapped_idx)
        swapped_name = converter.index_to_biome.get(swapped_idx, str(swapped_idx))
        swapped_biome_names.append(swapped_name)
    
    swapped_labels_tensor = torch.tensor(swapped_labels, dtype=torch.long, device=source_embeddings.device)
    
    print(f"\nBiome swap pairs:")
    for orig, swap in zip(biome_names, swapped_biome_names):
        print(f"  {orig} -> {swap}")
    
    experiments = [
        ("one_third_context", 2/3),
        ("two_thirds_context", 1/3),
    ]
    
    for exp_name, frac_inpaint in experiments:
        print(f"\n{'='*60}")
        print(f"Experiment: {exp_name} (biome swap)")
        print(f"{'='*60}")
        
        inpaint_mask = create_inpaint_mask_fraction(
            (B, C, D, H, W), frac_inpaint, mode="half_x"
        )
        inpaint_mask = inpaint_mask.to(source_embeddings.device)
        
        print(f"  Running time-aligned inpainting with swapped biomes...")
        inpainted_embeddings = inpainter.inpaint_time_aligned(
            source_embeddings,
            inpaint_mask,
            swapped_labels_tensor,
            cond_scale=cond_scale,
        )
        
        print(f"  Converting embeddings to voxels...")
        inpainted_voxels = embeddings_to_voxels(inpainted_embeddings, decode_embedding_matrix, decode_block_ids)
        
        exp_results = {}
        for i, biome_name in enumerate(biome_names):
            swapped_name = swapped_biome_names[i]
            folder_name = f"{biome_name}_to_{swapped_name}"
            biome_dir = output_dir / folder_name / exp_name
            biome_dir.mkdir(parents=True, exist_ok=True)
            
            source_i = source_voxels[i].cpu()
            inpainted_i = inpainted_voxels[i].cpu()
            mask_i = inpaint_mask[i, 0].cpu()
            
            biome_results = {
                "source": source_i,
                "inpainted": inpainted_i,
                "source_biome": biome_name,
                "target_biome": swapped_name,
            }
            torch.save(biome_results, biome_dir / "results.pt")
            
            source_blocks = converter.convert_to_original_blocks(source_i.unsqueeze(0))[0]
            inpainted_blocks = converter.convert_to_original_blocks(inpainted_i.unsqueeze(0))[0]
            
            AIR_BLOCK_ID = 5
            context_preview = source_blocks.clone()
            context_preview[mask_i] = AIR_BLOCK_ID
            
            try:
                render_side_by_side(
                    chunks=[source_blocks, context_preview, inpainted_blocks],
                    labels=[f"Source ({biome_name})", "Context", f"Inpainted ({swapped_name})"],
                    output_path=str(biome_dir / "comparison.png"),
                    visualizer=visualizer,
                    textured=textured,
                    image_size=image_size,
                )
            except Exception as e:
                print(f"    Warning: Could not render {biome_name}: {e}")
            
            exp_results[folder_name] = biome_results
        
        results[exp_name] = exp_results
        print(f"  Saved results for {len(biome_names)} biome swaps")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="DDPM Inpainting experiments")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--mappings", type=str, required=True, help="Path to mappings file")
    parser.add_argument("--embeddings", type=str, default=None, 
                        help="Optional path to block embeddings file (.npy or .pt). If not provided, uses checkpoint's decode matrix.")
    parser.add_argument("--source_folder", type=str, default=None,
                        help="Path to folder with generated_*.pt files. Required for standard and biome_swap modes.")
    parser.add_argument("--output_dir", type=str, default="./inpaint_ddpm_results", help="Output directory")
    parser.add_argument("--config", type=str, default=None, help="Optional path to config.json")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--cond_scale", type=float, default=6.0, help="CFG scale")
    parser.add_argument("--textures_dir", type=str, default="block_textures/", help="Textures directory")
    parser.add_argument("--image_size", type=int, default=512, help="Rendered image size")
    parser.add_argument("--no_render", action="store_true", help="Skip rendering visualizations")
    parser.add_argument("--experiment_mode", type=str, default="standard",
                        choices=["standard", "biome_swap", "seeded_inpaint"],
                        help="Experiment mode")
    parser.add_argument("--random_sample", action="store_true",
                        help="Randomly select sample from each source file")
    parser.add_argument("--num_variants", type=int, default=1,
                        help="Number of variants to generate per source context in standard or seeded_inpaint mode")
    parser.add_argument("--seed_context_files", type=str, nargs="*", default=None,
                        help="Explicit seed context .pt files for seeded_inpaint mode")
    parser.add_argument("--seed_context_dir", type=str, default="seed_context_outputs",
                        help="Default directory to scan for seed context .pt files in seeded_inpaint mode")
    parser.add_argument("--seed_biomes", type=str, nargs="*", default=None,
                        help="Optional biome conditioning names for seeded_inpaint mode. If omitted, uses unconditional generation via the CFG null branch.")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading DDPM model from {args.checkpoint}...")
    loaded = load_ddpm_model(
        args.checkpoint,
        config_path=args.config,
        mappings_path=args.mappings,
        embeddings_path=args.embeddings,
        device=args.device,
    )
    
    model = loaded["model"]
    diffusion = loaded["diffusion"]
    converter = loaded["converter"]
    device = loaded["device"]
    num_blocks = loaded["num_blocks"]
    embedding_table = loaded["index_embeddings"]  # Model index -> embedding (matches training path)
    decode_embedding_matrix = loaded["decode_embedding_matrix"]  # For embedding -> voxel
    decode_block_ids = loaded["decode_block_ids"]
    
    print(f"Model loaded: {num_blocks} blocks, {loaded['num_classes']} classes, {loaded['data_channels']}D embeddings")
    
    # Create inpainter
    inpainter = DDPMInpainter(diffusion, device)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.experiment_mode == "seeded_inpaint":
        if args.source_folder is not None:
            print("Ignoring --source_folder for seeded_inpaint mode.")

        resolved_seed_files = _resolve_seed_context_files(args.seed_context_files, args.seed_context_dir)
        resolved_seed_biomes = _resolve_seed_biome_requests(args.seed_biomes, len(resolved_seed_files))

        exp_config = {
            "checkpoint": args.checkpoint,
            "mappings": args.mappings,
            "embeddings": args.embeddings,
            "cond_scale": args.cond_scale,
            "num_blocks": num_blocks,
            "image_size": loaded["image_size"],
            "experiment_mode": args.experiment_mode,
            "mode": "file_seeded_inpaint",
            "seed_context_dir": args.seed_context_dir,
            "seed_context_files": [str(path) for path in resolved_seed_files],
            "seed_biomes": resolved_seed_biomes,
            "num_variants": args.num_variants,
        }
        with open(output_dir / "experiment_config.json", "w") as f:
            json.dump(exp_config, f, indent=2)

        print("\n" + "=" * 60)
        print("STARTING SEEDED INPAINT EXPERIMENTS")
        print("=" * 60)
        print(f"Seed contexts: {len(resolved_seed_files)}")
        for path, biome_name in zip(resolved_seed_files, resolved_seed_biomes):
            if _is_unconditional_biome_request(biome_name):
                print(f"  {path} -> unconditional")
            else:
                print(f"  {path} -> {biome_name}")
        print(f"Variants per context: {args.num_variants}")
        results = run_seeded_inpaint_experiments(
            model=model,
            inpainter=inpainter,
            converter=converter,
            embedding_table=embedding_table,
            decode_embedding_matrix=decode_embedding_matrix,
            decode_block_ids=decode_block_ids,
            output_dir=output_dir,
            context_files=[str(path) for path in resolved_seed_files],
            biome_names=resolved_seed_biomes,
            num_variants=args.num_variants,
            cond_scale=args.cond_scale,
            textures_dir=args.textures_dir,
            image_size=args.image_size,
            render=not args.no_render,
        )
        print("\n" + "=" * 60)
        print("SEEDED INPAINT EXPERIMENTS COMPLETE")
        print(f"Results saved to: {output_dir / 'seeded_inpaint'}")
        print("=" * 60)
        print("\nExperiment Summary:")
        for exp_name in results.keys():
            print(f"  {exp_name}")
        return

    if args.source_folder is None:
        raise ValueError(
            f"{args.experiment_mode} mode requires --source_folder with generated_<biome>.pt files."
        )

    if args.experiment_mode == "standard":
        print(f"\nLoading biomes from folder: {args.source_folder}")
        if args.random_sample:
            print("  Using random sample selection from each file")
        else:
            print("  Using first samples from each file")

        sampled_contexts, biome_names = _load_standard_biome_samples(
            args.source_folder,
            converter,
            device,
            samples_per_biome=STANDARD_SAMPLES_PER_BIOME,
            random_sample=args.random_sample,
        )
        print(
            f"Loaded {len(sampled_contexts)} source samples "
            f"across {len(biome_names)} biomes"
        )

        exp_config = {
            "checkpoint": args.checkpoint,
            "mappings": args.mappings,
            "embeddings": args.embeddings,
            "source_folder": args.source_folder,
            "biomes": biome_names,
            "cond_scale": args.cond_scale,
            "num_blocks": num_blocks,
            "image_size": loaded["image_size"],
            "experiment_mode": args.experiment_mode,
            "random_sample": args.random_sample,
            "mode": "standard_biome_variants",
            "samples_per_biome": STANDARD_SAMPLES_PER_BIOME,
            "num_variants": args.num_variants,
            "context_shape": [
                STANDARD_CONTEXT_CORNER_SIZE,
                STANDARD_CONTEXT_HEIGHT,
                STANDARD_CONTEXT_CORNER_SIZE,
            ],
        }
        with open(output_dir / "experiment_config.json", "w") as f:
            json.dump(exp_config, f, indent=2)

        print("\n" + "=" * 60)
        print(
            f"STARTING STANDARD INPAINT VARIANTS "
            f"({len(biome_names)} biomes, {len(sampled_contexts)} source maps)"
        )
        print("=" * 60)
        results = run_batched_inpainting_experiments(
            model=model,
            inpainter=inpainter,
            samples=sampled_contexts,
            converter=converter,
            embedding_table=embedding_table,
            decode_embedding_matrix=decode_embedding_matrix,
            decode_block_ids=decode_block_ids,
            output_dir=output_dir,
            num_variants=args.num_variants,
            cond_scale=args.cond_scale,
            textures_dir=args.textures_dir,
            image_size=args.image_size,
            render=not args.no_render,
        )

        print("\n" + "=" * 60)
        print("STANDARD EXPERIMENTS COMPLETE")
        print(f"Results saved to: {output_dir / 'standard'}")
        print("=" * 60)
        print("\nExperiment Summary:")
        for biome_name, sample_results in results.items():
            total_variants = sum(len(sample["variant_files"]) for sample in sample_results.values())
            print(
                f"  {biome_name}: {len(sample_results)} source samples, "
                f"{total_variants} generated variants"
            )
        return

    print(f"\nLoading biomes from folder: {args.source_folder}")
    source_voxels, source_labels, biome_names = load_biomes_from_folder(
        args.source_folder, converter, device, random_sample=args.random_sample
    )
    print(f"Loaded {len(biome_names)} biomes, shape: {source_voxels.shape}")

    source_embeddings = voxels_to_embeddings(source_voxels, embedding_table)
    print(f"Converted to embeddings: {source_embeddings.shape}")

    exp_config = {
        "checkpoint": args.checkpoint,
        "embeddings": args.embeddings,
        "source_folder": args.source_folder,
        "biomes": biome_names,
        "cond_scale": args.cond_scale,
        "experiment_mode": args.experiment_mode,
        "random_sample": args.random_sample,
    }
    with open(output_dir / "experiment_config.json", "w") as f:
        json.dump(exp_config, f, indent=2)

    print("\n" + "=" * 60)
    print(f"STARTING BIOME SWAP EXPERIMENTS ({len(biome_names)} biomes)")
    print("=" * 60)
    results = run_biome_swap_experiments(
        inpainter=inpainter,
        source_embeddings=source_embeddings,
        source_voxels=source_voxels,
        source_labels=source_labels,
        biome_names=biome_names,
        converter=converter,
        decode_embedding_matrix=decode_embedding_matrix,
        decode_block_ids=decode_block_ids,
        output_dir=output_dir,
        cond_scale=args.cond_scale,
        textures_dir=args.textures_dir,
        image_size=args.image_size,
    )

    print("\n" + "=" * 60)
    print("BIOME SWAP EXPERIMENTS COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
