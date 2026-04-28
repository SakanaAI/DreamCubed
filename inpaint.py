"""
Inpainting inference script for MD4 discrete diffusion models.

Implements time-aligned inpainting which starts from t* where the schedule 
expects the given fraction revealed.

Experiment modes:
- standard: Generate multiple inpainted variants from 2 samples per biome using a fixed corner context
- biome_swap: Generate configurable-context infills for selected source biomes and target biomes
- seeded_inpaint: Inpainting from notebook-authored seed context .pt files

Usage:
    python inpaint.py --checkpoint path/to/model.pt --mappings path/to/mappings.pt --source_folder path/to/folder
    python inpaint.py --checkpoint path/to/model.pt --mappings path/to/mappings.pt --source_folder path/to/folder --num_variants 4
    python inpaint.py --checkpoint path/to/model.pt --mappings path/to/mappings.pt --source_folder path/to/folder --experiment_mode biome_swap --source_biomes ocean,plains,village --target_biomes ice,desert,plains --num_variants 4 --biome_swap_context_fraction 0.5
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

from inference import (
    load_model_from_file,
    _default_device,
)
from discrete_diffusion_md4 import MD4Discrete3D
from data_utils import BlockBiomeConverter
from gif_utils import render_diffusion_gif
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
BIOME_SWAP_CONTEXT_FRACTION = 0.5


def find_start_time_for_fraction(fraction_known: float) -> float:
    """
    Given fraction of volume known, find the corresponding time in the cosine schedule.
    
    For cosine schedule: α(t) = 1 - cos((1-t) * π/2)
    We solve: α(t*) = fraction_known
    
    Returns t* in [0, 1] where t*=1 means fully masked, t*=0 means fully revealed.
    """
    fraction_known = max(0.001, min(0.999, fraction_known))
    # α(t) = 1 - cos((1-t) * π/2) = fraction_known
    # cos((1-t) * π/2) = 1 - fraction_known
    # (1-t) * π/2 = arccos(1 - fraction_known)
    # t = 1 - (2/π) * arccos(1 - fraction_known)
    t_star = 1 - (2 / math.pi) * math.acos(1 - fraction_known)
    return max(0.0, min(1.0, t_star))


def _select_logit_save_steps(
    total_steps: int,
    num_logit_intermediates: int,
    *,
    bias_power: float = 1.0,
) -> List[int]:
    """
    Choose denoising loop indices to snapshot.

    ``bias_power=1`` is uniform. Values ``>1`` bias snapshots toward earlier
    denoising steps (the noisy start of the reverse process), while values in
    ``(0, 1)`` bias toward later steps.
    """
    if total_steps <= 0:
        return []
    count = min(total_steps, max(1, int(num_logit_intermediates)))
    if bias_power <= 0:
        raise ValueError(f"bias_power must be > 0, got {bias_power}")
    if count == total_steps:
        return list(range(total_steps))

    positions = torch.linspace(0.0, 1.0, steps=count).pow(float(bias_power))
    raw_indices = (positions * float(total_steps - 1)).round().long().tolist()

    indices: List[int] = []
    next_min = 0
    for j, raw_idx in enumerate(raw_indices):
        max_allowed = total_steps - (count - j)
        idx = max(int(raw_idx), next_min)
        idx = min(idx, max_allowed)
        indices.append(idx)
        next_min = idx + 1
    return indices


class MD4Inpainter:
    """
    Inpainting helper for MD4 discrete diffusion models.
    Supports both naive and time-aligned inpainting.
    """
    
    def __init__(self, accelerator: Accelerator, num_classes: int, device: torch.device, mask_token_id: Optional[int] = None):
        self.accelerator = accelerator
        self.num_classes = int(num_classes)
        self.device = device
        self.mask_token_id = int(num_classes) if mask_token_id is None else int(mask_token_id)
        
        # Cosine masking schedule (same as MD4Discrete3D)
        self.masking_schedule = lambda t: 1 - torch.cos((1 - t) * math.pi / 2)
    
    def _match_last_dims(self, data: torch.Tensor, size: Tuple) -> torch.Tensor:
        assert len(data.size()) == 1
        out = data
        for _ in range(len(size) - 1):
            out = out.unsqueeze(-1)
        return out.repeat(1, *(size[1:]))
    
    def _compute_mask_prob(self, ti: torch.Tensor, si: torch.Tensor, spatial_shape: Tuple) -> torch.Tensor:
        """Compute unmask probability between times si and ti."""
        alphat = self.masking_schedule(ti)
        alphas = self.masking_schedule(si)
        denom = torch.clamp(1 - alphat, min=1e-6)
        first_factor = (alphas - alphat) / denom
        first_factor = torch.clamp(first_factor, 0.0, 1.0)
        return self._match_last_dims(first_factor, spatial_shape)
    
    def _sanitize_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
    
    def _probs_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        logits = self._sanitize_logits(logits)
        probs = torch.softmax(logits, dim=1)
        probs = torch.nan_to_num(probs, nan=0.0)
        sum_p = probs.sum(dim=1, keepdim=True)
        probs = probs / torch.clamp(sum_p, min=1e-8)
        probs = torch.clamp(probs, min=0.0)
        sum_p2 = probs.sum(dim=1, keepdim=True)
        probs = probs / torch.clamp(sum_p2, min=1e-8)
        return probs
    
    def _safe_multinomial(self, probs: torch.Tensor, B: int, H: int, W: int, D: int) -> torch.Tensor:
        K = self.num_classes
        probs_f = probs.view(B, K, -1).transpose(1, 2).contiguous().view(-1, K)
        probs_f = torch.nan_to_num(probs_f, nan=0.0)
        row_sums = probs_f.sum(dim=1, keepdim=True)
        uniform = torch.full_like(probs_f, 1.0 / float(K))
        bad = (row_sums <= 0) | (~torch.isfinite(row_sums))
        safe_probs = torch.where(bad, uniform, probs_f / torch.clamp(row_sums, min=1e-8))
        sampled = torch.multinomial(safe_probs, num_samples=1).view(B, -1)
        return sampled.view(B, H, W, D)

    def _force_unmask_remaining(
        self,
        model,
        xt: torch.Tensor,
        fill_mask: torch.Tensor,
        class_cond: Optional[torch.Tensor] = None,
        cond_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, int]:
        """Force-reveal any residual mask tokens in the fillable region."""
        remaining_mask = (xt == self.mask_token_id) & fill_mask
        if not remaining_mask.any():
            return xt, 0

        B, H, W, D = xt.shape
        ti = torch.zeros((B,), device=xt.device)

        x_in = F.one_hot(
            torch.clamp(xt, 0, self.mask_token_id),
            num_classes=self.mask_token_id + 1,
        )
        x_in = x_in.permute(0, 4, 1, 2, 3).float()
        x_in = x_in.permute(0, 1, 4, 2, 3)

        with self.accelerator.autocast():
            if class_cond is not None:
                if hasattr(model, 'forward_with_cond_scale'):
                    logits, _ = model.forward_with_cond_scale(
                        x_in, ti, class_cond, cond_scale=cond_scale
                    )
                else:
                    logits = model(x_in, ti, class_cond)
            else:
                logits = model(x_in, ti)

        logits = logits.permute(0, 1, 3, 4, 2).contiguous()
        probs = self._probs_from_logits(logits)
        sampled = self._safe_multinomial(probs, B, H, W, D)

        xt = xt.clone()
        xt[remaining_mask] = sampled[remaining_mask]
        return xt, int(remaining_mask.sum().item())
    
    @torch.inference_mode()
    def inpaint_time_aligned(
        self,
        model,
        known_indices: torch.Tensor,
        inpaint_mask: torch.Tensor,
        reverse_steps: int = 1000,
        progress: bool = True,
        air_index_fallback: int = 0,
        class_cond: Optional[torch.Tensor] = None,
        cond_scale: float = 1.0,
        force_unmask_final: bool = True,
        return_logit_intermediates: bool = False,
        num_logit_intermediates: int = 40,
        logit_save_bias_power: float = 1.0,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Time-aligned inpainting: start from t* where the schedule expects the given fraction revealed.
        
        Args:
            known_indices: [B, H, W, D] tensor with known block indices
            inpaint_mask: [B, H, W, D] boolean tensor, True = generate here, False = keep fixed
            
        Returns:
            [B, H, W, D] inpainted result, or ``(result, trace)`` when
            ``return_logit_intermediates`` is enabled. The trace contains a
            sparse sequence of full-volume samples drawn directly from the
            model logits at each saved step, without forcing context voxels to
            remain fixed in the visualization.
        """
        B, H, W, D = known_indices.shape
        device = self.device
        total_voxels = H * W * D
        
        # Calculate fraction known (averaged across batch for simplicity)
        num_known = (~inpaint_mask).float().sum(dim=(1, 2, 3))  # [B]
        fraction_known = (num_known / total_voxels).mean().item()
        
        # Find starting time that matches this fraction
        t_start = find_start_time_for_fraction(fraction_known)
        
        if progress:
            print(f"  Time-aligned: {fraction_known*100:.1f}% known -> starting at t={t_start:.4f}")
        
        # Initialize: known regions filled, inpaint regions masked
        xt = known_indices.clone().to(device)
        xt[inpaint_mask] = self.mask_token_id
        inpaint_mask = inpaint_mask.to(device)
        
        # Compute effective steps (proportional to t_start)
        effective_steps = max(1, int(reverse_steps * t_start))
        total_steps = max(0, effective_steps - 1)
        save_step_indices = (
            _select_logit_save_steps(
                total_steps,
                num_logit_intermediates,
                bias_power=logit_save_bias_power,
            )
            if return_logit_intermediates
            else []
        )
        save_step_set = set(save_step_indices)
        snapshots: List[torch.Tensor] = [] if return_logit_intermediates else []
        
        # Time grid from 0 to t_start
        tgrid = torch.linspace(0, t_start, effective_steps + 1, device=device)
        step_sequence = list(range(effective_steps, 1, -1))
        rng = tqdm(step_sequence, desc="Time-aligned inpaint") if progress else step_sequence
        
        for step_num, i in enumerate(rng):
            ti = torch.full((B,), tgrid[i], device=device)
            si = torch.full((B,), tgrid[i - 1], device=device)
            
            mask_prob = self._compute_mask_prob(ti, si, (B, H, W, D))
            
            # Build one-hot input with mask channel
            x_in = F.one_hot(torch.clamp(xt, 0, self.mask_token_id), num_classes=self.mask_token_id + 1)
            x_in = x_in.permute(0, 4, 1, 2, 3).float()  # [B,K+1,H,W,D]
            x_in = x_in.permute(0, 1, 4, 2, 3)  # [B,K+1,D,H,W]
            
            # Model forward
            with self.accelerator.autocast():
                if class_cond is not None:
                    if hasattr(model, 'forward_with_cond_scale'):
                        logits, _ = model.forward_with_cond_scale(x_in, ti, class_cond, cond_scale=cond_scale)
                    else:
                        logits = model(x_in, ti, class_cond)
                else:
                    logits = model(x_in, ti)
            
            logits = logits.permute(0, 1, 3, 4, 2).contiguous()  # [B,K,H,W,D]
            probs = self._probs_from_logits(logits)
            sampled = self._safe_multinomial(probs, B, H, W, D)

            if return_logit_intermediates and step_num in save_step_set:
                # For visualization, keep the raw full-volume sample from the
                # current logits so fixed context is not artificially enforced.
                snapshots.append(sampled.detach().cpu())
            
            # Only update positions in inpaint region that are still masked
            still_masked = (xt == self.mask_token_id)
            force_reveal_now = force_unmask_final and (i == 2)
            if force_reveal_now:
                update_mask = inpaint_mask & still_masked
            else:
                to_unmask = torch.rand((B, H, W, D), device=device) < mask_prob
                update_mask = inpaint_mask & still_masked & to_unmask
            xt[update_mask] = sampled[update_mask]

            del x_in, logits, probs, sampled, mask_prob, ti, si, still_masked, update_mask
            if not force_reveal_now:
                del to_unmask
        
        remaining_mask = (xt == self.mask_token_id) & inpaint_mask
        if force_unmask_final and remaining_mask.any():
            xt, num_forced = self._force_unmask_remaining(
                model,
                xt,
                inpaint_mask,
                class_cond=class_cond,
                cond_scale=cond_scale,
            )
            if progress and num_forced > 0:
                print(f"  Forced final unmask for {num_forced} residual masked voxels.")

        remaining_mask = (xt == self.mask_token_id) & inpaint_mask
        if remaining_mask.any():
            xt[remaining_mask] = int(air_index_fallback)
        if return_logit_intermediates:
            if not snapshots:
                ti = torch.zeros((B,), device=device)
                x_in = F.one_hot(
                    torch.clamp(xt, 0, self.mask_token_id),
                    num_classes=self.mask_token_id + 1,
                )
                x_in = x_in.permute(0, 4, 1, 2, 3).float()
                x_in = x_in.permute(0, 1, 4, 2, 3)

                with self.accelerator.autocast():
                    if class_cond is not None:
                        if hasattr(model, 'forward_with_cond_scale'):
                            logits, _ = model.forward_with_cond_scale(
                                x_in, ti, class_cond, cond_scale=cond_scale
                            )
                        else:
                            logits = model(x_in, ti, class_cond)
                    else:
                        logits = model(x_in, ti)

                logits = logits.permute(0, 1, 3, 4, 2).contiguous()
                probs = self._probs_from_logits(logits)
                sampled = self._safe_multinomial(probs, B, H, W, D)
                snapshots.append(sampled.detach().cpu())
            return xt, {
                "intermediates": torch.stack(snapshots, dim=1),
                "fraction_known": float(fraction_known),
                "t_start": float(t_start),
                "effective_steps": int(effective_steps),
                "save_step_indices": save_step_indices,
                "logit_save_bias_power": float(logit_save_bias_power),
                "trace_mode": "full_volume_logit_samples",
            }
        return xt


def create_inpaint_mask_fraction(
    shape: Tuple[int, int, int, int],
    fraction_to_inpaint: float,
    mode: str = "half_x",
    ring_thickness: int = 1,
    corner_size: int = 8,
) -> torch.Tensor:
    """
    Create an inpainting mask where True = generate, False = keep.
    
    Axis convention for shape (B, H, W, D):
        H (dim 1) = horizontal ground axis
        W (dim 2) = Y = vertical height (rendered as up)
        D (dim 3) = horizontal ground axis
    The ground plane is H x D (dims 1 and 3). W (dim 2) is vertical.
    
    Args:
        shape: (B, H, W, D) — (batch, ground_a, height, ground_b)
        fraction_to_inpaint: Fraction of volume to inpaint (0 to 1).
        mode: Masking strategy (see code for each mode's description)
        ring_thickness: Thickness of ring walls in blocks
        corner_size: Size of corner context blocks on the ground plane
            
    Returns:
        Boolean tensor [B, H, W, D] where True = inpaint this position
    """
    B, H, W, D = shape  # H=ground, W=height(Y), D=ground
    mask = torch.zeros(shape, dtype=torch.bool)
    
    if mode == "single_voxel":
        mask[:] = True
        mask[:, H // 2, W // 2, D // 2] = False
        
    elif mode == "half_x":
        # Split along H (ground axis, dim 1). Inpaint the front portion.
        split = int(H * fraction_to_inpaint)
        mask[:, :split, :, :] = True
        
    elif mode == "half_y":
        # Split along D (ground axis, dim 3). Inpaint the front portion.
        split = int(D * fraction_to_inpaint)
        mask[:, :, :, :split] = True
        
    elif mode == "half_z":
        # Split along W (vertical, dim 2). Inpaint the bottom portion.
        split = int(W * fraction_to_inpaint)
        mask[:, :, :split, :] = True
        
    elif mode == "corner":
        frac_per_dim = fraction_to_inpaint ** (1/3)
        mask[:, :int(H * frac_per_dim), :int(W * frac_per_dim), :int(D * frac_per_dim)] = True
        
    elif mode == "partial_ring":
        # Horseshoe-shaped context on the ground plane (H x D):
        # - Keep side walls only (no top/bottom Y caps — Y is dim 2 = W)
        # - Remove one wall closest to camera
        # - Keep 2/3 of the remaining horseshoe
        t = ring_thickness
        
        mask[:] = True
        
        if t < H // 2 and t < D // 2:
            # Keep back wall along H (full D extent, full height W)
            mask[:, -t:, :, :] = False
            
            # Partial side walls along D: keep back 2/3 of each
            side_length = H - t
            keep_length = (2 * side_length) // 3
            start_h = H - t - keep_length
            
            # Side wall D=0..t, from start_h to H-t along H, full height W
            mask[:, start_h:H-t, :, :t] = False
            
            # Side wall D=D-t..D, from start_h to H-t along H, full height W
            mask[:, start_h:H-t, :, -t:] = False
        else:
            mask[:] = False
            
    elif mode == "opposite_corners":
        # Keep two vertical columns at diagonally opposite corners of the ground plane.
        # Each column is corner_size x corner_size on H x D, full height W.
        c = corner_size
        
        mask[:] = True
        
        # Corner 1: H=[0,c), D=[0,c), full W height
        mask[:, :c, :, :c] = False
        
        # Corner 2: H=[H-c,H), D=[D-c,D), full W height
        mask[:, -c:, :, -c:] = False
        
    elif mode == "sandwich":
        # Keep strips on opposite edges of ground axis H, inpaint the middle.
        strip_width = corner_size
        
        mask[:] = True
        
        # Front strip (H = 0 to strip_width, full D, full W height)
        mask[:, :strip_width, :, :] = False
        
        # Back strip (H = H-strip_width to H, full D, full W height)
        mask[:, -strip_width:, :, :] = False
        
    elif mode == "ring_no_caps":
        # Full ring on the ground plane (H x D walls), NO top or bottom (W/Y) caps.
        # Keeps all 4 vertical walls, inpaints the interior column.
        t = ring_thickness
        
        mask[:] = True
        
        if t < H // 2 and t < D // 2:
            mask[:, :t, :, :] = False    # H=0 wall
            mask[:, -t:, :, :] = False   # H=H wall
            mask[:, :, :, :t] = False    # D=0 wall
            mask[:, :, :, -t:] = False   # D=D wall
        else:
            mask[:] = False
            
    elif mode == "ring_4_no_caps":
        # 4-block thick ring on the ground plane (H x D), NO W/Y caps
        t = 4
        
        mask[:] = True
        
        if t < H // 2 and t < D // 2:
            mask[:, :t, :, :] = False
            mask[:, -t:, :, :] = False
            mask[:, :, :, :t] = False
            mask[:, :, :, -t:] = False
        else:
            mask[:] = False
            
    elif mode == "strip_edge":
        # Keep a strip on one H edge of the ground plane (full D, full W height)
        strip_width = int(H * (1 - fraction_to_inpaint))
        
        mask[:] = True
        mask[:, -strip_width:, :, :] = False
        
    else:
        raise ValueError(f"Unknown inpaint mode: {mode}")
    
    return mask


def load_biomes_from_folder(
    folder_path: str,
    converter: BlockBiomeConverter,
    device: torch.device,
    pattern: str = "generated_*.pt",
    random_sample: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, list]:
    """
    Load one sample from each biome file in a folder.
    
    Args:
        folder_path: Path to folder containing generated_*.pt files
        converter: BlockBiomeConverter for index conversion
        device: Target device
        pattern: Glob pattern to match files (default: generated_*.pt)
        random_sample: If True, randomly select a sample from each file.
                       If False, always take the first sample.
        
    Returns:
        (voxel_indices, class_labels, biome_names)
        - voxel_indices: [N, H, W, D] tensor of block indices
        - class_labels: [N] tensor of class labels
        - biome_names: List of biome name strings
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
        # Extract biome name from filename like "generated_forest" -> "forest"
        if fname.startswith("generated_"):
            biome_name = fname[len("generated_"):]
        else:
            biome_name = fname
        
        # Look up biome index
        if converter.biome_to_index is not None and biome_name in converter.biome_to_index:
            biome_idx = converter.biome_to_index[biome_name]
        else:
            print(f"  Warning: Unknown biome '{biome_name}', skipping {fpath}")
            continue
        
        # Load the file
        data = torch.load(fpath, map_location="cpu", weights_only=False)
        
        if isinstance(data, torch.Tensor):
            voxels = data
        elif isinstance(data, dict):
            voxels = data.get("voxels", data.get("samples", None))
            if voxels is None:
                print(f"  Warning: No voxels in {fpath}, skipping")
                continue
        else:
            print(f"  Warning: Unexpected format in {fpath}, skipping")
            continue
        
        # Select sample from file
        if voxels.dim() == 4:
            num_samples = voxels.shape[0]
            if random_sample and num_samples > 1:
                sample_idx = random.randint(0, num_samples - 1)
                voxels = voxels[sample_idx]  # [H, W, D]
            else:
                voxels = voxels[0]  # [H, W, D]
        elif voxels.dim() == 3:
            pass  # Already [H, W, D]
        else:
            print(f"  Warning: Unexpected shape {voxels.shape} in {fpath}, skipping")
            continue
        
        # Convert original block IDs to model indices if needed
        num_blocks = len(converter.block_to_index)
        if voxels.max().item() >= num_blocks:
            voxels = converter.convert_to_indices(voxels.unsqueeze(0))[0]
        
        all_voxels.append(voxels)
        all_labels.append(biome_idx)
        biome_names.append(biome_name)
    
    if len(all_voxels) == 0:
        raise ValueError(f"No valid biome samples found in {folder_path}")
    
    # Stack into batches
    voxel_batch = torch.stack(all_voxels, dim=0).to(device)  # [N, H, W, D]
    label_batch = torch.tensor(all_labels, dtype=torch.long, device=device)  # [N]
    
    print(f"  Loaded {len(biome_names)} biomes: {biome_names}")
    
    return voxel_batch, label_batch, biome_names


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
    biome_filter: Optional[List[str]] = None,
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
    biome_filter_set = set(biome_filter) if biome_filter is not None else None
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

        if biome_filter_set is not None and biome_name not in biome_filter_set:
            continue

        data = torch.load(path, map_location="cpu", weights_only=False)
        voxels = _load_voxels_from_pt_payload(data, str(path))

        if voxels.dim() == 3:
            voxels = voxels.unsqueeze(0)
        elif voxels.dim() != 4:
            raise ValueError(
                f"Expected [N, H, W, D] or [H, W, D] voxels in {path}, got {tuple(voxels.shape)}"
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

    Tensor layout is [B, X, Y, Z], where Y is the vertical axis. Standard mode
    preserves the low-X / low-Z corner across the bottom-to-top vertical extent.
    """
    if len(shape) != 4:
        raise ValueError(f"Expected 4D shape [B, X, Y, Z], got {shape}")

    B, X, Y, Z = shape
    if corner_size > X or corner_size > Z:
        raise ValueError(
            f"Corner size {corner_size} does not fit inside spatial shape {(X, Y, Z)}"
        )
    if vertical_size > Y:
        raise ValueError(
            f"Vertical context size {vertical_size} exceeds chunk height {Y}"
        )

    mask = torch.ones((B, X, Y, Z), dtype=torch.bool)
    mask[:, :corner_size, :vertical_size, :corner_size] = False
    return mask


def create_biome_swap_inpaint_mask(
    shape: Tuple[int, int, int, int],
    context_fraction: float = BIOME_SWAP_CONTEXT_FRACTION,
) -> torch.Tensor:
    """
    Keep a leading X slice as context for biome-swap experiments.

    This preserves the opposite side from the old half_x setup so renders show
    context on the close-left side from the current camera view.
    """
    if len(shape) != 4:
        raise ValueError(f"Expected 4D shape [B, H, W, D], got {shape}")
    if not (0.0 < context_fraction < 1.0):
        raise ValueError(f"context_fraction must be between 0 and 1, got {context_fraction}")

    B, H, W, D = shape
    keep_width = max(1, int(H * context_fraction))
    if keep_width >= H:
        raise ValueError(
            f"context_fraction {context_fraction} keeps the full chunk width for shape {(H, W, D)}"
        )

    mask = torch.ones((B, H, W, D), dtype=torch.bool)
    mask[:, :keep_width, :, :] = False
    return mask


def load_source_sample(
    source_path: str,
    converter: BlockBiomeConverter,
    device: torch.device,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Load a source sample from a .pt file.
    
    Expected format:
        - 'voxels': [N, H, W, D] tensor of block indices (or original block IDs)
        - 'classes' or 'biomes': [N] tensor of class labels (optional)
        
    Returns:
        (voxel_indices, class_labels) - voxel_indices are in model index space
    """
    data = torch.load(source_path, map_location="cpu", weights_only=False)
    
    if isinstance(data, torch.Tensor):
        voxels = data
        labels = None
    elif isinstance(data, dict):
        voxels = data.get("voxels", data.get("samples", None))
        labels = data.get("classes", data.get("biomes", data.get("labels", None)))
        if voxels is None:
            raise ValueError(f"Source file missing 'voxels' key. Keys: {list(data.keys())}")
    else:
        raise ValueError(f"Unexpected source format: {type(data)}")
    
    # Take first sample if batch
    if voxels.dim() == 4:
        voxels = voxels[0:1]  # Keep batch dim
    elif voxels.dim() == 3:
        voxels = voxels.unsqueeze(0)
    
    if labels is not None:
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
        labels = labels[0:1]
    
    # Convert original block IDs to model indices if needed
    # Check if values exceed num_blocks (suggesting original IDs)
    num_blocks = len(converter.block_to_index)
    if voxels.max().item() >= num_blocks:
        print(f"  Converting original block IDs to model indices...")
        voxels = converter.convert_to_indices(voxels)
    
    return voxels.to(device), labels.to(device) if labels is not None else None


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


def render_chunk_to_file_fitted_iso(
    chunk: torch.Tensor,
    output_path: str,
    visualizer: MinecraftVisualizerPyVista,
    textured: bool = True,
    image_size: int = 512,
    fit_padding: float = 1.04,
) -> None:
    """
    Render with a tighter fitted isometric camera.

    This is separate from render_chunk_to_file() so default render behavior
    remains unchanged for all other callers.
    """
    visualizer.render_chunk_isometric_fitted(
        chunk,
        output_path,
        image_height_px=image_size,
        show_axis=False,
        use_textures=textured,
        fit_padding=fit_padding,
        interactive=False,
    )


def render_side_by_side(
    chunks: list,
    labels: list,
    output_path: str,
    visualizer: MinecraftVisualizerPyVista,
    textured: bool = True,
    image_size: int = 512,
    zoom: float = 1.0,
) -> None:
    """
    Render multiple chunks side by side into a single image with labels.
    
    Args:
        chunks: List of [H, W, D] tensors to render
        labels: List of label strings for each chunk
        output_path: Where to save the combined image
        visualizer: MinecraftVisualizerPyVista instance
        textured: Whether to use textured rendering
        image_size: Size of each individual render
        zoom: Camera zoom factor
    """
    from PIL import Image, ImageDraw, ImageFont
    import tempfile
    
    temp_files = []
    images = []
    
    # Render each chunk to a temp file
    for i, chunk in enumerate(chunks):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name
            temp_files.append(temp_path)
        
        render_chunk_to_file(chunk, temp_path, visualizer, textured, image_size, zoom)
        images.append(Image.open(temp_path))
    
    # Calculate combined image size
    label_height = 40
    total_width = image_size * len(chunks)
    total_height = image_size + label_height
    
    # Create combined image
    combined = Image.new('RGB', (total_width, total_height), color=(255, 255, 255))
    
    # Try to get a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
    
    draw = ImageDraw.Draw(combined)
    
    # Paste images and draw labels
    for i, (img, label) in enumerate(zip(images, labels)):
        x_offset = i * image_size
        
        # Draw label at top
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = x_offset + (image_size - text_width) // 2
        draw.text((text_x, 8), label, fill=(0, 0, 0), font=font)
        
        # Paste image below label
        combined.paste(img, (x_offset, label_height))
    
    # Save combined image
    combined.save(output_path)
    
    # Clean up temp files
    for temp_path in temp_files:
        try:
            os.remove(temp_path)
        except:
            pass
    
    # Close PIL images
    for img in images:
        img.close()


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

    Like semantic_super_sample.py, we keep passing a valid class label but force
    cond_scale=0 so forward_with_cond_scale() uses the learned null-conditioning
    branch.
    """
    if not getattr(model, "class_conditional", False):
        return None
    if not hasattr(model, "forward_with_cond_scale"):
        raise ValueError(
            "This class-conditional model does not expose forward_with_cond_scale(), "
            "so unconditional sampling via the learned null-conditioning branch "
            "is not supported by inpaint.py."
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
    - voxels or source_voxels: [H, Y, D] tensor in model index space or original block IDs
    - context_mask or known_mask: [H, Y, D] bool tensor marking fixed voxels
    - inpaint_mask: optional [H, Y, D] bool tensor marking generated voxels
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
            f"Seed context voxels must be 3D [H, Y, D], got shape {tuple(voxels.shape)} "
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
    """
    Build a visualization tensor that shows only the kept context.

    This matches the older inpainting experiment rendering path exactly:
    convert the authored source to original block IDs, then overwrite every
    masked/inpainted position with literal AIR block ID 5 so preview renders see
    a true sparse context chunk instead of whatever filler happened to be in the
    authored source tensor.
    """
    context_preview = source_blocks.clone()
    context_preview[inpaint_mask.cpu()] = int(air_block_id)
    return context_preview


def _save_inpaint_diffusion_artifacts(
    *,
    sample_dir: Path,
    artifact_stem: str,
    diffusion_trace: Optional[Dict[str, Any]],
    sample_index: int,
    converter: BlockBiomeConverter,
    textures_dir: str,
    image_size: int,
    fps: int,
    source_indices: Optional[torch.Tensor] = None,
    inpaint_mask: Optional[torch.Tensor] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Optional[str]]:
    """Persist per-sample logit intermediates and render a matching GIF."""
    if diffusion_trace is None:
        return {"intermediates_path": None, "gif_path": None}

    sample_dir.mkdir(parents=True, exist_ok=True)
    intermediates = diffusion_trace["intermediates"]
    if intermediates.dim() != 5:
        raise ValueError(
            f"Expected batched logit intermediates [B,T,H,W,D], got {tuple(intermediates.shape)}"
        )
    if not (0 <= sample_index < intermediates.shape[0]):
        raise IndexError(
            f"sample_index {sample_index} out of range for intermediates batch {intermediates.shape[0]}"
        )

    sample_intermediates = intermediates[sample_index].cpu()
    tensor_path = sample_dir / f"{artifact_stem}_logit_intermediates.pt"
    gif_path = sample_dir / f"{artifact_stem}_logit.gif"

    payload: Dict[str, Any] = {
        "intermediates": sample_intermediates,
        "sample_index": int(sample_index),
        "fraction_known": float(diffusion_trace["fraction_known"]),
        "t_start": float(diffusion_trace["t_start"]),
        "effective_steps": int(diffusion_trace["effective_steps"]),
        "save_step_indices": [int(idx) for idx in diffusion_trace.get("save_step_indices", [])],
        "logit_save_bias_power": float(diffusion_trace.get("logit_save_bias_power", 1.0)),
        "trace_mode": diffusion_trace.get("trace_mode", "full_volume_logit_samples"),
    }
    if source_indices is not None:
        payload["source_indices"] = source_indices.cpu()
    if inpaint_mask is not None:
        payload["inpaint_mask"] = inpaint_mask.cpu()
    if metadata is not None:
        payload["metadata"] = metadata
    torch.save(payload, tensor_path)

    textured = bool(textures_dir and os.path.exists(textures_dir))
    gif_output_path: Optional[str] = str(gif_path)
    try:
        render_diffusion_gif(
            sample_intermediates,
            str(gif_path),
            converter,
            num_frames=int(sample_intermediates.shape[0]),
            image_size=image_size,
            fps=fps,
            textured=textured,
            include_initial_empty=False,
            textures_dir=(textures_dir if textured else None),
        )
    except Exception as e:
        print(f"  Warning: Could not render diffusion GIF for {artifact_stem}: {e}")
        gif_output_path = None

    return {
        "intermediates_path": str(tensor_path),
        "gif_path": gif_output_path,
    }


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


def _run_seeded_batch(
    model,
    inpainter: MD4Inpainter,
    batch_name: str,
    samples: List[Dict[str, Any]],
    biome_name: str,
    converter: BlockBiomeConverter,
    output_dir: Path,
    reverse_steps: int = 1000,
    cond_scale: float = 4.0,
    air_idx: int = 0,
    textures_dir: str = "block_textures/",
    image_size: int = 512,
    render: bool = True,
    save_gif: bool = False,
    gif_timesteps: int = 40,
    gif_fps: int = 10,
    gif_early_bias: float = 2.0,
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
    inpaint_mask = torch.stack(
        [sample["inpaint_mask"].clone() for sample in samples],
        dim=0,
    ).to(inpainter.device)
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

    known_counts = (~inpaint_mask).sum(dim=(1, 2, 3))
    if not torch.all(known_counts == known_counts[0]):
        raise ValueError(
            f"Seeded batch '{batch_name}' mixes different context counts, which "
            "would invalidate the shared time-aligned schedule"
        )

    visualizer = None
    textured = False
    if render:
        if textures_dir and os.path.exists(textures_dir):
            visualizer = MinecraftVisualizerPyVista(textures_dir=textures_dir, build_textures=True)
            textured = True
        else:
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
    diffusion_trace = None
    if save_gif:
        inpainted_result, diffusion_trace = inpainter.inpaint_time_aligned(
            model,
            source_indices,
            inpaint_mask,
            reverse_steps=reverse_steps,
            progress=True,
            air_index_fallback=air_idx,
            class_cond=class_labels,
            cond_scale=effective_cond_scale,
            return_logit_intermediates=True,
            num_logit_intermediates=gif_timesteps,
            logit_save_bias_power=gif_early_bias,
        )
    else:
        inpainted_result = inpainter.inpaint_time_aligned(
            model,
            source_indices,
            inpaint_mask,
            reverse_steps=reverse_steps,
            progress=True,
            air_index_fallback=air_idx,
            class_cond=class_labels,
            cond_scale=effective_cond_scale,
        )

    result_dir = output_dir / _sanitize_output_name(batch_name)
    result_dir.mkdir(parents=True, exist_ok=True)

    source_blocks = converter.convert_to_original_blocks(source_indices.cpu())
    inpainted_blocks = converter.convert_to_original_blocks(inpainted_result.cpu())
    context_preview = _build_context_preview_blocks(source_blocks, inpaint_mask)

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
        "inpaint_mask": inpaint_mask.cpu(),
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
            "inpaint_mask": inpaint_mask[batch_idx].cpu(),
            "inpainted": inpainted_result[batch_idx].cpu(),
            "source_blocks": source_blocks[batch_idx].cpu(),
            "context_preview_blocks": context_preview[batch_idx].cpu(),
            "inpainted_blocks": inpainted_blocks[batch_idx].cpu(),
        }
        if save_gif:
            gif_artifacts = _save_inpaint_diffusion_artifacts(
                sample_dir=result_dir,
                artifact_stem=stem,
                diffusion_trace=diffusion_trace,
                sample_index=batch_idx,
                converter=converter,
                textures_dir=textures_dir,
                image_size=image_size,
                fps=gif_fps,
                source_indices=source_indices[batch_idx].cpu(),
                inpaint_mask=inpaint_mask[batch_idx].cpu(),
                metadata={
                    "batch_name": batch_name,
                    "sample_name": sample["name"],
                    "biome_name": biome_name,
                    "unconditional": unconditional,
                },
            )
            sample_result["logit_intermediates_path"] = gif_artifacts["intermediates_path"]
            sample_result["logit_gif_path"] = gif_artifacts["gif_path"]
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
    inpainter: MD4Inpainter,
    converter: BlockBiomeConverter,
    output_dir: Path,
    context_files: Optional[List[str]] = None,
    context_dir: str = "seed_context_outputs",
    biome_names: Optional[List[str]] = None,
    num_variants: int = 1,
    reverse_steps: int = 1000,
    cond_scale: float = 4.0,
    air_idx: int = 0,
    textures_dir: str = "block_textures/",
    image_size: int = 512,
    render: bool = True,
    save_gif: bool = False,
    gif_timesteps: int = 40,
    gif_fps: int = 10,
    gif_early_bias: float = 2.0,
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
            output_dir=root_dir,
            reverse_steps=reverse_steps,
            cond_scale=cond_scale,
            air_idx=air_idx,
            textures_dir=textures_dir,
            image_size=image_size,
            render=render,
            save_gif=save_gif,
            gif_timesteps=gif_timesteps,
            gif_fps=gif_fps,
            gif_early_bias=gif_early_bias,
        )

    return results


def run_inpainting_experiment(
    model,
    inpainter: MD4Inpainter,
    source_indices: torch.Tensor,
    source_labels: Optional[torch.Tensor],
    converter: BlockBiomeConverter,
    output_dir: Path,
    reverse_steps: int = 1000,
    cond_scale: float = 4.0,
    air_idx: int = 0,
    textures_dir: str = "block_textures/",
    image_size: int = 512,
    save_gif: bool = False,
    gif_timesteps: int = 40,
    gif_fps: int = 10,
    gif_early_bias: float = 2.0,
) -> Dict[str, Any]:
    """
    Run time-aligned inpainting experiments.
    
    Tests multiple context configurations including rings.
    Outputs side-by-side comparison images: Source | Context | Inpainted
    """
    B, H, W, D = source_indices.shape
    results = {}
    
    # Initialize visualizer
    if textures_dir and os.path.exists(textures_dir):
        visualizer = MinecraftVisualizerPyVista(textures_dir=textures_dir, build_textures=True)
        textured = True
    else:
        visualizer = MinecraftVisualizerPyVista()
        textured = False
    
    # Experiment configurations: (name, mode, fraction_to_inpaint, ring_thickness)
    experiments = [
        ("single_voxel", "single_voxel", 0.9999, 0),        # ~0% context (single voxel)
        ("quarter_context", "half_x", 0.75, 0),             # 25% context, 75% inpaint
        ("three_quarter_context", "half_x", 0.25, 0),       # 75% context, 25% inpaint
        # Partial ring - horseshoe shaped, side edges only, 2/3 kept
        ("partial_ring", "partial_ring", 0.0, 1),           # Horseshoe context
        # Opposite corners - two 8x32x8 vertical slices at opposite corners
        ("opposite_corners", "opposite_corners", 0.0, 0),   # Diagonal corner context
    ]
    
    for exp_name, mode, frac_inpaint, ring_thickness in experiments:
        print(f"\n{'='*60}")
        print(f"Experiment: {exp_name}")
        if mode == "ring":
            print(f"  Mode: {mode}, Ring thickness: {ring_thickness} blocks")
        else:
            print(f"  Mode: {mode}, Fraction to inpaint: {frac_inpaint*100:.1f}%")
        print(f"{'='*60}")
        
        # Create inpainting mask
        inpaint_mask = create_inpaint_mask_fraction((B, H, W, D), frac_inpaint, mode, ring_thickness=ring_thickness)
        inpaint_mask = inpaint_mask.to(source_indices.device)
        
        num_inpaint = inpaint_mask.sum().item()
        num_context = (~inpaint_mask).sum().item()
        total = H * W * D
        print(f"  Context voxels: {num_context} ({num_context/total*100:.1f}%), Inpaint voxels: {num_inpaint} ({num_inpaint/total*100:.1f}%)")
        
        # Run time-aligned inpainting
        print("\n  Running time-aligned inpainting...")
        diffusion_trace = None
        if save_gif:
            inpainted_result, diffusion_trace = inpainter.inpaint_time_aligned(
                model,
                source_indices,
                inpaint_mask,
                reverse_steps=reverse_steps,
                progress=True,
                air_index_fallback=air_idx,
                class_cond=source_labels,
                cond_scale=cond_scale,
                return_logit_intermediates=True,
                num_logit_intermediates=gif_timesteps,
                logit_save_bias_power=gif_early_bias,
            )
        else:
            inpainted_result = inpainter.inpaint_time_aligned(
                model,
                source_indices,
                inpaint_mask,
                reverse_steps=reverse_steps,
                progress=True,
                air_index_fallback=air_idx,
                class_cond=source_labels,
                cond_scale=cond_scale,
            )
        
        # Save results
        exp_results = {
            "source": source_indices.cpu(),
            "inpainted": inpainted_result.cpu(),
            "inpaint_mask": inpaint_mask.cpu(),
            "fraction_inpainted": frac_inpaint,
            "mode": mode,
        }
        results[exp_name] = exp_results
        
        # Save to disk
        exp_dir = output_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Convert to original block IDs for visualization
        source_blocks = converter.convert_to_original_blocks(source_indices.cpu())
        inpainted_blocks = converter.convert_to_original_blocks(inpainted_result.cpu())
        
        # Create context preview: source with inpaint region set to air (block ID 5)
        AIR_BLOCK_ID = 5
        context_preview = source_blocks.clone()
        context_preview[inpaint_mask.cpu()] = AIR_BLOCK_ID
        
        # Save .pt files
        torch.save(source_blocks, exp_dir / "source_blocks.pt")
        torch.save(inpainted_blocks, exp_dir / "inpainted_blocks.pt")
        torch.save(context_preview, exp_dir / "context_preview_blocks.pt")
        if save_gif:
            gif_artifacts = _save_inpaint_diffusion_artifacts(
                sample_dir=exp_dir,
                artifact_stem=exp_name,
                diffusion_trace=diffusion_trace,
                sample_index=0,
                converter=converter,
                textures_dir=textures_dir,
                image_size=image_size,
                fps=gif_fps,
                source_indices=source_indices[0].cpu(),
                inpaint_mask=inpaint_mask[0].cpu(),
                metadata={
                    "experiment": exp_name,
                    "mode": mode,
                    "fraction_inpainted": float(frac_inpaint),
                },
            )
            exp_results["logit_intermediates_path"] = gif_artifacts["intermediates_path"]
            exp_results["logit_gif_path"] = gif_artifacts["gif_path"]
        torch.save(exp_results, exp_dir / "results.pt")
        
        # Render side-by-side comparison: Source | Context | Inpainted
        try:
            print("  Rendering side-by-side comparison...")
            render_side_by_side(
                chunks=[source_blocks[0], context_preview[0], inpainted_blocks[0]],
                labels=["Source", "Context", "Inpainted"],
                output_path=str(exp_dir / "comparison.png"),
                visualizer=visualizer,
                textured=textured,
                image_size=image_size,
            )
            print(f"  Saved comparison to {exp_dir / 'comparison.png'}")
        except Exception as e:
            print(f"  Warning: Could not render visualization: {e}")
    
    return results


def run_batched_inpainting_experiments(
    model,
    inpainter: MD4Inpainter,
    samples: List[Dict[str, Any]],
    converter: BlockBiomeConverter,
    output_dir: Path,
    num_variants: int = 1,
    reverse_steps: int = 1000,
    cond_scale: float = 4.0,
    air_idx: int = 0,
    textures_dir: str = "block_textures/",
    image_size: int = 512,
    render: bool = True,
    save_gif: bool = False,
    gif_timesteps: int = 40,
    gif_fps: int = 10,
    gif_early_bias: float = 2.0,
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
        inpaint_mask = create_standard_corner_inpaint_mask(tuple(source_batch.shape)).to(inpainter.device)

        if getattr(model, "class_conditional", False):
            class_labels = torch.full(
                (num_variants,),
                int(sample["biome_index"]),
                dtype=torch.long,
                device=inpainter.device,
            )
        else:
            class_labels = None

        num_context = int((~inpaint_mask[0]).sum().item())
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

        diffusion_trace = None
        if save_gif:
            inpainted_result, diffusion_trace = inpainter.inpaint_time_aligned(
                model,
                source_batch,
                inpaint_mask,
                reverse_steps=reverse_steps,
                progress=True,
                air_index_fallback=air_idx,
                class_cond=class_labels,
                cond_scale=cond_scale,
                return_logit_intermediates=True,
                num_logit_intermediates=gif_timesteps,
                logit_save_bias_power=gif_early_bias,
            )
        else:
            inpainted_result = inpainter.inpaint_time_aligned(
                model,
                source_batch,
                inpaint_mask,
                reverse_steps=reverse_steps,
                progress=True,
                air_index_fallback=air_idx,
                class_cond=class_labels,
                cond_scale=cond_scale,
            )

        source_blocks = converter.convert_to_original_blocks(source_batch.cpu())
        inpainted_blocks = converter.convert_to_original_blocks(inpainted_result.cpu())
        context_preview = _build_context_preview_blocks(source_blocks, inpaint_mask.cpu())

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
            "inpaint_mask": inpaint_mask[0:1].cpu(),
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
                "inpaint_mask": inpaint_mask[variant_idx].cpu(),
                "context_preview_blocks": context_preview[variant_idx].cpu(),
                "inpainted": inpainted_result[variant_idx].cpu(),
                "inpainted_blocks": inpainted_blocks[variant_idx].cpu(),
            }
            if save_gif:
                gif_artifacts = _save_inpaint_diffusion_artifacts(
                    sample_dir=sample_dir,
                    artifact_stem=variant_stem,
                    diffusion_trace=diffusion_trace,
                    sample_index=variant_idx,
                    converter=converter,
                    textures_dir=textures_dir,
                    image_size=image_size,
                    fps=gif_fps,
                    source_indices=source_single[0].cpu(),
                    inpaint_mask=inpaint_mask[variant_idx].cpu(),
                    metadata={
                        "mode": "standard_biome_variants",
                        "biome_name": biome_name,
                        "sample_name": sample["sample_name"],
                        "variant_idx": variant_idx + 1,
                    },
                )
                variant_payload["logit_intermediates_path"] = gif_artifacts["intermediates_path"]
                variant_payload["logit_gif_path"] = gif_artifacts["gif_path"]
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


def run_biome_swap_experiments(
    model,
    inpainter: MD4Inpainter,
    samples: List[Dict[str, Any]],
    available_biomes: List[str],
    converter: BlockBiomeConverter,
    output_dir: Path,
    num_variants: int = 1,
    context_fraction: float = BIOME_SWAP_CONTEXT_FRACTION,
    target_biomes: Optional[List[str]] = None,
    reverse_steps: int = 1000,
    cond_scale: float = 4.0,
    air_idx: int = 0,
    textures_dir: str = "block_textures/",
    image_size: int = 512,
    render: bool = True,
    save_gif: bool = False,
    gif_timesteps: int = 40,
    gif_fps: int = 10,
    gif_early_bias: float = 2.0,
) -> Dict[str, Any]:
    """
    Run biome-swap variants for 2 samples per biome.

    For each source biome, pick one random source sample, keep a configurable
    fraction of the chunk as context, and batch-generate
    inpainted results for either:
    - explicit target biomes repeated num_variants times each, or
    - num_variants randomly selected alternate biomes
    """
    if not samples:
        raise ValueError("Biome swap mode requires at least one loaded source sample")
    if num_variants < 1:
        raise ValueError(f"num_variants must be >= 1, got {num_variants}")
    if not getattr(model, "class_conditional", False):
        raise ValueError("biome_swap mode requires a class-conditional model")
    if not (0.0 < context_fraction < 1.0):
        raise ValueError(f"biome_swap context_fraction must be between 0 and 1, got {context_fraction}")

    unique_biomes = sorted(set(available_biomes))
    if len(unique_biomes) < 2:
        raise ValueError("biome_swap mode requires at least 2 distinct biomes")

    results: Dict[str, Dict[str, Any]] = {}
    root_dir = output_dir / "biome_swap"
    root_dir.mkdir(parents=True, exist_ok=True)

    visualizer = None
    textured = False
    if render and textures_dir and os.path.exists(textures_dir):
        visualizer = MinecraftVisualizerPyVista(textures_dir=textures_dir, build_textures=True)
        textured = True
    elif render:
        visualizer = MinecraftVisualizerPyVista()
    for sample in samples:
        source_biome = sample["biome_name"]
        source_biome_idx = int(sample["biome_index"])
        if target_biomes:
            selected_target_biomes = [
                target_name
                for target_name in target_biomes
                for _ in range(num_variants)
            ]
        else:
            random_target_pool = [name for name in unique_biomes if name != source_biome]
            if len(random_target_pool) < num_variants:
                raise ValueError(
                    f"Requested {num_variants} biome-swap variants for source biome "
                    f"'{source_biome}', but only {len(random_target_pool)} other biomes are available."
                )
            selected_target_biomes = random.sample(random_target_pool, num_variants)

        selected_target_indices = [
            int(converter.biome_to_index[target_name]) for target_name in selected_target_biomes
        ]
        total_outputs = len(selected_target_biomes)

        biome_dir = root_dir / _sanitize_output_name(source_biome)
        sample_dir = biome_dir / _sanitize_output_name(sample["sample_name"])
        sample_dir.mkdir(parents=True, exist_ok=True)

        source_single = sample["source_indices"].unsqueeze(0).to(inpainter.device)
        source_batch = source_single.repeat(total_outputs, 1, 1, 1)
        inpaint_mask = create_biome_swap_inpaint_mask(
            tuple(source_batch.shape),
            context_fraction=context_fraction,
        ).to(inpainter.device)
        class_labels = torch.tensor(
            selected_target_indices,
            dtype=torch.long,
            device=inpainter.device,
        )

        num_context = int((~inpaint_mask[0]).sum().item())
        total = int(source_batch.shape[1] * source_batch.shape[2] * source_batch.shape[3])
        num_inpaint = total - num_context

        print("\n" + "=" * 60)
        print(f"Biome swap variants: {source_biome} / {sample['sample_name']}")
        print(f"  Source file: {sample['source_file']}")
        print(f"  Source sample index: {sample['source_sample_index']}")
        print(
            f"  Context voxels: {num_context} ({num_context/total*100:.2f}%), "
            f"Inpaint voxels: {num_inpaint} ({num_inpaint/total*100:.2f}%)"
        )
        print(f"  Context fraction: {context_fraction:.4f}")
        if target_biomes:
            print(f"  Target biomes: {target_biomes}")
            print(f"  Variants per target biome: {num_variants}")
            print(f"  Total generated outputs: {total_outputs}")
        else:
            print(f"  Target biomes: {selected_target_biomes}")
        print("=" * 60)

        diffusion_trace = None
        if save_gif:
            inpainted_result, diffusion_trace = inpainter.inpaint_time_aligned(
                model,
                source_batch,
                inpaint_mask,
                reverse_steps=reverse_steps,
                progress=True,
                air_index_fallback=air_idx,
                class_cond=class_labels,
                cond_scale=cond_scale,
                return_logit_intermediates=True,
                num_logit_intermediates=gif_timesteps,
                logit_save_bias_power=gif_early_bias,
            )
        else:
            inpainted_result = inpainter.inpaint_time_aligned(
                model,
                source_batch,
                inpaint_mask,
                reverse_steps=reverse_steps,
                progress=True,
                air_index_fallback=air_idx,
                class_cond=class_labels,
                cond_scale=cond_scale,
            )

        source_blocks = converter.convert_to_original_blocks(source_batch.cpu())
        inpainted_blocks = converter.convert_to_original_blocks(inpainted_result.cpu())
        context_preview = _build_context_preview_blocks(source_blocks, inpaint_mask.cpu())

        sample_summary = {
            "mode": "biome_swap_variants",
            "source_biome": source_biome,
            "source_biome_idx": source_biome_idx,
            "source_file": sample["source_file"],
            "source_sample_index": int(sample["source_sample_index"]),
            "sample_name": sample["sample_name"],
            "num_variants": int(num_variants),
            "total_generated_outputs": int(total_outputs),
            "context_fraction": float(context_fraction),
            "mask_mode": "leading_x_context",
            "source_indices": source_single.cpu(),
            "source_blocks": source_blocks[0:1].cpu(),
            "inpaint_mask": inpaint_mask[0:1].cpu(),
            "source_render": str(sample_dir / "source_chunk.png"),
            "variant_files": [],
            "target_biomes": target_biomes if target_biomes is not None else selected_target_biomes,
        }

        sample_result_summary = {
            "sample_name": sample["sample_name"],
            "source_file": sample["source_file"],
            "source_sample_index": int(sample["source_sample_index"]),
            "source_render": str(sample_dir / "source_chunk.png"),
            "variant_files": [],
            "target_biomes": target_biomes if target_biomes is not None else selected_target_biomes,
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

        per_target_counts: Dict[str, int] = {}
        for variant_idx, (target_biome, target_biome_idx) in enumerate(
            zip(selected_target_biomes, selected_target_indices),
            start=1,
        ):
            target_repeat_idx = per_target_counts.get(target_biome, 0) + 1
            per_target_counts[target_biome] = target_repeat_idx
            variant_name = (
                f"{sample['sample_name']}_to_{_sanitize_output_name(target_biome)}_variant_{target_repeat_idx:02d}"
            )
            variant_stem = _sanitize_output_name(variant_name)
            variant_file = sample_dir / f"{variant_stem}.pt"
            comparison_file = sample_dir / f"{variant_stem}_comparison.png"

            variant_payload = {
                "mode": "biome_swap_variants",
                "source_biome": source_biome,
                "source_biome_idx": source_biome_idx,
                "target_biome": target_biome,
                "target_biome_idx": int(target_biome_idx),
                "source_file": sample["source_file"],
                "source_sample_index": int(sample["source_sample_index"]),
                "sample_name": sample["sample_name"],
                "variant_idx": variant_idx,
                "target_variant_idx": target_repeat_idx,
                "source_indices": source_single[0].cpu(),
                "source_blocks": source_blocks[variant_idx - 1].cpu(),
                "inpaint_mask": inpaint_mask[variant_idx - 1].cpu(),
                "context_preview_blocks": context_preview[variant_idx - 1].cpu(),
                "inpainted": inpainted_result[variant_idx - 1].cpu(),
                "inpainted_blocks": inpainted_blocks[variant_idx - 1].cpu(),
            }
            if save_gif:
                gif_artifacts = _save_inpaint_diffusion_artifacts(
                    sample_dir=sample_dir,
                    artifact_stem=variant_stem,
                    diffusion_trace=diffusion_trace,
                    sample_index=variant_idx - 1,
                    converter=converter,
                    textures_dir=textures_dir,
                    image_size=image_size,
                    fps=gif_fps,
                    source_indices=source_single[0].cpu(),
                    inpaint_mask=inpaint_mask[variant_idx - 1].cpu(),
                    metadata={
                        "mode": "biome_swap_variants",
                        "source_biome": source_biome,
                        "target_biome": target_biome,
                        "sample_name": sample["sample_name"],
                        "variant_idx": variant_idx,
                    },
                )
                variant_payload["logit_intermediates_path"] = gif_artifacts["intermediates_path"]
                variant_payload["logit_gif_path"] = gif_artifacts["gif_path"]
            torch.save(variant_payload, variant_file)

            sample_summary["variant_files"].append(str(variant_file))
            sample_result_summary["variant_files"].append(str(variant_file))

            if render and visualizer is not None:
                try:
                    render_side_by_side(
                        chunks=[context_preview[variant_idx - 1], inpainted_blocks[variant_idx - 1]],
                        labels=[
                            f"Context ({context_fraction * 100:.2f}% source)",
                            f"Biome Swap: {target_biome}",
                        ],
                        output_path=str(comparison_file),
                        visualizer=visualizer,
                        textured=textured,
                        image_size=image_size,
                    )
                except Exception as e:
                    print(f"  Warning: Could not render {variant_name}: {e}")

        torch.save(sample_summary, sample_dir / "results.pt")
        results.setdefault(source_biome, {})[sample["sample_name"]] = sample_result_summary

    return results


def run_village_experiments(
    model,
    inpainter: MD4Inpainter,
    source_indices: torch.Tensor,
    source_labels: torch.Tensor,
    biome_names: List[str],
    converter: BlockBiomeConverter,
    output_dir: Path,
    village_source: Optional[str] = None,
    num_infills: int = 5,
    reverse_steps: int = 1000,
    cond_scale: float = 4.0,
    air_idx: int = 0,
    textures_dir: str = "block_textures/",
    image_size: int = 512,
    save_gif: bool = False,
    gif_timesteps: int = 40,
    gif_fps: int = 10,
    gif_early_bias: float = 2.0,
) -> Dict[str, Any]:
    """
    Run village-focused inpainting experiments.
    
    Two main experiment types:
    1. Village ring infill: Take 1-ring (no caps) from each biome, infill with village
    2. Village diversity: Take 3 village samples with different contexts, generate N infills each
    """
    B, H, W, D = source_indices.shape
    results = {}
    
    # Initialize visualizer
    if textures_dir and os.path.exists(textures_dir):
        visualizer = MinecraftVisualizerPyVista(textures_dir=textures_dir, build_textures=True)
        textured = True
    else:
        visualizer = MinecraftVisualizerPyVista()
        textured = False
    
    # Find village biome index
    village_idx = None
    for idx, name in converter.index_to_biome.items():
        if "village" in name.lower():
            village_idx = idx
            break
    
    if village_idx is None:
        print("Warning: Could not find 'village' biome. Using biome index 0.")
        village_idx = 0
    else:
        village_name = converter.index_to_biome.get(village_idx, str(village_idx))
        print(f"Found village biome: {village_name} (index {village_idx})")
    
    village_label = torch.tensor([village_idx], dtype=torch.long, device=source_indices.device)
    
    # ==========================================================================
    # EXPERIMENT 1: Village ring infill - 1-ring from each biome, fill with village
    # ==========================================================================
    print(f"\n{'='*60}")
    print(f"Experiment: village_ring_infill ({B} biomes as context)")
    print(f"  1-block ring (no caps) from each biome, infill village")
    print(f"{'='*60}")
    
    # Create ring mask (no caps)
    ring_mask = create_inpaint_mask_fraction(
        (B, H, W, D), 0.0, mode="ring_no_caps", ring_thickness=1
    )
    ring_mask = ring_mask.to(source_indices.device)
    
    num_inpaint = ring_mask[0].sum().item()
    num_context = (~ring_mask[0]).sum().item()
    total = H * W * D
    print(f"  Context voxels: {num_context} ({num_context/total*100:.1f}%), Inpaint voxels: {num_inpaint} ({num_inpaint/total*100:.1f}%)")
    
    # Run with village conditioning for all samples
    village_labels_batch = torch.full((B,), village_idx, dtype=torch.long, device=source_indices.device)
    
    print(f"  Running ring infill with village conditioning...")
    ring_diffusion_trace = None
    if save_gif:
        inpainted_result, ring_diffusion_trace = inpainter.inpaint_time_aligned(
            model,
            source_indices,
            ring_mask,
            reverse_steps=reverse_steps,
            progress=True,
            air_index_fallback=air_idx,
            class_cond=village_labels_batch,
            cond_scale=cond_scale,
            return_logit_intermediates=True,
            num_logit_intermediates=gif_timesteps,
            logit_save_bias_power=gif_early_bias,
        )
    else:
        inpainted_result = inpainter.inpaint_time_aligned(
            model,
            source_indices,
            ring_mask,
            reverse_steps=reverse_steps,
            progress=True,
            air_index_fallback=air_idx,
            class_cond=village_labels_batch,
            cond_scale=cond_scale,
        )
    
    # Save per-biome results
    ring_results = {}
    for i, biome_name in enumerate(biome_names):
        biome_dir = output_dir / "village_ring_infill" / biome_name
        biome_dir.mkdir(parents=True, exist_ok=True)
        
        source_i = source_indices[i:i+1].cpu()
        inpainted_i = inpainted_result[i:i+1].cpu()
        mask_i = ring_mask[i:i+1].cpu()
        
        biome_results = {
            "source": source_i,
            "inpainted": inpainted_i,
            "inpaint_mask": mask_i,
            "context_biome": biome_name,
            "infill_biome": "village",
            "experiment": "village_ring_infill",
        }
        if save_gif:
            gif_artifacts = _save_inpaint_diffusion_artifacts(
                sample_dir=biome_dir,
                artifact_stem="village_ring_infill",
                diffusion_trace=ring_diffusion_trace,
                sample_index=i,
                converter=converter,
                textures_dir=textures_dir,
                image_size=image_size,
                fps=gif_fps,
                source_indices=source_i[0],
                inpaint_mask=mask_i[0],
                metadata={
                    "experiment": "village_ring_infill",
                    "context_biome": biome_name,
                    "infill_biome": "village",
                },
            )
            biome_results["logit_intermediates_path"] = gif_artifacts["intermediates_path"]
            biome_results["logit_gif_path"] = gif_artifacts["gif_path"]
        torch.save(biome_results, biome_dir / "results.pt")
        
        # Visualize
        source_blocks = converter.convert_to_original_blocks(source_i)
        inpainted_blocks = converter.convert_to_original_blocks(inpainted_i)
        
        AIR_BLOCK_ID = 5
        context_preview = source_blocks.clone()
        context_preview[mask_i] = AIR_BLOCK_ID
        
        try:
            render_side_by_side(
                chunks=[source_blocks[0], context_preview[0], inpainted_blocks[0]],
                labels=[
                    f"Source ({biome_name})",
                    "Context (ring)",
                    "Infilled (village)",
                ],
                output_path=str(biome_dir / "comparison.png"),
                visualizer=visualizer,
                textured=textured,
                image_size=image_size,
            )
        except Exception as e:
            print(f"    Warning: Could not render {biome_name}: {e}")
        
        ring_results[biome_name] = biome_results
    
    results["village_ring_infill"] = ring_results
    print(f"  Saved village ring infill results for {B} biomes")
    
    # ==========================================================================
    # EXPERIMENT 2: Village diversity - multiple infills from same context
    # ==========================================================================
    if village_source is not None:
        print(f"\n{'='*60}")
        print(f"Experiment: village_diversity")
        print(f"  Loading village samples from: {village_source}")
        print(f"  Generating {num_infills} infills per context type")
        print(f"{'='*60}")
        
        # Load village samples
        village_data = torch.load(village_source, map_location="cpu")
        if isinstance(village_data, dict):
            if "voxels" in village_data:
                village_voxels = village_data["voxels"]
            elif "samples" in village_data:
                village_voxels = village_data["samples"]
            else:
                village_voxels = village_data.get("indices", next(iter(village_data.values())))
        else:
            village_voxels = village_data
        
        # Convert to model indices if needed
        if village_voxels.max() > converter.num_blocks:
            village_voxels = converter.convert_to_model_indices(village_voxels)
        
        # Need at least 3 samples
        if village_voxels.shape[0] < 3:
            print(f"  Warning: Need at least 3 village samples, got {village_voxels.shape[0]}. Duplicating...")
            while village_voxels.shape[0] < 3:
                village_voxels = torch.cat([village_voxels, village_voxels[:1]], dim=0)
        
        # Take first 3 samples
        village_samples = village_voxels[:3].to(source_indices.device)
        print(f"  Using 3 village source samples, shape: {village_samples.shape}")
        
        # Context configurations for each sample
        context_configs = [
            ("strip_1_3", "strip_edge", 1/3, 0),           # 1/3 strip on edge
            ("ring_4_no_caps", "ring_4_no_caps", 0.0, 4),  # 4-block ring, no caps
            ("opposite_corners", "opposite_corners", 0.0, 0),  # 8x8 opposing corners
        ]
        
        diversity_results = {}
        
        for sample_idx, (ctx_name, mode, frac, ring_t) in enumerate(context_configs):
            print(f"\n  Sample {sample_idx + 1}: {ctx_name}")
            
            source_single = village_samples[sample_idx:sample_idx+1]
            exp_dir = output_dir / "village_diversity" / ctx_name
            exp_dir.mkdir(parents=True, exist_ok=True)
            
            # Create mask
            mask_single = create_inpaint_mask_fraction(
                source_single.shape, frac, mode=mode, ring_thickness=ring_t
            ).to(source_indices.device)
            
            num_ctx = (~mask_single[0]).sum().item()
            num_inp = mask_single[0].sum().item()
            print(f"    Context: {num_ctx}, Inpaint: {num_inp}")
            
            # Generate N infills
            infills = []
            infill_artifacts = []
            for infill_idx in range(num_infills):
                print(f"    Generating infill {infill_idx + 1}/{num_infills}...")
                diversity_diffusion_trace = None
                if save_gif:
                    infilled, diversity_diffusion_trace = inpainter.inpaint_time_aligned(
                        model,
                        source_single,
                        mask_single,
                        reverse_steps=reverse_steps,
                        progress=False,
                        air_index_fallback=air_idx,
                        class_cond=village_label,
                        cond_scale=cond_scale,
                        return_logit_intermediates=True,
                        num_logit_intermediates=gif_timesteps,
                        logit_save_bias_power=gif_early_bias,
                    )
                else:
                    infilled = inpainter.inpaint_time_aligned(
                        model,
                        source_single,
                        mask_single,
                        reverse_steps=reverse_steps,
                        progress=False,
                        air_index_fallback=air_idx,
                        class_cond=village_label,
                        cond_scale=cond_scale,
                    )
                infills.append(infilled.cpu())
                if save_gif:
                    artifact_stem = f"infill_{infill_idx + 1:02d}"
                    infill_artifacts.append(
                        _save_inpaint_diffusion_artifacts(
                            sample_dir=exp_dir,
                            artifact_stem=artifact_stem,
                            diffusion_trace=diversity_diffusion_trace,
                            sample_index=0,
                            converter=converter,
                            textures_dir=textures_dir,
                            image_size=image_size,
                            fps=gif_fps,
                            source_indices=source_single[0].cpu(),
                            inpaint_mask=mask_single[0].cpu(),
                            metadata={
                                "experiment": "village_diversity",
                                "context_type": ctx_name,
                                "infill_idx": infill_idx + 1,
                            },
                        )
                    )
            
            # Save and visualize
            source_blocks = converter.convert_to_original_blocks(source_single.cpu())
            AIR_BLOCK_ID = 5
            context_preview = source_blocks.clone()
            context_preview[mask_single.cpu()] = AIR_BLOCK_ID
            
            sample_results = {
                "source": source_single.cpu(),
                "infills": infills,
                "inpaint_mask": mask_single.cpu(),
                "context_type": ctx_name,
                "num_infills": num_infills,
            }
            if save_gif:
                sample_results["infill_artifacts"] = infill_artifacts
            torch.save(sample_results, exp_dir / "results.pt")
            
            # Create composite visualization: 1 source context + N infills
            all_chunks = [source_blocks[0], context_preview[0]]
            all_labels = ["Source", "Context"]
            
            for idx, infill in enumerate(infills):
                infill_blocks = converter.convert_to_original_blocks(infill)
                all_chunks.append(infill_blocks[0])
                all_labels.append(f"Infill {idx + 1}")
            
            # Render grid
            try:
                render_side_by_side(
                    chunks=all_chunks,
                    labels=all_labels,
                    output_path=str(exp_dir / "diversity_comparison.png"),
                    visualizer=visualizer,
                    textured=textured,
                    image_size=image_size,
                )
            except Exception as e:
                print(f"    Warning: Could not render diversity comparison: {e}")
            
            diversity_results[ctx_name] = sample_results
        
        results["village_diversity"] = diversity_results
        print(f"\n  Saved village diversity results for 3 context types, {num_infills} infills each")
    else:
        print("\n  Skipping village_diversity (no --village_source provided)")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Inpainting experiments for MD4 discrete diffusion")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--mappings", type=str, required=True, help="Path to mappings file")
    parser.add_argument("--source", type=str, default=None, help="Path to source .pt file with voxels (single biome mode)")
    parser.add_argument("--source_folder", type=str, default=None, 
                        help="Path to folder with generated_*.pt files. Required for standard and biome_swap modes.")
    parser.add_argument("--output_dir", type=str, default="./inpaint_results", help="Output directory")
    parser.add_argument("--config", type=str, default=None, help="Optional path to config.json")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--reverse_steps", type=int, default=None, help="Override reverse steps")
    parser.add_argument("--cond_scale", type=float, default=None, help="Conditioning scale for CFG")
    parser.add_argument("--biome", type=str, default=None, 
                        help="Biome condition (name or index). Overrides source file label if provided.")
    parser.add_argument("--textures_dir", type=str, default="block_textures/", 
                        help="Path to block textures directory for rendering")
    parser.add_argument("--image_size", type=int, default=512, help="Rendered image size in pixels")
    parser.add_argument("--no_render", action="store_true", help="Skip rendering visualizations")
    parser.add_argument("--save_gif", action="store_true",
                        help="Save per-output logit-based diffusion GIFs and intermediate tensors")
    parser.add_argument("--gif_timesteps", type=int, default=40,
                        help="Number of denoising snapshots to save per output when --save_gif is enabled")
    parser.add_argument("--gif_fps", type=int, default=10,
                        help="Frames per second for saved diffusion GIFs")
    parser.add_argument("--gif_early_bias", type=float, default=2.0,
                        help="Bias GIF snapshots toward earlier denoising steps. 1.0 = uniform, >1 = earlier-heavy, <1 = later-heavy")
    parser.add_argument("--experiment_mode", type=str, default="standard", 
                        choices=["standard", "biome_swap", "village", "seeded_inpaint"],
                        help="Experiment mode: 'standard' for 2-samples-per-biome corner-context variants, 'biome_swap' for configurable-context target-biome variants, 'village' for village infilling tests, or 'seeded_inpaint' for synthetic human-guided seeding experiments")
    parser.add_argument("--random_sample", action="store_true",
                        help="Randomly select source samples from each multi-sample generated file instead of always taking the first ones")
    parser.add_argument("--village_source", type=str, default=None,
                        help="Path to village .pt file for village diversity experiments")
    parser.add_argument("--num_infills", type=int, default=5,
                        help="Number of infills per source sample for village diversity experiments")
    parser.add_argument("--seed_context_files", type=str, nargs="*", default=None,
                        help="Explicit seed context .pt files for seeded_inpaint mode")
    parser.add_argument("--seed_context_dir", type=str, default="seed_context_outputs",
                        help="Default directory to scan for seed context .pt files in seeded_inpaint mode")
    parser.add_argument("--seed_biomes", type=str, nargs="*", default=None,
                        help="Optional biome conditioning names for seeded_inpaint mode. If omitted, uses unconditional generation via the CFG null branch. Provide one per context file, one value to broadcast, or aliases like 'unconditional'/'none'.")
    parser.add_argument("--num_variants", type=int, default=1,
                        help="Number of variants to generate per source context in standard, biome_swap, or seeded_inpaint mode")
    parser.add_argument("--biome_swap_context_fraction", type=float, default=BIOME_SWAP_CONTEXT_FRACTION,
                        help="Fraction of each source chunk to keep as fixed context in biome_swap mode (e.g. 0.5 for half, 0.0625 for 1/16)")
    parser.add_argument("--source_biomes", type=str, default=None,
                        help="Comma-separated source biome filter for biome_swap mode, e.g. ocean,plains,village")
    parser.add_argument("--target_biomes", type=str, default=None,
                        help="Comma-separated target biome list for biome_swap mode, e.g. ice,desert,plains")
    
    args = parser.parse_args()
    if args.save_gif and args.gif_timesteps < 1:
        raise ValueError(f"--gif_timesteps must be >= 1 when --save_gif is enabled, got {args.gif_timesteps}")
    if args.gif_early_bias <= 0:
        raise ValueError(f"--gif_early_bias must be > 0, got {args.gif_early_bias}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    loaded = load_model_from_file(
        args.checkpoint,
        config_path=args.config,
        mappings_path=args.mappings,
        device=args.device,
        use_ema_if_available=True,
    )
    
    model = loaded["model"]
    converter = loaded["converter"]
    device = loaded["device"]
    config = loaded["config"]

    resolved_source_biomes = _resolve_biome_list(converter, args.source_biomes)
    resolved_target_biomes = _resolve_biome_list(converter, args.target_biomes)
    
    reverse_steps = args.reverse_steps or loaded.get("reverse_steps", 1000)
    cond_scale = args.cond_scale or loaded.get("default_cond_scale", 4.0)
    
    print(f"Model loaded: {loaded['num_blocks']} blocks, {loaded.get('num_classes', 'N/A')} classes")
    print(f"Image size: {loaded['image_size']}, Reverse steps: {reverse_steps}")
    
    # Setup accelerator - use simple creation to avoid state conflicts
    accelerator = Accelerator(mixed_precision="no")
    
    # Create inpainter
    num_blocks = loaded["num_blocks"]
    inpainter = MD4Inpainter(
        accelerator=accelerator,
        num_classes=num_blocks,
        device=device,
    )
    
    # Get air index for fallback
    try:
        air_idx = converter.get_air_block_index()
    except Exception:
        air_idx = 0
    
    # Seeded inpaint mode can synthesize its own source chunks, so it does not need --source.
    if args.experiment_mode == "seeded_inpaint":
        if args.source is not None or args.source_folder is not None:
            print("Ignoring --source/--source_folder for seeded_inpaint mode.")
        if args.biome is not None:
            print("Ignoring --biome for seeded_inpaint mode. Use --seed_biomes instead.")

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        resolved_seed_files = _resolve_seed_context_files(args.seed_context_files, args.seed_context_dir)
        resolved_seed_biomes = _resolve_seed_biome_requests(args.seed_biomes, len(resolved_seed_files))

        exp_config = {
            "checkpoint": args.checkpoint,
            "mappings": args.mappings,
            "reverse_steps": reverse_steps,
            "cond_scale": cond_scale,
            "num_blocks": num_blocks,
            "image_size": loaded["image_size"],
            "experiment_mode": args.experiment_mode,
            "mode": "file_seeded_inpaint",
            "seed_context_dir": args.seed_context_dir,
            "seed_context_files": [str(path) for path in resolved_seed_files],
            "seed_biomes": resolved_seed_biomes,
            "num_variants": args.num_variants,
            "save_gif": args.save_gif,
            "gif_timesteps": args.gif_timesteps,
            "gif_fps": args.gif_fps,
            "gif_early_bias": args.gif_early_bias,
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
            output_dir=output_dir,
            context_files=[str(path) for path in resolved_seed_files],
            biome_names=resolved_seed_biomes,
            num_variants=args.num_variants,
            reverse_steps=reverse_steps,
            cond_scale=cond_scale,
            air_idx=air_idx,
            textures_dir=args.textures_dir,
            image_size=args.image_size,
            render=not args.no_render,
            save_gif=args.save_gif,
            gif_timesteps=args.gif_timesteps,
            gif_fps=args.gif_fps,
            gif_early_bias=args.gif_early_bias,
        )
        print("\n" + "=" * 60)
        print("SEEDED INPAINT EXPERIMENTS COMPLETE")
        print(f"Results saved to: {output_dir / 'seeded_inpaint'}")
        print("=" * 60)
        print("\nExperiment Summary:")
        for exp_name in results.keys():
            print(f"  {exp_name}")
        return

    if args.experiment_mode in {"standard", "biome_swap"} and args.source_folder is None:
        raise ValueError(
            f"{args.experiment_mode} mode now requires --source_folder with generated_<biome>.pt files. "
            f"--source is no longer supported for {args.experiment_mode} mode."
        )

    # Validate source arguments
    if args.source is None and args.source_folder is None:
        raise ValueError("Must specify either --source (single file) or --source_folder (multi-biome folder)")
    if args.source is not None and args.source_folder is not None:
        raise ValueError("Cannot specify both --source and --source_folder. Choose one.")
    
    # Multi-biome folder mode
    if args.source_folder is not None:
        print(f"\nLoading biomes from folder: {args.source_folder}")
        if args.random_sample:
            print("  Using random sample selection from each file")
        else:
            print("  Using first sample from each file")
        sampled_contexts: Optional[List[Dict[str, Any]]] = None
        if args.experiment_mode in {"standard", "biome_swap"}:
            samples_per_biome = STANDARD_SAMPLES_PER_BIOME
            loader_random_sample = args.random_sample
            loader_biome_filter = None
            if args.experiment_mode == "biome_swap":
                samples_per_biome = 1
                loader_random_sample = True
                loader_biome_filter = resolved_source_biomes

            sampled_contexts, biome_names = _load_standard_biome_samples(
                args.source_folder,
                converter,
                device,
                samples_per_biome=samples_per_biome,
                random_sample=loader_random_sample,
                biome_filter=loader_biome_filter,
            )
            print(
                f"Loaded {len(sampled_contexts)} source samples "
                f"across {len(biome_names)} biomes"
            )
            if args.experiment_mode == "biome_swap":
                print("  Using one random source sample per selected source biome")
            source_indices = None
            source_labels = None
        else:
            source_indices, source_labels, biome_names = load_biomes_from_folder(
                args.source_folder, converter, device, random_sample=args.random_sample
            )
            print(f"Loaded {len(biome_names)} biomes, shape: {source_indices.shape}")
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save experiment config
        exp_config = {
            "checkpoint": args.checkpoint,
            "mappings": args.mappings,
            "source_folder": args.source_folder,
            "biomes": biome_names,
            "reverse_steps": reverse_steps,
            "cond_scale": cond_scale,
            "num_blocks": num_blocks,
            "image_size": loaded["image_size"],
            "experiment_mode": args.experiment_mode,
            "random_sample": args.random_sample,
            "mode": (
                "standard_biome_variants"
                if args.experiment_mode == "standard"
                else "biome_swap_variants"
                if args.experiment_mode == "biome_swap"
                else "multi_biome_batched"
            ),
            "samples_per_biome": (
                STANDARD_SAMPLES_PER_BIOME
                if args.experiment_mode == "standard"
                else 1
                if args.experiment_mode == "biome_swap"
                else None
            ),
            "num_variants": args.num_variants if args.experiment_mode in {"standard", "biome_swap"} else None,
            "context_shape": [
                STANDARD_CONTEXT_CORNER_SIZE,
                STANDARD_CONTEXT_HEIGHT,
                STANDARD_CONTEXT_CORNER_SIZE,
            ] if args.experiment_mode == "standard" else None,
            "context_fraction": args.biome_swap_context_fraction if args.experiment_mode == "biome_swap" else None,
            "mask_mode": "leading_x_context" if args.experiment_mode == "biome_swap" else None,
            "source_biomes": resolved_source_biomes if args.experiment_mode == "biome_swap" else None,
            "target_biomes": resolved_target_biomes if args.experiment_mode == "biome_swap" else None,
            "random_source_sample_per_biome": True if args.experiment_mode == "biome_swap" else None,
            "save_gif": args.save_gif,
            "gif_timesteps": args.gif_timesteps,
            "gif_fps": args.gif_fps,
            "gif_early_bias": args.gif_early_bias,
        }
        with open(output_dir / "experiment_config.json", "w") as f:
            json.dump(exp_config, f, indent=2)
        
        # Run experiments based on mode
        print("\n" + "="*60)
        if args.experiment_mode == "standard":
            print(
                f"STARTING STANDARD INPAINT VARIANTS "
                f"({len(biome_names)} biomes, {len(sampled_contexts) if sampled_contexts is not None else 0} source maps)"
            )
            print("="*60)

            results = run_batched_inpainting_experiments(
                model=model,
                inpainter=inpainter,
                samples=sampled_contexts if sampled_contexts is not None else [],
                converter=converter,
                output_dir=output_dir,
                num_variants=args.num_variants,
                reverse_steps=reverse_steps,
                cond_scale=cond_scale,
                air_idx=air_idx,
                textures_dir=args.textures_dir,
                image_size=args.image_size,
                render=not args.no_render,
                save_gif=args.save_gif,
                gif_timesteps=args.gif_timesteps,
                gif_fps=args.gif_fps,
                gif_early_bias=args.gif_early_bias,
            )
        elif args.experiment_mode == "biome_swap":
            print(
                f"STARTING BIOME SWAP VARIANTS "
                f"({len(biome_names)} biomes, {len(sampled_contexts) if sampled_contexts is not None else 0} source maps)"
            )
            print("="*60)
            
            results = run_biome_swap_experiments(
                model=model,
                inpainter=inpainter,
                samples=sampled_contexts if sampled_contexts is not None else [],
                available_biomes=biome_names,
                converter=converter,
                output_dir=output_dir,
                num_variants=args.num_variants,
                context_fraction=args.biome_swap_context_fraction,
                target_biomes=resolved_target_biomes,
                reverse_steps=reverse_steps,
                cond_scale=cond_scale,
                air_idx=air_idx,
                textures_dir=args.textures_dir,
                image_size=args.image_size,
                render=not args.no_render,
                save_gif=args.save_gif,
                gif_timesteps=args.gif_timesteps,
                gif_fps=args.gif_fps,
                gif_early_bias=args.gif_early_bias,
            )
        elif args.experiment_mode == "village":
            print(f"STARTING VILLAGE EXPERIMENTS ({len(biome_names)} biomes as ring context)")
            print("="*60)
            
            results = run_village_experiments(
                model=model,
                inpainter=inpainter,
                source_indices=source_indices,
                source_labels=source_labels,
                biome_names=biome_names,
                converter=converter,
                output_dir=output_dir,
                village_source=args.village_source,
                num_infills=args.num_infills,
                reverse_steps=reverse_steps,
                cond_scale=cond_scale,
                air_idx=air_idx,
                textures_dir=args.textures_dir,
                image_size=args.image_size,
                save_gif=args.save_gif,
                gif_timesteps=args.gif_timesteps,
                gif_fps=args.gif_fps,
                gif_early_bias=args.gif_early_bias,
            )
        print("\n" + "="*60)
        print(f"{args.experiment_mode.upper()} EXPERIMENTS COMPLETE")
        if args.experiment_mode == "standard":
            print(f"Results saved to: {output_dir / 'standard'}")
        elif args.experiment_mode == "biome_swap":
            print(f"Results saved to: {output_dir / 'biome_swap'}")
        else:
            print(f"Results saved to: {output_dir}")
        print("="*60)
        
        print("\nExperiment Summary:")
        if args.experiment_mode in {"standard", "biome_swap"}:
            for biome_name, sample_results in results.items():
                total_variants = sum(len(sample["variant_files"]) for sample in sample_results.values())
                print(
                    f"  {biome_name}: {len(sample_results)} source samples, "
                    f"{total_variants} generated variants"
                )
        else:
            for exp_name in results.keys():
                print(f"  {exp_name}: {len(results[exp_name])} samples processed")
        
        return
    
    # Single file mode
    print(f"\nLoading source sample from {args.source}...")
    source_indices, source_labels = load_source_sample(args.source, converter, device)
    print(f"Source shape: {source_indices.shape}")
    
    # Resolve biome condition
    # Priority: --biome argument > source file label > error
    if args.biome is not None:
        # User specified biome explicitly
        biome_arg = args.biome.strip()
        
        # Try to parse as integer index first
        try:
            biome_idx = int(biome_arg)
            if biome_idx < 0 or biome_idx >= loaded.get("num_classes", 0):
                raise ValueError(f"Biome index {biome_idx} out of range [0, {loaded.get('num_classes', 0) - 1}]")
            source_labels = torch.tensor([biome_idx], dtype=torch.long, device=device)
            biome_name = converter.index_to_biome.get(biome_idx, str(biome_idx)) if converter.index_to_biome else str(biome_idx)
            print(f"Using biome from --biome argument: {biome_name} (index {biome_idx})")
        except ValueError:
            # Try to look up by name
            if converter.biome_to_index is not None and biome_arg in converter.biome_to_index:
                biome_idx = converter.biome_to_index[biome_arg]
                source_labels = torch.tensor([biome_idx], dtype=torch.long, device=device)
                print(f"Using biome from --biome argument: {biome_arg} (index {biome_idx})")
            else:
                available = list(converter.biome_to_index.keys()) if converter.biome_to_index else []
                raise ValueError(
                    f"Unknown biome '{biome_arg}'. "
                    f"Available biomes: {available[:10]}{'...' if len(available) > 10 else ''}"
                )
    elif source_labels is not None:
        # Use label from source file
        label_idx = source_labels[0].item()
        label_name = converter.index_to_biome.get(label_idx, str(label_idx)) if converter.index_to_biome else str(label_idx)
        print(f"Using biome from source file: {label_name} (index {label_idx})")
    else:
        # No biome specified - error for conditional models
        if model.class_conditional:
            available = list(converter.biome_to_index.keys()) if converter.biome_to_index else []
            raise ValueError(
                f"Model is biome-conditional but no biome specified. "
                f"Use --biome to specify, or ensure source file has 'classes'/'biomes' key. "
                f"Available biomes: {available[:10]}{'...' if len(available) > 10 else ''}"
            )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Resolve biome name for logging
    biome_idx = source_labels[0].item() if source_labels is not None else None
    biome_name = converter.index_to_biome.get(biome_idx, str(biome_idx)) if (converter.index_to_biome and biome_idx is not None) else str(biome_idx)
    
    # Save experiment config
    exp_config = {
        "checkpoint": args.checkpoint,
        "mappings": args.mappings,
        "source": args.source,
        "reverse_steps": reverse_steps,
        "cond_scale": cond_scale,
        "num_blocks": num_blocks,
        "image_size": loaded["image_size"],
        "biome_index": biome_idx,
        "biome_name": biome_name,
        "save_gif": args.save_gif,
        "gif_timesteps": args.gif_timesteps,
        "gif_fps": args.gif_fps,
        "gif_early_bias": args.gif_early_bias,
    }
    with open(output_dir / "experiment_config.json", "w") as f:
        json.dump(exp_config, f, indent=2)
    
    print(f"\nBiome condition: {biome_name} (index {biome_idx})")
    
    # Run experiments
    print("\n" + "="*60)
    print("STARTING INPAINTING EXPERIMENTS")
    print("="*60)
    
    results = run_inpainting_experiment(
        model=model,
        inpainter=inpainter,
        source_indices=source_indices,
        source_labels=source_labels,
        converter=converter,
        output_dir=output_dir,
        reverse_steps=reverse_steps,
        cond_scale=cond_scale,
        air_idx=air_idx,
        textures_dir=args.textures_dir,
        image_size=args.image_size,
        save_gif=args.save_gif,
        gif_timesteps=args.gif_timesteps,
        gif_fps=args.gif_fps,
        gif_early_bias=args.gif_early_bias,
    )
    
    print("\n" + "="*60)
    print("EXPERIMENTS COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("="*60)
    
    # Summary
    print("\nExperiment Summary:")
    for exp_name, exp_data in results.items():
        frac = exp_data["fraction_inpainted"]
        print(f"  {exp_name}: {(1-frac)*100:.1f}% context, {frac*100:.1f}% inpainted")


if __name__ == "__main__":
    main()
