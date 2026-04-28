# Taken from https://github.com/darioShar/pytorch-md4/blob/main/pytorch_md4.py

import wandb
import math
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.distributed as dist

# import model.unet as unet
from pathlib import Path
from accelerate import Accelerator
from ema_pytorch import EMA
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from transformers import get_wsd_schedule
from visualization_utils import save_chunks, MinecraftVisualizerPyVista
# from cleanfid import fid as cleanfid
from data_utils import BlockBiomeConverter, rotate_voxels_90_fix_stairs_torch
from render_dataset import render_dataset_to_images
from accelerate.utils import TorchDynamoPlugin
import logging
import os
from copy import deepcopy
import json
import time

logger = logging.getLogger(__name__)
# local helpers / metadata
__version__ = "md4-3d-0.1"

def exists(x):
    return x is not None

def divisible_by(numer, denom):
    return (numer % denom) == 0

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

# Voxel dataset
class VoxelDatasetConditional(torch.utils.data.Dataset):
    def __init__(self, file_path, one_hot_on_load=False, mappings_file_path=None, rotation_aug_prob: float = 0.0):
        super().__init__()
        self.file_path = file_path
        self.one_hot_on_load = one_hot_on_load
        self.mappings_file_path = mappings_file_path
        self.rotation_aug_prob = float(rotation_aug_prob)
        self._warned_one_hot_rotation = False
        
        data = torch.load(self.file_path)
        voxels = data['voxels'].contiguous()
        self.biomes = data['biomes'].contiguous()  # Class labels (one-hot or indices)

        # Keep voxels dtype; may be one-hot/embeddings [B,C,H,W,D] (float) or indices [B,H,W,D] (int)
        self.voxels = voxels
        
        print(f'Dataset: orig biomes shape: {self.biomes.shape}')
        # Convert one-hot to class indices if needed
        if self.biomes.dim() > 1 and self.biomes.shape[-1] > 1:
            self.biomes = torch.argmax(self.biomes, dim=-1)

        print(f'Dataset: Voxels shape: {self.voxels.shape}')
        print(f'Dataset: biomes shape: {self.biomes.shape}')

    def __len__(self):
        return self.voxels.shape[0]

    def __getitem__(self, index):
        voxel_data = self.voxels[index]
        if self.rotation_aug_prob > 0.0:
            if voxel_data.dtype.is_floating_point:
                if not self._warned_one_hot_rotation:
                    print("[VoxelDatasetConditional] rotation augmentation skipped because voxels are one-hot; set one_hot_on_load=False to enable orientation-aware rotations.")
                    self._warned_one_hot_rotation = True
            else:
                if torch.rand(1).item() < self.rotation_aug_prob:
                    k = int(torch.randint(1, 4, (1,), device=voxel_data.device).item())
                    voxel_data = rotate_voxels_90_fix_stairs_torch(voxel_data, k=k)
        class_label = self.biomes[index]
        return voxel_data, class_label


class VoxelDatasetMemmapConditional(torch.utils.data.Dataset):
    def __init__(self, dir_path, one_hot_on_load=False, mappings_file_path=None, rotation_aug_prob: float = 0.0):
        super().__init__()
        self.dir_path = str(dir_path)
        self.one_hot_on_load = one_hot_on_load
        self.mappings_file_path = mappings_file_path
        self.rotation_aug_prob = float(rotation_aug_prob)
        self._warned_one_hot_rotation = False

        manifest_path = os.path.join(self.dir_path, 'manifest.json')
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"manifest.json not found in {self.dir_path}")
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        voxels_rel = manifest['paths']['voxels']
        labels_rel = manifest['paths']['biome_labels']
        self.num_blocks = int(manifest.get('num_blocks', 0) or 0)
        self.num_classes = int(manifest.get('num_classes', 0) or 0)
        self.labels_format = manifest.get('class_labels_format', 'indices')

        vox_path = os.path.join(self.dir_path, voxels_rel)
        lbl_path = os.path.join(self.dir_path, labels_rel)

        # Load as numpy memmaps
        self.voxels_np = np.load(str(vox_path), mmap_mode='r')
        self.biomes_np = np.load(str(lbl_path), mmap_mode='r')

        # Shapes for quick checks
        self._is_label_one_hot = (self.biomes_np.ndim == 2)

        print(f"Memmap dataset loaded: {self.voxels_np.shape}, labels shape={self.biomes_np.shape}")

    def __len__(self):
        return int(self.voxels_np.shape[0])

    def __getitem__(self, index):
        # Voxels are indices [H,W,D] (np.int32). Convert to torch.
        vx_np = self.voxels_np[index]
        # vox = torch.from_numpy(vx_np)
        vox = torch.tensor(vx_np)

        # Optional augmentation only for index grids
        if self.rotation_aug_prob > 0.0:
            if vox.dtype.is_floating_point:
                if not self._warned_one_hot_rotation:
                    print("[VoxelDatasetMemmapConditional] rotation augmentation skipped because voxels are one-hot; set one_hot_on_load=False to enable orientation-aware rotations.")
                    self._warned_one_hot_rotation = True
            else:
                if torch.rand(1).item() < self.rotation_aug_prob:
                    k = int(torch.randint(1, 4, (1,), device=vox.device).item())
                    vox = rotate_voxels_90_fix_stairs_torch(vox, k=k)

        # If one_hot_on_load is requested, expand to [C,H,W,D]
        if self.one_hot_on_load:
            if self.num_blocks in (None, 0):
                raise ValueError("num_blocks not found in manifest; cannot one-hot encode on load")
            vox = F.one_hot(vox.long(), num_classes=int(self.num_blocks)).permute(3, 0, 1, 2).float()

        # Labels: either indices or one-hot
        if self._is_label_one_hot or (self.labels_format == 'one_hot'):
            lbl_np = self.biomes_np[index]
            # np.memmap -> numpy array; convert to tensor then argmax
            lbl = torch.from_numpy(np.asarray(lbl_np)).long()
            class_label = torch.argmax(lbl).long()
        else:
            # 1D labels array; indexing returns scalar-like
            lbl_val = self.biomes_np[index]
            # Ensure tensor long
            class_label = torch.tensor(int(lbl_val)).long()

        return vox, class_label

#############################################
# Helper function and MD4Generation class
#############################################

def match_last_dims(data, size):
    """
    Repeat a 1D tensor so that its last dimensions [1:] match `size[1:]`.
    Useful for working with batched data.
    """
    assert len(data.size()) == 1, "Data must be 1-dimensional (one value per batch)"
    for _ in range(len(size) - 1):
        data = data.unsqueeze(-1)
    return data.repeat(1, *(size[1:]))



class MD4Discrete3D:
    def __init__(self, accelerator, num_classes, device, mask_token_id=None):
        self.num_classes = int(num_classes)
        self.device = device
        self.accelerator = accelerator
        # reserve mask token id; default to K (append mask channel)
        self.mask_token_id = int(num_classes) if mask_token_id is None else int(mask_token_id)

        # masking schedule and loss weight per MD4
        self.masking_schedule = lambda t: 1 - torch.cos((1 - t) * math.pi / 2)
        self.ce_loss_weight = lambda t: math.pi * torch.tan((1 - t) * math.pi / 2) / 2

    def _match_last_dims(self, data, size):
        assert len(data.size()) == 1
        out = data
        for _ in range(len(size) - 1):
            out = out.unsqueeze(-1)
        return out.repeat(1, *(size[1:]))

    def _compute_mask_prob(self, ti, si, spatial_shape):
        """
        Compute the unmask probability between times si and ti with numerical guards.
        """
        alphat = self.masking_schedule(ti)
        alphas = self.masking_schedule(si)
        # avoid division by zero when ti == 1
        denom = torch.clamp(1 - alphat, min=1e-6)
        first_factor = (alphas - alphat) / denom
        first_factor = torch.clamp(first_factor, 0.0, 1.0)
        return self._match_last_dims(first_factor, spatial_shape)

    def _sanitize_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Replace NaN/Inf with finite sentinels to stabilize softmax, preserving dtype.
        """
        return torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

    def _probs_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert logits [B,K,H,W,D] to probabilities with guards and renormalization.
        """
        logits = self._sanitize_logits(logits)
        probs = torch.softmax(logits, dim=1)
        probs = torch.nan_to_num(probs, nan=0.0)
        # ensure rows sum to 1 across class dim
        sum_p = probs.sum(dim=1, keepdim=True)
        probs = probs / torch.clamp(sum_p, min=1e-8)
        probs = torch.clamp(probs, min=0.0)
        # renormalize after clamp
        sum_p2 = probs.sum(dim=1, keepdim=True)
        probs = probs / torch.clamp(sum_p2, min=1e-8)
        return probs

    def _safe_multinomial(self, probs: torch.Tensor, B: int, H: int, W: int, D: int) -> torch.Tensor:
        """
        Sample from probs [B,K,H,W,D] with uniform fallback for invalid rows.
        Returns indices shaped [B,H,W,D].
        """
        # flatten to [B*N, K]
        K = self.num_classes
        probs_f = probs.view(B, K, -1).transpose(1, 2).contiguous().view(-1, K)
        probs_f = torch.nan_to_num(probs_f, nan=0.0)
        row_sums = probs_f.sum(dim=1, keepdim=True)
        # uniform fallback for rows with zero/invalid sums
        uniform = torch.full_like(probs_f, 1.0 / float(K))
        bad = (row_sums <= 0) | (~torch.isfinite(row_sums))
        safe_probs = torch.where(bad, uniform, probs_f / torch.clamp(row_sums, min=1e-8))
        sampled = torch.multinomial(safe_probs, num_samples=1).view(B, -1)
        return sampled.view(B, H, W, D)

    def _force_unmask_remaining(self, model, xt, class_cond=None, cond_scale=1.0):
        """Force-reveal any residual mask tokens using one final model prediction.
        """
        remaining_mask = (xt == self.mask_token_id)
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
    def sample(self, model, batch_size, spatial_size, reverse_steps=1000, progress=True, air_index_fallback=0, class_cond=None, cond_scale=1.0, return_intermediates: bool = False, force_unmask_final: bool = True):
        # spatial_size: (H, W, D)
        H, W, D = spatial_size
        device = self.device

        # start fully masked
        xt = (torch.ones((batch_size, H, W, D), device=device, dtype=torch.long) * self.mask_token_id)
        intermediates = [] if return_intermediates else None

        # time grid
        tgrid = torch.linspace(0, 1, reverse_steps + 1, device=device)
        rng = tqdm(range(reverse_steps, 1, -1)) if progress else range(reverse_steps, 1, -1)

        for i in rng:
            ti = torch.ones((batch_size,), device=device) * tgrid[i]
            si = torch.ones((batch_size,), device=device) * tgrid[i - 1]

            # compute unmask prob between si and ti (guarded)
            mask_prob = self._compute_mask_prob(ti, si, (batch_size, H, W, D))

            # build one-hot input with mask channel
            x_in = F.one_hot(torch.clamp(xt, 0, self.mask_token_id), num_classes=self.mask_token_id + 1)  # [B,H,W,D,K+1]
            x_in = x_in.permute(0, 4, 1, 2, 3).float()  # [B,K+1,H,W,D]
            x_in = x_in.permute(0, 1, 4, 2, 3)  # [B,K+1,D,H,W]

            # model predicts logits over K real classes (no mask)
            # allow optional class conditioning + classifier-free guidance
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

            # sample categories
            # flatten spatial for multinomial sampling per-voxel
            B = batch_size
            sampled = self._safe_multinomial(probs, B, H, W, D)

            # select positions to unmask
            mask_t = (xt == self.mask_token_id)
            force_reveal_now = force_unmask_final and (i == 2)
            if force_reveal_now:
                update_mask = mask_t
            else:
                to_unmask = (torch.rand((batch_size, H, W, D), device=device) < mask_prob)
                update_mask = mask_t & to_unmask
            xt[update_mask] = sampled[update_mask]

            if return_intermediates:
                intermediates.append(xt.clone().detach())

            del x_in, logits, probs, sampled, mask_prob, ti, si, mask_t, update_mask
            if not force_reveal_now:
                del to_unmask

        remaining_mask = (xt == self.mask_token_id)
        if force_unmask_final and remaining_mask.any():
            xt, num_forced = self._force_unmask_remaining(
                model, xt, class_cond=class_cond, cond_scale=cond_scale
            )
            if progress and num_forced > 0:
                print(f"Forced final unmask for {num_forced} residual masked voxels.")
            if return_intermediates:
                intermediates.append(xt.clone().detach())

        remaining_mask = (xt == self.mask_token_id)
        if remaining_mask.any():
            xt[remaining_mask] = int(air_index_fallback)
        if return_intermediates:
            return xt, torch.stack(intermediates, dim=1)
        return xt  # [B,H,W,D] integer indices

    def sample_shape(self, model, shape, spatial_size, batch_size, reverse_steps, progress=True, air_index_fallback=0, class_cond=None, cond_scale=1.0, step=1, temp=0.8, force_unmask_final: bool = True):
        # spatial_size: (H, W, D)
        H_small, W_small, D_small = spatial_size
        H_target, W_target, D_target = shape
        
        x_lim = H_target - H_small
        y_lim = W_target - W_small
        z_lim = D_target - D_small
        
        device = self.device

        # start fully masked
        xt = (torch.ones((batch_size,) + shape, device=device, dtype=torch.long) * self.mask_token_id)
        
        # time grid
        tgrid = torch.linspace(0, 1, reverse_steps + 1, device=device)
        rng = tqdm(range(reverse_steps, 1, -1)) if progress else range(reverse_steps, 1, -1)

        for t_step in rng:
            # keep track of PoE probabilities
            # shape: [B, H, W, D, K]
            x_0_probs = torch.zeros((batch_size,) + shape + (self.num_classes,), device=device)
            
            ti = torch.ones((batch_size,), device=device) * tgrid[t_step]
            si = torch.ones((batch_size,), device=device) * tgrid[t_step - 1]

            # compute unmask prob between si and ti
            mask_prob = self._compute_mask_prob(ti, si, (batch_size,) + shape)

            # build one-hot input with mask channel
            # xt: [B, H, W, D]
            x_in_full = F.one_hot(torch.clamp(xt, 0, self.mask_token_id), num_classes=self.mask_token_id + 1)  # [B,H,W,D,K+1]
            x_in_full = x_in_full.permute(0, 4, 1, 2, 3).float()  # [B,K+1,H,W,D]
            x_in_full = x_in_full.permute(0, 1, 4, 2, 3)  # [B,K+1,D,H,W]

            # Sliding window
            for i in range(0, x_lim + 1, step):
                for j in range(0, y_lim + 1, step):
                    for k in range(0, z_lim + 1, step):
                        
                        # Slice input: [B, C, D, H, W]
                        # i->H (dim 3), j->W (dim 4), k->D (dim 2)
                        x_t_part = x_in_full[:, :, k:k+D_small, i:i+H_small, j:j+W_small]
                        
                        # Model forward
                        with self.accelerator.autocast():
                            if class_cond is not None:
                                if hasattr(model, 'forward_with_cond_scale'):
                                    logits_part, _ = model.forward_with_cond_scale(x_t_part, ti, class_cond, cond_scale=cond_scale)
                                else:
                                    logits_part = model(x_t_part, ti, class_cond)
                            else:
                                logits_part = model(x_t_part, ti)
                        
                        # logits_part: [B, K, D, H, W] -> [B, K, H, W, D]
                        logits_part = logits_part.permute(0, 1, 3, 4, 2).contiguous()
                        probs_part = self._probs_from_logits(logits_part) # [B, K, H, W, D]
                        
                        # Accumulate probs
                        # Permute probs to [B, H, W, D, K] to match x_0_probs
                        probs_part = probs_part.permute(0, 2, 3, 4, 1)
                        x_0_probs[:, i:i+H_small, j:j+W_small, k:k+D_small, :] += probs_part

            # Mixture with Temperature
            # Normalize accumulated probabilities
            x_0_probs = x_0_probs / torch.clamp(x_0_probs.sum(-1, keepdim=True), min=1e-8)
            
            C_val = torch.tensor(x_0_probs.size(-1)).float()
            x_0_probs = torch.softmax((torch.log(torch.clamp(x_0_probs, min=1e-8)) + torch.log(C_val)) / temp, dim=-1)
            
            # Sample
            # _safe_multinomial expects [B, K, H, W, D]
            x_0_probs_permuted = x_0_probs.permute(0, 4, 1, 2, 3)
            sampled = self._safe_multinomial(x_0_probs_permuted, batch_size, H_target, W_target, D_target)

            # Update mask
            mask_t = (xt == self.mask_token_id)
            force_reveal_now = force_unmask_final and (t_step == 2)
            if force_reveal_now:
                update_mask = mask_t
            else:
                to_unmask = (torch.rand((batch_size,) + shape, device=device) < mask_prob)
                update_mask = mask_t & to_unmask
            xt[update_mask] = sampled[update_mask]

            del x_in_full, x_0_probs, x_0_probs_permuted, sampled, mask_prob, ti, si, mask_t, update_mask
            if not force_reveal_now:
                del to_unmask

        remaining_mask = (xt == self.mask_token_id)
        if force_unmask_final and remaining_mask.any():
            xt, num_forced = self._force_unmask_remaining(
                model, xt, class_cond=class_cond, cond_scale=cond_scale
            )
            if progress and num_forced > 0:
                print(f"Forced final unmask for {num_forced} residual masked voxels.")

        remaining_mask = (xt == self.mask_token_id)
        if remaining_mask.any():
            xt[remaining_mask] = int(air_index_fallback)
        return xt  # [B,H,W,D] integer indices


    @torch.inference_mode()
    def ddim_sample(self, model, batch_size, spatial_size, sampling_timesteps, *, reverse_steps=1000, progress=True, air_index_fallback=0, class_cond=None, cond_scale=1.0, return_intermediates: bool = False, force_unmask_final: bool = True):
        """
        Reduced-step sampling analogous to DDIM: select a coarse schedule of time indices
        and perform MD4 updates only at those indices.
        """
        H, W, D = spatial_size
        device = self.device

        # start fully masked
        xt = (torch.ones((batch_size, H, W, D), device=device, dtype=torch.long) * self.mask_token_id)
        intermediates = [] if return_intermediates else None

        # original time grid [0..1]
        tgrid = torch.linspace(0, 1, reverse_steps + 1, device=device)
        # coarse indices from 0..reverse_steps
        idxs = torch.linspace(0, reverse_steps, steps=int(sampling_timesteps) + 1, device=device).round().long().tolist()
        # ensure uniqueness and proper bounds
        idxs = sorted(set(idxs))
        if idxs[-1] != reverse_steps:
            idxs.append(reverse_steps)
        if idxs[0] == 0:
            start = len(idxs) - 1
        else:
            idxs = [0] + idxs
            start = len(idxs) - 1

        rng = tqdm(range(start, 1, -1)) if progress else range(start, 1, -1)
        for k in rng:
            i_idx = idxs[k]
            s_idx = idxs[k - 1]
            ti = torch.ones((batch_size,), device=device) * tgrid[i_idx]
            si = torch.ones((batch_size,), device=device) * tgrid[s_idx]

            # compute unmask prob between si and ti
            mask_prob = self._compute_mask_prob(ti, si, (batch_size, H, W, D))

            # one-hot with mask channel
            x_in = F.one_hot(torch.clamp(xt, 0, self.mask_token_id), num_classes=self.mask_token_id + 1)
            x_in = x_in.permute(0, 4, 1, 2, 3).float()
            x_in = x_in.permute(0, 1, 4, 2, 3)

            # logits (conditional+CFG if provided) with mixed precision
            with self.accelerator.autocast():
                if class_cond is not None and hasattr(model, 'forward_with_cond_scale'):
                    logits, _ = model.forward_with_cond_scale(x_in, ti, class_cond, cond_scale=cond_scale)
                elif class_cond is not None:
                    logits = model(x_in, ti, class_cond)
                else:
                    logits = model(x_in, ti)
            logits = logits.permute(0, 1, 3, 4, 2).contiguous()
            probs = self._probs_from_logits(logits)

            # sample categories per voxel
            B = batch_size
            sampled = self._safe_multinomial(probs, B, H, W, D)

            # select positions to unmask
            mask_t = (xt == self.mask_token_id)
            force_reveal_now = force_unmask_final and (k == 2)
            if force_reveal_now:
                update_mask = mask_t
            else:
                to_unmask = (torch.rand((batch_size, H, W, D), device=device) < mask_prob)
                update_mask = mask_t & to_unmask
            xt[update_mask] = sampled[update_mask]

            if return_intermediates:
                intermediates.append(xt.clone().detach())

            del x_in, logits, probs, sampled, mask_prob, ti, si, mask_t, update_mask
            if not force_reveal_now:
                del to_unmask

        remaining_mask = (xt == self.mask_token_id)
        if force_unmask_final and remaining_mask.any():
            xt, num_forced = self._force_unmask_remaining(
                model, xt, class_cond=class_cond, cond_scale=cond_scale
            )
            if progress and num_forced > 0:
                print(f"Forced final unmask for {num_forced} residual masked voxels.")
            if return_intermediates:
                intermediates.append(xt.clone().detach())

        remaining_mask = (xt == self.mask_token_id)
        if remaining_mask.any():
            xt[remaining_mask] = int(air_index_fallback)
        if return_intermediates:
            return xt, torch.stack(intermediates, dim=1)
        return xt


    def training_loss(self, model, x_indices, class_cond=None):
        # x_indices: [B,H,W,D] ints in [0..K-1]
        device = x_indices.device
        B, H, W, D = x_indices.shape

        # sample times per example
        t = torch.rand((B,), device=device)
        # masking probability per-voxel
        mask_prob = 1 - self.masking_schedule(t)
        mask_prob = self._match_last_dims(mask_prob, (B, H, W, D))
        mask = (torch.rand((B, H, W, D), device=device) < mask_prob)

        # create masked input
        x_masked = x_indices.clone()
        x_masked[mask] = self.mask_token_id

        # one-hot with mask channel
        x_in = F.one_hot(torch.clamp(x_masked, 0, self.mask_token_id), num_classes=self.mask_token_id + 1)  # [B,H,W,D,K+1]
        x_in = x_in.permute(0, 4, 1, 2, 3).float()  # [B,K+1,H,W,D]
        x_in = x_in.permute(0, 1, 4, 2, 3)  # [B,K+1,D,H,W]

        # forward (optionally conditional)
        if class_cond is not None:
            logits = model(x_in, t, class_cond)  # [B,K,D,H,W]
        else:
            logits = model(x_in, t)  # [B,K,D,H,W]
        logits = logits.permute(0, 1, 3, 4, 2).contiguous()  # [B,K,H,W,D]

        # per-voxel cross-entropy
        ce = F.cross_entropy(logits, x_indices, reduction='none')  # [B,H,W,D]

        # weight by t-dependent factor and mask
        weights = self.ce_loss_weight(t)
        weights = self._match_last_dims(weights, (B, H, W, D))
        loss = (weights * ce)[mask]

        # avoid empty mask (rare at very low t) - fall back to mean over all
        if loss.numel() == 0:
            loss = (weights * ce).mean()
        else:
            loss = loss.mean()

        return loss


class VoxelTrainerMD4:
    def __init__(
        self,
        model,
        dataset,
        results_folder,
        *,
        train_batch_size = 8,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        optimizer = 'adamw',
        adam_betas = (0.9, 0.99),
        weight_decay = 0.01,
        warmup_steps = 0,
        scheduler = 'cosine',
        min_lr_ratio = 0.0,
        save_and_sample_every = 1000,
        num_samples = 16,
        amp = False,
        mixed_precision_type = 'fp16',
        compile_model = False,
        max_grad_norm = 1.,
        evaluate_ema_model = True,
        mappings_file_path = None,
        run_name = 'md4_discrete',
        save_only_last_checkpoint = False,
        reverse_steps = 1000,
        sampling_timesteps = None,
        # optional conditioning config
        num_classes = None,
        default_cond_scale = None,
        village_label = 'village',
        village_label_index = None,
        # validation support
        val_dataloader = None,
        val_dataset = None,
        val_batch_size = None,
        val_every_n_steps = None,
        val_max_batches = None,
        # new validation controls
        val_steps = None,
        val_progress = True,
        use_wandb = False,
        wandb_log_images = True,
        # FID options
        # fid_ref_images_dir = None,
        # fid_num_samples = None,
        # fid_every_steps = None,
        # fid_textures_dir = 'block_textures/',
        # fid_batch_size = 32,
        # fid_num_workers = 2,
    ):
        super().__init__()

        if compile_model:
            dynamo_plugin = TorchDynamoPlugin(
                backend="inductor",
                mode="default",
                dynamic=False,
            )
        else:
            dynamo_plugin = None

        self.accelerator = Accelerator(
            split_batches = False,
            mixed_precision = mixed_precision_type if amp else 'no',
            step_scheduler_with_optimizer=False,
            dynamo_plugin=dynamo_plugin,
        )

        self.model = model
        self.save_and_sample_every = save_and_sample_every
        self.save_only_last_checkpoint = save_only_last_checkpoint
        self.num_samples = num_samples
        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        self.max_grad_norm = max_grad_norm
        self.evaluate_ema_model = evaluate_ema_model
        self.mappings_file_path = mappings_file_path
        self.run_name = run_name
        self.reverse_steps = int(reverse_steps)
        self.sampling_timesteps = None if sampling_timesteps in (None, 0) else int(sampling_timesteps)

        # optional conditioning
        self.num_classes = None if num_classes in (None, 0) else int(num_classes)
        self.default_cond_scale = None if default_cond_scale in (None, 0) else float(default_cond_scale)
        self.village_label = village_label
        self.village_label_index = None if village_label_index is None else int(village_label_index)


        # loss tracking
        self.losses = []
        self.steps = []
        # validation tracking
        self.val_losses = []
        self.val_steps = []
        self.best_val_loss = float('inf')
        self.best_val_step = None

        # dataset and dataloader
        self.ds = dataset
        assert len(self.ds) > 0, 'your dataset is empty'
        num_workers = 2
        min(8, os.cpu_count() // max(1, torch.cuda.device_count()))
        dl = DataLoader(self.ds, batch_size=train_batch_size, shuffle=True, pin_memory=True,
                num_workers=num_workers, persistent_workers=True, drop_last=True,
                prefetch_factor=2)
        # dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = num_workers)

        # validation
        self.val_max_batches = None if val_max_batches in (None, 0) else int(val_max_batches)
        _val_loader = None
        if val_dataset is not None:
            _val_loader = DataLoader(val_dataset, batch_size = (val_batch_size or train_batch_size), shuffle = False, pin_memory = True, num_workers = num_workers)
        elif val_dataloader is not None:
            _val_loader = val_dataloader
        self.val_dl = None if _val_loader is None else self.accelerator.prepare(_val_loader)

        # decide validation interval (steps) and tqdm behavior
        if val_steps not in (None, 0):
            self.val_interval_steps = int(val_steps)
        elif val_every_n_steps not in (None, 0):
            self.val_interval_steps = int(val_every_n_steps)
        else:
            self.val_interval_steps = int(save_and_sample_every)
        self.val_progress = bool(val_progress)

        # optimizer
        if optimizer.lower() == 'adamw':
            self.opt = AdamW(self.model.parameters(), lr = train_lr, betas = adam_betas, weight_decay = weight_decay)
        elif optimizer.lower() == 'adam':
            if weight_decay != 0:
                raise ValueError('Adam does not support weight decay currently')
            self.opt = Adam(self.model.parameters(), lr = train_lr, betas = adam_betas)
        else:
            raise ValueError(f'Unknown optimizer {optimizer}')

        self.scheduler = get_wsd_schedule(
            optimizer=self.opt,
            num_warmup_steps=warmup_steps,
            num_decay_steps=train_num_steps - warmup_steps,
            num_training_steps=train_num_steps,
            min_lr_ratio=min_lr_ratio,

            decay_type=scheduler.lower(),
        )

        self.results_folder = Path(results_folder)

        self.step = 0

        # prepare model and optimizer
        self.model, self.opt, self.scheduler, dl = self.accelerator.prepare(
            self.model, self.opt, self.scheduler, dl)
        
        base_model = self.accelerator.unwrap_model(
            self.model,
            keep_torch_compile=True,
            )
        # your ema wrapper seems to expose .ema_model and .update()
        self.ema = EMA(base_model, beta=ema_decay, update_every=ema_update_every)  # or your ctor
        self.ema.ema_model.to(self.accelerator.device)
        self.ema.ema_model.eval().requires_grad_(False)

        self.dl = cycle(dl)

        # converter for visualization and block-id mapping
        self.converter = None
        if self.mappings_file_path is not None:
            mappings = torch.load(self.mappings_file_path, weights_only=False)
            self.converter = BlockBiomeConverter(mappings['block_mappings'], mappings['biome_mappings'])

        # lazy init after seeing first batch
        self._md4 = None
        self._spatial_size = None  # (H,W,D)
        self.use_wandb = use_wandb
        self.wandb_log_images = bool(wandb_log_images)

        # FID configuration
        # self.fid_ref_images_dir = fid_ref_images_dir
        # self.fid_num_samples = None if fid_num_samples in (None, 0) else int(fid_num_samples)
        # self.fid_every_steps = None if fid_every_steps in (None, 0) else int(fid_every_steps)
        # self.fid_textures_dir = fid_textures_dir
        # self.fid_batch_size = int(fid_batch_size)
        # self.fid_num_workers = int(fid_num_workers)
        # self.fid_scores = []
        # self.fid_steps = []


    def _align_scheduler_to_step(self, step_value: int):
        """
        Align LR scheduler state with a given global step without needing a checkpointed
        scheduler state. This makes resuming from older checkpoints (that did not save
        the scheduler) produce the correct LR going forward.
        """
        try:
            if self.scheduler is None:
                return
            # Prefer loading an adjusted state_dict when possible
            try:
                state = self.scheduler.state_dict()
                if 'last_epoch' in state:
                    state['last_epoch'] = int(step_value) - 1
                    self.scheduler.load_state_dict(state)
                else:
                    # Fallback: set attribute directly if available
                    if hasattr(self.scheduler, 'last_epoch'):
                        self.scheduler.last_epoch = int(step_value) - 1
            except Exception:
                # Ultimate fallback: step through to desired position
                target = max(0, int(step_value))
                for _ in range(target):
                    self.scheduler.step()
            # Ensure optimizer LR reflects scheduler's current computed LR
            try:
                lrs = self.scheduler.get_last_lr()
                if isinstance(lrs, (list, tuple)):
                    for pg, lr in zip(self.opt.param_groups, lrs):
                        pg['lr'] = lr
                else:
                    for pg in self.opt.param_groups:
                        pg['lr'] = float(lrs)
            except Exception:
                pass
        except Exception as e:
            print(f"Warning: failed to align scheduler to step {step_value}: {e}")

    @property
    def device(self):
        return self.accelerator.device

    def _init_md4_if_needed(self, vox):
        """
        Initialize MD4 helper and spatial size from either one-hot/embedding volumes [B,C,H,W,D]
        or index grids [B,H,W,D]. Determine num_classes from converter if available, otherwise
        infer from tensor shape (one-hot) or max index + 1 (indices).
        """
        if vox.dim() == 5:
            # [B,C,H,W,D]
            _, C, H, W, D = vox.shape
            self._spatial_size = (H, W, D)
            # detect one-hot by sum across channels ~ 1
            is_one_hot = torch.allclose(vox.sum(dim=1).mean(), torch.ones((), device=vox.device), atol=1e-2)
            if self.converter is not None:
                num_classes = len(self.converter.block_to_index)
            else:
                # if one-hot, C is K; otherwise we cannot infer reliably -> fall back to C
                num_classes = int(C) if is_one_hot else int(C)
        elif vox.dim() == 4:
            # [B,H,W,D] indices
            _, H, W, D = vox.shape
            self._spatial_size = (H, W, D)
            if self.converter is not None:
                num_classes = len(self.converter.block_to_index)
            else:
                # best-effort: infer from max index in batch
                try:
                    num_classes = int(vox.max().item()) + 1
                except Exception:
                    num_classes = 256  # conservative fallback
        else:
            raise ValueError(f"Unexpected vox shape: {tuple(vox.shape)}")

        if self._md4 is None:
            self._md4 = MD4Discrete3D(accelerator=self.accelerator, num_classes=num_classes, device=self.device)

    def _to_indices(self, vox):
        """
        Normalize input to integer indices [B,H,W,D]:
        - If vox is [B,C,H,W,D] and one-hot, take argmax.
        - If vox is [B,C,H,W,D] and embeddings, use converter if available; else argmax fallback.
        - If vox is [B,H,W,D], return as long.
        """
        if vox.dim() == 4:
            return vox.long()
        if vox.dim() != 5:
            raise ValueError(f"Unexpected vox shape: {tuple(vox.shape)}")
        if self.converter is not None:
            sums = vox.sum(dim=1)
            is_one_hot = torch.allclose(sums.mean(), torch.ones((), device=vox.device), atol=1e-2)
            if is_one_hot:
                return torch.argmax(vox, dim=1)
            with torch.no_grad():
                return self.converter.convert_emb_to_indices(vox.detach().cpu()).to(vox.device)
        return torch.argmax(vox, dim=1)

    def _extract_batch(self, batch):
        """
        Normalize a batch from the dataloader to a tuple (voxels, class_cond or None).
        Supports:
        - tensor: vox
        - (vox, cond)
        - dict: { 'voxels' | 'vox' | 'x': ..., optional 'cond' | 'class_cond' | 'classes' | 'y' | 'label': ... }
        """
        vox = None
        class_cond = None
        if isinstance(batch, (list, tuple)):
            if len(batch) == 0:
                raise ValueError('Empty batch encountered')
            vox = batch[0]
            if len(batch) > 1:
                class_cond = batch[1]
        elif isinstance(batch, dict):
            vox = batch.get('voxels', None)
            if vox is None:
                vox = batch.get('vox', None)
            if vox is None:
                vox = batch.get('x', None)
            class_cond = (
                batch.get('cond', None)
                if 'cond' in batch else batch.get('class_cond', None)
            )
            if class_cond is None:
                class_cond = batch.get('classes', None)
            if class_cond is None:
                class_cond = batch.get('y', None)
            if class_cond is None:
                class_cond = batch.get('label', None)
            if vox is None:
                raise ValueError('Dictionary batch missing voxel tensor under keys voxels/vox/x')
        else:
            vox = batch
        return vox, class_cond

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'scheduler': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
            'ema': self.ema.state_dict() if hasattr(self, 'ema') else None,
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'best_val_loss': float(self.best_val_loss) if exists(self.best_val_loss) else None,
            'best_val_step': int(self.best_val_step) if exists(self.best_val_step) and self.best_val_step is not None else None,
            'version': __version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device
        # allow flexible loading: explicit path, 'best', or numeric milestone
        ckpt_path = None
        if isinstance(milestone, (str, Path)) and os.path.exists(str(milestone)):
            ckpt_path = str(milestone)
        else:
            if isinstance(milestone, str) and (
                milestone in ('best', 'model_best', 'best.pt', 'model_best.pt')
            ):
                ckpt_path = str(self.results_folder / 'model_best.pt')
            else:
                ckpt_path = str(self.results_folder / f'model-{milestone}.pt')

        data = torch.load(ckpt_path, map_location=device, weights_only=True)

        model = self.accelerator.unwrap_model(
            self.model,
            keep_torch_compile=True,
            )
        model.load_state_dict(data['model'])
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        scheduler_loaded = False
        if data.get('scheduler') is not None and hasattr(self, 'scheduler'):
            try:
                self.scheduler.load_state_dict(data['scheduler'])
                scheduler_loaded = True
            except Exception as e:
                print(f"Warning: failed to load scheduler state: {e}")
        if data.get('ema') is not None:
            # load EMA on all ranks to keep evaluation consistent
            self.ema.load_state_dict(data['ema'])
        # restore best validation tracking if present
        if 'best_val_loss' in data and data['best_val_loss'] is not None:
            self.best_val_loss = float(data['best_val_loss'])
        if 'best_val_step' in data and data['best_val_step'] is not None:
            self.best_val_step = int(data['best_val_step'])
        if 'version' in data:
            print(f"loading from version {data['version']}")
        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])
        # If scheduler wasn't in the checkpoint, align it to the saved step
        if not scheduler_loaded:
            self._align_scheduler_to_step(self.step)

    def _save_best_checkpoint(self):
        if not self.accelerator.is_local_main_process:
            return
        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'scheduler': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
            'ema': self.ema.state_dict() if hasattr(self, 'ema') else None,
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'best_val_loss': float(self.best_val_loss) if exists(self.best_val_loss) else None,
            'best_val_step': int(self.best_val_step) if exists(self.best_val_step) and self.best_val_step is not None else None,
            'version': __version__
        }
        torch.save(data, str(self.results_folder / 'model_best.pt'))

    def plot_losses(self):
        if not self.accelerator.is_main_process:
            return
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.plot(self.steps, self.losses, 'b-', linewidth=1, alpha=0.8, label='Train')
        if len(self.val_losses) > 0:
            plt.plot(self.val_steps, self.val_losses, 'orange', linewidth=1.5, alpha=0.9, label='Validation')
            try:
                import numpy as np
                best_idx = int(np.argmin(self.val_losses))
                best_step = self.val_steps[best_idx]
                plt.axvline(x=best_step, color='gray', linestyle='--', alpha=0.6, label='Best Val')
            except Exception:
                pass
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss (MD4)')
        plt.grid(True, alpha=0.3)
        if len(self.losses) > 50:
            import numpy as np
            window_size = min(50, len(self.losses) // 10)
            smoothed = np.convolve(self.losses, np.ones(window_size)/window_size, mode='valid')
            steps = self.steps[window_size-1:]
            plt.plot(steps, smoothed, 'r-', linewidth=2, label=f'Train (smoothed={window_size})')
            if len(self.val_losses) > 10:
                v_window = min(20, max(5, len(self.val_losses) // 5))
                v_smoothed = np.convolve(self.val_losses, np.ones(v_window)/v_window, mode='valid')
                v_steps = self.val_steps[v_window-1:]
                plt.plot(v_steps, v_smoothed, 'g-', linewidth=2, label=f'Val (smoothed={v_window})')
            plt.legend()
        plt.tight_layout()
        plt.savefig(str(self.results_folder / 'training_loss.png'), dpi=150, bbox_inches='tight')
        plt.close()
        torch.save({'steps': self.steps, 'losses': self.losses, 'val_steps': self.val_steps, 'val_losses': self.val_losses}, str(self.results_folder / 'training_loss_data.pt'))

    def plot_validation_only(self):
        if not self.accelerator.is_main_process or len(self.val_losses) == 0:
            return
        import matplotlib.pyplot as plt
        import numpy as np
        plt.figure(figsize=(10, 5))
        plt.plot(self.val_steps, self.val_losses, color='orange', linewidth=1.8, label='Validation')
        try:
            best_idx = int(np.argmin(self.val_losses))
            best_step = self.val_steps[best_idx]
            best_val = self.val_losses[best_idx]
            plt.axvline(x=best_step, color='gray', linestyle='--', alpha=0.6, label='Best Val')
            plt.scatter([best_step], [best_val], color='red', zorder=3)
        except Exception:
            pass
        plt.xlabel('Training Step')
        plt.ylabel('Validation Loss')
        plt.title('Validation Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(str(self.results_folder / 'validation_loss.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # def plot_fid(self):
    #     if not self.accelerator.is_main_process or len(self.fid_scores) == 0:
    #         return
    #     import matplotlib.pyplot as plt
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(self.fid_steps, self.fid_scores, color='purple', linewidth=1.8, label='FID')
    #     plt.xlabel('Training Step')
    #     plt.ylabel('FID (lower is better)')
    #     plt.title('FID over time')
    #     plt.grid(True, alpha=0.3)
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig(str(self.results_folder / 'fid_over_time.png'), dpi=150, bbox_inches='tight')
    #     plt.close()

    # @torch.inference_mode()
    # def _save_fid_samples(self):
    #     """
    #     Generate and save samples for later FID computation.
    #     This is much faster than rendering+computing FID during training.
    #     """
    #     # if self.fid_ref_images_dir in (None, "") or self.fid_num_samples in (None, 0):
    #     #     return None

    #     accelerator = self.accelerator

    #     # Ensure MD4 helper and spatial size are initialized for sampling
    #     if self._md4 is None or self._spatial_size is None:
    #         if self.val_dl is not None:
    #             try:
    #                 for batch in self.val_dl:
    #                     vox, _ = self._extract_batch(batch)
    #                     self._init_md4_if_needed(vox)
    #                     break
    #             except Exception as e:
    #                 print(f"Failed to initialize MD4 for sampling: {e}")
    #                 return None
    #         else:
    #             print("Cannot save FID samples: MD4 not initialized and no validation dataloader")
    #             return None

    #     # Build class labels if using class-conditional model
    #     class_labels = None
    #     if self.num_classes not in (None, 0):
    #         class_labels = torch.randint(0, int(self.num_classes), (int(self.fid_num_samples),), device=accelerator.device)

    #     # Save directory for this step's samples
    #     samples_dir = os.path.join(str(self.results_folder), "fid_samples")
        
    #     t_start = time.time()
    #     # Generate and save samples (no rendering, just raw indices)
    #     all_samples_cpu, all_labels_cpu = self._distributed_sample_indices(
    #         int(self.fid_num_samples), 
    #         class_labels, 
    #         save_samples=True, 
    #         save_samples_dir=samples_dir,
    #         use_ema_model=True
    #     )
    #     t_end = time.time()
        
    #     if accelerator.is_main_process:
    #         print(f"Saved {self.fid_num_samples} FID samples at step {self.step} ({t_end - t_start:.2f}s)")
        
    #     accelerator.wait_for_everyone()
    #     return None
    

    @torch.inference_mode()
    def _compute_validation_loss(self, model_to_eval):
        if self.val_dl is None:
            return None
        accelerator = self.accelerator
        device = accelerator.device
        model_to_eval.eval()
        loss_sum = torch.tensor(0.0, device=device)
        n_items = torch.tensor(0, device=device, dtype=torch.long)
        total_batches = None
        try:
            total_batches = len(self.val_dl)
        except Exception:
            total_batches = None
        _tqdm_disable = not (accelerator.is_main_process and self.val_progress)
        _pbar = tqdm(total=total_batches, disable=_tqdm_disable, desc='Validating')
        for bi, batch in enumerate(self.val_dl):
            vox, class_cond = self._extract_batch(batch)
            # if unconditional mode, ignore labels even if present
            if self.num_classes in (None, 0):
                class_cond = None
            vox = vox.to(device)
            if class_cond is not None:
                class_cond = class_cond.to(device)
            if self._md4 is None:
                self._init_md4_if_needed(vox)
            x_indices = self._to_indices(vox)
            with accelerator.autocast():
                loss = self._md4.training_loss(model_to_eval, x_indices, class_cond=class_cond)
            bsz = x_indices.shape[0]
            loss_sum += loss.detach() * bsz
            n_items += bsz
            if self.val_max_batches is not None and bi + 1 >= self.val_max_batches:
                if _pbar is not None:
                    _pbar.update(1)
                break
            if _pbar is not None:
                _pbar.update(1)
        if _pbar is not None:
            _pbar.close()
        loss_sum = accelerator.reduce(loss_sum, reduction='sum')
        n_items = accelerator.reduce(n_items, reduction='sum')
        model_to_eval.train()
        return (loss_sum / max(1, n_items)).item()

    def _render_and_save_samples(self, samples_indices, milestone, classes=None):
        # samples_indices: [B,H,W,D] int indices -> convert to original block IDs for visualization
        if self.converter is None:
            block_ids = samples_indices
        else:
            block_ids = self.converter.convert_to_original_blocks(samples_indices.cpu()).to('cpu')

        # Save PT
        torch.save(block_ids, str(self.results_folder / f'sample-{milestone}.pt'))

        # Prepare class strings when provided
        class_strs = None
        if classes is not None:
            classes_cpu = classes.detach().to('cpu') if isinstance(classes, torch.Tensor) else classes
            try:
                if self.converter is not None and hasattr(self.converter, 'convert_class_to_biomes'):
                    class_strs = self.converter.convert_class_to_biomes(classes_cpu)
                else:
                    class_strs = [str(int(c)) for c in classes_cpu]
            except Exception:
                class_strs = [str(int(c)) for c in classes_cpu]

        # Render grid image
        save_chunks(
            block_ids,
            self.step,
            self.results_folder,
            self.converter if self.converter is not None else BlockBiomeConverter(),
            classes=class_strs,
            textured=True,
            # textures_dir='block_textures/'
        )
        # Log rendered image to wandb if enabled
        if self.use_wandb and self.wandb_log_images and self.accelerator.is_main_process:
            try:
                img_path = str(self.results_folder / f'sampled_chunks_ep_{self.step}.png')
                # log under a unique key per step so W&B shows separate panels
                wandb.log({f"samples/image_step_{self.step}": wandb.Image(img_path)}, step=self.step)
            except Exception as e:
                print(f"wandb image log failed: {e}")

    def _sample_batch(self, model_to_eval, count, class_cond=None, cond_scale=None):
        assert self._md4 is not None and self._spatial_size is not None
        # try to use air index from converter
        air_idx = 0
        if self.converter is not None:
            try:
                air_idx = self.converter.get_air_block_index()
            except Exception:
                air_idx = 0
        kwargs = {
            'reverse_steps': self.reverse_steps,
            'progress': False,
            'air_index_fallback': air_idx,
        }
        if class_cond is not None:
            effective_scale = cond_scale if cond_scale is not None else (self.default_cond_scale if self.default_cond_scale is not None else 4.0)
            kwargs.update({'class_cond': class_cond, 'cond_scale': float(effective_scale)})
        if self.sampling_timesteps is not None and self.sampling_timesteps > 0 and self.sampling_timesteps < self.reverse_steps:
            samples = self._md4.ddim_sample(model_to_eval, count, self._spatial_size, self.sampling_timesteps, **kwargs)
        else:
            samples = self._md4.sample(model_to_eval, count, self._spatial_size, **kwargs)
        return samples

    @torch.inference_mode()
    def _distributed_sample_indices(self, total_samples: int, class_labels: torch.Tensor | None = None, save_samples: bool = False, save_samples_dir: str | None = None, use_ema_model: bool = False):
        """
        Generate samples in a distributed manner across all ranks, then gather to main process.
        
        Args:
            total_samples: Total number of samples to generate (split across all ranks)
            class_labels: Optional tensor of class labels [total_samples] for conditional generation
            save_samples: If True, save raw samples to disk (main process only)
            save_samples_dir: Directory to save samples (only used if save_samples=True)
            use_ema_model: If True and EMA is available, use EMA weights for sampling
            
        Returns:
            (all_samples, all_labels) on main process, (None, None) on other ranks
            all_samples: [total_samples, H, W, D] tensor on CPU (main process only)
            all_labels: [total_samples] tensor on CPU or None (main process only)
        """
        accelerator = self.accelerator
        device = accelerator.device

        # EMA model is NOT wrapped by Accelerate, base model IS wrapped
        if bool(use_ema_model) and hasattr(self, 'ema') and hasattr(self.ema, 'ema_model') and (self.ema.ema_model is not None):
            # EMA model is already unwrapped 
            if dist.is_available() and dist.is_initialized():
                for param in self.ema.ema_model.state_dict().values():
                    if isinstance(param, torch.Tensor):
                        dist.broadcast(param.data, src=0)
            eval_model = self.ema.ema_model
        else:
            # unwrap it for inference
            eval_model = accelerator.unwrap_model(
                self.model,
                keep_torch_compile=True,
                )
        
        eval_model.eval()

        total = int(total_samples) 
        all_indices = list(range(total))
        local_samples = None
        local_labels = None
        
        with accelerator.split_between_processes(all_indices) as local_indices:
            local_n = len(local_indices)
            
            if class_labels is not None and local_n > 0:
                if not isinstance(class_labels, torch.Tensor):
                    class_labels = torch.tensor(class_labels, dtype=torch.long)
                assert class_labels.dim() == 1 and class_labels.shape[0] == total, "class_labels must be length total_samples"
                picked = [class_labels[i].item() for i in local_indices]
                local_labels = torch.tensor(picked, device=device, dtype=torch.long)
            
            # Generate samples on this rank
            if local_n > 0:
                if accelerator.is_main_process:
                    print(f'Generating {local_n} samples per rank ({total} total across {accelerator.num_processes} ranks)')
                local_samples = self._sample_batch(
                    eval_model,
                    local_n,
                    class_cond=local_labels,
                    cond_scale=(self.default_cond_scale if local_labels is not None else None)
                )
            else:
                local_samples = torch.empty((0, *self._spatial_size), dtype=torch.long, device=device)
                local_labels = torch.empty((0,), dtype=torch.long, device=device) if class_labels is not None else None
        
        local_count = torch.tensor([local_samples.shape[0]], device=device)
        all_counts = accelerator.gather(local_count) 
        
        # Pad tensors to max size 
        max_count = int(all_counts.max().item())
        local_samples = accelerator.pad_across_processes(local_samples, dim=0, pad_index=0, pad_first=False)
        if local_labels is not None:
            local_labels = accelerator.pad_across_processes(local_labels, dim=0, pad_index=0, pad_first=False)
        
        gathered_samples = accelerator.gather(local_samples)
        gathered_labels = accelerator.gather(local_labels) if local_labels is not None else None
        
        if accelerator.is_main_process:
            gathered_samples = gathered_samples.view(accelerator.num_processes, -1, *self._spatial_size)
            samples_list = [gathered_samples[i, :int(all_counts[i])] 
                          for i in range(accelerator.num_processes) if all_counts[i] > 0]
            gathered_samples = torch.cat(samples_list, dim=0).cpu() if samples_list else torch.empty((0, *self._spatial_size), dtype=torch.long)
            
            if gathered_labels is not None:
                gathered_labels = gathered_labels.view(accelerator.num_processes, -1)
                labels_list = [gathered_labels[i, :int(all_counts[i])] 
                             for i in range(accelerator.num_processes) if all_counts[i] > 0]
                gathered_labels = torch.cat(labels_list, dim=0).cpu() if labels_list else None
            else:
                gathered_labels = None
        else:
            gathered_samples = None
            gathered_labels = None
        
        # Save raw samples to disk if requested for fid
        if save_samples and save_samples_dir is not None and accelerator.is_main_process:
            os.makedirs(save_samples_dir, exist_ok=True)
            save_data = {'samples': gathered_samples, 'step': self.step}
            if gathered_labels is not None:
                save_data['labels'] = gathered_labels
            save_path = os.path.join(save_samples_dir, f'samples_step_{self.step}.pt')
            torch.save(save_data, save_path)
            if accelerator.is_main_process:
                print(f'Saved {gathered_samples.shape[0]} samples to {save_path}')
        
        accelerator.wait_for_everyone()
        
        # Return gathered data on main process, None on others
        if accelerator.is_main_process:
            return gathered_samples, gathered_labels
        else:
            return None, None

    def _get_village_index(self):
        if self.village_label_index is not None:
            return int(self.village_label_index)
        if self.converter is not None and hasattr(self.converter, 'biome_to_index') and self.village_label in getattr(self.converter, 'biome_to_index', {}):
            return int(self.converter.biome_to_index[self.village_label])
        return 0

    def _load_indices_from_path(self, path: str):
        data = torch.load(path, map_location=self.device)
        if isinstance(data, dict) and 'voxels' in data:
            vox = data['voxels']
            if vox.dim() == 5:
                if vox.shape[1] > 1 and torch.all((vox.sum(dim=1) - 1.0).abs() < 1e-3):
                    indices = torch.argmax(vox, dim=1)
                    return indices.to(self.device)
                if self.converter is None:
                    raise ValueError(f"Converter required to map embeddings to indices for inpaint previews (shape={tuple(vox.shape)})")
                indices = self.converter.convert_emb_to_indices(vox)
                return indices.to(self.device)
            if vox.dim() == 4:
                indices = torch.argmax(vox, dim=0).long().unsqueeze(0)
                return indices.to(self.device)
            if vox.dim() == 3:
                return vox.long().unsqueeze(0).to(self.device)
            raise ValueError(f"Unsupported 'voxels' tensor shape in {path}: {tuple(vox.shape)}")
        if isinstance(data, torch.Tensor):
            t = data
            if t.dim() == 4:
                indices = torch.argmax(t, dim=0).long().unsqueeze(0)
                return indices.to(self.device)
            if t.dim() == 3:
                return t.long().unsqueeze(0).to(self.device)
            if t.dim() == 5:
                if t.shape[1] > 1 and torch.all((t.sum(dim=1) - 1.0).abs() < 1e-3):
                    indices = torch.argmax(t, dim=1)
                    return indices.to(self.device)
                if self.converter is None:
                    raise ValueError(f"Converter required to convert embeddings to indices for tensor of shape={tuple(t.shape)}")
                return self.converter.convert_emb_to_indices(t).to(self.device)
            raise ValueError(f"Unsupported preview tensor shape at {path}: {tuple(t.shape)}")
        raise ValueError(f"Unsupported preview source format at {path}: type={type(data)}")

        
    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                self.model.train()
                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    batch = next(self.dl)
                    vox, class_cond = self._extract_batch(batch)
                    # if unconditional mode, ignore labels even if present
                    if self.num_classes in (None, 0):
                        class_cond = None
                    vox = vox.to(device)
                    if class_cond is not None:
                        class_cond = class_cond.to(device)

                    if self._md4 is None:
                        self._init_md4_if_needed(vox)

                    x_indices = self._to_indices(vox)

                    with self.accelerator.autocast():
                        loss = self._md4.training_loss(self.model, x_indices, class_cond=class_cond)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.scheduler.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                # record loss
                if accelerator.is_main_process:
                    self.losses.append(total_loss)
                    self.steps.append(self.step)
                    if self.use_wandb:
                        wandb.log({
                            "train/loss": total_loss,
                            "train/step": self.step,
                            "train/lr": self.scheduler.get_last_lr()[0],
                            },
                            step=self.step)

                self.step += 1
                # Update EMA on main only, then sync everyone
                if accelerator.is_main_process and hasattr(self, 'ema'):
                    self.ema.update()
                self.accelerator.wait_for_everyone()

                if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                    milestone = self.step // self.save_and_sample_every
                    with torch.inference_mode():
                        # Unconditional preview
                        if self.num_classes in (None, 0):
                            _prev_t0 = time.perf_counter()
                            samples_indices, _ = self._distributed_sample_indices(self.num_samples)
                            self.accelerator.wait_for_everyone()
                            if self.accelerator.is_main_process and samples_indices is not None:
                                self._render_and_save_samples(samples_indices, milestone)
                            self.accelerator.wait_for_everyone()
                        else:
                            # Conditional: round-robin labels globally
                            labels = [(i % int(self.num_classes)) for i in range(int(self.num_samples))]
                            labels_t = torch.tensor(labels, dtype=torch.long)
                            _prev_t0 = time.perf_counter()
                            all_samples, all_labels = self._distributed_sample_indices(self.num_samples, class_labels=labels_t, use_ema_model=True)
                            self.accelerator.wait_for_everyone()
                            if self.accelerator.is_main_process and all_samples is not None:
                                try:
                                    self._render_and_save_samples(all_samples, milestone, classes=all_labels)
                                except Exception as e:
                                    print(f"Preview render failed: {e}")
                                    try:
                                        import traceback; traceback.print_exc()
                                    except Exception:
                                        pass
                            self.accelerator.wait_for_everyone()
                    if not self.save_only_last_checkpoint:
                        self.save(milestone)

                # with torch.inference_mode():
                    # should_save_fid = (
                    #     (self.fid_ref_images_dir not in (None, '')) and
                    #     (self.fid_num_samples not in (None, 0)) and
                    #     (self.fid_every_steps not in (None, 0)) and
                    #     divisible_by(self.step, self.fid_every_steps)
                    # )
                    # if should_save_fid:
                    #     self._save_fid_samples()

                # periodic validation
                if self.val_dl is not None and divisible_by(self.step, self.val_interval_steps):
                    with torch.inference_mode():
                        if (hasattr(self, 'ema') and self.evaluate_ema_model):
                            # Use EMA model for validation (EMA sync handled in distributed functions)
                            model_to_eval = self.ema.ema_model
                        else:
                            model_to_eval = self.model
                        model_to_eval.eval()
                        val_loss = self._compute_validation_loss(model_to_eval)
                    if self.accelerator.is_main_process and val_loss is not None:
                        self.val_losses.append(float(val_loss))
                        self.val_steps.append(self.step)
                        if self.use_wandb:
                            wandb.log({
                                "val/loss": float(val_loss),
                                "val/step": self.step,
                            }, step=self.step)
                        logging.info(f"Validation loss at step {self.step}: {val_loss:.4f}")
                        # update plot with validation curve
                        self.plot_losses()
                        self.plot_validation_only()
                        # save best checkpoint if improved
                        if float(val_loss) < float(self.best_val_loss):
                            self.best_val_loss = float(val_loss)
                            self.best_val_step = int(self.step)
                            self._save_best_checkpoint()
                        # Always log current best (even if unchanged) to show plateauing
                        if self.use_wandb:
                            wandb.log({
                                "val/best_loss": float(self.best_val_loss),
                                "val/best_step": int(self.best_val_step) if self.best_val_step is not None else -1,
                            }, step=self.step)
                        logging.info(f"Best validation so far: {self.best_val_loss:.4f} at step {self.best_val_step}")

                pbar.update(1)

        accelerator.print('training complete')
        if self.accelerator.is_main_process and self.save_only_last_checkpoint:
            self.save("final")
        self.plot_losses()
        
    