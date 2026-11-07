"""
Trainer for the autoregressive Dream-Cubed baseline.

Uses the MD4 optimizer, WSD schedule and EMA conventions with per-voxel
autoregressive NLL. Gradient accumulation avoids redundant DDP reductions;
preview generation is optional so it can run offline on utilization-enforced
clusters. Ordinary checkpoint restore preserves optimizer, scheduler and EMA.

The only substantive differences are:
  * the loss is an exact per-voxel negative log-likelihood (nats/voxel) instead
    of the MD4 masked-diffusion bound;
  * sampling calls `ARDiT3D.generate` (KV-cached patch and voxel decoding)
    instead of a reverse diffusion loop.
"""

import logging
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from accelerate.utils import TorchDynamoPlugin, DistributedDataParallelKwargs, gather_object
from ema_pytorch import EMA
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_wsd_schedule

from data_utils import BlockBiomeConverter
from discrete_diffusion_md4 import cycle, divisible_by, exists
from visualization_utils import save_chunks

logger = logging.getLogger(__name__)
__version__ = "ar-3d-0.1"

LN2 = float(np.log(2.0))


class VoxelTrainerAR:
    def __init__(
        self,
        model,
        dataset,
        results_folder,
        *,
        vocab_size,
        train_batch_size=8,
        gradient_accumulate_every=1,
        train_lr=1e-4,
        train_num_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        optimizer='adamw',
        adam_betas=(0.9, 0.99),
        weight_decay=0.01,
        warmup_steps=0,
        scheduler='cosine',
        min_lr_ratio=0.0,
        save_and_sample_every=1000,
        save_latest_every=0,
        num_samples=16,
        amp=False,
        mixed_precision_type='fp16',
        compile_model=False,
        sync_each_microbatch=False,
        num_workers=2,
        prefetch_factor=2,
        performance_every=10,
        max_train_updates=0,
        max_train_seconds=0,
        profile_steps=0,
        profile_start_step=5,
        sample_during_training=True,
        max_grad_norm=1.,
        evaluate_ema_model=True,
        mappings_file_path=None,
        run_name='ar_dit',
        save_only_last_checkpoint=False,
        # conditioning / sampling
        num_classes=None,
        default_cond_scale=None,
        sample_temperature=1.0,
        sample_top_k=0,
        sample_top_p=0.0,
        # validation
        val_dataloader=None,
        val_dataset=None,
        val_batch_size=None,
        val_every_n_steps=None,
        val_max_batches=None,
        val_steps=None,
        val_progress=True,
        use_wandb=False,
        wandb_log_images=True,
        # FID sample dumps
        fid_ref_images_dir=None,
        fid_num_samples=None,
        fid_every_steps=None,
        fid_textures_dir='block_textures/',
        fid_batch_size=32,
        fid_num_workers=2,
    ):
        super().__init__()

        if compile_model:
            dynamo_plugin = TorchDynamoPlugin(backend="inductor", mode="default", dynamic=False)
        else:
            dynamo_plugin = None

        self.accelerator = Accelerator(
            split_batches=False,
            mixed_precision=mixed_precision_type if amp else 'no',
            step_scheduler_with_optimizer=False,
            dynamo_plugin=dynamo_plugin,
            # AR buffers are immutable raster permutations; no BatchNorm stats.
            kwargs_handlers=[DistributedDataParallelKwargs(
                broadcast_buffers=False, gradient_as_bucket_view=True)],
        )
        self.sync_each_microbatch = bool(sync_each_microbatch)
        self.performance_every = max(1, int(performance_every))
        self.max_train_updates = int(max_train_updates)
        self.max_train_seconds = float(max_train_seconds)
        self.profile_steps = int(profile_steps)
        self.profile_start_step = int(profile_start_step)
        self.sample_during_training = bool(sample_during_training)
        self.profiler = None

        self.model = model
        self.vocab_size = int(vocab_size)
        self.save_and_sample_every = save_and_sample_every
        # Rolling checkpoint without sampling; save time depends on storage.
        self.save_latest_every = 0 if save_latest_every in (None, 0) else int(save_latest_every)
        self.save_only_last_checkpoint = save_only_last_checkpoint
        self.num_samples = num_samples
        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        self.train_lr = float(train_lr)
        self.warmup_steps = int(warmup_steps)
        self.scheduler_name = str(scheduler).lower()
        self.min_lr_ratio = float(min_lr_ratio)
        self.max_grad_norm = max_grad_norm
        self.evaluate_ema_model = evaluate_ema_model
        self.mappings_file_path = mappings_file_path
        self.run_name = run_name

        self.num_classes = None if num_classes in (None, 0) else int(num_classes)
        self.default_cond_scale = None if default_cond_scale in (None, 0) else float(default_cond_scale)
        self.sample_temperature = float(sample_temperature)
        self.sample_top_k = int(sample_top_k)
        self.sample_top_p = float(sample_top_p)

        self.losses = []
        self.steps = []
        self.val_losses = []
        self.val_steps = []
        self.best_val_loss = float('inf')
        self.best_val_step = None

        self.ds = dataset
        if len(dataset) < train_batch_size:
            raise ValueError('Dataset is smaller than the per-device batch size; drop_last would produce an empty loader.')
        assert len(self.ds) > 0, 'your dataset is empty'
        if num_workers < 0 or prefetch_factor < 1:
            raise ValueError('num_workers must be >= 0 and prefetch_factor >= 1')
        worker_kwargs = dict(num_workers=num_workers, pin_memory=True)
        if num_workers:
            worker_kwargs.update(persistent_workers=True, prefetch_factor=prefetch_factor)
        dl = DataLoader(self.ds, batch_size=train_batch_size, shuffle=True,
                        drop_last=True, **worker_kwargs)
        if len(dl) < self.accelerator.num_processes:
            raise ValueError('Dataset needs at least one full batch per distributed rank')

        self.val_max_batches = None if val_max_batches in (None, 0) else int(val_max_batches)
        _val_loader = None
        if val_dataset is not None:
            _val_loader = DataLoader(val_dataset, batch_size=(val_batch_size or train_batch_size),
                                     shuffle=False, **worker_kwargs)
        elif val_dataloader is not None:
            _val_loader = val_dataloader
        self.val_dl = (None if _val_loader is None else
                       self.accelerator.prepare_data_loader(_val_loader, device_placement=False))

        if val_steps not in (None, 0):
            self.val_interval_steps = int(val_steps)
        elif val_every_n_steps not in (None, 0):
            self.val_interval_steps = int(val_every_n_steps)
        else:
            self.val_interval_steps = int(save_and_sample_every)
        self.val_progress = bool(val_progress)

        if optimizer.lower() == 'adamw':
            self.opt = AdamW(self.model.parameters(), lr=train_lr, betas=adam_betas,
                             weight_decay=weight_decay)
        elif optimizer.lower() == 'adam':
            if weight_decay != 0:
                raise ValueError('Adam does not support weight decay currently')
            self.opt = Adam(self.model.parameters(), lr=train_lr, betas=adam_betas)
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

        self.model, self.opt, self.scheduler = self.accelerator.prepare(
            self.model, self.opt, self.scheduler)
        # Keep CPU batches pinned until the explicit non-blocking transfer in
        # the loop. Accelerate's default automatic transfer is blocking.
        dl = self.accelerator.prepare_data_loader(dl, device_placement=False)

        base_model = self.accelerator.unwrap_model(self.model, keep_torch_compile=True)
        self.ema = EMA(base_model, beta=ema_decay, update_every=ema_update_every)
        self.ema.ema_model.to(self.accelerator.device)
        self.ema.ema_model.eval().requires_grad_(False)

        self.dl = cycle(dl)

        self.converter = None
        if self.mappings_file_path is not None:
            mappings = torch.load(self.mappings_file_path, weights_only=False)
            self.converter = BlockBiomeConverter(mappings['block_mappings'], mappings['biome_mappings'])

        self._spatial_size = None
        self.use_wandb = use_wandb
        self.wandb_log_images = bool(wandb_log_images)

        self.fid_ref_images_dir = fid_ref_images_dir
        self.fid_num_samples = None if fid_num_samples in (None, 0) else int(fid_num_samples)
        self.fid_every_steps = None if fid_every_steps in (None, 0) else int(fid_every_steps)
        self.fid_textures_dir = fid_textures_dir
        self.fid_batch_size = int(fid_batch_size)
        self.fid_num_workers = int(fid_num_workers)
        self.fid_scores = []
        self.fid_steps = []

    # -- misc ---------------------------------------------------------------

    @property
    def device(self):
        return self.accelerator.device

    def _scheduler_impl(self):
        """Return the PyTorch scheduler underneath Accelerate's wrapper."""
        return getattr(self.scheduler, 'scheduler', self.scheduler)


    def _extract_batch(self, batch):
        if isinstance(batch, (list, tuple)):
            if len(batch) == 0:
                raise ValueError('Empty batch encountered')
            return batch[0], (batch[1] if len(batch) > 1 else None)
        if isinstance(batch, dict):
            vox = batch.get('voxels', batch.get('vox', batch.get('x', None)))
            if vox is None:
                raise ValueError('Dictionary batch missing voxel tensor under keys voxels/vox/x')
            cond = batch.get('cond', batch.get('class_cond', batch.get('classes',
                   batch.get('y', batch.get('label', None)))))
            return vox, cond
        return batch, None

    def _to_indices(self, vox):
        """Normalize a batch to integer block indices [B, H, W, D]."""
        if vox.dim() == 4:
            return vox.long()
        if vox.dim() == 5:
            return torch.argmax(vox, dim=1).long()
        raise ValueError(f"Unexpected vox shape: {tuple(vox.shape)}")

    def _loss(self, model, x_indices, class_cond):
        logits, targets = model(x_indices, class_cond)
        return F.cross_entropy(
            logits.reshape(-1, self.vocab_size).float(),
            targets.reshape(-1),
            reduction='mean',
        )

    # -- checkpointing ------------------------------------------------------

    def _checkpoint_payload(self):
        unwrapped = self.accelerator.unwrap_model(self.model, keep_torch_compile=True)
        parameter_names = [name for name, _ in unwrapped.named_parameters()]
        return {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'scheduler': self._scheduler_impl().state_dict()
                         if hasattr(self, 'scheduler') else None,
            'learning_rates': [float(pg['lr']) for pg in self.opt.param_groups],
            'ema': self.ema.state_dict() if hasattr(self, 'ema') else None,
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'train_num_steps': int(self.train_num_steps),
            'gradient_accumulate_every': int(self.gradient_accumulate_every),
            'optimizer_parameter_names': parameter_names,
            'schedule_config': {
                'train_lr': self.train_lr,
                'train_num_steps': int(self.train_num_steps),
                'warmup_steps': self.warmup_steps,
                'scheduler': self.scheduler_name,
                'min_lr_ratio': self.min_lr_ratio,
            },
            'best_val_loss': float(self.best_val_loss) if exists(self.best_val_loss) else None,
            'best_val_step': int(self.best_val_step) if self.best_val_step is not None else None,
            # loss curves live only in memory otherwise, so a resume would restart
            # the plots from the resume point
            'losses': list(self.losses),
            'loss_steps': list(self.steps),
            'val_losses': list(self.val_losses),
            'val_loss_steps': list(self.val_steps),
            'training_recipe': 'voxel-ar-clean-v1',
            'version': __version__,
        }

    def _save_checkpoint(self, path):
        if not self.accelerator.is_main_process:
            return
        path = Path(path)
        tmp_path = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
        try:
            torch.save(self._checkpoint_payload(), str(tmp_path))
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def save(self, milestone):
        self._save_checkpoint(self.results_folder / f'model-{milestone}.pt')

    def _save_best_checkpoint(self):
        self._save_checkpoint(self.results_folder / 'model_best.pt')

    def load(self, milestone):
        """Restore training state without changing weights, momentum or LR policy."""
        if isinstance(milestone, (str, Path)) and os.path.exists(str(milestone)):
            path = Path(milestone)
        elif milestone in ('best', 'model_best', 'best.pt', 'model_best.pt'):
            path = self.results_folder / 'model_best.pt'
        else:
            path = self.results_folder / f'model-{milestone}.pt'
        data = torch.load(path, map_location='cpu', weights_only=True)
        for key in ('step', 'model', 'opt', 'scheduler', 'ema'):
            if data.get(key) is None:
                raise ValueError(f'Checkpoint {path} lacks {key}; cannot resume full training state.')

        # Legacy recovery runs changed the model or optimization recipe. Their
        # weights cannot become a clean baseline merely by removing those flags.
        if any((data.get('local_ada_stability_config') or {}).values()) or data.get(
                'resume_training_mode', 'exact') != 'exact':
            raise ValueError('This checkpoint used clamps or recovery interventions. '
                             'Start the clean comparison with resume=false.')
        model = self.accelerator.unwrap_model(self.model, keep_torch_compile=True)
        names = [name for name, _ in model.named_parameters()]
        if data.get('optimizer_parameter_names', names) != names:
            raise ValueError('Optimizer parameter order differs from checkpoint.')
        schedule = dict(train_lr=self.train_lr, train_num_steps=int(self.train_num_steps),
                        warmup_steps=self.warmup_steps, scheduler=self.scheduler_name,
                        min_lr_ratio=self.min_lr_ratio)
        if data.get('schedule_config', schedule) != schedule or data.get(
                'train_num_steps', self.train_num_steps) != self.train_num_steps:
            raise ValueError('Keep the original LR schedule when resuming training.')

        model.load_state_dict(data['model'], strict=True)
        self.opt.load_state_dict(data['opt'])
        self._scheduler_impl().load_state_dict(data['scheduler'])
        self.ema.load_state_dict(data['ema'], strict=True)
        if exists(self.accelerator.scaler) and data.get('scaler') is not None:
            self.accelerator.scaler.load_state_dict(data['scaler'])
        self.step = int(data['step'])
        self.losses = list(data.get('losses', []))
        self.steps = list(data.get('loss_steps', []))
        self.val_losses = list(data.get('val_losses', []))
        self.val_steps = list(data.get('val_loss_steps', []))
        self.best_val_loss = (float('inf') if data.get('best_val_loss') is None
                              else float(data['best_val_loss']))
        self.best_val_step = data.get('best_val_step')
        self.accelerator.print(f'Loaded {path}: step={self.step}, '
                               f'lr={self.opt.param_groups[0]["lr"]:.6g}')

    # -- plots --------------------------------------------------------------

    def plot_losses(self):
        if not self.accelerator.is_main_process:
            return
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.plot(self.steps, self.losses, 'b-', linewidth=1, alpha=0.8, label='Train')
        if len(self.val_losses) > 0:
            plt.plot(self.val_steps, self.val_losses, 'orange', linewidth=1.5, alpha=0.9, label='Validation')
            best_idx = int(np.argmin(self.val_losses))
            plt.axvline(x=self.val_steps[best_idx], color='gray', linestyle='--', alpha=0.6, label='Best Val')
        if len(self.losses) > 50:
            window_size = min(50, len(self.losses) // 10)
            smoothed = np.convolve(self.losses, np.ones(window_size) / window_size, mode='valid')
            plt.plot(self.steps[window_size - 1:], smoothed, 'r-', linewidth=2,
                     label=f'Train (smoothed={window_size})')
        plt.xlabel('Training Step')
        plt.ylabel('NLL (nats / voxel)')
        plt.title('Training and Validation Loss (Autoregressive)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(str(self.results_folder / 'training_loss.png'), dpi=150, bbox_inches='tight')
        plt.close()
        torch.save({'steps': self.steps, 'losses': self.losses,
                    'val_steps': self.val_steps, 'val_losses': self.val_losses},
                   str(self.results_folder / 'training_loss_data.pt'))

    def plot_validation_only(self):
        if not self.accelerator.is_main_process or len(self.val_losses) == 0:
            return
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(self.val_steps, self.val_losses, color='orange', linewidth=1.8, label='Validation')
        best_idx = int(np.argmin(self.val_losses))
        plt.axvline(x=self.val_steps[best_idx], color='gray', linestyle='--', alpha=0.6, label='Best Val')
        plt.scatter([self.val_steps[best_idx]], [self.val_losses[best_idx]], color='red', zorder=3)
        plt.xlabel('Training Step')
        plt.ylabel('NLL (nats / voxel)')
        plt.title('Validation Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(str(self.results_folder / 'validation_loss.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # -- evaluation ---------------------------------------------------------

    @torch.inference_mode()
    def _compute_validation_loss(self, model_to_eval):
        if self.val_dl is None:
            return None
        accelerator = self.accelerator
        device = accelerator.device
        was_training = model_to_eval.training
        model_to_eval.eval()
        loss_sum = torch.tensor(0.0, device=device)
        n_items = torch.tensor(0, device=device, dtype=torch.long)
        try:
            total_batches = len(self.val_dl)
        except Exception:
            total_batches = None
        pbar = tqdm(total=total_batches, disable=not (accelerator.is_main_process and self.val_progress),
                    desc='Validating')
        for bi, batch in enumerate(self.val_dl):
            vox, class_cond = self._extract_batch(batch)
            if self.num_classes in (None, 0):
                class_cond = None
            vox = vox.to(device, non_blocking=True)
            if class_cond is not None:
                class_cond = class_cond.to(device, non_blocking=True)
            x_indices = self._to_indices(vox)
            if self._spatial_size is None:
                self._spatial_size = tuple(x_indices.shape[1:])
            with accelerator.autocast():
                # cond_drop_prob=0: report the conditional likelihood, not the
                # CFG-dropout-averaged one.
                logits, targets = model_to_eval(x_indices, class_cond, 0.0)
                loss = F.cross_entropy(logits.reshape(-1, self.vocab_size).float(),
                                       targets.reshape(-1), reduction='mean')
            bsz = x_indices.shape[0]
            loss_sum += loss.detach() * bsz
            n_items += bsz
            pbar.update(1)
            if self.val_max_batches is not None and bi + 1 >= self.val_max_batches:
                break
        pbar.close()
        loss_sum = accelerator.reduce(loss_sum, reduction='sum')
        n_items = accelerator.reduce(n_items, reduction='sum')
        model_to_eval.train(was_training)
        return (loss_sum / max(1, n_items)).item()

    # -- sampling -----------------------------------------------------------

    def _preview_labels(self):
        # Same randomly selected biomes at every checkpoint, on every rank.
        # A private CPU generator leaves training's dropout/data RNG untouched.
        order = torch.randperm(int(self.num_classes), generator=torch.Generator().manual_seed(0))
        repeats = (int(self.num_samples) + len(order) - 1) // len(order)
        return order.repeat(repeats)[:int(self.num_samples)]

    def _sample_batch(self, model_to_eval, count, class_cond=None, cond_scale=None):
        scale = 1.0
        if class_cond is not None:
            scale = cond_scale if cond_scale is not None else (
                self.default_cond_scale if self.default_cond_scale is not None else 1.0)
        return model_to_eval.generate(
            count,
            class_cond=class_cond,
            cond_scale=float(scale),
            temperature=self.sample_temperature,
            top_k=self.sample_top_k,
            top_p=self.sample_top_p,
            progress=False,
            device=self.device,
        )

    @torch.inference_mode()
    def _distributed_sample_indices(self, total_samples: int, class_labels=None,
                                    save_samples: bool = False, save_samples_dir: str = None,
                                    use_ema_model: bool = False):
        """Shard sampling across ranks and gather onto the main process."""
        accelerator = self.accelerator
        device = accelerator.device

        if bool(use_ema_model) and hasattr(self, 'ema') and getattr(self.ema, 'ema_model', None) is not None:
            if dist.is_available() and dist.is_initialized():
                for param in self.ema.ema_model.state_dict().values():
                    if isinstance(param, torch.Tensor):
                        dist.broadcast(param.data, src=0)
            eval_model = self.ema.ema_model
        else:
            eval_model = accelerator.unwrap_model(self.model, keep_torch_compile=True)
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
                assert class_labels.dim() == 1 and class_labels.shape[0] == total
                local_labels = torch.tensor([class_labels[i].item() for i in local_indices],
                                            device=device, dtype=torch.long)
            if local_n > 0:
                if accelerator.is_main_process:
                    print(f'Generating {local_n} samples per rank ({total} total across '
                          f'{accelerator.num_processes} ranks)')
                with accelerator.autocast():
                    local_samples = self._sample_batch(
                        eval_model, local_n, class_cond=local_labels,
                        cond_scale=(self.default_cond_scale if local_labels is not None else None))
            else:
                local_samples = torch.empty((0, *self._spatial_size), dtype=torch.long, device=device)
                local_labels = torch.empty((0,), dtype=torch.long, device=device) if class_labels is not None else None

        local_count = torch.tensor([local_samples.shape[0]], device=device)
        all_counts = accelerator.gather(local_count)

        local_samples = accelerator.pad_across_processes(local_samples, dim=0, pad_index=0, pad_first=False)
        if local_labels is not None:
            local_labels = accelerator.pad_across_processes(local_labels, dim=0, pad_index=0, pad_first=False)

        gathered_samples = accelerator.gather(local_samples)
        gathered_labels = accelerator.gather(local_labels) if local_labels is not None else None

        if accelerator.is_main_process:
            gathered_samples = gathered_samples.view(accelerator.num_processes, -1, *self._spatial_size)
            samples_list = [gathered_samples[i, :int(all_counts[i])]
                            for i in range(accelerator.num_processes) if all_counts[i] > 0]
            gathered_samples = (torch.cat(samples_list, dim=0).cpu() if samples_list
                                else torch.empty((0, *self._spatial_size), dtype=torch.long))
            if gathered_labels is not None:
                gathered_labels = gathered_labels.view(accelerator.num_processes, -1)
                labels_list = [gathered_labels[i, :int(all_counts[i])]
                               for i in range(accelerator.num_processes) if all_counts[i] > 0]
                gathered_labels = torch.cat(labels_list, dim=0).cpu() if labels_list else None
        else:
            gathered_samples = None
            gathered_labels = None

        if save_samples and save_samples_dir is not None and accelerator.is_main_process:
            os.makedirs(save_samples_dir, exist_ok=True)
            save_data = {'samples': gathered_samples, 'step': self.step}
            if gathered_labels is not None:
                save_data['labels'] = gathered_labels
            save_path = os.path.join(save_samples_dir, f'samples_step_{self.step}.pt')
            torch.save(save_data, save_path)
            print(f'Saved {gathered_samples.shape[0]} samples to {save_path}')

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            return gathered_samples, gathered_labels
        return None, None

    @torch.inference_mode()
    def _save_fid_samples(self):
        """Dump raw samples in the same format as the diffusion trainers so the
        existing `calc_fid.py` pipeline can consume them unchanged."""
        if self.fid_ref_images_dir in (None, "") or self.fid_num_samples in (None, 0):
            return None
        if self._spatial_size is None:
            print("Cannot save FID samples: spatial size unknown (no batch seen yet)")
            return None

        class_labels = None
        if self.num_classes not in (None, 0):
            class_labels = torch.randint(0, int(self.num_classes), (int(self.fid_num_samples),),
                                         device=self.accelerator.device)

        samples_dir = os.path.join(str(self.results_folder), "fid_samples")
        t_start = time.time()
        self._distributed_sample_indices(int(self.fid_num_samples), class_labels,
                                         save_samples=True, save_samples_dir=samples_dir,
                                         use_ema_model=True)
        if self.accelerator.is_main_process:
            print(f"Saved {self.fid_num_samples} FID samples at step {self.step} "
                  f"({time.time() - t_start:.2f}s)")
        self.accelerator.wait_for_everyone()
        return None

    def _render_and_save_samples(self, samples_indices, milestone, classes=None):
        if self.converter is None:
            block_ids = samples_indices
        else:
            block_ids = self.converter.convert_to_original_blocks(samples_indices.cpu()).to('cpu')

        torch.save(block_ids, str(self.results_folder / f'sample-{milestone}.pt'))

        class_strs = None
        if classes is not None:
            classes_cpu = classes.detach().to('cpu') if isinstance(classes, torch.Tensor) else classes
            try:
                class_strs = self.converter.convert_class_to_biomes(classes_cpu)
            except Exception:
                class_strs = [str(int(c)) for c in classes_cpu]

        save_chunks(
            block_ids,
            self.step,
            self.results_folder,
            self.converter if self.converter is not None else BlockBiomeConverter(),
            classes=class_strs,
            textured=True,
        )
        if self.use_wandb and self.wandb_log_images and self.accelerator.is_main_process:
            try:
                img_path = str(self.results_folder / f'sampled_chunks_ep_{self.step}.png')
                wandb.log({"samples/previews": wandb.Image(
                    img_path, caption=f'EMA samples at training step {self.step}')}, step=self.step)
            except Exception as e:
                print(f"wandb image log failed: {e}")

    # -- training loop ------------------------------------------------------

    def train(self):
        if self.profile_steps > 0:
            trace_dir = self.results_folder / 'traces'
            trace_dir.mkdir(parents=True, exist_ok=True)
            activities = [torch.profiler.ProfilerActivity.CPU]
            if self.device.type == 'cuda':
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            self.profiler = torch.profiler.profile(
                activities=activities,
                schedule=torch.profiler.schedule(wait=self.profile_start_step, warmup=1,
                                                active=self.profile_steps, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    str(trace_dir), worker_name=f'rank{self.accelerator.process_index}'),
                record_shapes=True,
            )
        try:
            if self.profiler is not None:
                self.profiler.start()
            self._train_loop()
        finally:
            if self.profiler is not None:
                self.profiler.stop()
                self.profiler = None

    def _profile_region(self, name):
        return torch.profiler.record_function(name) if self.profiler is not None else nullcontext()

    def _accumulation_context(self, micro):
        sync = self.sync_each_microbatch or micro == self.gradient_accumulate_every - 1
        return nullcontext() if sync else self.accelerator.no_sync(self.model)

    def _log_performance(self, record):
        if not self.use_wandb:
            return
        # All ranks participate, but only rank 0 owns the W&B run. Gather only
        # at the reporting cadence, never in the microbatch loop.
        records = gather_object([record])
        if not self.accelerator.is_main_process:
            return
        metrics = {
            f'perf/rank_{item["rank"]}/{key}': value
            for item in records for key, value in item.items()
            if key not in ('rank', 'step')
        }
        metrics['perf/seconds_per_update'] = max(r['seconds_per_update'] for r in records)
        metrics['perf/global_samples_per_second'] = min(
            r['global_samples_per_second'] for r in records)
        # Match the loss record just logged for this completed optimizer update.
        wandb.log(metrics, step=self.step - 1)

    def _train_loop(self):
        accelerator = self.accelerator
        device = accelerator.device

        start_step = self.step
        started = time.perf_counter()
        window_started = started
        data_wait = 0.0
        window_updates = 0
        perf_path = self.results_folder / f'performance-rank{accelerator.process_index}.jsonl'
        accelerator.print(
            f'AR runtime: precision={accelerator.mixed_precision}, ranks={accelerator.num_processes}, '
            f'microbatch={self.batch_size}, accumulation={self.gradient_accumulate_every}, '
            f'gradient syncs/update={self.gradient_accumulate_every if self.sync_each_microbatch else 1}, '
            f'online sampling={self.sample_during_training}')
        with tqdm(initial=self.step, total=self.train_num_steps,
                  disable=not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                self.model.train()
                # kept on device: one host sync per optimizer step instead of one
                # per micro-batch, which matters because the AR model runs a much
                # smaller per-device batch and therefore many more micro-batches.
                loss_accum = torch.zeros((), device=device)

                for micro in range(self.gradient_accumulate_every):
                    waited = time.perf_counter()
                    with self._profile_region('ar/data_wait'):
                        batch = next(self.dl)
                    data_wait += time.perf_counter() - waited
                    vox, class_cond = self._extract_batch(batch)
                    if self.num_classes in (None, 0):
                        class_cond = None
                    vox = vox.to(device, non_blocking=True)
                    if class_cond is not None:
                        class_cond = class_cond.to(device, non_blocking=True)

                    x_indices = self._to_indices(vox)
                    if self._spatial_size is None:
                        self._spatial_size = tuple(x_indices.shape[1:])

                    # Both forward AND backward must be inside no_sync. The
                    # final backward averages the accumulated gradients once.
                    # Accelerator accumulation stays at 1: we scale loss here.
                    with self._accumulation_context(micro), self._profile_region('ar/forward_backward'):
                        with accelerator.autocast():
                            loss = self._loss(self.model, x_indices, class_cond)
                            loss = loss / self.gradient_accumulate_every
                        loss_accum += loss.detach()
                        accelerator.backward(loss)

                # Queue optimizer/EMA work before reading scalars on the host.
                # DDP's final backward already synchronized gradients; extra
                # barriers here only stall all four GPUs.
                with self._profile_region('ar/optimizer'):
                    grad_norm = accelerator.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm)
                    self.opt.step()
                    self.scheduler.step()
                    self.opt.zero_grad(set_to_none=True)
                with self._profile_region('ar/ema'):
                    if hasattr(self, 'ema'):
                        self.ema.update()

                # Log the global effective-batch mean. Rank 0's local shard alone
                # is much noisier and can make a healthy distributed resume look
                # unstable.
                time_limit = torch.full_like(loss_accum, float(self.max_train_seconds > 0
                    and time.perf_counter() - started >= self.max_train_seconds))
                metrics = torch.stack((loss_accum.detach(), grad_norm.detach().float(), time_limit))
                total_loss, grad_norm_value, time_limit_fraction = accelerator.reduce(
                    metrics, reduction='mean').cpu().tolist()
                pbar.set_description(f'nll/voxel: {total_loss:.4f}')

                if accelerator.is_main_process:
                    self.losses.append(total_loss)
                    self.steps.append(self.step)
                    if self.use_wandb:
                        wandb.log({
                            "train/loss": total_loss,
                            "train/nats_per_voxel": total_loss,
                            "train/bits_per_voxel": total_loss / LN2,
                            "train/step": self.step,
                            "train/lr": self.scheduler.get_last_lr()[0],
                            "train/grad_norm": grad_norm_value,
                            "train/epochs": ((self.step + 1) * self.batch_size
                                * self.gradient_accumulate_every * accelerator.num_processes
                                / len(self.ds)),
                        }, step=self.step)

                self.step += 1

                window_updates += 1
                if window_updates >= self.performance_every:
                    elapsed = time.perf_counter() - window_started
                    record = dict(step=self.step, rank=accelerator.process_index,
                                  updates=window_updates, seconds=elapsed,
                                  seconds_per_update=elapsed / window_updates,
                                  data_wait_seconds=data_wait,
                                  data_wait_fraction=data_wait / elapsed,
                                  global_samples_per_second=(window_updates * self.batch_size
                                      * self.gradient_accumulate_every * accelerator.num_processes / elapsed))
                    if device.type == 'cuda':
                        record['peak_allocated_gib'] = torch.cuda.max_memory_allocated(device) / 2**30
                    with perf_path.open('a') as f:
                        f.write(json.dumps(record) + '\n')
                    accelerator.print(f'AR performance: {record}')
                    self._log_performance(record)
                    window_started, data_wait, window_updates = time.perf_counter(), 0.0, 0

                # All ranks agree on a wall-clock stop; rank-local decisions
                # could otherwise strand peers in a collective.
                bounded_stop = (self.max_train_updates > 0
                                and self.step - start_step >= self.max_train_updates)
                bounded_stop |= time_limit_fraction > 0
                if bounded_stop:
                    self.save('latest')
                    accelerator.wait_for_everyone()
                    accelerator.print(f'Bounded run stopped at step {self.step}; saved model-latest.pt. '
                                      'Full training schedule is unchanged.')
                    if self.profiler is not None:
                        self.profiler.step()
                    return

                if self.save_latest_every and divisible_by(self.step, self.save_latest_every):
                    self.save('latest')
                    self.accelerator.wait_for_everyone()

                if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                    milestone = self.step // self.save_and_sample_every
                    # Preserve this exact step before entering slow generation.
                    if not self.save_only_last_checkpoint:
                        self.save(milestone)
                    accelerator.wait_for_everyone()
                    preview_started = time.perf_counter()
                    with torch.inference_mode():
                        if not self.sample_during_training:
                            samples_indices = None
                        elif self.num_classes in (None, 0):
                            samples_indices, _ = self._distributed_sample_indices(
                                self.num_samples, use_ema_model=True)
                            labels_out = None
                        else:
                            labels = self._preview_labels()
                            accelerator.print(f'Preview biomes (class IDs): {labels.tolist()}')
                            samples_indices, labels_out = self._distributed_sample_indices(
                                self.num_samples, class_labels=labels,
                                use_ema_model=True)
                        self.accelerator.wait_for_everyone()
                        if self.accelerator.is_main_process and samples_indices is not None:
                            try:
                                self._render_and_save_samples(samples_indices, milestone, classes=labels_out)
                            except Exception as e:
                                print(f"Preview render failed: {e}")
                        self.accelerator.wait_for_everyone()
                    if self.sample_during_training:
                        preview_seconds = time.perf_counter() - preview_started
                        accelerator.print(f'Preview at step {self.step}: {self.num_samples} samples '
                                          f'in {preview_seconds:.1f}s including rendering.')
                        if self.use_wandb and accelerator.is_main_process:
                            wandb.log({'samples/seconds': preview_seconds,
                                       'samples/count': self.num_samples}, step=self.step)

                with torch.inference_mode():
                    should_save_fid = (
                        (self.fid_ref_images_dir not in (None, '')) and
                        (self.fid_num_samples not in (None, 0)) and
                        (self.fid_every_steps not in (None, 0)) and
                        divisible_by(self.step, self.fid_every_steps)
                    )
                    if should_save_fid:
                        self._save_fid_samples()

                if self.val_dl is not None and divisible_by(self.step, self.val_interval_steps):
                    with torch.inference_mode():
                        if hasattr(self, 'ema') and self.evaluate_ema_model:
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
                                "val/nats_per_voxel": float(val_loss),
                                "val/bits_per_voxel": float(val_loss) / LN2,
                                "val/step": self.step,
                            }, step=self.step)
                        logging.info(f"Validation NLL at step {self.step}: {val_loss:.4f} nats/voxel")
                        self.plot_losses()
                        self.plot_validation_only()
                        if float(val_loss) < float(self.best_val_loss):
                            self.best_val_loss = float(val_loss)
                            self.best_val_step = int(self.step)
                            self._save_best_checkpoint()
                        if self.use_wandb:
                            wandb.log({
                                "val/best_loss": float(self.best_val_loss),
                                "val/best_step": int(self.best_val_step) if self.best_val_step is not None else -1,
                            }, step=self.step)
                        logging.info(f"Best validation so far: {self.best_val_loss:.4f} at step {self.best_val_step}")

                pbar.update(1)
                if self.profiler is not None:
                    self.profiler.step()

        accelerator.print('training complete')
        if self.accelerator.is_main_process and self.save_only_last_checkpoint:
            self.save("final")
        self.plot_losses()
