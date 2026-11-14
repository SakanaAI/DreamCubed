"""
Training entrypoint for the autoregressive Dream-Cubed baseline.

Mirrors `train_discrete.py` (same JSON-config + argparse-override scheme, same
dataset plumbing, same results layout) so the AR run can be launched with the
same workflow as the MD4 and DDPM runs:

    bash launch_ar.sh 8 32_ar_patch4_h100
    python train_autoregressive.py --config training_configs/32_ar_patch4_h100.json

The patch-4 comparison configs fix the MD4 backbone depth at 25. Optional
--target_params selects depth by total parameter count for other experiments.
"""

import argparse
import datetime
import json
import logging
import os
import re
from pathlib import Path

import torch
from accelerate.utils import set_seed

from autoregressive_dit import ARDiT3D, depth_for_target_params
from autoregressive_trainer import VoxelTrainerAR
from data_utils import BlockBiomeConverter
from discrete_diffusion_md4 import VoxelDatasetConditional, VoxelDatasetMemmapConditional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_NUMBERED_CHECKPOINT_RE = re.compile(r'^model-(\d+)\.pt$')


def resolve_resume_checkpoint(save_dir, resume_path=''):
    """Resolve an explicit checkpoint, rolling latest, or newest numbered save.

    Both rolling and numbered checkpoints are preferred over ``model_best.pt``
    because "best" can be substantially older than the interrupted process.
    The integer suffix is a save milestone rather than a global step, so the
    checkpoint's embedded step is still authoritative.
    """
    save_dir = Path(save_dir)
    if resume_path:
        explicit = Path(resume_path).expanduser()
        if not explicit.is_absolute() and not explicit.exists():
            explicit = save_dir / explicit
        return explicit

    if save_dir.exists():
        rolling = save_dir / 'model-latest.pt'
        if rolling.exists():
            return rolling

    numbered = []
    if save_dir.exists():
        for path in save_dir.iterdir():
            match = _NUMBERED_CHECKPOINT_RE.fullmatch(path.name)
            if match is not None:
                numbered.append((int(match.group(1)), path))
    if numbered:
        return max(numbered, key=lambda item: item[0])[1]

    # Compatibility fallbacks for runs which did not retain numbered saves.
    for filename in ('model-final.pt', 'model_best.pt'):
        candidate = save_dir / filename
        if candidate.exists():
            return candidate
    return save_dir / 'model-latest.pt'


def wandb_init(args, run_name, group_name, log_dir, *, resume=False):
    import wandb

    config_dict = {k: v for k, v in vars(args).items() if k != "config"}
    config_dict["log_dir"] = log_dir
    config_dict["wandb_run_name"] = run_name
    config_dict["wandb_group_name"] = group_name

    init_kwargs = dict(
        project=config_dict["wandb_project"],
        group=group_name[:127],
        name=run_name[:127],
        config=config_dict,
    )

    meta_path = os.path.join(log_dir, 'wandb_run.json')
    resume_id = None
    if resume and os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                resume_id = json.load(f).get('id')
        except Exception:
            resume_id = None
    if resume_id:
        init_kwargs.update(dict(id=resume_id, resume='allow'))

    run = wandb.init(**init_kwargs)
    if not resume_id:
        try:
            os.makedirs(log_dir, exist_ok=True)
            with open(meta_path, 'w') as f:
                json.dump({'id': run.id, 'name': run.name}, f)
        except Exception:
            pass
    return wandb


def str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def load_config(config_path):
    if not config_path:
        return {}
    require_path(config_path, '--config')
    with open(config_path, 'r') as f:
        return json.load(f)


def require_path(path, what):
    """Fail with the resolved path rather than several frames later on an empty string."""
    if not path:
        raise FileNotFoundError(f'{what} is empty; set it in the config or pass it on the command line')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{what} not found: {path!r} -> {os.path.abspath(path)} (cwd {os.getcwd()})')


def get_total_devices():
    world_size = os.environ.get("WORLD_SIZE")
    return int(world_size) if world_size is not None else 1


def compute_accumulation_steps(train_batch_size, per_device_train_batch_size):
    div = per_device_train_batch_size * get_total_devices()
    steps = train_batch_size / div
    if not steps.is_integer():
        raise ValueError(
            f"train_batch_size must be divisible by per_device_batch*total_devices={div}")
    return int(steps)


def compute_per_device_train_batch_size(train_batch_size, gradient_accumulate_every):
    div = gradient_accumulate_every * get_total_devices()
    per_device = train_batch_size / div
    if not per_device.is_integer():
        raise ValueError(
            f"train_batch_size must be divisible by grad_accum*total_devices={div}")
    return int(per_device)


def is_main_process():
    if "RANK" in os.environ:
        return int(os.environ["RANK"]) == 0
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"]) == 0
    return True


def log_if_main(message):
    if is_main_process():
        logger.info(message)


def launch():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', type=str, help='Path to JSON configuration file')
    config_args, remaining_argv = config_parser.parse_known_args()
    config = load_config(config_args.config)

    parser = argparse.ArgumentParser(
        description='3D autoregressive baseline training (biome-conditional)',
        parents=[config_parser])

    parser.add_argument('--mode', type=str, default=config.get('mode', 'conditional'),
                        choices=['conditional', 'unconditional'])

    # backbone (shared with the diffusion models)
    parser.add_argument('--image_size', type=int, default=config.get('image_size', 32))
    parser.add_argument('--time_dim', type=int, default=config.get('time_dim', 256))
    parser.add_argument('--hidden_channels', type=int, default=config.get('hidden_channels', 768))
    parser.add_argument('--patch_size', type=int, default=config.get('patch_size', 4))
    parser.add_argument('--depth', type=int, default=config.get('depth', 25))
    parser.add_argument('--num_heads', type=int, default=config.get('num_heads', 8))
    parser.add_argument('--mlp_ratio', type=float, default=config.get('mlp_ratio', 4.0))
    parser.add_argument('--attn_drop', type=float, default=config.get('attn_drop', 0.0))
    parser.add_argument('--proj_drop', type=float, default=config.get('proj_drop', 0.0))

    # autoregressive specifics
    parser.add_argument('--local_head_depth', type=int, default=config.get('local_head_depth', 2),
                        help='Layers in the within-patch decoder. 0 predicts all p^3 voxels of a '
                             'patch independently (patch-factorized ablation).')
    parser.add_argument('--local_head_dim', type=int, default=config.get('local_head_dim', 384))
    parser.add_argument('--local_head_heads', type=int, default=config.get('local_head_heads', 6))
    parser.add_argument('--token_order', type=str, default=config.get('token_order', 'y_slow'),
                        choices=['y_slow', 'x_slow', 'conv'],
                        help="Raster order over patches and voxels. 'y_slow' generates bottom-up.")
    parser.add_argument('--grad_checkpoint', type=str_to_bool, default=config.get('grad_checkpoint', False),
                        help='Activation checkpointing; the local head is memory hungry at large patch sizes.')
    parser.add_argument('--target_params', type=float, default=config.get('target_params', 0),
                        help='If > 0, pick the backbone depth that lands closest to this many '
                             'million parameters (for parity with the diffusion baselines).')

    # dataset
    parser.add_argument('--dataset_path', type=str, default=config.get('dataset_path', ''))
    parser.add_argument('--mappings_file_path', type=str, default=config.get('mappings_file_path', ''))
    parser.add_argument('--val_dataset_path', type=str, default=config.get('val_dataset_path', ''))
    parser.add_argument('--val_mappings_file_path', type=str, default=config.get('val_mappings_file_path', ''))
    parser.add_argument('--one_hot_on_load', type=str_to_bool, default=config.get('one_hot_on_load', False))
    parser.add_argument('--rotation_aug_prob', type=float, default=config.get('rotation_aug_prob', 0.0))

    # trainer
    parser.add_argument('--train_batch_size', type=int, default=config.get('train_batch_size', 128))
    parser.add_argument('--per_device_train_batch_size', type=int,
                        default=config.get('per_device_train_batch_size', 8))
    parser.add_argument('--gradient_accumulate_every', type=int,
                        default=config.get('gradient_accumulate_every', -1))
    parser.add_argument('--train_lr', type=float, default=config.get('train_lr', 1e-4))
    parser.add_argument('--compile_model', type=str_to_bool, default=config.get('compile_model', False))
    parser.add_argument('--sync_each_microbatch', type=str_to_bool,
                        default=config.get('sync_each_microbatch', False),
                        help='Legacy DDP synchronization for an A/B comparison; normally false.')
    parser.add_argument('--num_workers', type=int, default=config.get('num_workers', 2),
                        help='DataLoader workers per rank, not per node.')
    parser.add_argument('--prefetch_factor', type=int, default=config.get('prefetch_factor', 2))
    parser.add_argument('--performance_every', type=int, default=config.get('performance_every', 10))
    parser.add_argument('--max_train_updates', type=int, default=config.get('max_train_updates', 0),
                        help='Stop and save latest after this many updates this invocation; '
                             'does not change the training LR schedule. 0 disables.')
    parser.add_argument('--max_train_seconds', type=float, default=config.get('max_train_seconds', 0),
                        help='Stop at an update boundary and save latest after this many seconds '
                             'of the training loop; reserve time for startup/evaluation/save.')
    parser.add_argument('--profile_steps', type=int, default=config.get('profile_steps', 0),
                        help='Record this many optimizer updates in a per-rank PyTorch trace; 0 disables.')
    parser.add_argument('--profile_start_step', type=int, default=config.get('profile_start_step', 5),
                        help='Updates to skip before profiler warmup, relative to this invocation.')

    parser.add_argument('--train_num_steps', type=int, default=config.get('train_num_steps', -1))
    parser.add_argument('--train_num_epochs', type=int, default=config.get('train_num_epochs', 10))
    parser.add_argument('--validate_every', type=int, default=config.get('validate_every', -1))
    parser.add_argument('--validate_every_epoch', type=int, default=config.get('validate_every_epoch', 1))
    parser.add_argument('--val_batch_size', type=int, default=config.get('val_batch_size', 0),
                        help='Defaults to the per-device train batch size.')
    parser.add_argument('--val_max_batches', type=int, default=config.get('val_max_batches', 0),
                        help='Cap the batches per validation pass; 0 uses the whole split.')
    parser.add_argument('--ema_update_every', type=int, default=config.get('ema_update_every', 10))
    parser.add_argument('--ema_decay', type=float, default=config.get('ema_decay', 0.995))
    parser.add_argument('--optimizer', type=str, default=config.get('optimizer', 'adamw'))
    parser.add_argument('--weight_decay', type=float, default=config.get('weight_decay', 0.01))
    parser.add_argument('--adam_betas', type=str, default=config.get('adam_betas', '0.9,0.99'))
    parser.add_argument('--warmup_steps', type=int, default=config.get('warmup_steps', 0))
    parser.add_argument('--scheduler', type=str, default=config.get('scheduler', 'cosine'),
                        choices=['linear', 'cosine'])
    parser.add_argument('--min_lr_ratio', type=float, default=config.get('min_lr_ratio', 0.0))

    parser.add_argument('--save_and_sample_every', type=int, default=config.get('save_and_sample_every', -1))
    parser.add_argument('--save_and_sample_every_epoch', type=int,
                        default=config.get('save_and_sample_every_epoch', 20))
    parser.add_argument('--save_latest_every', type=int, default=config.get('save_latest_every', 0),
                        help='Save an atomic model-latest.pt without sampling every N steps; '
                             '0 disables rolling checkpoints.')
    parser.add_argument('--num_samples', type=int, default=config.get('num_samples', 16))
    parser.add_argument('--sample_during_training', type=str_to_bool,
                        default=config.get('sample_during_training', True),
                        help='Disable for utilization-enforced jobs; checkpoint cadence is preserved. '
                             'Use ar_generate_eval_samples.py for offline previews/FID.')
    parser.add_argument('--results_folder', type=str, default=config.get('results_folder', './results'))
    parser.add_argument('--run_name', type=str, default=config.get('run_name', 'ar_dit'))
    parser.add_argument('--exp_name', type=str, default=config.get('exp_name', ''))
    parser.add_argument('--amp', type=str_to_bool, default=config.get('amp', True))
    parser.add_argument('--mixed_precision_type', type=str, default=config.get('mixed_precision_type', 'bf16'),
                        choices=['fp16', 'bf16'])
    parser.add_argument('--max_grad_norm', type=float, default=config.get('max_grad_norm', 1.0))
    parser.add_argument('--evaluate_ema_model', type=str_to_bool, default=config.get('evaluate_ema_model', True))
    parser.add_argument('--save_only_last_checkpoint', type=str_to_bool,
                        default=config.get('save_only_last_checkpoint', True))

    # conditioning / sampling
    parser.add_argument('--num_classes', type=int, default=config.get('num_classes', 15))
    parser.add_argument('--cond_drop_prob', type=float, default=config.get('cond_drop_prob', 0.2))
    parser.add_argument('--comb_method', type=str, default=config.get('comb_method', 'add'),
                        choices=['add', 'cat'])
    parser.add_argument('--default_cond_scale', type=float, default=config.get('default_cond_scale', 4.0))
    parser.add_argument('--sample_temperature', type=float, default=config.get('sample_temperature', 1.0))
    parser.add_argument('--sample_top_k', type=int, default=config.get('sample_top_k', 0))
    parser.add_argument('--sample_top_p', type=float, default=config.get('sample_top_p', 0.0))

    parser.add_argument('--device', type=str, default=config.get('device', 'cuda'))
    parser.add_argument('--seed', type=int, default=config.get('seed', 42),
                        help='Base seed; each distributed rank uses seed + rank.')

    # wandb
    parser.add_argument('--use_wandb', type=str_to_bool, default=config.get('use_wandb', True))
    parser.add_argument('--wandb_project', type=str, default=config.get('wandb_project', 'dream-cubed'))
    parser.add_argument('--wandb_group_name', type=str, default=config.get('wandb_group_name', 'autoregressive_dit'))
    parser.add_argument('--wandb_run_name', type=str, default=config.get('wandb_run_name', ''))
    parser.add_argument('--wandb_log_images', type=str_to_bool, default=config.get('wandb_log_images', True))
    parser.add_argument('--wandb_resume', type=str_to_bool, default=config.get('wandb_resume', False),
                        help='Resume the previous W&B run ID. This is independent of --resume; '
                             'leave false to log checkpoint continuation to a new W&B run.')

    # FID sample dumps
    parser.add_argument('--fid_ref_images_dir', type=str, default=config.get('fid_ref_images_dir', None))
    parser.add_argument('--fid_num_samples', type=int, default=config.get('fid_num_samples', 0))
    parser.add_argument('--fid_every_steps', type=int, default=config.get('fid_every_steps', 0))
    parser.add_argument('--fid_every_epoch', type=int, default=config.get('fid_every_epoch', 0))
    parser.add_argument('--fid_textures_dir', type=str, default=config.get('fid_textures_dir', 'block_textures/'))
    parser.add_argument('--fid_batch_size', type=int, default=config.get('fid_batch_size', 32))
    parser.add_argument('--fid_num_workers', type=int, default=config.get('fid_num_workers', 2))

    # resume
    parser.add_argument('--resume', type=str_to_bool, default=config.get('resume', False))
    parser.add_argument('--resume_path', type=str, default=config.get('resume_path', ''),
                        help='Explicit checkpoint path (or filename inside the run directory). '
                             'By default, --resume selects model-latest.pt when present, then '
                             'the highest numbered model-N.pt.')

    args = parser.parse_args(remaining_argv)
    args.training_recipe = 'voxel-ar-clean-v1'
    # Bind before set_seed/model.to: otherwise all torchrun processes initially
    # allocate their model and a CUDA context on GPU 0.
    if torch.device(args.device).type == 'cuda' and 'LOCAL_RANK' in os.environ:
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        args.device = f"cuda:{os.environ['LOCAL_RANK']}"
    process_rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', 0)))
    set_seed(int(args.seed) + process_rank)

    require_path(args.mappings_file_path, '--mappings_file_path')
    require_path(args.dataset_path, '--dataset_path')
    if args.val_dataset_path:
        require_path(args.val_dataset_path, '--val_dataset_path')

    if len(args.exp_name) == 0:
        args.exp_name = f"exp_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if len(args.wandb_run_name) == 0:
        args.wandb_run_name = f"{args.run_name}"
    adam_betas = tuple(map(float, args.adam_betas.split(',')))

    if args.per_device_train_batch_size is not None and args.per_device_train_batch_size <= 0:
        effective_grad_accum = args.gradient_accumulate_every if (args.gradient_accumulate_every or 0) > 0 else 1
        args.per_device_train_batch_size = compute_per_device_train_batch_size(
            args.train_batch_size, effective_grad_accum)
        log_if_main(f"Computed per_device_train_batch_size={args.per_device_train_batch_size}")

    save_dir = os.path.join(args.results_folder, args.run_name, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = resolve_resume_checkpoint(save_dir, args.resume_path)
        if not resume_checkpoint.exists():
            raise FileNotFoundError(
                f"--resume was set but checkpoint not found: {resume_checkpoint}. "
                "Pass --resume_path explicitly.")
        args.resolved_resume_path = str(resume_checkpoint)

    # vocabulary size K from the dataset mappings (same source as MD4)
    mappings = torch.load(args.mappings_file_path, map_location='cpu', weights_only=False)
    converter = BlockBiomeConverter(mappings['block_mappings'], mappings['biome_mappings'])
    vocab_size = len(converter.block_to_index)
    log_if_main(f"Vocabulary: {vocab_size} block types")

    model_kwargs = dict(
        vocab_size=vocab_size,
        hidden_channels=args.hidden_channels,
        image_size=args.image_size,
        patch_size=args.patch_size,
        time_dim=args.time_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        attn_drop=args.attn_drop,
        proj_drop=args.proj_drop,
        comb_method=args.comb_method,
        class_conditional=(args.mode == 'conditional'),
        num_classes=args.num_classes,
        cond_drop_prob=args.cond_drop_prob,
        local_head_depth=args.local_head_depth,
        local_head_dim=args.local_head_dim,
        local_head_heads=args.local_head_heads,
        token_order=args.token_order,
        use_gradient_checkpointing=args.grad_checkpoint,
    )

    if args.target_params and args.target_params > 0:
        target = int(args.target_params * 1_000_000)
        depth, predicted = depth_for_target_params(target, model_kwargs)
        log_if_main(f"Parameter matching: depth={depth} gives {predicted / 1e6:.1f}M "
                    f"(target {args.target_params}M)")
        args.depth = depth
        model_kwargs['depth'] = depth

    model = ARDiT3D(**model_kwargs).to(args.device)
    breakdown = model.param_breakdown()
    log_if_main("Parameter breakdown (M): " + ", ".join(
        f"{k}={v / 1e6:.2f}" for k, v in breakdown.items()))
    log_if_main(f"Training model with {breakdown['total'] // 1_000_000}M parameters "
                f"({model.num_tokens} patch tokens of {model.patch_voxels} voxels)")

    def _make_ds(path, mappings_file_path, rotation_aug_prob):
        if not path:
            return None
        if os.path.isdir(path) and os.path.exists(os.path.join(path, 'manifest.json')):
            return VoxelDatasetMemmapConditional(
                dir_path=path,
                one_hot_on_load=args.one_hot_on_load,
                mappings_file_path=mappings_file_path,
                rotation_aug_prob=rotation_aug_prob,
            )
        return VoxelDatasetConditional(
            file_path=path,
            one_hot_on_load=args.one_hot_on_load,
            mappings_file_path=mappings_file_path,
            rotation_aug_prob=rotation_aug_prob,
        )

    dataset = _make_ds(args.dataset_path, args.mappings_file_path, args.rotation_aug_prob)
    val_dataset = _make_ds(args.val_dataset_path, args.val_mappings_file_path, 0.0)

    if args.gradient_accumulate_every and args.gradient_accumulate_every > 0:
        gradient_accumulate_every = args.gradient_accumulate_every
    else:
        gradient_accumulate_every = compute_accumulation_steps(
            args.train_batch_size, args.per_device_train_batch_size)

    effective_batch = (args.per_device_train_batch_size * get_total_devices()
                       * gradient_accumulate_every)
    if gradient_accumulate_every < 1 or effective_batch != args.train_batch_size:
        raise ValueError(f'Effective batch {effective_batch} differs from train_batch_size '
                         f'{args.train_batch_size}; fix per-device batch / accumulation.')
    if not args.sample_during_training and args.fid_num_samples:
        raise ValueError('sample_during_training=false requires fid_num_samples=0; '
                         'run FID generation offline.')

    steps_per_epoch = max(1, len(dataset) // args.train_batch_size)
    if args.train_num_steps <= 0:
        assert args.train_num_epochs > 0, "Either train_num_steps or train_num_epochs must be > 0"
        args.train_num_steps = steps_per_epoch * args.train_num_epochs
        log_if_main(f"Setting train_num_steps to {args.train_num_steps} ({args.train_num_epochs} epochs)")
    else:
        assert args.train_num_epochs <= 0, "Only one of train_num_steps or train_num_epochs can be > 0"
        args.train_num_epochs = args.train_num_steps // steps_per_epoch

    log_if_main("Training for %d steps (%d epochs), %d samples, batch %d across %d devices "
                "(grad accum %d, per-device %d)" %
                (args.train_num_steps, args.train_num_epochs, len(dataset), args.train_batch_size,
                 get_total_devices(), gradient_accumulate_every, args.per_device_train_batch_size))

    if args.save_and_sample_every <= 0:
        assert args.save_and_sample_every_epoch > 0
        args.save_and_sample_every = steps_per_epoch * args.save_and_sample_every_epoch
    if args.validate_every <= 0:
        assert args.validate_every_epoch > 0
        args.validate_every = steps_per_epoch * args.validate_every_epoch
    if args.fid_every_steps <= 0 and args.fid_every_epoch > 0:
        args.fid_every_steps = steps_per_epoch * args.fid_every_epoch

    resolved_config = dict(vars(args))
    if is_main_process():
        config_path = os.path.join(save_dir, 'config.json')
        if args.resume and os.path.exists(config_path):
            stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            config_path = os.path.join(save_dir, f'config_resume_{stamp}.json')
        with open(config_path, 'w') as f:
            json.dump(resolved_config, f, indent=2, sort_keys=True)

    log_if_main(f"Sampling/saving every {args.save_and_sample_every} steps, "
                f"validating every {args.validate_every} steps")

    trainer = VoxelTrainerAR(
        model,
        dataset,
        save_dir,
        vocab_size=vocab_size,
        train_batch_size=args.per_device_train_batch_size,
        gradient_accumulate_every=gradient_accumulate_every,
        train_lr=args.train_lr,
        train_num_steps=args.train_num_steps,
        ema_update_every=args.ema_update_every,
        ema_decay=args.ema_decay,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        adam_betas=adam_betas,
        warmup_steps=args.warmup_steps,
        scheduler=args.scheduler,
        min_lr_ratio=args.min_lr_ratio,
        save_and_sample_every=args.save_and_sample_every,
        save_latest_every=args.save_latest_every,
        num_samples=args.num_samples,
        amp=args.amp,
        mixed_precision_type=args.mixed_precision_type,
        compile_model=args.compile_model,
        sync_each_microbatch=args.sync_each_microbatch,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        performance_every=args.performance_every,
        max_train_updates=args.max_train_updates,
        max_train_seconds=args.max_train_seconds,
        profile_steps=args.profile_steps,
        profile_start_step=args.profile_start_step,
        sample_during_training=args.sample_during_training,
        max_grad_norm=args.max_grad_norm,
        evaluate_ema_model=args.evaluate_ema_model,
        save_only_last_checkpoint=args.save_only_last_checkpoint,
        mappings_file_path=args.mappings_file_path,
        run_name=args.run_name,
        num_classes=(args.num_classes if args.mode == 'conditional' else None),
        default_cond_scale=args.default_cond_scale,
        sample_temperature=args.sample_temperature,
        sample_top_k=args.sample_top_k,
        sample_top_p=args.sample_top_p,
        val_dataset=val_dataset,
        val_batch_size=(args.val_batch_size or None),
        val_max_batches=(args.val_max_batches or None),
        val_every_n_steps=args.validate_every,
        use_wandb=args.use_wandb,
        wandb_log_images=args.wandb_log_images,
        fid_ref_images_dir=args.fid_ref_images_dir,
        fid_num_samples=args.fid_num_samples,
        fid_every_steps=args.fid_every_steps,
        fid_textures_dir=args.fid_textures_dir,
        fid_batch_size=args.fid_batch_size,
        fid_num_workers=args.fid_num_workers,
    )

    if args.resume:
        log_if_main(f"Resuming from checkpoint: {resume_checkpoint}")
        trainer.load(resume_checkpoint)

    # Initialize logging after loading the model and optimizer, if resuming.
    if args.use_wandb and is_main_process():
        wandb_init(args, run_name=args.wandb_run_name, group_name=args.wandb_group_name,
                   log_dir=save_dir, resume=args.wandb_resume)
        log_if_main(f"WandB logging enabled (project: {args.wandb_project}, "
                    f"group: {args.wandb_group_name}, resume: {args.wandb_resume})")
    trainer.accelerator.wait_for_everyone()

    try:
        trainer.train()
    finally:
        if args.use_wandb and is_main_process():
            import wandb
            wandb.finish()


if __name__ == '__main__':
    launch()
