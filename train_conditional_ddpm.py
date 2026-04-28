import argparse
import datetime
import json
import os

from denoising_diffusion_pytorch_3d import (
    DDPMoxelDatasetMemmapConditional,
    GaussianDiffusion3D_CFG,
    VoxelTrainerClassConditional,
)
from dit import DiT3D




def str_to_bool(v):
    """Convert string to boolean for argparse."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def load_config(config_path):
    """Load configuration from JSON file."""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}

def wandb_init(args: argparse.Namespace, run_name: str, group_name: str, log_dir: str, *, resume: bool = False):
    import wandb

    # match train_discrete.py behavior
    args_dict = vars(args)
    config_dict = {k: v for k, v in args_dict.items() if k != "config"}
    config_dict["log_dir"] = log_dir
    config_dict["wandb_run_name"] = run_name
    config_dict["wandb_group_name"] = group_name

    init_kwargs = dict(
        project=config_dict.get("wandb_project", "dream-cubed"),
        group=group_name[:127],
        name=run_name[:127],
        config=config_dict,
    )

    meta_path = os.path.join(log_dir, 'wandb_run.json')
    resume_id = None
    if resume and os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                resume_id = meta.get('id')
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

def get_total_devices():
    world_size = os.environ.get("WORLD_SIZE")
    if world_size is not None:
        return int(world_size)
    return 1

def compute_accumulation_steps(train_batch_size, per_device_train_batch_size):
    total_devices = get_total_devices()
    div = per_device_train_batch_size * total_devices
    steps = train_batch_size / div
    if not steps.is_integer():
        raise ValueError(
            "train_batch_size must be divisible by "
            f"per_device_batch*total_devices={div}"
        )
    return int(steps)

def compute_per_device_train_batch_size(train_batch_size, gradient_accumulate_every):
    total_devices = get_total_devices()
    div = gradient_accumulate_every * total_devices
    per_device = train_batch_size / div
    if not per_device.is_integer():
        raise ValueError(
            "train_batch_size must be divisible by "
            f"grad_accum*total_devices={div}"
        )
    return int(per_device)

def launch():
    # First, create a parser just to get the config file path
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', type=str, help='Path to JSON configuration file')
    config_args, remaining_argv = config_parser.parse_known_args()
    
    # Load config file if provided
    config = load_config(config_args.config)
    
    parser = argparse.ArgumentParser(
        description='3D Conditional DDPM Training (DiT backbone, embeddings-as-continuous data)',
        parents=[config_parser]
    )

    # Model (DiT only)
    parser.add_argument('--image_size', type=int, default=config.get('image_size', 16), help='Voxel cube size')
    parser.add_argument('--time_dim', type=int, default=config.get('time_dim', 256), help='Time embedding dimension')
    parser.add_argument('--data_channels', type=int, default=config.get('data_channels', 64), help='Embedding channels (E)')

    parser.add_argument('--hidden_channels', type=int, default=config.get('hidden_channels', 768), help='DiT hidden dim')
    parser.add_argument('--patch_size', type=int, default=config.get('patch_size', 2), help='DiT patch size')
    parser.add_argument('--depth', type=int, default=config.get('depth', 6), help='DiT layers')
    parser.add_argument('--num_heads', type=int, default=config.get('num_heads', 8), help='DiT heads')
    parser.add_argument('--mlp_ratio', type=float, default=config.get('mlp_ratio', 4.0), help='DiT MLP ratio')
    parser.add_argument('--attn_drop', type=float, default=config.get('attn_drop', 0.0), help='DiT attn drop')
    parser.add_argument('--proj_drop', type=float, default=config.get('proj_drop', 0.0), help='DiT proj drop')
    parser.add_argument('--comb_method', type=str, default=config.get('comb_method', 'add'), choices=['add', 'cat'], help='How to combine time+class conditioning in DiT')

    # Conditioning / CFG training knob
    parser.add_argument('--num_classes', type=int, default=config.get('num_classes', 11), help='Number of conditioning classes')
    parser.add_argument('--cond_drop_prob', type=float, default=config.get('cond_drop_prob', 0.2), help='Classifier-free guidance drop probability during training')
    parser.add_argument('--default_cond_scale', type=float, default=config.get('default_cond_scale', 4.0), help='Default CFG scale used for sampling previews')

    # Dataset (processed memmap format; indices -> embeddings)
    parser.add_argument('--dataset_path', type=str, default=config.get('dataset_path', 'data/voxel_dataset_test_processed_emb_norm.pt'), help='Path to processed dataset directory (must contain manifest.json)')
    parser.add_argument('--mappings_file_path', type=str, default=config.get('mappings_file_path', 'data/voxel_dataset_test_mappings_norm.pt'), help='Mappings file (needed to map dataset indices -> original block IDs)')
    parser.add_argument('--val_dataset_path', type=str, default=config.get('val_dataset_path', ''), help='Optional validation dataset directory (manifest.json)')
    parser.add_argument('--val_mappings_file_path', type=str, default=config.get('val_mappings_file_path', ''), help='Optional validation mappings file (defaults to mappings_file_path when empty)')
    parser.add_argument('--rotation_aug_prob', type=float, default=config.get('rotation_aug_prob', 0.0), help='Probability of applying 90° rotation augmentation per sample (indices only)')
    parser.add_argument('--crop', type=str_to_bool, default=config.get('crop', False), help='Crop dataset to smaller size')
    parser.add_argument('--crop_size', type=int, default=config.get('crop_size', 16), help='Size to crop dataset to')
    parser.add_argument('--embeddings_path', type=str, default=config.get('embeddings_path', 'assets/block_embeddings_norm.npy'), help='Path to block embeddings (.npy table row=block_id, or .pt dict)')
    parser.add_argument('--embeddings_dict_key', type=str, default=config.get('embeddings_dict_key', None), help='If embeddings_path is a .pt with nested dict, optional key to select')

    # Diffusion
    parser.add_argument('--timesteps', type=int, default=config.get('timesteps', 2000), help='Number of diffusion timesteps')
    parser.add_argument('--sampling_timesteps', type=int, default=config.get('sampling_timesteps', None), help='Number of sampling timesteps (for DDIM)')
    parser.add_argument('--objective', type=str, default=config.get('objective', 'pred_v'), choices=['pred_noise', 'pred_x0', 'pred_v'], help='Diffusion objective')
    parser.add_argument('--beta_schedule', type=str, default=config.get('beta_schedule', 'cosine'), choices=['linear', 'cosine', 'sigmoid'], help='Beta noise schedule')
    parser.add_argument('--ddim_sampling_eta', type=float, default=config.get('ddim_sampling_eta', 0.0), help='DDIM sampling eta parameter')
    parser.add_argument('--auto_normalize', type=str_to_bool, default=config.get('auto_normalize', False), help='Automatically normalize data')
    parser.add_argument('--offset_noise_strength', type=float, default=config.get('offset_noise_strength', 0.0), help='Offset noise strength')
    parser.add_argument('--min_snr_loss_weight', type=str_to_bool, default=config.get('min_snr_loss_weight', False), help='Use min-SNR loss weighting')
    parser.add_argument('--min_snr_gamma', type=float, default=config.get('min_snr_gamma', 5.0), help='Min-SNR gamma parameter')
    
    # Trainer (train_discrete-style global batch semantics)
    parser.add_argument('--train_batch_size', type=int, default=config.get('train_batch_size', 128), help='Global / effective batch size across devices')
    parser.add_argument('--per_device_train_batch_size', type=int, default=config.get('per_device_train_batch_size', 16), help='Per-device dataloader batch size')
    parser.add_argument('--gradient_accumulate_every', type=int, default=config.get('gradient_accumulate_every', -1), help='Override gradient accumulation steps (if > 0)')
    parser.add_argument('--train_lr', type=float, default=config.get('train_lr', 8e-5), help='Learning rate')
    parser.add_argument('--optimizer', type=str, default=config.get('optimizer', 'adamw'), choices=['adamw', 'adam', 'AdamW', 'Adam'], help='Optimizer (DDPM trainer supports AdamW/Adam)')
    parser.add_argument('--weight_decay', type=float, default=config.get('weight_decay', 0.01))
    parser.add_argument('--warmup_steps', type=int, default=config.get('warmup_steps', 0))
    parser.add_argument('--scheduler', type=str, default=config.get('scheduler', 'cosine'), choices=['linear', 'cosine', 'Linear', 'Cosine'])
    parser.add_argument('--min_lr_ratio', type=float, default=config.get('min_lr_ratio', 0.0))
    parser.add_argument('--compile_model', type=str_to_bool, default=config.get('compile_model', False), help='Compile model with torch.compile (Accelerate dynamo plugin)')
    # dataloader perf
    parser.add_argument('--dataloader_num_workers', type=int, default=config.get('dataloader_num_workers', 2))
    parser.add_argument('--dataloader_prefetch_factor', type=int, default=config.get('dataloader_prefetch_factor', 2))
    parser.add_argument('--dataloader_persistent_workers', type=str_to_bool, default=config.get('dataloader_persistent_workers', True))

    # note: either num steps or num epochs must be set to > 0, not both
    parser.add_argument('--train_num_steps', type=int, default=config.get('train_num_steps', -1))
    parser.add_argument('--train_num_epochs', type=int, default=config.get('train_num_epochs', 10))
    # validation cadence (MD4-style)
    parser.add_argument('--validate_every', type=int, default=config.get('validate_every', -1), help='Validate every N steps (<=0 to derive from validate_every_epoch)')
    parser.add_argument('--validate_every_epoch', type=int, default=config.get('validate_every_epoch', 1), help='Validate every N epochs (used if validate_every <= 0)')

    # note: either save_and_sample_every or save_and_sample_every_epoch must be set to > 0, not both
    parser.add_argument('--save_and_sample_every', type=int, default=config.get('save_and_sample_every', 500))
    parser.add_argument('--save_and_sample_every_epoch', type=int, default=config.get('save_and_sample_every_epoch', -1))

    parser.add_argument('--ema_update_every', type=int, default=config.get('ema_update_every', 10), help='EMA update frequency')
    parser.add_argument('--ema_decay', type=float, default=config.get('ema_decay', 0.995), help='EMA decay rate')
    parser.add_argument('--adam_betas', type=str, default=config.get('adam_betas', '0.9,0.99'), help='Adam beta parameters (comma-separated)')
    parser.add_argument('--num_samples', type=int, default=config.get('num_samples', 25), help='Number of samples to generate')
    parser.add_argument('--results_folder', type=str, default=config.get('results_folder', './results'), help='Results output folder')
    parser.add_argument('--run_name', type=str, default=config.get('run_name', 'ddpm_dit_conditional'), help='Run name for logging')
    parser.add_argument('--exp_name', type=str, default=config.get('exp_name', ''), help='Experiment name (subfolder under run_name)')
    parser.add_argument('--amp', type=str_to_bool, default=config.get('amp', True), help='Use automatic mixed precision')
    parser.add_argument('--mixed_precision_type', type=str, default=config.get('mixed_precision_type', 'fp16'), choices=['fp16', 'bf16'], help='Mixed precision type')
    parser.add_argument('--split_batches', type=str_to_bool, default=config.get('split_batches', False), help='Accelerate split_batches (should be False when using per_device_train_batch_size semantics)')
    parser.add_argument('--max_grad_norm', type=float, default=config.get('max_grad_norm', 1.0), help='Maximum gradient norm for clipping')
    parser.add_argument('--evaluate_ema_model', type=str_to_bool, default=config.get('evaluate_ema_model', True), help='Evaluate EMA model during training')
    parser.add_argument('--save_only_last_checkpoint', type=str_to_bool, default=config.get('save_only_last_checkpoint', False), help='Only save final checkpoint at end of training')

    parser.add_argument('--device', type=str, default=config.get('device', 'cuda'), help='Device to use for training')

    # wandb (match train_discrete.py config keys)
    parser.add_argument('--use_wandb', type=str_to_bool, default=config.get('use_wandb', False))
    parser.add_argument('--wandb_project', type=str, default=config.get('wandb_project', 'dream-cubed'))
    parser.add_argument('--wandb_group_name', type=str, default=config.get('wandb_group_name', 'ddpm_dit'))
    parser.add_argument('--wandb_run_name', type=str, default=config.get('wandb_run_name', ''))
    parser.add_argument('--wandb_log_images', type=str_to_bool, default=config.get('wandb_log_images', True))
    
    args = parser.parse_args(remaining_argv)

    if len(args.exp_name) == 0:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.exp_name = f"exp_{timestamp}"
    if len(getattr(args, "wandb_run_name", "")) == 0:
        args.wandb_run_name = f"{args.run_name}"

    # split_batches=False required for per_device_train_batch_size semantics in multi-GPU
    if bool(args.split_batches) and get_total_devices() > 1:
        raise ValueError(
            "split_batches=True is incompatible with per_device_train_batch_size semantics. "
            "Set --split_batches false for DDPM conditional runs."
        )

    adam_betas = tuple(map(float, args.adam_betas.split(',')))

    # derive per-device batch size if requested (<=0)
    if args.per_device_train_batch_size is not None and args.per_device_train_batch_size <= 0:
        effective_grad_accum = args.gradient_accumulate_every if (args.gradient_accumulate_every and args.gradient_accumulate_every > 0) else 1
        args.per_device_train_batch_size = compute_per_device_train_batch_size(args.train_batch_size, effective_grad_accum)

    # dataset (must be MD4 processed memmap dir)
    if not (os.path.isdir(args.dataset_path) and os.path.exists(os.path.join(args.dataset_path, "manifest.json"))):
        raise ValueError(
            f"--dataset_path must point to a processed dataset directory containing manifest.json (MD4 processed memmap format). Got: {args.dataset_path}"
        )

    voxel_dataset = DDPMoxelDatasetMemmapConditional(
        dir_path=args.dataset_path,
        mappings_file_path=args.mappings_file_path,
        rotation_aug_prob=float(args.rotation_aug_prob),
        block_embeddings_path=args.embeddings_path,
        embeddings_dict_key=args.embeddings_dict_key,
        crop=args.crop,
        crop_size=args.crop_size,
    )
    # strict sanity: embedding dim must match model input channels
    if int(getattr(voxel_dataset, "E", -1)) != int(args.data_channels):
        raise ValueError(
            f"Embedding dim mismatch: dataset embeddings have E={getattr(voxel_dataset, 'E', None)} "
            f"but config/data_channels={args.data_channels}. "
            f"Check --embeddings_path and how it was generated."
        )

    # optional validation dataset
    val_dataset = None
    val_path = str(args.val_dataset_path or "").strip()
    if val_path:
        if not (os.path.isdir(val_path) and os.path.exists(os.path.join(val_path, "manifest.json"))):
            raise ValueError(f"--val_dataset_path must be a processed dataset dir containing manifest.json. Got: {val_path}")
        val_mappings = str(args.val_mappings_file_path or "").strip() or args.mappings_file_path
        val_dataset = DDPMoxelDatasetMemmapConditional(
            dir_path=val_path,
            mappings_file_path=val_mappings,
            rotation_aug_prob=0.0,
            block_embeddings_path=args.embeddings_path,
            embeddings_dict_key=args.embeddings_dict_key,
            crop=args.crop,
            crop_size=args.crop_size,
        )
        if int(getattr(val_dataset, "E", -1)) != int(args.data_channels):
            raise ValueError(
                f"Validation embedding dim mismatch: val dataset embeddings have E={getattr(val_dataset, 'E', None)} "
                f"but config/data_channels={args.data_channels}."
            )

    # derive gradient accumulation steps (train_discrete semantics)
    if args.gradient_accumulate_every and args.gradient_accumulate_every > 0:
        gradient_accumulate_every = int(args.gradient_accumulate_every)
    else:
        gradient_accumulate_every = compute_accumulation_steps(
            args.train_batch_size, args.per_device_train_batch_size
        )

    # derive steps/epochs
    if args.train_num_steps <= 0:
        assert args.train_num_epochs > 0, "Either train_num_steps or train_num_epochs must be > 0"
        steps_per_epoch = max(1, len(voxel_dataset) // int(args.train_batch_size))
        args.train_num_steps = int(steps_per_epoch * int(args.train_num_epochs))
    else:
        assert args.train_num_epochs <= 0, "Only one of train_num_steps or train_num_epochs can be > 0"

    if args.save_and_sample_every <= 0:
        assert args.save_and_sample_every_epoch > 0, "Either save_and_sample_every or save_and_sample_every_epoch must be > 0"
        steps_per_epoch = max(1, len(voxel_dataset) // int(args.train_batch_size))
        args.save_and_sample_every = int(steps_per_epoch * int(args.save_and_sample_every_epoch))
    else:
        assert args.save_and_sample_every_epoch <= 0, "Only one of save_and_sample_every or save_and_sample_every_epoch can be > 0"

    # derive validation cadence (MD4-style)
    steps_per_epoch = max(1, len(voxel_dataset) // int(args.train_batch_size))
    if args.validate_every <= 0:
        if val_dataset is not None:
            assert args.validate_every_epoch > 0, "Either validate_every or validate_every_epoch must be > 0 when val_dataset is set"
            args.validate_every = int(steps_per_epoch * int(args.validate_every_epoch))
        else:
            args.validate_every = -1
    else:
        assert args.validate_every_epoch <= 0 or int(args.validate_every_epoch) <= 0, "Only one of validate_every or validate_every_epoch can be > 0"

    # save resolved config next to training results (results/run_name/exp_name/config.json)
    save_dir = os.path.join(args.results_folder, args.run_name, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(dict(vars(args), gradient_accumulate_every=int(gradient_accumulate_every)), f, indent=2, sort_keys=True)

    # wandb init (entrypoint owns init; trainer only calls wandb.log)
    is_main = True
    if "LOCAL_RANK" in os.environ:
        is_main = int(os.environ["LOCAL_RANK"]) == 0
    elif "RANK" in os.environ:
        is_main = int(os.environ["RANK"]) == 0
    if args.use_wandb and is_main:
        wandb_init(args, run_name=args.wandb_run_name, group_name=args.wandb_group_name, log_dir=save_dir, resume=False)

    # build model (DiT class-conditional)
    model = DiT3D(
        in_dim=args.data_channels,
        out_dim=args.data_channels,
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
        class_conditional=True,
        num_classes=args.num_classes,
        cond_drop_prob=args.cond_drop_prob,
    ).to(args.device)


    # Create diffusion model
    diffusion_kwargs = {
        'image_size': args.image_size,
        'timesteps': args.timesteps,
        'objective': args.objective,
        'beta_schedule': args.beta_schedule,
        'ddim_sampling_eta': args.ddim_sampling_eta,
        'offset_noise_strength': args.offset_noise_strength,
        'min_snr_loss_weight': args.min_snr_loss_weight,
        'min_snr_gamma': args.min_snr_gamma,
        'sampling_timesteps': args.sampling_timesteps,
        'auto_normalize': args.auto_normalize
        # 'mappings_file': args.mappings_file_path
    }


    diffusion = GaussianDiffusion3D_CFG(model, **diffusion_kwargs)

    trainer = VoxelTrainerClassConditional(
        diffusion,
        voxel_dataset, 
        train_batch_size=int(args.per_device_train_batch_size),
        train_lr=args.train_lr,
        train_num_steps=args.train_num_steps,
        gradient_accumulate_every=int(gradient_accumulate_every),
        ema_update_every=args.ema_update_every,
        ema_decay=args.ema_decay,
        optimizer=args.optimizer,
        weight_decay=float(args.weight_decay),
        adam_betas=adam_betas,
        warmup_steps=int(args.warmup_steps),
        scheduler=args.scheduler,
        min_lr_ratio=float(args.min_lr_ratio),
        amp=args.amp,
        mixed_precision_type=args.mixed_precision_type,
        save_and_sample_every=args.save_and_sample_every,
        num_samples=args.num_samples,
        results_folder=args.results_folder,
        run_name=os.path.join(args.run_name, args.exp_name),
        split_batches=args.split_batches,
        max_grad_norm=args.max_grad_norm,
        evaluate_ema_model=args.evaluate_ema_model,
        save_only_last_checkpoint=args.save_only_last_checkpoint,
        mappings_file_path=args.mappings_file_path,
        num_classes=args.num_classes,
        default_cond_scale=args.default_cond_scale,
        dataloader_num_workers=int(args.dataloader_num_workers),
        dataloader_prefetch_factor=(None if int(args.dataloader_prefetch_factor) <= 0 else int(args.dataloader_prefetch_factor)),
        dataloader_persistent_workers=bool(args.dataloader_persistent_workers),
        val_dataset=val_dataset,
        val_every_n_steps=(args.validate_every if (val_dataset is not None and args.validate_every and args.validate_every > 0) else None),
        compile_model=bool(args.compile_model),
        use_wandb=bool(args.use_wandb),
        wandb_log_images=bool(args.wandb_log_images),
    )

    trainer.train()
    # train(args)


if __name__ == '__main__':
    launch()