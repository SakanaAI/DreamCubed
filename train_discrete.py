import argparse
import json
import os
import torch
import logging
import datetime

# from denoising_diffusion_pytorch_3d import VoxelDataset, VoxelDatasetConditional
from dit import DiT3D
from discrete_diffusion_md4 import VoxelTrainerMD4, VoxelDatasetConditional, VoxelDatasetMemmapConditional
from data_utils import BlockBiomeConverter

def wandb_init(args: argparse.Namespace, run_name: str, group_name: str, log_dir: str, *, resume: bool = False):
    import wandb

    # wandb has a 128-size character limit on the group name
    args_dict = vars(args)
    config_dict = {k: v for k, v in args_dict.items() if k != "config"}
    config_dict["log_dir"] = log_dir
    config_dict["wandb_run_name"] = run_name
    config_dict["wandb_group_name"] = group_name

    # try to resume existing run if requested
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
                meta = json.load(f)
                resume_id = meta.get('id')
        except Exception:
            resume_id = None
    if resume_id:
        init_kwargs.update(dict(id=resume_id, resume='allow'))

    run = wandb.init(**init_kwargs)

    # persist run id for future resumes (only when starting a fresh run)
    if not resume_id:
        try:
            os.makedirs(log_dir, exist_ok=True)
            with open(meta_path, 'w') as f:
                json.dump({'id': run.id, 'name': run.name}, f)
        except Exception:
            pass
    return wandb

# unified logger (matches conditional script behavior)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def load_config(config_path):
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}


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
    print(f"total_devices: {total_devices}")
    print(f"gradient_accumulate_every: {gradient_accumulate_every}")
    print(f"train_batch_size: {train_batch_size}")
    div = gradient_accumulate_every * total_devices
    per_device = train_batch_size / div
    print(f"per_device: {per_device}")
    if not per_device.is_integer():
        raise ValueError(
            "train_batch_size must be divisible by "
            f"grad_accum*total_devices={div}"
        )
    return int(per_device)


def is_main_process():
    if "LOCAL_RANK" in os.environ:
        is_main_process = int(os.environ["LOCAL_RANK"]) == 0
    elif "RANK" in os.environ:
        is_main_process = int(os.environ["RANK"]) == 0
    else:
        is_main_process = True
    return is_main_process


def log_if_main(message):
    if is_main_process():
        logger.info(message)


def launch():
    # first parse config path
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', type=str, help='Path to JSON configuration file')
    config_args, remaining_argv = config_parser.parse_known_args()

    config = load_config(config_args.config)

    parser = argparse.ArgumentParser(description='3D Discrete (MD4) Training (unified conditional/unconditional)', parents=[config_parser])

    # mode selection
    parser.add_argument('--mode', type=str, default=config.get('mode', 'conditional'), choices=['conditional', 'unconditional'], help="Training mode: 'conditional' or 'unconditional'")

    # model selection
    parser.add_argument('--model_type', type=str, default=config.get('model_type', 'unet'), help="Model type: 'unet' or 'dit'")

    # shared / unet args
    parser.add_argument('--image_size', type=int, default=config.get('image_size', 16), help='Voxel cube size')
    parser.add_argument('--time_dim', type=int, default=config.get('time_dim', 256), help='Time embedding dimension')
    parser.add_argument('--is_ddim_sampling', type=str_to_bool, default=config.get('is_ddim_sampling', False), help='Enable DDIM (Unimplemented)')

    # DiT args
    parser.add_argument('--hidden_channels', type=int, default=config.get('hidden_channels', 768), help='DiT hidden dim')
    parser.add_argument('--patch_size', type=int, default=config.get('patch_size', 1), help='DiT patch size')
    parser.add_argument('--depth', type=int, default=config.get('depth', 6), help='DiT layers')
    parser.add_argument('--num_heads', type=int, default=config.get('num_heads', 8), help='DiT heads')
    parser.add_argument('--mlp_ratio', type=float, default=config.get('mlp_ratio', 4.0), help='DiT MLP ratio')
    parser.add_argument('--attn_drop', type=float, default=config.get('attn_drop', 0.0), help='DiT attn drop')
    parser.add_argument('--proj_drop', type=float, default=config.get('proj_drop', 0.0), help='DiT proj drop')

    # dataset
    parser.add_argument('--dataset_path', type=str, default=config.get('dataset_path', 'data/voxel_dataset_test_processed_emb_norm.pt'), help='Path to processed voxel dataset (.pt)')
    parser.add_argument('--mappings_file_path', type=str, default=config.get('mappings_file_path', 'data/voxel_dataset_test_mappings_norm.pt'), help='Path to mappings file')
    # conditional dataset/val
    parser.add_argument('--val_mappings_file_path', type=str, default=config.get('val_mappings_file_path', 'data/voxel_dataset_test_mappings_norm.pt'), help='Path to mappings file (validation)')
    parser.add_argument('--val_dataset_path', type=str, default=config.get('val_dataset_path', 'data/voxel_dataset_test_processed_emb_norm.pt'), help='Path to processed voxel dataset (.pt) for validation')
    parser.add_argument('--one_hot_on_load', type=str_to_bool, default=config.get('one_hot_on_load', False), help='One-hot encode voxels at load time (requires mappings)')
    parser.add_argument('--rotation_aug_prob', type=float, default=config.get('rotation_aug_prob', 0.0), help='Probability of applying 90° rotation augmentation per sample')

    # trainer
    parser.add_argument('--train_batch_size', type=int, default=config.get('train_batch_size', 128))
    parser.add_argument('--per_device_train_batch_size', type=int, default=config.get('per_device_train_batch_size', 16))
    parser.add_argument('--gradient_accumulate_every', type=int, default=config.get('gradient_accumulate_every', -1), help='Override gradient accumulation steps (if > 0)')
    parser.add_argument('--train_lr', type=float, default=config.get('train_lr', 1e-4))
    parser.add_argument('--compile_model', type=str_to_bool, default=config.get('compile_model', False), help='Use torch.compile to compile the model (may improve speed)')

    # note: either num steps or num epochs must be set to > 0, not both
    parser.add_argument('--train_num_steps', type=int, default=config.get('train_num_steps', -1))
    parser.add_argument('--train_num_epochs', type=int, default=config.get('train_num_epochs', 10))
    parser.add_argument('--validate_every', type=int, default=config.get('validate_every', -1))
    parser.add_argument('--validate_every_epoch', type=int, default=config.get('validate_every_epoch', 1))
    parser.add_argument('--ema_update_every', type=int, default=config.get('ema_update_every', 10))
    parser.add_argument('--ema_decay', type=float, default=config.get('ema_decay', 0.995))
    parser.add_argument('--optimizer', type=str, default=config.get('optimizer', 'adamw'), choices=['adamw', 'adam'])
    parser.add_argument('--weight_decay', type=float, default=config.get('weight_decay', 0.01))
    parser.add_argument('--adam_betas', type=str, default=config.get('adam_betas', '0.9,0.99'))

    parser.add_argument('--warmup_steps', type=int, default=config.get('warmup_steps', 0))
    parser.add_argument('--scheduler', type=str, default=config.get('scheduler', 'linear'), choices=['linear', 'cosine'])
    parser.add_argument('--min_lr_ratio', type=float, default=config.get('min_lr_ratio', 0.0))

    # note: either save_and_sample_every or save_and_sample_every_epoch must be set to > 0, not both
    parser.add_argument('--save_and_sample_every', type=int, default=config.get('save_and_sample_every', 500))
    parser.add_argument('--save_and_sample_every_epoch', type=int, default=config.get('save_and_sample_every_epoch', -1))
    parser.add_argument('--num_samples', type=int, default=config.get('num_samples', 16))
    parser.add_argument('--results_folder', type=str, default=config.get('results_folder', './results'))
    parser.add_argument('--run_name', type=str, default=config.get('run_name', 'md4_discrete_unified'))
    parser.add_argument('--exp_name', type=str, default=config.get('exp_name', ''))
    parser.add_argument('--amp', type=str_to_bool, default=config.get('amp', True))
    parser.add_argument('--mixed_precision_type', type=str, default=config.get('mixed_precision_type', 'bf16'), choices=['fp16', 'bf16'])
    parser.add_argument('--split_batches', type=str_to_bool, default=config.get('split_batches', True), help='Only used in unconditional trainer')
    parser.add_argument('--max_grad_norm', type=float, default=config.get('max_grad_norm', 1.0))
    parser.add_argument('--evaluate_ema_model', type=str_to_bool, default=config.get('evaluate_ema_model', True))
    parser.add_argument('--save_only_last_checkpoint', type=str_to_bool, default=config.get('save_only_last_checkpoint', True))

    # conditional / guidance args
    parser.add_argument('--num_classes', type=int, default=config.get('num_classes', 11), help='Number of class labels for conditioning')
    parser.add_argument('--default_cond_scale', type=float, default=config.get('default_cond_scale', 4.0), help='Classifier-free guidance scale at sampling')
    parser.add_argument('--cond_drop_prob', type=float, default=config.get('cond_drop_prob', 0.2), help='conditional drop probability')
    parser.add_argument('--comb_method',  type=str, default=config.get('comb_method', 'add'), choices=['add', 'cat'], help='Method for combining time and class conditioning in film layer (add or cat)')

    parser.add_argument('--device', type=str, default=config.get('device', 'cuda'))

    # diffusion step schedule
    parser.add_argument('--reverse_steps', type=int, default=config.get('reverse_steps', 1000))
    parser.add_argument('--sampling_timesteps', type=int, default=config.get('sampling_timesteps', 250))

    # wandb
    parser.add_argument('--use_wandb', type=str_to_bool, default=config.get('use_wandb', False))
    parser.add_argument('--wandb_project', type=str, default=config.get('wandb_project', 'dream-cubed'))
    parser.add_argument('--wandb_group_name', type=str, default=config.get('wandb_group_name', 'discrete_dit'))
    parser.add_argument('--wandb_run_name', type=str, default=config.get('wandb_run_name', ''))
    parser.add_argument('--wandb_log_images', type=str_to_bool, default=config.get('wandb_log_images', True), help='Enable logging images to wandb')

    # FID args
    parser.add_argument('--fid_ref_images_dir', type=str, default=config.get('fid_ref_images_dir', None), help='Directory of reference images for FID (renders). If provided, FID will be computed periodically.')
    parser.add_argument('--fid_num_samples', type=int, default=config.get('fid_num_samples', 0), help='Number of samples to generate for FID per evaluation (0 to disable).')
    parser.add_argument('--fid_every_steps', type=int, default=config.get('fid_every_steps', 0), help='Compute FID every N training steps (0 to defer to fid_every_epoch).')
    parser.add_argument('--fid_every_epoch', type=int, default=config.get('fid_every_epoch', 0), help='Compute FID every N epochs (used if fid_every_steps <= 0).')
    parser.add_argument('--fid_textures_dir', type=str, default=config.get('fid_textures_dir', 'block_textures/'), help='Textures directory used to render generated samples for FID.')
    parser.add_argument('--fid_batch_size', type=int, default=config.get('fid_batch_size', 32), help='Batch size for Inception feature extraction during FID.')
    parser.add_argument('--fid_num_workers', type=int, default=config.get('fid_num_workers', 2), help='Number of workers for FID dataloaders.')
    # resume
    parser.add_argument('--resume', type=str_to_bool, default=config.get('resume', False), help='Resume training from a checkpoint')
    parser.add_argument('--resume_path', type=str, default=config.get('resume_path', ''), help='Optional explicit path to checkpoint (.pt). Defaults to model_best.pt in save dir when --resume=true')
    parser.add_argument('--resume_override_step', type=int, default=config.get('resume_override_step', -1), help='Optional override for global step when resuming (aligns scheduler too)')
    args = parser.parse_args(remaining_argv)

    if len(args.exp_name) == 0:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.exp_name = f"exp_{timestamp}" 
    if len(args.wandb_run_name) == 0:
        args.wandb_run_name = f"{args.run_name}"
    # parse composites
    adam_betas = tuple(map(float, args.adam_betas.split(',')))

    # resolve per-device batch size if requested (-1)
    if args.per_device_train_batch_size is not None and args.per_device_train_batch_size <= 0:
        effective_grad_accum = args.gradient_accumulate_every if (args.gradient_accumulate_every and args.gradient_accumulate_every > 0) else 1
        computed_pdbs = compute_per_device_train_batch_size(args.train_batch_size, effective_grad_accum)
        args.per_device_train_batch_size = computed_pdbs
        log_if_main(
            f"Computed per_device_train_batch_size={computed_pdbs} from train_batch_size={args.train_batch_size}, "
            f"total_devices={get_total_devices()}, gradient_accumulate_every={effective_grad_accum}"
        )

    # save resolved config next to training results (results/run_name/exp_name, config.json)
    save_dir = os.path.join(args.results_folder, args.run_name, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    resolved_config = dict(vars(args))
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(resolved_config, f, indent=2, sort_keys=True)

    # load mappings to determine K
    mappings = torch.load(args.mappings_file_path, map_location='cpu', weights_only=False)
    converter = BlockBiomeConverter(mappings['block_mappings'], mappings['biome_mappings'])
    num_blocks = len(converter.block_to_index)  # K
    in_dim = num_blocks + 1  # add mask channel
    out_dim = num_blocks     # predict K logits

    # build model (conditional vs unconditional)
    if args.model_type == 'dit':
        model = DiT3D(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_channels=args.hidden_channels,
            image_size=args.image_size,
            patch_size=args.patch_size,
            time_dim=args.time_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            attn_drop=args.attn_drop,
            proj_drop=args.proj_drop,
            class_conditional=(args.mode == 'conditional'),
            num_classes=args.num_classes,
            comb_method=args.comb_method,
            cond_drop_prob=args.cond_drop_prob,
        ).to(args.device)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    print(f"Training model with {sum(p.numel() for p in model.parameters() if p.requires_grad) // 1_000_000}M parameters")
    # dataset
    # Use conditional dataset for both modes; unconditional simply ignores labels inside trainer
    def _make_ds(path, mappings_file_path, rotation_aug_prob):
        if path is None:
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
    val_dataset = _make_ds(args.val_dataset_path, args.val_mappings_file_path, 0.0) if args.mode == 'conditional' else None

    # derive gradient accumulation steps
    if args.gradient_accumulate_every and args.gradient_accumulate_every > 0:
        gradient_accumulate_every = args.gradient_accumulate_every
    else:
        gradient_accumulate_every = compute_accumulation_steps(
            args.train_batch_size, args.per_device_train_batch_size)

    # derive steps/epochs
    if args.train_num_steps <= 0:
        assert args.train_num_epochs > 0, "Either train_num_steps or train_num_epochs must be > 0"
        args.train_num_steps = (len(dataset) // args.train_batch_size) * args.train_num_epochs
        log_if_main(f"Setting train_num_steps to {args.train_num_steps} ({args.train_num_epochs} epochs)")
    else:
        assert args.train_num_epochs <= 0, "Only one of train_num_steps or train_num_epochs can be > 0"
        args.train_num_epochs = args.train_num_steps // (len(dataset) // args.train_batch_size)
    log_if_main("Training for %d steps (%d epochs), with %d total samples, and a batch size of %d (across %d devices, accumulating gradients over %d steps)" %
                (args.train_num_steps, args.train_num_epochs, len(dataset), args.train_batch_size, get_total_devices(), gradient_accumulate_every))
    
    # Print batch size configuration
    log_if_main(f"Batch size configuration:")
    log_if_main(f"  - Effective batch size: {args.train_batch_size}")
    log_if_main(f"  - Per device batch size: {args.per_device_train_batch_size}")
    log_if_main(f"  - Gradient accumulation steps: {gradient_accumulate_every}")
    log_if_main(f"  - Total devices: {get_total_devices()}")

    if args.save_and_sample_every <= 0:
        assert args.save_and_sample_every_epoch > 0, "Either save_and_sample_every or save_and_sample_every_epoch must be > 0"
        args.save_and_sample_every = (len(dataset) // args.train_batch_size) * args.save_and_sample_every_epoch
        log_if_main(f"Setting save_and_sample_every to {args.save_and_sample_every} steps ({args.save_and_sample_every_epoch} epochs)")
    else:
        assert args.save_and_sample_every_epoch <= 0, "Only one of save_and_sample_every or save_and_sample_every_epoch can be > 0"
        args.save_and_sample_every_epoch = args.save_and_sample_every // (len(dataset) // args.train_batch_size)
    log_if_main(f"Will save and sample every {args.save_and_sample_every} steps ({args.save_and_sample_every_epoch} epochs)")

    if args.validate_every <= 0:
        assert args.validate_every_epoch > 0, "Either validate_every or validate_every_epoch must be > 0"
        args.validate_every = (len(dataset) // args.train_batch_size) * args.validate_every_epoch
        log_if_main(f"Setting validate_every to {args.validate_every} steps ({args.validate_every_epoch} epochs)")
    else:
        assert args.validate_every_epoch <= 0, "Only one of validate_every or validate_every_epoch can be > 0"
        args.validate_every_epoch = args.validate_every // (len(dataset) // args.train_batch_size)
    log_if_main(f"Will validate every {args.validate_every} steps ({args.validate_every_epoch} epochs)")

    # Resolve FID cadence (optional)
    if args.fid_every_steps <= 0:
        if args.fid_every_epoch > 0:
            args.fid_every_steps = (len(dataset) // args.train_batch_size) * args.fid_every_epoch
            log_if_main(f"Setting fid_every_steps to {args.fid_every_steps} steps ({args.fid_every_epoch} epochs)")
        else:
            # leave at 0 → disabled cadence; FID only runs if other triggers call it
            log_if_main("FID cadence not set (fid_every_steps<=0 and fid_every_epoch<=0). FID will not run periodically.")

    log_if_main(f"Using LR: {args.train_lr} Optimizer: {args.optimizer}, Scheduler: {args.scheduler}, Warmup steps: {args.warmup_steps}, Min LR ratio: {args.min_lr_ratio}")

    if args.use_wandb and is_main_process():
        wandb = wandb_init(args, run_name=args.wandb_run_name, group_name=args.wandb_group_name, log_dir=save_dir, resume=args.resume)
        log_if_main(f"WandB logging enabled (project: {args.wandb_project}, group: {args.wandb_group_name}, run name: {args.wandb_run_name})")
    # trainer
    if args.mode == 'conditional':
        trainer = VoxelTrainerMD4(
            model,
            dataset,
            save_dir,
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
            num_samples=args.num_samples,
            amp=args.amp,
            mixed_precision_type=args.mixed_precision_type,
            compile_model=args.compile_model,
            max_grad_norm=args.max_grad_norm,
            evaluate_ema_model=args.evaluate_ema_model,
            save_only_last_checkpoint=args.save_only_last_checkpoint,
            mappings_file_path=args.mappings_file_path,
            run_name=args.run_name,
            num_classes=args.num_classes,
            default_cond_scale=args.default_cond_scale,
            reverse_steps=args.reverse_steps,
            sampling_timesteps=args.sampling_timesteps,
            val_dataset=val_dataset,
            val_every_n_steps=args.validate_every,
            inpaint_preview_paths=[
                "scratch/clean_71.pt",
                "scratch/clean_131.pt",
            ],
            inpaint_bounds=[
                dict(x_start=0, x_end=16, y_start=0, y_end=16, z_start=5, z_end=16),
                dict(x_start=0, x_end=16, y_start=0, y_end=16, z_start=5, z_end=16),
            ],
            inpaint_num_variants=4,
            use_wandb=args.use_wandb,
            wandb_log_images=args.wandb_log_images,
            # FID
            fid_ref_images_dir=args.fid_ref_images_dir,
            fid_num_samples=args.fid_num_samples,
            fid_every_steps=args.fid_every_steps,
            fid_textures_dir=args.fid_textures_dir,
            fid_batch_size=args.fid_batch_size,
            fid_num_workers=args.fid_num_workers,
        )
    else:
        trainer = VoxelTrainerMD4(
            model,
            dataset,
            save_dir,
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
            num_samples=args.num_samples,
            amp=args.amp,
            mixed_precision_type=args.mixed_precision_type,
            max_grad_norm=args.max_grad_norm,
            evaluate_ema_model=args.evaluate_ema_model,
            save_only_last_checkpoint=args.save_only_last_checkpoint,
            mappings_file_path=args.mappings_file_path,
            run_name=args.run_name,
            reverse_steps=args.reverse_steps,
            sampling_timesteps=args.sampling_timesteps,
            use_wandb=args.use_wandb,
            wandb_log_images=args.wandb_log_images,
            # FID
            fid_ref_images_dir=args.fid_ref_images_dir,
            fid_num_samples=args.fid_num_samples,
            fid_every_steps=args.fid_every_steps,
            fid_textures_dir=args.fid_textures_dir,
            fid_batch_size=args.fid_batch_size,
            fid_num_workers=args.fid_num_workers,
        )

    # optional resume
    if args.resume:
        ckpt_path = args.resume_path if (args.resume_path and len(args.resume_path) > 0) else os.path.join(save_dir, 'model_best.pt')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"--resume was set but checkpoint not found: {ckpt_path}. Pass --resume_path to specify a checkpoint.")
        log_if_main(f"Resuming from checkpoint: {ckpt_path}")
        trainer.load(ckpt_path)
        # Optional manual step override (and scheduler alignment)
        if args.resume_override_step is not None and int(args.resume_override_step) >= 0:
            trainer.step = int(args.resume_override_step)
            try:
                trainer._align_scheduler_to_step(trainer.step)
            except Exception:
                pass

    trainer.train()


if __name__ == '__main__':
    launch()


