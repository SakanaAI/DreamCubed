"""
Train the 3D feature extractor used for voxel-native FID.

    bash launch_fid3d.sh 4 fid3d_32
    python train_fid3d.py --config training_configs/fid3d_32.json

The network is a small 3D CNN trained on real chunks only, with two heads: biome
classification (the Inception analogue) and classification of structure-
destroying but histogram-preserving corruptions, which is what stops the feature
space from collapsing onto block frequencies.  See `fid3d_model.py` for the
rationale.

Training finishes with a sanity check that is worth reading before trusting the
metric: it reports the real-vs-real Frechet distance (the noise floor at that
sample count) and the distance induced by each corruption at increasing
severity.  A usable metric has a floor near zero and rises monotonically with
severity; if it does not, the feature space -- not the generative model -- is
what needs fixing.
"""

import argparse
import datetime
import json
import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import get_wsd_schedule

from data_utils import BlockBiomeConverter
from discrete_diffusion_md4 import VoxelDatasetConditional, VoxelDatasetMemmapConditional
from fid3d_model import (
    CORRUPTION_FNS,
    CORRUPTION_TYPES,
    VoxelFeatureExtractor3D,
    apply_corruptions,
    extract_features,
    fid_from_features,
    histogram_features,
    load_feature_extractor,
    save_feature_extractor,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# severity ladders for the end-of-training sensitivity check
SANITY_SWEEPS = {
    'block_noise': (0.01, 0.02, 0.05, 0.10, 0.20),
    'vertical_roll': (1.0, 2.0, 4.0, 8.0),
    'patch_shuffle': (16.0, 8.0, 4.0),
    'column_shuffle': (1.0,),
    'vertical_flip': (1.0,),
    'half_swap': (1.0,),
    'horizontal_splice': (1.0,),
}


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


def make_dataset(path, mappings_path, rotation_aug_prob):
    if not path:
        return None
    if os.path.isdir(path) and os.path.exists(os.path.join(path, 'manifest.json')):
        return VoxelDatasetMemmapConditional(
            dir_path=path, one_hot_on_load=False,
            mappings_file_path=mappings_path, rotation_aug_prob=rotation_aug_prob)
    return VoxelDatasetConditional(
        file_path=path, one_hot_on_load=False,
        mappings_file_path=mappings_path, rotation_aug_prob=rotation_aug_prob)


@torch.inference_mode()
def evaluate(model, loader, accelerator, args, corruption_types):
    model.eval()
    device = accelerator.device
    totals = torch.zeros(5, device=device)  # biome_correct, biome_n, corr_correct, corr_n, loss_sum

    # fixed seed for the corruption draw, so validation numbers move only when
    # the model does
    fork_devices = [torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(1234)
        for batch in tqdm(loader, disable=not accelerator.is_main_process, desc='Validating'):
            vox, labels = batch[0].to(device), batch[1].to(device)
            corrupted, corr_labels = apply_corruptions(
                vox, args.corruption_prob, corruption_types)
            with accelerator.autocast():
                _, biome_logits, corr_logits = model(corrupted)
            clean = corr_labels == 0
            if bool(clean.any()):
                totals[0] += (biome_logits[clean].argmax(-1) == labels[clean]).sum()
                totals[1] += int(clean.sum())
                totals[4] += F.cross_entropy(biome_logits[clean].float(), labels[clean],
                                             reduction='sum')
            if corr_logits is not None:
                totals[2] += (corr_logits.argmax(-1) == corr_labels).sum()
                totals[3] += corr_labels.numel()

    totals = accelerator.reduce(totals, reduction='sum')
    biome_acc = (totals[0] / totals[1].clamp(min=1)).item()
    corr_acc = (totals[2] / totals[3]).item() if totals[3] > 0 else None
    biome_loss = (totals[4] / totals[1].clamp(min=1)).item()
    model.train()
    return biome_acc, corr_acc, biome_loss


@torch.inference_mode()
def run_sanity_check(model, val_ds, args, device, out_path):
    """Real-vs-real floor plus a corruption severity sweep.

    Also scores every corruption with raw block histograms as features.  That
    control is the point of the exercise: histogram features are blind to any
    corruption that preserves block counts, so if the learned features behave
    the same way, they have collapsed onto block statistics.
    """
    n = int(args.sanity_num_samples)
    usable = min(2 * n, len(val_ds))
    if usable < 512:
        logger.warning('Validation split too small for a meaningful sanity check')
        return None
    half = usable // 2

    rng = np.random.default_rng(0)
    order = rng.permutation(len(val_ds))[:usable]

    def _materialize(indices):
        loader = DataLoader(Subset(val_ds, indices.tolist()),
                            batch_size=args.eval_batch_size, num_workers=2)
        return torch.cat([b[0] for b in loader], dim=0)

    vols_a = _materialize(order[:half])
    vols_b = _materialize(order[half:])

    t0 = time.time()
    feats_a = extract_features(model, vols_a, batch_size=args.eval_batch_size, device=device)
    feats_b = extract_features(model, vols_b, batch_size=args.eval_batch_size, device=device)
    elapsed = time.time() - t0

    results = {
        'num_samples_per_group': int(half),
        'feature_dim': int(model.feature_dim),
        'seconds_to_featurize': round(elapsed, 2),
        'real_vs_real': fid_from_features(feats_a, feats_b),
        'corruptions': {},
    }
    logger.info(f"[sanity] real-vs-real floor at N={half}: {results['real_vs_real']:.4f} "
                f"({elapsed:.1f}s to featurize {2 * half} chunks)")

    keep = None
    if args.sanity_histogram_baseline:
        hist_a = histogram_features(vols_a, model.vocab_size, args.eval_batch_size)
        hist_b = histogram_features(vols_b, model.vocab_size, args.eval_batch_size)
        # drop blocks that never vary in the reference set, they only make the
        # covariance singular
        keep = hist_a.std(axis=0) > 1e-12
        hist_a = hist_a[:, keep]
        results['histogram_baseline'] = {
            'feature_dim': int(keep.sum()),
            'real_vs_real': fid_from_features(hist_a, hist_b[:, keep]),
            'corruptions': {},
        }
        logger.info(f"[sanity] histogram-baseline real-vs-real: "
                    f"{results['histogram_baseline']['real_vs_real']:.4f}")

    for name, severities in SANITY_SWEEPS.items():
        if name not in CORRUPTION_FNS:
            continue
        results['corruptions'][name] = {}
        if keep is not None:
            results['histogram_baseline']['corruptions'][name] = {}
        for sev in severities:
            corrupted = torch.cat([
                CORRUPTION_FNS[name](chunk.to(device), sev).cpu()
                for chunk in torch.split(vols_b, args.eval_batch_size)
            ], dim=0)
            feats_c = extract_features(model, corrupted, batch_size=args.eval_batch_size,
                                       device=device)
            score = fid_from_features(feats_a, feats_c)
            results['corruptions'][name][str(sev)] = score

            hist_score = None
            if keep is not None:
                try:
                    hist_c = histogram_features(corrupted, model.vocab_size, args.eval_batch_size)
                    hist_score = fid_from_features(hist_a, hist_c[:, keep])
                    results['histogram_baseline']['corruptions'][name][str(sev)] = hist_score
                except Exception as e:
                    logger.warning(f'histogram baseline failed for {name}/{sev}: {e}')

            suffix = '' if hist_score is None else f'   (histogram baseline {hist_score:.4f})'
            logger.info(f"[sanity] {name:<18} severity {sev:<6} -> {score:.4f}{suffix}")

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f'[sanity] wrote {out_path}')
    return results


def launch():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', type=str, help='Path to JSON configuration file')
    config_args, remaining_argv = config_parser.parse_known_args()
    config = load_config(config_args.config)

    parser = argparse.ArgumentParser(description='Train the 3D FID feature extractor',
                                     parents=[config_parser])

    # data
    parser.add_argument('--dataset_path', type=str, default=config.get('dataset_path', ''))
    parser.add_argument('--val_dataset_path', type=str, default=config.get('val_dataset_path', ''))
    parser.add_argument('--mappings_file_path', type=str, default=config.get('mappings_file_path', ''))
    parser.add_argument('--rotation_aug_prob', type=float, default=config.get('rotation_aug_prob', 0.75))
    parser.add_argument('--num_classes', type=int, default=config.get('num_classes', 0),
                        help='Biome classes; read from the dataset manifest when 0.')

    # architecture
    parser.add_argument('--image_size', type=int, default=config.get('image_size', 32))
    parser.add_argument('--embed_dim', type=int, default=config.get('embed_dim', 32))
    parser.add_argument('--channels', type=str, default=config.get('channels', '64,128,256'))
    parser.add_argument('--blocks_per_stage', type=str, default=config.get('blocks_per_stage', '2,2,2'))
    parser.add_argument('--stem_stride', type=int, default=config.get('stem_stride', 2))
    parser.add_argument('--feature_channels', type=int, default=config.get('feature_channels', 128))
    parser.add_argument('--y_bins', type=int, default=config.get('y_bins', 4),
                        help='Vertical bands kept by the pooling stage. feature_dim = '
                             'feature_channels * y_bins. Set 1 for plain global pooling.')
    parser.add_argument('--width_mult', type=float, default=config.get('width_mult', 1.0))

    # objective
    parser.add_argument('--use_corruptions', type=str_to_bool, default=config.get('use_corruptions', True),
                        help='False trains a plain biome classifier with no corruption head. '
                             'Run this ablation to check whether the classifier-only feature '
                             'space is blind to histogram-preserving corruptions.')
    parser.add_argument('--corruption_prob', type=float, default=config.get('corruption_prob', 0.35))
    parser.add_argument('--corruption_types', type=str,
                        default=config.get('corruption_types', ','.join(CORRUPTION_TYPES[1:])))
    parser.add_argument('--corruption_loss_weight', type=float,
                        default=config.get('corruption_loss_weight', 1.0))
    parser.add_argument('--label_smoothing', type=float, default=config.get('label_smoothing', 0.0))

    # optimization
    parser.add_argument('--train_batch_size', type=int, default=config.get('train_batch_size', 512))
    parser.add_argument('--per_device_train_batch_size', type=int,
                        default=config.get('per_device_train_batch_size', 128))
    parser.add_argument('--eval_batch_size', type=int, default=config.get('eval_batch_size', 256))
    parser.add_argument('--train_lr', type=float, default=config.get('train_lr', 3e-4))
    parser.add_argument('--weight_decay', type=float, default=config.get('weight_decay', 0.05))
    parser.add_argument('--adam_betas', type=str, default=config.get('adam_betas', '0.9,0.99'))
    parser.add_argument('--warmup_steps', type=int, default=config.get('warmup_steps', 500))
    parser.add_argument('--scheduler', type=str, default=config.get('scheduler', 'cosine'),
                        choices=['linear', 'cosine'])
    parser.add_argument('--min_lr_ratio', type=float, default=config.get('min_lr_ratio', 0.0))
    parser.add_argument('--train_num_epochs', type=int, default=config.get('train_num_epochs', 10))
    parser.add_argument('--max_grad_norm', type=float, default=config.get('max_grad_norm', 1.0))
    parser.add_argument('--num_workers', type=int, default=config.get('num_workers', 4))
    parser.add_argument('--amp', type=str_to_bool, default=config.get('amp', True))
    parser.add_argument('--mixed_precision_type', type=str,
                        default=config.get('mixed_precision_type', 'bf16'), choices=['fp16', 'bf16'])

    # bookkeeping
    parser.add_argument('--results_folder', type=str, default=config.get('results_folder', './results'))
    parser.add_argument('--run_name', type=str, default=config.get('run_name', 'fid3d'))
    parser.add_argument('--exp_name', type=str, default=config.get('exp_name', ''))
    parser.add_argument('--log_every', type=int, default=config.get('log_every', 50))
    parser.add_argument('--use_wandb', type=str_to_bool, default=config.get('use_wandb', False))
    parser.add_argument('--wandb_project', type=str, default=config.get('wandb_project', 'dream-cubed'))
    parser.add_argument('--wandb_group_name', type=str, default=config.get('wandb_group_name', 'fid3d'))

    # sanity check
    parser.add_argument('--sanity_check', type=str_to_bool, default=config.get('sanity_check', True))
    parser.add_argument('--sanity_num_samples', type=int, default=config.get('sanity_num_samples', 5000),
                        help='Chunks per group. The real-vs-real floor is reported at this N; '
                             'use the same N when comparing models.')
    parser.add_argument('--sanity_histogram_baseline', type=str_to_bool,
                        default=config.get('sanity_histogram_baseline', True),
                        help='Also score every corruption using raw block histograms as features.')

    args = parser.parse_args(remaining_argv)

    require_path(args.mappings_file_path, '--mappings_file_path')
    require_path(args.dataset_path, '--dataset_path')
    if args.val_dataset_path:
        require_path(args.val_dataset_path, '--val_dataset_path')

    if not args.exp_name:
        args.exp_name = f"exp_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir = os.path.join(args.results_folder, args.run_name, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)

    accelerator = Accelerator(
        split_batches=False,
        mixed_precision=args.mixed_precision_type if args.amp else 'no',
        step_scheduler_with_optimizer=False,
    )
    device = accelerator.device

    mappings = torch.load(args.mappings_file_path, map_location='cpu', weights_only=False)
    converter = BlockBiomeConverter(mappings['block_mappings'], mappings['biome_mappings'])
    vocab_size = len(converter.block_to_index)

    train_ds = make_dataset(args.dataset_path, args.mappings_file_path, args.rotation_aug_prob)
    val_ds = make_dataset(args.val_dataset_path, args.mappings_file_path, 0.0)
    assert train_ds is not None and len(train_ds) > 0, 'empty training dataset'

    num_biomes = args.num_classes or getattr(train_ds, 'num_classes', 0)
    assert num_biomes > 1, 'could not determine the number of biome classes; pass --num_classes'

    corruption_types = []
    if args.use_corruptions and args.corruption_prob > 0:
        corruption_types = [t for t in args.corruption_types.split(',') if t and t != 'clean']
        unknown = [t for t in corruption_types if t not in CORRUPTION_FNS]
        assert not unknown, f'unknown corruption types: {unknown}'
    use_corruptions = len(corruption_types) > 0
    # labels index the global CORRUPTION_TYPES list, so the head is sized to it
    # even when only a subset is enabled
    num_corruptions = len(CORRUPTION_TYPES) if use_corruptions else 0

    model = VoxelFeatureExtractor3D(
        vocab_size=vocab_size,
        num_biomes=num_biomes,
        embed_dim=args.embed_dim,
        channels=tuple(int(c) for c in args.channels.split(',')),
        blocks_per_stage=tuple(int(c) for c in args.blocks_per_stage.split(',')),
        stem_stride=args.stem_stride,
        feature_channels=args.feature_channels,
        y_bins=args.y_bins,
        width_mult=args.width_mult,
        num_corruptions=num_corruptions,
        image_size=args.image_size,
    )

    if accelerator.is_main_process:
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f'Vocabulary {vocab_size} blocks, {num_biomes} biomes, '
                    f'{len(train_ds)} training chunks')
        logger.info(f'Feature extractor: {n_params / 1e6:.2f}M params, '
                    f'feature_dim={model.feature_dim}')
        if use_corruptions:
            logger.info(f'Corruptions: {corruption_types} (p={args.corruption_prob})')
        else:
            logger.info('Corruption head DISABLED: training a plain biome classifier '
                        '(classifier-only ablation)')
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump({**vars(args), 'vocab_size': vocab_size, 'num_biomes': num_biomes,
                       'feature_dim': model.feature_dim, 'num_params': n_params,
                       'effective_corruption_types': corruption_types},
                      f, indent=2, sort_keys=True)

    train_dl = DataLoader(train_ds, batch_size=args.per_device_train_batch_size, shuffle=True,
                          pin_memory=True, num_workers=args.num_workers, drop_last=True,
                          persistent_workers=args.num_workers > 0)
    val_dl = None if val_ds is None else DataLoader(
        val_ds, batch_size=args.eval_batch_size, shuffle=False, pin_memory=True,
        num_workers=args.num_workers)

    world = accelerator.num_processes
    grad_accum = max(1, args.train_batch_size // (args.per_device_train_batch_size * world))
    steps_per_epoch = max(1, len(train_dl) // grad_accum)
    total_steps = steps_per_epoch * args.train_num_epochs

    opt = AdamW(model.parameters(), lr=args.train_lr,
                betas=tuple(float(b) for b in args.adam_betas.split(',')),
                weight_decay=args.weight_decay)
    scheduler = get_wsd_schedule(
        optimizer=opt,
        num_warmup_steps=args.warmup_steps,
        num_decay_steps=max(1, total_steps - args.warmup_steps),
        num_training_steps=total_steps,
        min_lr_ratio=args.min_lr_ratio,
        decay_type=args.scheduler,
    )

    model, opt, scheduler, train_dl = accelerator.prepare(model, opt, scheduler, train_dl)
    if val_dl is not None:
        val_dl = accelerator.prepare(val_dl)

    if args.use_wandb and accelerator.is_main_process:
        import wandb
        wandb.init(project=args.wandb_project, group=args.wandb_group_name[:127],
                   name=args.run_name[:127], config=vars(args))

    accelerator.print(f'Training for {total_steps} steps '
                      f'({args.train_num_epochs} epochs x {steps_per_epoch} steps, '
                      f'grad accum {grad_accum}, {world} processes)')

    best_score = -1.0
    step = 0
    model.train()

    for epoch in range(args.train_num_epochs):
        pbar = tqdm(total=len(train_dl), disable=not accelerator.is_main_process,
                    desc=f'epoch {epoch + 1}/{args.train_num_epochs}')
        for micro, batch in enumerate(train_dl):
            vox, labels = batch[0].to(device), batch[1].to(device)
            corrupted, corr_labels = apply_corruptions(
                vox, args.corruption_prob, corruption_types)

            with accelerator.autocast():
                _, biome_logits, corr_logits = model(corrupted)
                clean = corr_labels == 0
                # corrupted chunks get no biome target: splicing mixes two
                # biomes, so the label is no longer well defined
                if bool(clean.any()):
                    biome_loss = F.cross_entropy(biome_logits[clean].float(), labels[clean],
                                                 label_smoothing=args.label_smoothing)
                else:
                    biome_loss = biome_logits.sum() * 0.0
                loss = biome_loss
                corr_loss = torch.zeros((), device=device)
                if corr_logits is not None:
                    corr_loss = F.cross_entropy(corr_logits.float(), corr_labels,
                                                label_smoothing=args.label_smoothing)
                    loss = loss + args.corruption_loss_weight * corr_loss

            accelerator.backward(loss / grad_accum)

            if (micro + 1) % grad_accum == 0:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                opt.step()
                scheduler.step()
                opt.zero_grad(set_to_none=True)
                step += 1

                if accelerator.is_main_process and step % args.log_every == 0:
                    msg = {
                        'train/loss': float(loss.detach()),
                        'train/biome_loss': float(biome_loss.detach()),
                        'train/corruption_loss': float(corr_loss.detach()),
                        'train/lr': scheduler.get_last_lr()[0],
                        'train/step': step,
                    }
                    if args.use_wandb:
                        import wandb
                        wandb.log(msg, step=step)
                    pbar.set_postfix(loss=f"{msg['train/loss']:.3f}",
                                     biome=f"{msg['train/biome_loss']:.3f}",
                                     corr=f"{msg['train/corruption_loss']:.3f}")
            pbar.update(1)
        pbar.close()

        if val_dl is not None:
            biome_acc, corr_acc, biome_loss_val = evaluate(
                model, val_dl, accelerator, args, corruption_types)
            score = biome_acc if corr_acc is None else 0.5 * (biome_acc + corr_acc)
            accelerator.print(f'epoch {epoch + 1}: biome acc {biome_acc:.4f}, '
                              f'corruption acc {"n/a" if corr_acc is None else f"{corr_acc:.4f}"}, '
                              f'biome nll {biome_loss_val:.4f}')
            if args.use_wandb and accelerator.is_main_process:
                import wandb
                metrics = {'val/biome_acc': biome_acc, 'val/biome_loss': biome_loss_val,
                           'val/score': score}
                if corr_acc is not None:
                    metrics['val/corruption_acc'] = corr_acc
                wandb.log(metrics, step=step)
            if accelerator.is_main_process and score > best_score:
                best_score = score
                unwrapped = accelerator.unwrap_model(model)
                h = save_feature_extractor(
                    unwrapped, os.path.join(save_dir, 'feature_extractor_best.pt'),
                    extra={'epoch': epoch + 1, 'step': step, 'biome_acc': biome_acc,
                           'corruption_acc': corr_acc,
                           'use_corruptions': use_corruptions,
                           'mappings_file_path': args.mappings_file_path,
                           'train_dataset': args.dataset_path})
                accelerator.print(f'  saved best feature extractor (hash {h})')
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        h = save_feature_extractor(
            unwrapped, os.path.join(save_dir, 'feature_extractor_final.pt'),
            extra={'step': step, 'mappings_file_path': args.mappings_file_path,
                   'train_dataset': args.dataset_path})
        accelerator.print(f'saved final feature extractor (hash {h})')

        if args.sanity_check and val_ds is not None:
            best_path = os.path.join(save_dir, 'feature_extractor_best.pt')
            eval_model = unwrapped
            if os.path.exists(best_path):
                eval_model, _ = load_feature_extractor(best_path, device=device)
            eval_model.eval()
            run_sanity_check(eval_model, val_ds, args, device,
                             os.path.join(save_dir, 'sanity_check.json'))

    accelerator.wait_for_everyone()
    accelerator.print('done')


if __name__ == '__main__':
    launch()
