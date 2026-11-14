"""
Time autoregressive sampling from a training checkpoint.

    python ar_sample_timing.py --checkpoint results/32_ar_patch4_.../model-1.pt

Loads the EMA weights, generates one batch with the sampling progress bar on,
and extrapolates to a full evaluation set.  Nothing is saved; this exists only
to answer "how long will 100 samples per biome take".

Sampling cost is dominated by a sequential loop of (num_tokens x patch_voxels)
steps that runs once per *batch*, not per sample, so wall-clock per sample falls
almost linearly with batch size until the GPU saturates.  Sweep --batch_size
(1, 8, 32, 64) to find where that stops being true before planning a big run.
"""

import argparse
import json
import math
import os
import time

import torch

from autoregressive_dit import ARDiT3D
from data_utils import BlockBiomeConverter


def find_config(checkpoint_path, explicit=None):
    if explicit:
        return explicit
    checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    # HF exports keep weights in inference_weights/ and config.json one level up.
    for directory in (checkpoint_dir, os.path.dirname(checkpoint_dir)):
        candidate = os.path.join(directory, 'config.json')
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f'no config.json next to or above the checkpoint ({checkpoint_path}); pass --config')


def extract_weights(ckpt, prefer_ema=True):
    """Pull a plain model state dict out of a trainer checkpoint."""
    if prefer_ema and ckpt.get('ema') is not None:
        ema = ckpt['ema']
        # ema_pytorch stores the online model and its own step/initted buffers in
        # the same dict, so keep only the averaged weights
        weights = {k[len('ema_model.'):]: v for k, v in ema.items()
                   if isinstance(k, str) and k.startswith('ema_model.')}
        if weights:
            return weights, 'ema'
    return ckpt['model'], 'online'


def build_model(config, vocab_size, device):
    model = ARDiT3D(
        vocab_size=vocab_size,
        hidden_channels=config['hidden_channels'],
        image_size=config['image_size'],
        patch_size=config['patch_size'],
        time_dim=config['time_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        mlp_ratio=config.get('mlp_ratio', 4.0),
        attn_drop=0.0,
        proj_drop=0.0,
        comb_method=config.get('comb_method', 'add'),
        class_conditional=(config.get('mode', 'conditional') == 'conditional'),
        num_classes=config.get('num_classes', 15),
        cond_drop_prob=config.get('cond_drop_prob', 0.2),
        local_head_depth=config.get('local_head_depth', 2),
        local_head_dim=config.get('local_head_dim', 384),
        local_head_heads=config.get('local_head_heads', 6),
        token_order=config.get('token_order', 'y_slow'),
        use_gradient_checkpointing=False,
    )
    return model.to(device).eval()


def main():
    p = argparse.ArgumentParser(description='Time AR sampling from a checkpoint')
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--config', type=str, default=None,
                   help='Defaults to config.json in the checkpoint directory.')
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--biome', type=str, default=None,
                   help='Biome name or index. Defaults to cycling through all biomes.')
    p.add_argument('--cond_scale', type=float, default=None,
                   help='Defaults to the default_cond_scale of the training run.')
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--top_k', type=int, default=0)
    p.add_argument('--top_p', type=float, default=0.0)
    p.add_argument('--use_ema', action='store_true', default=True)
    p.add_argument('--no_ema', dest='use_ema', action='store_false')
    p.add_argument('--amp', action='store_true', default=True)
    p.add_argument('--no_amp', dest='amp', action='store_false')
    p.add_argument('--target_per_biome', type=int, default=100,
                   help='Used only for the extrapolation printed at the end.')
    p.add_argument('--device', type=str, default='cuda')
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')

    with open(find_config(args.checkpoint, args.config)) as f:
        config = json.load(f)

    mappings = torch.load(config['mappings_file_path'], map_location='cpu', weights_only=False)
    converter = BlockBiomeConverter(mappings['block_mappings'], mappings['biome_mappings'])
    vocab_size = len(converter.block_to_index)

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    weights, source = extract_weights(ckpt, prefer_ema=args.use_ema)

    model = build_model(config, vocab_size, device)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if missing or unexpected:
        print(f'state dict mismatch: {len(missing)} missing, {len(unexpected)} unexpected')
        if missing:
            print(f'  first missing: {missing[:5]}')
        if unexpected:
            print(f'  first unexpected: {unexpected[:5]}')

    num_classes = int(config.get('num_classes', 15)) if model.class_conditional else 0
    print(f'Loaded {source} weights from step {ckpt.get("step", "?")}, '
          f'{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params')
    print(f'{model.num_tokens} patch tokens x {model.patch_voxels} voxels = '
          f'{model.num_tokens * model.patch_voxels} sequential steps per batch')

    class_cond = None
    if num_classes:
        if args.biome is None:
            labels = [i % num_classes for i in range(args.batch_size)]
        elif args.biome.isdigit():
            labels = [int(args.biome)] * args.batch_size
        else:
            labels = [converter.biome_to_index[args.biome]] * args.batch_size
        class_cond = torch.tensor(labels, dtype=torch.long, device=device)
        print(f'Biomes: {converter.convert_class_to_biomes(class_cond.cpu())}')

    cond_scale = args.cond_scale
    if cond_scale is None:
        cond_scale = float(config.get('default_cond_scale', 1.0)) if num_classes else 1.0
    if cond_scale != 1.0:
        print(f'Classifier-free guidance at scale {cond_scale} doubles the batch internally')

    autocast = (torch.autocast(device.type, dtype=torch.bfloat16) if args.amp
                else torch.autocast(device.type, enabled=False))

    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode(), autocast:
        samples = model.generate(
            args.batch_size,
            class_cond=class_cond,
            cond_scale=cond_scale,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            progress=True,
            device=device,
        )
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print(f'\nGenerated {tuple(samples.shape)} in {elapsed:.1f}s '
          f'({elapsed / args.batch_size:.1f}s per sample)')
    if device.type == 'cuda':
        print(f'Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB')

    if num_classes:
        total = args.target_per_biome * num_classes
        batches = math.ceil(total / args.batch_size)
        hours = batches * elapsed / 3600
        print(f'\nExtrapolation: {args.target_per_biome} per biome x {num_classes} biomes '
              f'= {total} samples')
        print(f'  {batches} batches of {args.batch_size} -> {hours:.1f} GPU-hours '
              f'({hours / 4:.1f}h on 4 GPUs)')
        print('  This scales poorly with batch_size=1; re-run with a larger batch '
              'before trusting it.')


if __name__ == '__main__':
    main()
