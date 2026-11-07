"""
Generate a per-biome evaluation set from an autoregressive checkpoint.

    python ar_generate_eval_samples.py --checkpoint results/<run>/<exp>/model_best.pt \
        --samples_per_biome 100 --batch_size 25

Writes the same layout the diffusion models produce via `calc_fid.py`, so the
existing evaluation scripts can be pointed straight at the output directory:

    <out_dir>/<run_id>/generated_<biome>.pt   payload of original block IDs
    <out_dir>/<run_id>/<biome>/chunk_*.png    textured renders for 2D FID

Rendering goes through `calc_fid.render_samples_to_images`, the same function
the diffusion path uses, so the images are directly comparable.

Work is resumable: a biome whose payload already holds enough samples is
skipped, and rendering is skipped when the image count is already complete.
Use --biomes to split the run across several jobs.
"""

import argparse
import json
import os
import time

import torch
from tqdm import tqdm

from ar_sample_timing import build_model, extract_weights, find_config
from calc_fid import _count_images, render_samples_to_images
from data_utils import BlockBiomeConverter


def default_run_id(checkpoint_path):
    """Mirror the <model>_<experiment>_<checkpoint> id that calc_fid builds."""
    ckpt = os.path.normpath(os.path.abspath(checkpoint_path))
    stem = os.path.splitext(os.path.basename(ckpt))[0]
    exp_name = os.path.basename(os.path.dirname(ckpt))
    model_name = os.path.basename(os.path.dirname(os.path.dirname(ckpt)))
    parts = [p for p in (model_name, exp_name, stem) if p not in ('', '.', os.sep)]
    return '_'.join(parts) if parts else stem


def existing_sample_count(payload_path):
    if not os.path.exists(payload_path):
        return 0
    try:
        payload = torch.load(payload_path, map_location='cpu', weights_only=False)
        voxels = payload['voxels'] if isinstance(payload, dict) else payload
        return int(voxels.shape[0])
    except Exception as e:
        print(f'[warn] could not read {payload_path} ({e}); regenerating')
        return 0


@torch.inference_mode()
def generate_for_biome(model, converter, class_idx, num_samples, *, batch_size,
                       cond_scale, temperature, top_k, top_p, amp, device):
    """Return original block IDs [N, S, S, S] on CPU."""
    chunks = []
    remaining = int(num_samples)
    autocast = (torch.autocast(device.type, dtype=torch.bfloat16) if amp
                else torch.autocast(device.type, enabled=False))
    pbar = tqdm(total=remaining, desc=f'class {class_idx}', unit='sample')
    while remaining > 0:
        bsz = min(remaining, int(batch_size))
        labels = torch.full((bsz,), int(class_idx), dtype=torch.long, device=device)
        with autocast:
            indices = model.generate(
                bsz,
                class_cond=labels,
                cond_scale=float(cond_scale),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                progress=False,
                device=device,
            )
        chunks.append(converter.convert_to_original_blocks(indices.detach().cpu()))
        remaining -= bsz
        pbar.update(bsz)
    pbar.close()
    return torch.cat(chunks, dim=0)


def main():
    p = argparse.ArgumentParser(description='Generate per-biome AR samples for evaluation')
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--config', type=str, default=None,
                   help='Defaults to config.json in the checkpoint directory.')
    p.add_argument('--out_dir', type=str, default='renders/per_biome_gen',
                   help='Base directory; samples land in <out_dir>/<run_id>.')
    p.add_argument('--run_id', type=str, default=None,
                   help='Defaults to <model>_<experiment>_<checkpoint>.')
    p.add_argument('--samples_per_biome', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=25,
                   help='Samples per generate() call. AR sampling time is nearly '
                        'flat in batch size, so bigger is much cheaper per sample.')
    p.add_argument('--biomes', type=str, default=None,
                   help='Comma-separated subset, for splitting across jobs.')
    p.add_argument('--overwrite', action='store_true',
                   help='Regenerate biomes that already have a complete payload.')

    # sampling
    p.add_argument('--cond_scale', type=float, default=None,
                   help='Defaults to the default_cond_scale of the training run.')
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--top_k', type=int, default=0)
    p.add_argument('--top_p', type=float, default=0.0)
    p.add_argument('--use_ema', action='store_true', default=True)
    p.add_argument('--no_ema', dest='use_ema', action='store_false')
    p.add_argument('--amp', action='store_true', default=True)
    p.add_argument('--no_amp', dest='amp', action='store_false')
    p.add_argument('--seed', type=int, default=0)

    # rendering (defaults match calc_fid.py)
    p.add_argument('--render', action='store_true', default=True)
    p.add_argument('--no_render', dest='render', action='store_false',
                   help='Write payloads only; skip the 2D renders.')
    p.add_argument('--textures_dir', type=str, default='block_textures/')
    p.add_argument('--image_px', type=int, default=256)
    p.add_argument('--zoom', type=float, default=1.0)
    p.add_argument('--show_axis', action='store_true', default=False)

    p.add_argument('--device', type=str, default='cuda')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')

    with open(find_config(args.checkpoint, args.config)) as f:
        config = json.load(f)

    mappings = torch.load(config['mappings_file_path'], map_location='cpu', weights_only=False)
    converter = BlockBiomeConverter(mappings['block_mappings'], mappings['biome_mappings'])
    if converter.index_to_biome is None or converter.biome_to_index is None:
        raise ValueError('mappings carry no biome names; per-biome generation needs a '
                         'class-conditional dataset')
    vocab_size = len(converter.block_to_index)

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    weights, source = extract_weights(ckpt, prefer_ema=args.use_ema)
    model = build_model(config, vocab_size, device)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if missing or unexpected:
        print(f'state dict mismatch: {len(missing)} missing, {len(unexpected)} unexpected')
    if not model.class_conditional:
        raise ValueError('checkpoint is unconditional; per-biome generation needs a '
                         'class-conditional model')

    # convert_to_original_blocks treats a leading dimension equal to the vocabulary
    # size as one-hot channels, so a batch of exactly that size would be mangled
    if args.batch_size == vocab_size:
        raise ValueError(f'--batch_size must not equal the vocabulary size ({vocab_size})')

    cond_scale = args.cond_scale
    if cond_scale is None:
        cond_scale = float(config.get('default_cond_scale', 1.0))

    run_id = args.run_id or default_run_id(args.checkpoint)
    out_root = os.path.join(args.out_dir, run_id)
    os.makedirs(out_root, exist_ok=True)

    biome_names = [converter.index_to_biome[i] for i in sorted(converter.index_to_biome.keys())]
    if args.biomes:
        requested = [b.strip() for b in args.biomes.split(',') if b.strip()]
        unknown = [b for b in requested if b not in biome_names]
        if unknown:
            raise ValueError(f'unknown biomes {unknown}; available: {biome_names}')
        biome_names = requested

    textures_dir = args.textures_dir if (args.textures_dir and os.path.exists(args.textures_dir)) else None
    if args.render and textures_dir is None:
        print(f'[warn] textures not found at {args.textures_dir}; rendering untextured, '
              f'which will NOT be comparable to the diffusion renders')

    print(f'Loaded {source} weights from step {ckpt.get("step", "?")}')
    print(f'Output root: {out_root}')
    print(f'{len(biome_names)} biomes x {args.samples_per_biome} samples, '
          f'batch {args.batch_size}, guidance {cond_scale}')

    t_start = time.time()
    for biome_name in biome_names:
        class_idx = int(converter.biome_to_index[biome_name])
        payload_path = os.path.join(out_root, f'generated_{biome_name}.pt')
        gen_dir = os.path.join(out_root, biome_name)

        have = 0 if args.overwrite else existing_sample_count(payload_path)
        if have >= args.samples_per_biome:
            print(f"[skip] '{biome_name}' already has {have} samples")
            samples = torch.load(payload_path, map_location='cpu',
                                 weights_only=False)['voxels'][:args.samples_per_biome]
        else:
            t_biome = time.time()
            samples = generate_for_biome(
                model, converter, class_idx, args.samples_per_biome,
                batch_size=args.batch_size, cond_scale=cond_scale,
                temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                amp=args.amp, device=device,
            )
            torch.save({
                'voxels': samples,
                'classes': torch.full((samples.shape[0],), class_idx, dtype=torch.long),
                'class_names': [biome_name] * int(samples.shape[0]),
                'image_size': int(config.get('image_size', 32)),
            }, payload_path)
            elapsed = time.time() - t_biome
            print(f"[done] '{biome_name}' {samples.shape[0]} samples in {elapsed:.0f}s "
                  f"({elapsed / max(1, samples.shape[0]):.1f}s per sample) -> {payload_path}")

        if args.render:
            rendered = _count_images(gen_dir) if os.path.isdir(gen_dir) else 0
            if rendered >= int(samples.shape[0]) and not args.overwrite:
                print(f"[skip] '{biome_name}' already has {rendered} renders")
            else:
                render_samples_to_images(
                    samples,
                    gen_dir,
                    textures_dir=textures_dir,
                    image_px=args.image_px,
                    show_axis=args.show_axis,
                    zoom=args.zoom,
                    converter=None,  # payload already holds original block IDs
                )

    total = time.time() - t_start
    print(f'\nFinished {len(biome_names)} biomes in {total / 60:.1f} min')
    print(f'Point the evaluation scripts at: {out_root}')


if __name__ == '__main__':
    main()
