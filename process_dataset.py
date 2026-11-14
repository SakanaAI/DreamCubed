"""Build balanced memmapped training/validation data from raw biome archives."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--biome_dir', required=True, help='Directory containing <biome>_chunks*.npz')
    parser.add_argument('--biomes', required=True, help='Comma-separated source biome names')
    parser.add_argument('--output_path', required=True, help='New raw memmap directory (without a suffix)')
    parser.add_argument('--total_size', type=int, default=1_000_000)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=100, help='Preprocessing batch size')
    parser.add_argument('--boosted_biome', default=None)
    parser.add_argument('--boost_factor', type=float, default=1.0)
    parser.add_argument('--block_compression', action='store_true')
    parser.add_argument('--compression_mapping_path', default='assets/block_compression_mapping.json')
    parser.add_argument('--compression_mapping_by_name_path', default='assets/block_compression_mapping_by_name.json')
    parser.add_argument('--compression_block_types_path', default='assets/block_types_updated.json')
    args = parser.parse_args()
    names = [name.strip() for name in args.biomes.split(',') if name.strip()]
    if not names or args.total_size <= 0 or args.batch_size <= 0:
        parser.error('Provide at least one biome and positive total_size and batch_size.')
    if not 0 <= args.val_split < 1:
        parser.error('val_split must be in [0, 1).')
    if not Path(args.biome_dir).is_dir():
        parser.error(f'Raw data directory not found: {args.biome_dir}')
    output = Path(args.output_path)
    if output.suffix:
        parser.error('output_path must be a directory name without a file suffix.')
    for path in (output, output.with_name(output.name+'_cc_dir'),
                 output.with_name(output.name+'_cc_val_dir'),
                 output.with_name(output.name+'_cc_oh_mappings.pt')):
        if path.exists():
            parser.error(f'Output already exists; choose a new output_path: {path}')
    from data_utils import build_biome_dataset_from_parts_memmap_balanced, process_class_conditional_dataset_memmap
    raw = build_biome_dataset_from_parts_memmap_balanced(
        biome_dir=args.biome_dir, biome_names=names, output_path=str(output),
        total_size=args.total_size, seed=args.seed,
        boosted_biome=args.boosted_biome, boost_factor=args.boost_factor,
        apply_block_compression=args.block_compression,
        compression_mapping_path=args.compression_mapping_path,
        compression_mapping_by_name_path=args.compression_mapping_by_name_path,
        compression_block_types_path=args.compression_block_types_path,
    )
    train, val = process_class_conditional_dataset_memmap(
        raw, val_split=args.val_split, val_seed=args.seed, batch_size=args.batch_size)
    print(json.dumps({'raw':raw,'train':train,'validation':val,
                      'mappings':str(output.with_name(output.name+'_cc_oh_mappings.pt'))},indent=2))


if __name__ == '__main__':
    main()
