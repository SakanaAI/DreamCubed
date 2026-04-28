import os
import argparse
from typing import Optional, List, Tuple, Union, Dict, Any
import torch
from tqdm import tqdm
import numpy as np
import json

from visualization_utils import MinecraftVisualizerPyVista
from data_utils import BlockBiomeConverter


def bool_flag(value: str) -> bool:
    if isinstance(value, bool):
        return value
    val = value.lower()
    if val in {"yes", "true", "t", "y", "1"}:
        return True
    if val in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def render_dataset_to_images(
    data_path: str,
    mappings_path: str,
    output_dir: str,
    textures_dir: Optional[str] = None,
    image_size: int = 256,
    show_axis: bool = False,
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
    skip_existing: bool = True,
    zoom: float = 1.0,
    corrupt_prob: float = 0.0,
):
    # Create subdirectory under output_dir named after the data file (without extension)
    dataset_name = os.path.splitext(os.path.basename(data_path))[0]
    if corrupt_prob and corrupt_prob > 0.0:
        pct = int(round(corrupt_prob * 100))
        dataset_name = f"{dataset_name}_corrupted_p{pct}"
    save_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    # Load dataset (supports .pt tensor bundles and memmap dataset directories)
    voxels, class_labels = _load_processed_dataset_any_format(data_path)
    mappings = torch.load(mappings_path, weights_only=False)
    converter = BlockBiomeConverter(mappings['block_mappings'], mappings['biome_mappings'])

    # Validate dims (support numpy arrays and torch tensors)
    vox_ndim = voxels.dim() if isinstance(voxels, torch.Tensor) else voxels.ndim
    if vox_ndim not in (4, 5):
        raise ValueError(f"Expected 'voxels' to have 4 dims [N,H,W,D] (indices) or 5 dims [N,C,H,W,D] (one-hot/embeddings), got {voxels.shape}")

    num_samples = voxels.shape[0]
    s_idx = 0 if start_index is None else max(0, start_index)
    e_idx = num_samples if end_index is None else min(end_index, num_samples)
    if s_idx >= e_idx:
        return

    # Initialize visualizer (textured if textures_dir is provided)
    if textures_dir:
        visualizer = MinecraftVisualizerPyVista(textures_dir=textures_dir, build_textures=True)
        render_fn = visualizer.visualize_chunk_textured
    else:
        visualizer = MinecraftVisualizerPyVista()
        render_fn = visualizer.visualize_chunk

    # Determine corruption settings if requested
    num_blocks = None
    air_idx = None
    if corrupt_prob and corrupt_prob > 0.0:
        num_blocks = len(converter.block_to_index)
        try:
            air_idx = converter.get_air_block_index()
        except Exception:
            air_idx = 0

    for i in tqdm(range(s_idx, e_idx), total=e_idx - s_idx, desc="Rendering", unit="img"):
        out_path = os.path.join(save_dir, f"chunk_{i:06d}.png")
        if skip_existing and os.path.exists(out_path):
            continue

        # Normalize sample to integer indices [H,W,D]
        if vox_ndim == 5:
            sample = voxels[i]  # [C,H,W,D]
            if isinstance(sample, torch.Tensor):
                indices = torch.argmax(sample, dim=0)
            else:
                # numpy array channel-first -> argmax over channel
                indices = torch.from_numpy(np.asarray(sample)).argmax(dim=0)
        else:  # [N,H,W,D]
            sample = voxels[i]
            indices = sample.long() if isinstance(sample, torch.Tensor) else torch.from_numpy(np.asarray(sample)).long()

        # Optionally corrupt non-air voxels with probability p by assigning random indices
        if corrupt_prob and corrupt_prob > 0.0:
            rnd = torch.rand(indices.shape)
            mask = (rnd < float(corrupt_prob))
            if air_idx is not None:
                mask = mask & (indices != int(air_idx))
            random_vals = torch.randint(low=0, high=int(num_blocks), size=indices.shape, dtype=torch.long)
            indices = indices.clone()
            indices[mask] = random_vals[mask]

        # Map indices -> original block IDs for rendering
        converted_sample = converter.convert_to_original_blocks(indices)

        plotter = render_fn(converted_sample, interactive=False, show_axis=show_axis)
        if zoom != 1.0 and hasattr(plotter, "camera") and hasattr(plotter.camera, "zoom"):
            plotter.camera.zoom(zoom)
        plotter.screenshot(
            filename=out_path,
            window_size=(image_size, image_size),
            transparent_background=False,
        )
        plotter.close()


def _load_processed_dataset_any_format(data_path: str) -> Tuple[Union[torch.Tensor, np.ndarray], Optional[torch.Tensor]]:
    """
    Load processed dataset either from a .pt file (torch.save) or a memmap directory with manifest.json.
    Returns:
      voxels: [N,C,H,W,D] one-hot (torch) or [N,H,W,D] indices (torch/numpy)
      class_labels: torch.LongTensor of shape [N] (indices) or [N,C] (one-hot), or None
    """
    if os.path.isdir(data_path) and os.path.exists(os.path.join(data_path, 'manifest.json')):
        # Memmap directory
        man_path = os.path.join(data_path, 'manifest.json')
        with open(man_path, 'r') as f:
            manifest = json.load(f)
        vox_rel = manifest['paths']['voxels']
        lbl_rel = manifest['paths']['biome_labels']
        labels_format = manifest.get('class_labels_format', 'indices')
        vox_np = np.load(os.path.join(data_path, vox_rel), mmap_mode='r')
        lbl_np = np.load(os.path.join(data_path, lbl_rel), mmap_mode='r')
        # Convert labels to torch tensor for convenience
        if labels_format == 'one_hot' or (isinstance(lbl_np, np.ndarray) and lbl_np.ndim == 2):
            class_labels = torch.from_numpy(np.asarray(lbl_np)).long()
        else:
            class_labels = torch.from_numpy(np.asarray(lbl_np)).long().view(-1)
        return vox_np, class_labels
    else:
        # Torch .pt file
        data = torch.load(data_path, map_location="cpu", weights_only=False)
        if "voxels" not in data:
            raise KeyError("Loaded file does not contain a 'voxels' key.")
        vox = data["voxels"]
        cls = data.get("biomes", None)
        return vox, cls


def render_dataset_to_images_by_biome(
    data_path: str,
    mappings_path: str,
    output_dir: str,
    *,
    textures_dir: Optional[str] = None,
    image_size: int = 256,
    show_axis: bool = False,
    per_biome: Optional[int] = None,
    skip_existing: bool = True,
    zoom: float = 1.0,
    corrupt_prob: float = 0.0,
    only_biomes: Optional[List[str]] = None,
):
    """
    Render up to N samples per biome into biome-specific subdirectories.

    Output layout: <output_dir>/<dataset_name>/<biome_name>/chunk_XXXXXX.png
    """
    # Base directory derived from dataset file name
    dataset_name = os.path.splitext(os.path.basename(data_path))[0]
    if corrupt_prob and corrupt_prob > 0.0:
        pct = int(round(corrupt_prob * 100))
        dataset_name = f"{dataset_name}_corrupted_p{pct}"
    base_save_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(base_save_dir, exist_ok=True)

    # Load dataset (supports .pt tensor bundles and memmap dataset directories)
    voxels, class_labels = _load_processed_dataset_any_format(data_path)
    if class_labels is None:
        raise KeyError("Loaded data does not contain class labels 'biomes' / 'biome_labels'.")
    mappings = torch.load(mappings_path, weights_only=False)
    converter = BlockBiomeConverter(mappings['block_mappings'], mappings['biome_mappings'])

    # Normalize class labels to integer indices [N]
    if isinstance(class_labels, torch.Tensor) and class_labels.dim() == 2 and class_labels.shape[1] > 1:
        biome_indices = torch.argmax(class_labels, dim=1).long()
    else:
        biome_indices = class_labels.long().view(-1)

    # Initialize visualizer (textured if available)
    if textures_dir:
        visualizer = MinecraftVisualizerPyVista(textures_dir=textures_dir, build_textures=True)
        render_fn = visualizer.visualize_chunk_textured
    else:
        visualizer = MinecraftVisualizerPyVista()
        render_fn = visualizer.visualize_chunk

    # Corruption settings (optional)
    num_blocks = None
    air_idx = None
    if corrupt_prob and corrupt_prob > 0.0:
        num_blocks = len(converter.block_to_index)
        try:
            air_idx = converter.get_air_block_index()
        except Exception:
            air_idx = 0

    # Determine biomes to render from mappings (name -> index)
    idx_to_biome = converter.index_to_biome or {}
    biome_to_idx = converter.biome_to_index or {}

    # Optional restriction to a subset of biome names
    biome_names = [idx_to_biome[i] for i in sorted(idx_to_biome.keys())]
    if only_biomes is not None:
        requested = set(map(str, only_biomes))
        biome_names = [b for b in biome_names if b in requested]

    # For each biome, select up to N indices and render
    for biome_name in biome_names:
        if biome_name not in biome_to_idx:
            continue
        b_idx = int(biome_to_idx[biome_name])

        # Find dataset indices for this biome
        sel = (biome_indices == b_idx).nonzero(as_tuple=False).squeeze(1).tolist()
        if len(sel) == 0:
            continue

        if per_biome is not None and per_biome >= 0:
            sel = sel[: int(per_biome)]

        biome_dir = os.path.join(base_save_dir, biome_name)
        os.makedirs(biome_dir, exist_ok=True)

        for i in tqdm(sel, total=len(sel), desc=f"Rendering [{biome_name}]", unit="img"):
            out_path = os.path.join(biome_dir, f"chunk_{i:06d}.png")
            if skip_existing and os.path.exists(out_path):
                continue

            # Normalize sample to integer indices [H,W,D]
            vox_ndim = voxels.dim() if isinstance(voxels, torch.Tensor) else voxels.ndim
            if vox_ndim == 5:
                sample = voxels[i]  # [C,H,W,D]
                if isinstance(sample, torch.Tensor):
                    indices = torch.argmax(sample, dim=0)
                else:
                    indices = torch.from_numpy(np.asarray(sample)).argmax(dim=0)
            else:  # [N,H,W,D]
                sample = voxels[i]
                indices = sample.long() if isinstance(sample, torch.Tensor) else torch.from_numpy(np.asarray(sample)).long()

            # Optional corruption of non-air voxels
            if corrupt_prob and corrupt_prob > 0.0:
                rnd = torch.rand(indices.shape)
                mask = (rnd < float(corrupt_prob))
                if air_idx is not None:
                    mask = mask & (indices != int(air_idx))
                random_vals = torch.randint(low=0, high=int(num_blocks), size=indices.shape, dtype=torch.long)
                indices = indices.clone()
                indices[mask] = random_vals[mask]

            # Map indices -> original block IDs for rendering
            converted_sample = converter.convert_to_original_blocks(indices)

            plotter = render_fn(converted_sample, interactive=False, show_axis=show_axis)
            if zoom != 1.0 and hasattr(plotter, "camera") and hasattr(plotter.camera, "zoom"):
                plotter.camera.zoom(zoom)
            plotter.screenshot(
                filename=out_path,
                window_size=(image_size, image_size),
                transparent_background=False,
            )
            plotter.close()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a dataset of voxel chunks into images for FID.")
    parser.add_argument("--data", required=True, help="Path to processed dataset (.pt with 'voxels'/'biomes' or memmap dir with manifest.json).")
    parser.add_argument("--mapping", required=True, help="Path to .pt mapping file.")
    parser.add_argument("--out_dir", default='dataset_renders/', help="Base directory to save rendered images.")
    parser.add_argument("--textures_dir", default='block_textures/', help="Optional textures dir to enable textured rendering.")
    parser.add_argument("--image_size", type=int, default=256, help="Square image width/height in pixels.")
    parser.add_argument("--show_axis", type=bool_flag, default=False, help="Whether to show axes on renders.")
    parser.add_argument("--start", type=int, default=None, help="Start index (inclusive) in dataset.")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive) in dataset.")
    parser.add_argument("--no_skip_existing", action="store_true", help="Do not skip images that already exist.")
    parser.add_argument("--zoom", type=float, default=1.0, help="Camera zoom factor (>1 zooms in, <1 zooms out).")
    parser.add_argument("--corrupt_prob", type=float, default=0.0, help="Probability to replace each non-air voxel with a random block index.")
    # Biome-split rendering options
    parser.add_argument("--split_by_biome", action="store_true", help="Render images split into biome-specific directories.")
    parser.add_argument("--per_biome", type=int, default=None, help="Max number of samples to render per biome; default=all.")
    return parser

# python calc_fid.py --data data/voxel_dataset_processed.pt --out_dir renders --start 0 --end 1000
def main():
    parser = build_argparser()
    args = parser.parse_args()

    if args.split_by_biome:
        render_dataset_to_images_by_biome(
            data_path=args.data,
            mappings_path=args.mapping,
            output_dir=args.out_dir,
            textures_dir=args.textures_dir,
            image_size=args.image_size,
            show_axis=args.show_axis,
            per_biome=args.per_biome,
            skip_existing=not args.no_skip_existing,
            zoom=args.zoom,
            corrupt_prob=args.corrupt_prob,
        )
    else:
        render_dataset_to_images(
            data_path=args.data,
            mappings_path=args.mapping,
            output_dir=args.out_dir,
            textures_dir=args.textures_dir,
            image_size=args.image_size,
            show_axis=args.show_axis,
            start_index=args.start,
            end_index=args.end,
            skip_existing=not args.no_skip_existing,
            zoom=args.zoom,
            corrupt_prob=args.corrupt_prob,
        )


if __name__ == "__main__":
    main()

