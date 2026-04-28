# DreamCubed

This repository contains the training, inference, and evaluation code for Dream-Cubed. The two training entry points are:

- `train_discrete.py` for the discrete MD4 model
- `train_conditional_ddpm.py` for the continuous DDPM model

## Pretrained Models

- [MD4 Patch 4 (balanced natural biome dataset)](https://huggingface.co/SakanaAI/DreamCubedDP4)
- [MD4 Patch 2 (balanced natural biome dataset)](https://huggingface.co/SakanaAI/DreamCubedDP2)
- [DDPM Patch 2 (balanced natural biome dataset)](https://huggingface.co/SakanaAI/DreamCubedCP2)

## Dataset

Raw Minecraft chunk data and the processed datasets used for training are available here:

- [Dream-Cubed Dataset](https://huggingface.co/datasets/SakanaAI/DreamCubed2M)

## Setup

Run commands from the repository root.

## Training

Models are configured through JSON files passed with `--config`. The launchers forward any additional CLI flags to the underlying trainer, so you can override individual config values from the command line.

### MD4

```bash
bash launch.sh <num_gpus> <path/to/config.json>
```

This calls `train_discrete.py`, which trains the discrete MD4 model. Important config fields usually include:

- dataset paths: `dataset_path`, `mappings_file_path`, optionally `val_dataset_path` and `val_mappings_file_path`
- model settings: `image_size` (block dimension), `patch_size`, `hidden_channels`, `depth`, `num_heads`
- optimization: `train_batch_size`, `per_device_train_batch_size`, `train_lr`, `train_num_steps` or `train_num_epochs`
- sampling and evaluation: `reverse_steps`, `sampling_timesteps`
- output: `results_folder`, `run_name`, `exp_name`

Each run writes its resolved training config to `results/<run_name>/<exp_name>/config.json`. Inference and FID scripts reuse that file later.

### DDPM

```bash
bash launch_ddpm.sh <num_gpus> <path/to/config.json>
```

This calls `train_conditional_ddpm.py`, which trains the continuous DDPM model with a DiT backbone. Its configs follow the same structure as MD4, but also require the embedding-related fields used by the continuous representation, especially `embeddings_path` and `data_channels`.

Training config examples are not bundled in this clean repo, but any JSON passed to the launchers can set the arguments exposed by the trainer entry points.

## Inference

`inference.py` provides the shared checkpoint-loading and sampling helpers used by the other inference utilities. The main inference workflows we use are `inpaint.py`, `inpaint_ddpm.py`, and `semantic_super_sample.py`.

### Inpainting

`inpaint.py` runs MD4 inpainting experiments. The main modes are:

- `standard`: uses two source samples per biome from a folder of `generated_<biome>.pt` files and produces multiple stochastic inpainted variants from a fixed corner context
- `biome_swap`: keeps a configurable fraction of a source chunk fixed and regenerates the rest while conditioning on a different target biome
- `seeded_inpaint`: starts from notebook-authored seed context `.pt` files and can run either unconditional or biome-conditioned infilling

Standard inpainting:

```bash
python inpaint.py \
  --checkpoint path/to/model_best.pt \
  --mappings path/to/mappings.pt \
  --source_folder path/to/generated_samples \
  --experiment_mode standard \
  --num_variants 4
```

Biome swap:

```bash
python inpaint.py \
  --checkpoint path/to/model_best.pt \
  --mappings path/to/mappings.pt \
  --source_folder path/to/generated_samples \
  --experiment_mode biome_swap \
  --source_biomes ocean,plains,village \
  --target_biomes ice,desert,plains \
  --biome_swap_context_fraction 0.5 \
  --num_variants 4
```

Seeded inpainting:

```bash
python inpaint.py \
  --checkpoint path/to/model_best.pt \
  --mappings path/to/mappings.pt \
  --experiment_mode seeded_inpaint \
  --seed_context_dir seed_context_outputs \
  --num_variants 4
```

### DDPM Inpainting

`inpaint_ddpm.py` provides the DDPM version of the same inpainting workflow. It supports the same main experiment modes documented above for `inpaint.py`:

- `standard`
- `biome_swap`
- `seeded_inpaint`

It can optionally take `--embeddings path/to/block_embeddings.npy`. If omitted, it falls back to the decode matrix stored in the checkpoint.

Standard inpainting:

```bash
python inpaint_ddpm.py \
  --checkpoint path/to/model_best.pt \
  --mappings path/to/mappings.pt \
  --source_folder path/to/generated_samples \
  --experiment_mode standard \
  --num_variants 4
```

Biome swap:

```bash
python inpaint_ddpm.py \
  --checkpoint path/to/model_best.pt \
  --mappings path/to/mappings.pt \
  --source_folder path/to/generated_samples \
  --experiment_mode biome_swap
```

Seeded inpainting:

```bash
python inpaint_ddpm.py \
  --checkpoint path/to/model_best.pt \
  --mappings path/to/mappings.pt \
  --experiment_mode seeded_inpaint \
  --seed_context_dir seed_context_outputs \
  --num_variants 4
```

### Semantic Super Sampling

`semantic_super_sample.py` generates larger worlds by stitching chunk-sized generations together with overlap-aware inpainting. It can run from:

- a predefined biome layout via `--layout`
- a custom JSON grid via `--custom_grid`
- a seed-context layout via `--context_grid`
- no layout at all, which falls back to an unconditional `5x5` generation

It also supports several chunk generation orders:

- `raster`: simple left-to-right, bottom-to-top generation
- `spiral`: generate the outer ring first and work inward
- `frontier`: always choose the next chunk with the most generated neighbors
- `checkerboard`: generate alternating cells first, then fill the gaps
- `biome`: finish one connected same-biome region at a time


List built-in layouts:

```bash
python semantic_super_sample.py --list_layouts
```

Generate a world from a predefined layout:

```bash
python semantic_super_sample.py \
  --checkpoint path/to/model_best.pt \
  --mappings path/to/mappings.pt \
  --layout coastline_4x4 \
  --scan_order frontier \
  --output_dir super_sample_results
```

## Visualization

`visualization_utils.py` contains the `MinecraftVisualizerPyVista` renderer used throughout the inference and evaluation scripts. This is the same visualizer used by `inpaint.py`, `inpaint_ddpm.py`, `semantic_super_sample.py`, and `render_dataset.py` to produce textured chunk renders.

The typical flow is:

1. convert model outputs back to original Minecraft block IDs
2. create a `MinecraftVisualizerPyVista` with `textures_dir="block_textures/"` and `build_textures=True`
3. call `visualize_chunk_textured(...)`
4. save the result with `plotter.screenshot(...)`
5. close the plotter

Minimal example:

```python
import torch

from data_utils import BlockBiomeConverter
from visualization_utils import MinecraftVisualizerPyVista

mappings = torch.load("path/to/mappings.pt", weights_only=False)
converter = BlockBiomeConverter(
    mappings["block_mappings"],
    mappings["biome_mappings"],
)

# chunk_indices is a [H, W, D] tensor in model index space
chunk_blocks = converter.convert_to_original_blocks(chunk_indices)

visualizer = MinecraftVisualizerPyVista(
    textures_dir="block_textures/",
    build_textures=True,
)

plotter = visualizer.visualize_chunk_textured(
    chunk_blocks,
    interactive=False,
    show_axis=False,
)
plotter.screenshot(
    filename="chunk_render.png",
    window_size=(512, 512),
    transparent_background=False,
)
plotter.close()
```

If you do not want textured rendering, the same visualizer can fall back to `visualize_chunk(...)` instead.

## FID Evaluation

`calc_fid.py` supports direct image-folder FID, per-biome FID from pre-generated samples, and end-to-end evaluation from a trained model.

For our main evaluation workflow, use the per-biome model mode:

```bash
python calc_fid.py \
  --per_biome_fid_model path/to/run_or_checkpoint \
  --mappings_path path/to/mappings.pt \
  --samples_per_biome 100 \
  --per_biome_ref_split val \
  --per_biome_gen_dir renders/per_biome_gen
```

This mode:

1. loads the trained model
2. generates `100` samples per biome by default
3. renders the generated samples
4. renders the reference dataset split if those images do not already exist
5. computes per-biome FID and writes JSON and NPZ summaries

If you already have saved per-biome sample tensors, use:

```bash
python calc_fid.py \
  --per_biome_fid_samples_dir path/to/per_biome_samples \
  --per_biome_samples_format block_ids \
  --mappings_path path/to/mappings.pt \
  --dataset_path path/to/reference_dataset \
  --samples_per_biome 100
```

## Notes

- Training, inference, and FID scripts all assume that checkpoint directories contain the `config.json` written during training.
- Rendered outputs depend on `block_textures/`; if textures are missing, some scripts can still fall back to untextured rendering.
- Processed datasets can be either saved `.pt` tensors or memmap directories containing `manifest.json`.
