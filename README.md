# DreamCubed

This repository contains the training, inference, and evaluation code for Dream-Cubed. The generative-model training entry points are:

- `train_discrete.py` for the discrete MD4 model
- `train_conditional_ddpm.py` for the continuous DDPM model
- `train_autoregressive.py` for the autoregressive (AR) baseline

## Pretrained Models

Pretrained model weights and configs using the core dataset are available in the public Hugging Face model: https://huggingface.co/dream-cubed/dream-cubed-models

Pretrained model weights and configs using the human-authored dataset are available in the public Hugging Face model: https://huggingface.co/dream-cubed/dream-cubed-human-models

## Dataset

Core dataset (collected from procedurally generated Minecraft worlds) available in the public Hugging Face dataset: https://huggingface.co/datasets/dream-cubed/DreamCubedNatural

Human-authored dataset (collected from human-authored Minecraft maps) available in the public Hugging Face dataset: https://huggingface.co/datasets/dream-cubed/DreamCubedHuman

## Setup

Run commands from the repository root.

Multiline shell examples below use Bash syntax. In PowerShell, put the command
on one line or use PowerShell backticks for line continuation.

Use Python 3.10 and an isolated environment. Install a matching CUDA PyTorch and
torchvision pair before the remaining dependencies. The following setup uses
PyTorch 2.5.1+cu121 and torchvision 0.20.1+cu121:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv/Scripts/Activate.ps1
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

The example CUDA wheel requires a compatible NVIDIA driver. Use hardware with
sufficient memory for the model and batch size in your configuration.
For the optional scraping utilities, also install
`python -m pip install -r data_scraping/requirements.txt`.

## Training

Models are configured through JSON files passed with `--config`. The launchers forward any additional CLI flags to the underlying trainer, so you can override individual config values from the command line.

With one GPU, launchers run the trainer directly; multiple GPUs use Accelerate.
Explicit config paths (including paths with spaces) are accepted, as is the
legacy `training_configs/<name>.json` shorthand. Set `DREAMCUBED_PYTHON` to select
a Python executable. On Windows, direct `python train_*.py --config ...` commands
also work without Bash. Set `dataloader_num_workers=0` for MD4/DDPM or
`num_workers=0` for AR/FID3D when debugging worker startup.

Download the resolved configs alongside the released model weights and update
dataset, mapping, embedding and output paths for your machine. Use the same AR
training schedule when resuming a checkpoint. MD4's legacy online `fid_*`
training options are unsupported; use the standalone evaluation scripts below.

For a new AR training run, override the published `resume: true` with
`--resume false`. DDPM resolved configs may contain both step and epoch values;
positive step values take precedence. Keep the published architecture settings
when loading released weights.

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

### Autoregressive baseline

`train_autoregressive.py` uses `autoregressive_dit.py` and `autoregressive_trainer.py`. Supply a training JSON with your dataset/mapping paths and model settings; use the resolved config from the evaluated run when reproducing results.

```bash
python train_autoregressive.py --config path/to/ar_config.json
python ar_generate_eval_samples.py \
  --checkpoint path/to/model_best.pt \
  --config path/to/config.json \
  --out_dir renders/per_biome_gen --run_id ar_baseline \
  --samples_per_biome 100 --no_render
```

`ar_generate_eval_samples.py` also requires `ar_sample_timing.py`. Set the mapping path in the resolved config to your local dataset mapping. The generated per-biome voxel files can be evaluated with the metrics below.

Time AR sampling separately (this generates a batch but does not save samples):

```bash
python ar_sample_timing.py --checkpoint path/to/inference_weights/model_best_inference.pt --batch_size 1
```

The config is discovered beside the checkpoint or one directory above it;
`--config path/to/config.json` overrides discovery. Timing depends strongly on
batch size and hardware, so the script's extrapolation is only an estimate.

## Inference

`inference.py` provides the shared checkpoint-loading and sampling helpers used by the other inference utilities. The main inference workflows we use are `inpaint.py`, `inpaint_ddpm.py`, and `semantic_super_sample.py`.

### Inpainting

`inpaint.py` runs MD4 inpainting experiments. The main modes are:

- `standard`: uses two source samples per biome from a folder of `generated_<biome>.pt` files and produces multiple stochastic inpainted variants from a fixed corner context
- `biome_swap`: keeps a configurable fraction of a source chunk fixed and regenerates the rest while conditioning on a different target biome
- `seeded_inpaint`: starts from notebook-authored seed context `.pt` files and can run either unconditional or biome-conditioned infilling

Standard inpainting:

The standard protocol expects 32-cubed source chunks and preserves an
8-by-32-by-8 corner. Use `--no_render` to save voxel results without rendering.

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

It can optionally take `--embeddings path/to/block_embeddings.npy` (or a `.pt`
table). If omitted, it first tries the config's `embeddings_path`, then an
embedded checkpoint decode table when available. The older natural DDPM export
has no embedded table; keep its configured `assets/full_embeddings_16_raw.pt`
available or supply the matching table explicitly.

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

Render a processed dataset directly:

```bash
python render_dataset.py --data path/to/validation_split --mapping path/to/mappings.pt --out_dir renders/reference --start 0 --end 100 --image_size 512
```

`--start` is inclusive and `--end` is exclusive. Use the matching mapping to
convert stored vocabulary indices to raw block IDs.

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

For untextured rendering, construct the visualizer with `build_textures=False`
and call `visualize_chunk(...)`. Textured rendering requires mappings and texture
files for the blocks actually present in the chunk. Missing used textures or
unknown block IDs raise an error in strict mode; unused atlas entries do not
prevent rendering.

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

Image-FID reference statistics are cached in `fid_stats_<mode>.npz`. Remove the
relevant cache when changing reference images or the feature-extraction setup.
Image FID needs at least two images to compute covariance. Use the sample counts
and evaluation protocol from your experiment for meaningful comparisons.

## Voxel-based evaluation

`train_fid3d.py` trains the 3D feature extractor defined in `fid3d_model.py`:

```bash
python train_fid3d.py --config path/to/fid3d_config.json
```

The evaluation scripts compute 3D FID (`fid3d_eval.py`), KID/MMD (`kernel3d_eval.py`), and MAUVE (`mauve3d_eval.py`). For example:

```bash
python fid3d_eval.py \
  --samples_dir path/to/generated_voxel_samples \
  --checkpoint path/to/feature_extractor_best.pt \
  --ref_dataset path/to/restored_validation_split \
  --mappings_path path/to/mappings.pt
```

Replace `fid3d_eval.py` with `kernel3d_eval.py` or `mauve3d_eval.py` to run the other metrics with the same inputs. Here `--checkpoint` is the trained feature extractor, not the generator. Samples default to raw block IDs; processed references default to vocabulary indices. Use the matching mapping and an unsharded/restored validation split. Both companion scripts depend on `fid3d_eval.py` and `fid3d_model.py`.

Use the same extractor, reference data, sample counts and protocol settings across models; see each script's `--help` for options. These are usage examples, not the complete settings for reproducing the paper's tables.

### Block-frequency statistics

Compare a processed reference split with generated per-biome voxel files:

```bash
python compute_voxel_block_stats.py --dataset-dir path/to/validation_split --dataset-mappings-path path/to/mappings.pt --samples-dir path/to/generated_samples --output-dir results/block_stats --no-plots
```

Generated files should follow `generated_<biome>.pt` naming and contain raw block
IDs. The processed reference uses vocabulary indices and its corresponding
mapping. Omit `--no-plots` to also create plots; see `--help` for output options.

## Notes

- Keep the training `config.json` beside the checkpoint or one directory above it, matching the published `inference_weights/` layout; use explicit config arguments where available.
- Rendered outputs depend on `block_textures/`; if textures are missing, some scripts can still fall back to untextured rendering.
- Processed datasets can be either saved `.pt` tensors or memmap directories containing `manifest.json`.

### Preprocess raw biome archives

`process_dataset.py` accepts raw `<biome>_chunks*.npz` archives and creates
balanced memmapped data, compressed vocabulary mappings, and training/validation
splits. It refuses to overwrite an existing output:

```bash
python process_dataset.py --biome_dir path/to/raw --biomes plains,desert \
  --output_path data/example --total_size 1000 --val_split 0.1
```

This writes `data/example`, `data/example_cc_dir`, `data/example_cc_val_dir` and
`data/example_cc_oh_mappings.pt`. Use the latter three for training/validation.
Raw block-ID compression is optional (`--block_compression`); see `--help` for
its mapping paths. This workflow consumes raw archives, not Hugging Face shards.


# Minecraft Data Scraping

Scripts for scraping data from Minecraft servers are available in the data_scraping folder. This extends the Evocraft API for interfacing with a running Minecraft server to select and save voxel chunks from a world, and uses Pyubiomes for targeted biome sampling and village location.

This release includes the scripts used to collect voxel chunks from Minecraft worlds and to estimate biome frequencies. The main entry points are:

- `gen_data.py`: scrape block, biome, and metadata chunks from a running Minecraft server.
- `get_biome_dist.py`: estimate biome frequencies by querying sparse biome samples from a running Minecraft server.
- `auto_collect.py` and `auto_collect_biome_dist.py`: optional wrappers that regenerate worlds for one or more seeds, start the server, run the scraper, and stop the server.

The scraper talks to a local Minecraft 1.12.2 server through the gRPC service defined in `src/main/proto/minecraft.proto`. The generated Python stubs are checked in under `clients/python/src/main/proto/`.

## System Setup

Use Java 8 for the Minecraft server.

Linux:

```bash
sudo apt-get update
sudo apt-get install -y openjdk-8-jdk build-essential python3-dev
```

macOS:

```bash
/usr/libexec/java_home -v 1.8
```

Set up Python dependencies:

```bash
cd data_scraping
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e Pyubiomes
```

The local `Pyubiomes` package builds a small C extension, so `python3-dev` or equivalent Python headers must be available.

Run the remaining scraping commands from `data_scraping/`. Hydra uses its basic
local launcher by default; Submitit is only needed if you explicitly configure
that cluster launcher. Generic RPC extraction can import without Pyubiomes;
seed-based biome and village lookup still requires the native helper.

## Minecraft Server Setup

Create a local `server/` directory containing:

- `spongevanilla-1.12.2-7.3.0.jar`
- `mods/minecraft-rpc-0.0.5.jar`

The Minecraft server jar and custom RPC mod jar are runtime artifacts and are not installed by `pip`. Provide them with the release artifacts or recreate them from the Minecraft RPC server project before running the scrapers.

Start the server once from inside `server/`:

```bash
cd server
java -jar spongevanilla-1.12.2-7.3.0.jar nogui
```

On the first launch, accept the Minecraft EULA in `server/eula.txt`, then start the server again. The server is ready for scraping when it logs that the gRPC service is listening on port `5001`.

To use a specific Java 8 executable on macOS:

```bash
cd server
/usr/libexec/java_home -v 1.8 --exec java -jar spongevanilla-1.12.2-7.3.0.jar nogui
```

To inspect the world manually, install Minecraft Java Edition `1.12.2`, open Multiplayer, choose Direct Connect, and join `localhost`.

World settings live in `server/server.properties`. To generate a fresh world after changing settings or `level-seed`, stop the server, delete the generated world directory, and restart the server.

## Fast Biome Locator

`gen_data.py` can use seed-based biome lookup for targeted biome scraping. Build the local cubiomes-backed helper once on a new Linux machine:

```bash
bash native/build_fast_biomes.sh
```

This produces `native/libdream_cubiomes.so` from the vendored sources in `Pyubiomes/cubiomes/`. If the shared library is missing, `seed_utils.py` will try to build it automatically on non-Windows systems; otherwise it falls back to the installed local `Pyubiomes` package.

## Running `gen_data.py`

Most settings live in `conf/config.yaml` and can be overridden from the command line through Hydra.

Collect general overworld voxel chunks:

```bash
python gen_data.py data_gen.mode=voxels data.num_samples=1000
```

Collect chunks from a configured rectangular area:

```bash
python gen_data.py data_gen.mode=voxels_in_area data.map_name=my_world data.area_corners='[[0,0],[512,512]]'
```

Collect chunks from a target biome:

```bash
python gen_data.py data_gen.mode=targeted_biome data.targeted_biome_label=river data.num_samples=1000
```

Other supported modes are `villages`, `caves`, and `nether`. Outputs are written under `data/Voxels/` as compressed `.npz` part files or combined chunk files, depending on mode.

## Running `get_biome_dist.py`

Estimate biome frequencies from the running server:

```bash
python get_biome_dist.py --dimension overworld --max-radius-blocks 10000 --window-size 32
```

The output JSON defaults to `analysis/biome_distributions/` and includes raw biome counts, simplified biome counts, and the normalized distribution.

## Player Selection

Most scraping modes do not need to move a connected player. RPC methods that do move a player use the first online player by default. To force a specific player, set `MCRPC_PLAYER` in the server environment before launch.

## Optional Automation

Run `gen_data.py` across fresh worlds:

```bash
python auto_collect.py --iterations 3 --mode targeted_biome --targeted-biome-label river
```

Run biome-distribution estimates across fresh worlds:

```bash
python auto_collect_biome_dist.py --iterations 3 --dimension overworld
```

Both automation scripts expect `server/server.properties` and the server jar to already exist. They update `level-seed`, delete the configured world directory, start the server, wait for readiness, run the scraper, and stop the server.
