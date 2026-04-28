"""
Semantic Super Sampling for MD4 Discrete Diffusion.

Generates large terrain by stitching 32x32x32 chunks in a configurable grid
layout. Each cell is assigned a biome and generated with overlapping context
strips from already-generated neighbors to ensure smooth transitions.

Supports five scan orders:
  - raster: left-to-right, bottom-to-top (each cell sees left + below neighbors)
  - spiral: outer ring first, spiraling inward (inner cells get context from
    all surrounding sides, reducing disjointness at biome boundaries)
  - frontier: choose the next cell with the most already-generated neighbors,
    so each step tends to maximize available local context
  - checkerboard: generate alternating cells first, then fill the gaps so the
    second pass usually gets context from two or more sides
  - biome: generate one connected same-biome region at a time, using raster
    order within each region

Usage:
    python semantic_super_sample.py \
        --checkpoint path/to/model.pt \
        --mappings path/to/mappings.json \
        --layout coastline_4x4 \
        --output_dir ./super_sample_results

    python semantic_super_sample.py --list_layouts
    python semantic_super_sample.py --checkpoint ... --list_biomes
"""

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from accelerate import Accelerator

from inference import load_model_from_file, _default_device
from data_utils import BlockBiomeConverter
from gif_utils import render_diffusion_gif
from visualization_utils import MinecraftVisualizerPyVista
from inpaint import (
    MD4Inpainter,
    _load_seed_context_payload,
    render_chunk_to_file,
    render_chunk_to_file_fitted_iso,
    render_side_by_side,
)

LANCZOS_RESAMPLE = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS


# =============================================================================
# Predefined World Layouts
# =============================================================================
# Grids are defined visually: row 0 = north/top (far from camera in render),
# last row = south/bottom (close to camera). The generation function reverses
# them so processing goes south-to-north, west-to-east.

WORLD_LAYOUTS = {
    "coastline_4x4": {
        "description": "Diagonal forest-beach-ocean coastline",
        "grid": [
            ["forest",  "forest",  "forest",  "forest"],
            ["forest",  "forest",  "forest",  "beach"],
            ["forest",  "forest",  "beach",   "ocean"],
            ["forest",  "beach",   "ocean",   "ocean"],
        ],
    },
    "delta_5x5": {
        "description": "River leading to the ocean",
        "grid": [
            ["ocean",  "ocean",  "ocean",  "ocean", "ocean"],
            ["beaches",  "beaches",  "river",  "beaches", "beaches"],
            ["plains",  "plains",  "river",  "plains", "plains"],
            ["forest",  "forest",  "river",  "forest", "forest"],
            ["forest",  "forest",  "extreme_hills",  "forest", "forest"],
        ],
    },
    "desert_river_4x4": {
            "description": "River in the desert",
            "grid": [
                ["desert",  "beaches",  "beaches",  "desert"],
                ["desert",  "river",  "desert",  "desert"],
                ["desert",  "river",  "desert",  "desert"],
                ["desert",  "desert",  "desert",  "desert"],
            ],
        },
    "coastline_3x3": {
        "description": "Small diagonal coastline",
        "grid": [
            ["forest",  "forest",  "beach"],
            ["forest",  "beach",   "ocean"],
            ["beach",   "ocean",   "ocean"],
        ],
    },
    "coastline_2x2": {
        "description": "Minimal coastline for quick testing",
        "grid": [
            ["forest",  "beach"],
            ["beach",   "ocean"],
        ],
    },
    "island_4x4": {
        "description": "Forest island surrounded by ocean",
        "grid": [
            ["ocean",   "ocean",   "ocean",   "ocean"],
            ["ocean",   "forest",  "forest",  "ocean"],
            ["ocean",   "forest",  "forest",  "ocean"],
            ["ocean",   "ocean",   "ocean",   "ocean"],
        ],
    },
    "biome_bands_4x4": {
        "description": "Horizontal biome bands",
        "grid": [
            ["ice", "ice", "ice", "ice"],
            ["forest",       "forest",       "forest",       "forest"],
            ["plains",       "plains",       "plains",       "plains"],
            ["desert",       "desert",       "desert",       "desert"],
        ],
    },
    "jungle_ocean_3x3": {
        "description": "Horizontal biome bands",
        "grid": [
            ["ocean", "beach", "jungle"],
            ["ocean", "beach", "jungle"],
            ["ocean", "beach", "jungle"],
        ],
    },
    "jungle_ocean_4x4": {
        "description": "Horizontal biome bands",
        "grid": [
            ["ocean", "beach", "plains", "jungle"],
            ["ocean", "beach", "plains", "jungle"],
            ["ocean", "beach", "plains", "jungle"],
            ["ocean", "beach", "plains", "jungle"],
        ],
    },
    "village_coast_3x3": {
        "description": "Village settlement near a coast",
        "grid": [
            ["forest",  "forest",  "beach"],
            ["forest",  "village", "beach"],
            ["forest",  "beach",   "ocean"],
        ],
    },
    "river_kingdom_5x5": {
        "description": "Mountain-fed river valley opening into settled lowlands",
        "grid": [
            ["extreme_hills", "extreme_hills", "forest",        "forest",        "birch_forest"],
            ["extreme_hills", "forest",        "river",         "forest",        "birch_forest"],
            ["forest",        "plains",        "river",         "plains",        "birch_forest"],
            ["plains",        "plains",        "river",         "village",       "plains"],
            ["plains",        "beaches",       "river",         "beaches",       "ocean"],
        ],
    },
    "jungle_delta_5x5": {
        "description": "Dense jungle river system dissolving into swampy coast",
        "grid": [
            ["jungle",        "jungle",        "river",         "swampland",     "ocean"],
            ["jungle",        "jungle",        "river",         "swampland",     "beaches"],
            ["jungle",        "forest",        "river",         "swampland",     "beaches"],
            ["forest",        "plains",        "river",         "beaches",       "ocean"],
            ["plains",        "plains",        "river",         "beaches",       "ocean"],
        ],
    },
    "frozen_thaw_5x5": {
        "description": "A cold biome front melting into forested river country",
        "grid": [
            ["ice",           "ice",           "ice",           "taiga",         "taiga"],
            ["ice",           "ice",           "taiga",         "taiga",         "forest"],
            ["ice",           "taiga",         "river",         "forest",        "forest"],
            ["taiga",         "taiga",         "river",         "plains",        "plains"],
            ["taiga",         "forest",        "river",         "plains",        "village"],
        ],
    },
    "savanna_oasis_5x5": {
        "description": "Dry badlands and savanna split by a life-giving river",
        "grid": [
            ["extreme_hills", "desert",        "desert",        "savanna",       "savanna"],
            ["desert",        "desert",        "river",         "savanna",       "savanna"],
            ["desert",        "plains",        "river",         "plains",        "savanna"],
            ["desert",        "plains",        "river",         "village",       "plains"],
            ["desert",        "beaches",       "river",         "beaches",       "ocean"],
        ],
    },
    "crown_island_6x6": {
        "description": "A layered island with beaches, inner forest, and a central settlement",
        "grid": [
            ["ocean",         "ocean",         "beaches",       "beaches",       "ocean",         "ocean"],
            ["ocean",         "beaches",       "plains",        "plains",        "beaches",       "ocean"],
            ["beaches",       "plains",        "forest",        "forest",        "plains",        "beaches"],
            ["beaches",       "plains",        "forest",        "village",       "plains",        "beaches"],
            ["ocean",         "beaches",       "plains",        "plains",        "beaches",       "ocean"],
            ["ocean",         "ocean",         "beaches",       "beaches",       "ocean",         "ocean"],
        ],
    },
    "all_biomes_3x5": {
        "description": "3x5 showcase covering all 15 supported biome labels",
        "grid": [
            ["beaches",       "birch_forest",  "cave",          "desert",        "extreme_hills"],
            ["forest",        "ice",           "jungle",        "ocean",         "plains"],
            ["river",         "savanna",       "swampland",     "taiga",         "village"],
        ],
    },
    "shattered_coast_6x6": {
        "description": "A broken coastline of forests, beaches, and open sea with a river outlet",
        "grid": [
            ["forest",        "forest",        "forest",        "beaches",       "ocean",         "ocean"],
            ["forest",        "forest",        "plains",        "beaches",       "ocean",         "ocean"],
            ["forest",        "plains",        "river",         "beaches",       "beaches",       "ocean"],
            ["birch_forest",  "plains",        "river",         "plains",        "beaches",       "ocean"],
            ["birch_forest",  "forest",        "river",         "plains",        "beaches",       "ocean"],
            ["forest",        "forest",        "river",         "beaches",       "ocean",         "ocean"],
        ],
    },
    "highland_basin_6x6": {
        "description": "A mountain basin draining through forests into settled plains",
        "grid": [
            ["extreme_hills", "extreme_hills", "extreme_hills", "forest",        "forest",        "birch_forest"],
            ["extreme_hills", "forest",        "forest",        "forest",        "birch_forest",  "birch_forest"],
            ["extreme_hills", "forest",        "river",         "plains",        "plains",        "birch_forest"],
            ["forest",        "forest",        "river",         "plains",        "village",       "plains"],
            ["forest",        "plains",        "river",         "plains",        "plains",        "plains"],
            ["taiga",         "plains",        "river",         "beaches",       "beaches",       "ocean"],
        ],
    },
    "jungle_atoll_6x6": {
        "description": "A lush jungle island ringed by beaches and shallow ocean approaches",
        "grid": [
            ["ocean",         "ocean",         "beaches",       "beaches",       "ocean",         "ocean"],
            ["ocean",         "beaches",       "jungle",        "jungle",        "beaches",       "ocean"],
            ["beaches",       "jungle",        "jungle",        "swampland",     "beaches",       "ocean"],
            ["beaches",       "jungle",        "river",         "swampland",     "beaches",       "ocean"],
            ["ocean",         "beaches",       "forest",        "plains",        "beaches",       "ocean"],
            ["ocean",         "ocean",         "beaches",       "beaches",       "ocean",         "ocean"],
        ],
    },
    "northern_frontier_6x6": {
        "description": "A cold northern frontier thawing into river plains and a lone village",
        "grid": [
            ["ice",           "ice",           "taiga",         "taiga",         "forest",        "forest"],
            ["ice",           "ice",           "taiga",         "forest",        "forest",        "birch_forest"],
            ["ice",           "taiga",         "river",         "forest",        "plains",        "birch_forest"],
            ["taiga",         "taiga",         "river",         "plains",        "plains",        "plains"],
            ["taiga",         "forest",        "river",         "plains",        "village",       "plains"],
            ["forest",        "forest",        "river",         "beaches",       "ocean",         "ocean"],
        ],
    },
    "sunset_estuary_6x6": {
        "description": "A warm estuary where desert and savanna meet the sea through a broad river mouth",
        "grid": [
            ["desert",        "desert",        "savanna",       "savanna",       "beaches",       "ocean"],
            ["desert",        "desert",        "savanna",       "plains",        "beaches",       "ocean"],
            ["desert",        "plains",        "river",         "plains",        "beaches",       "ocean"],
            ["savanna",       "plains",        "river",         "plains",        "beaches",       "ocean"],
            ["savanna",       "plains",        "river",         "village",       "beaches",       "ocean"],
            ["plains",        "plains",        "river",         "beaches",       "ocean",         "ocean"],
        ],
    },
    "continental_spine_8x8": {
        "description": "A full regional map with a mountain spine, twin watersheds, and coasts on both sides",
        "grid": [
            ["ocean",         "beaches",       "forest",        "extreme_hills", "extreme_hills", "forest",        "beaches",       "ocean"],
            ["ocean",         "beaches",       "forest",        "extreme_hills", "extreme_hills", "forest",        "beaches",       "ocean"],
            ["ocean",         "beaches",       "plains",        "forest",        "forest",        "plains",        "beaches",       "ocean"],
            ["ocean",         "beaches",       "river",         "plains",        "plains",        "river",         "beaches",       "ocean"],
            ["ocean",         "beaches",       "river",         "plains",        "village",       "river",         "beaches",       "ocean"],
            ["ocean",         "beaches",       "forest",        "plains",        "plains",        "forest",        "beaches",       "ocean"],
            ["ocean",         "beaches",       "birch_forest",  "forest",        "forest",        "taiga",         "beaches",       "ocean"],
            ["ocean",         "ocean",         "beaches",       "beaches",       "beaches",       "beaches",       "ocean",         "ocean"],
        ],
    },
    "ancient_empire_3x3": {
        "description": "A simple 3x3 showcase of the ancient_empire biome label",
        "grid": [
            ["ancient_empire", "ancient_empire", "ancient_empire"],
            ["ancient_empire", "ancient_empire", "ancient_empire"],
            ["ancient_empire", "ancient_empire", "ancient_empire"],
        ],
    },
    "kingdom_of_sarano_3x3": {
        "description": "A simple 3x3 showcase of the kingdom_of_sarano biome label",
        "grid": [
            ["kingdom_of_sarano", "kingdom_of_sarano", "kingdom_of_sarano"],
            ["kingdom_of_sarano", "kingdom_of_sarano", "kingdom_of_sarano"],
            ["kingdom_of_sarano", "kingdom_of_sarano", "kingdom_of_sarano"],
        ],
    },
    
    "marethea_3x3": {
        "description": "A simple 3x3 showcase of the marethea biome label",
        "grid": [
            ["marethea", "marethea", "marethea"],
            ["marethea", "marethea", "marethea"],
            ["marethea", "marethea", "marethea"],
        ],
    },
    "osirion_3x3": {
        "description": "A simple 3x3 showcase of the osirion biome label",
        "grid": [
            ["osirion", "osirion", "osirion"],
            ["osirion", "osirion", "osirion"],
            ["osirion", "osirion", "osirion"],
        ],
    },
    "sora_kingdom_3x3": {
        "description": "A simple 3x3 showcase of the sora_kingdom biome label",
        "grid": [
            ["sora_kingdom", "sora_kingdom", "sora_kingdom"],
            ["sora_kingdom", "sora_kingdom", "sora_kingdom"],
            ["sora_kingdom", "sora_kingdom", "sora_kingdom"],
        ],
    },
    "the_5_bridges_3x3": {
        "description": "A simple 3x3 showcase of the the_5_bridges biome label",
        "grid": [
            ["the_5_bridges", "the_5_bridges", "the_5_bridges"],
            ["the_5_bridges", "the_5_bridges", "the_5_bridges"],
            ["the_5_bridges", "the_5_bridges", "the_5_bridges"],
        ],
    },
    "the_5_bridges_5x5": {
        "description": "A simple 5x5 showcase of the the_5_bridges biome label",
        "grid": [
            ["the_5_bridges", "the_5_bridges", "the_5_bridges", "the_5_bridges", "the_5_bridges"],
            ["the_5_bridges", "the_5_bridges", "the_5_bridges", "the_5_bridges", "the_5_bridges"],
            ["the_5_bridges", "the_5_bridges", "the_5_bridges", "the_5_bridges", "the_5_bridges"],
            ["the_5_bridges", "the_5_bridges", "the_5_bridges", "the_5_bridges", "the_5_bridges"],
            ["the_5_bridges", "the_5_bridges", "the_5_bridges", "the_5_bridges", "the_5_bridges"],
        ],
    },
    "ancient_empire_5x5": {
        "description": "A simple 5x5 showcase of the ancient_empire biome label",
        "grid": [
            ["ancient_empire", "ancient_empire", "ancient_empire", "ancient_empire", "ancient_empire"],
            ["ancient_empire", "ancient_empire", "ancient_empire", "ancient_empire", "ancient_empire"],
            ["ancient_empire", "ancient_empire", "ancient_empire", "ancient_empire", "ancient_empire"],
            ["ancient_empire", "ancient_empire", "ancient_empire", "ancient_empire", "ancient_empire"],
            ["ancient_empire", "ancient_empire", "ancient_empire", "ancient_empire", "ancient_empire"],
        ],
    },
    "sora_kingdom_5x5": {
        "description": "A simple 5x5 showcase of the sora_kingdom biome label",
        "grid": [
            ["sora_kingdom", "sora_kingdom", "sora_kingdom", "sora_kingdom", "sora_kingdom"],
            ["sora_kingdom", "sora_kingdom", "sora_kingdom", "sora_kingdom", "sora_kingdom"],
            ["sora_kingdom", "sora_kingdom", "sora_kingdom", "sora_kingdom", "sora_kingdom"],
            ["sora_kingdom", "sora_kingdom", "sora_kingdom", "sora_kingdom", "sora_kingdom"],
            ["sora_kingdom", "sora_kingdom", "sora_kingdom", "sora_kingdom", "sora_kingdom"],
        ],
    },
    "osirion_5x5": {
        "description": "A simple 5x5 showcase of the osirion biome label",
        "grid": [
            ["osirion", "osirion", "osirion", "osirion", "osirion"],
            ["osirion", "osirion", "osirion", "osirion", "osirion"],
            ["osirion", "osirion", "osirion", "osirion", "osirion"],
            ["osirion", "osirion", "osirion", "osirion", "osirion"],
            ["osirion", "osirion", "osirion", "osirion", "osirion"],
        ],
    },
    "marethea_5x5": {
        "description": "A simple 5x5 showcase of the marethea biome label",
        "grid": [
            ["marethea", "marethea", "marethea", "marethea", "marethea"],
            ["marethea", "marethea", "marethea", "marethea", "marethea"],
            ["marethea", "marethea", "marethea", "marethea", "marethea"],
            ["marethea", "marethea", "marethea", "marethea", "marethea"],
            ["marethea", "marethea", "marethea", "marethea", "marethea"],
        ],
    },
    "kingdom_of_sarano_5x5": {
        "description": "A simple 5x5 showcase of the kingdom_of_sarano biome label",
        "grid": [
            ["kingdom_of_sarano", "kingdom_of_sarano", "kingdom_of_sarano", "kingdom_of_sarano", "kingdom_of_sarano"],
            ["kingdom_of_sarano", "kingdom_of_sarano", "kingdom_of_sarano", "kingdom_of_sarano", "kingdom_of_sarano"],
            ["kingdom_of_sarano", "kingdom_of_sarano", "kingdom_of_sarano", "kingdom_of_sarano", "kingdom_of_sarano"],
            ["kingdom_of_sarano", "kingdom_of_sarano", "kingdom_of_sarano", "kingdom_of_sarano", "kingdom_of_sarano"],
            ["kingdom_of_sarano", "kingdom_of_sarano", "kingdom_of_sarano", "kingdom_of_sarano", "kingdom_of_sarano"],
        ],
    },
    "emerald_reaches_6x6": {
        "description": "A broad inland kingdom of forests and plains split by a central river valley",
        "grid": [
            ["forest",        "forest",        "birch_forest",  "plains",        "plains",        "savanna"],
            ["forest",        "forest",        "river",         "plains",        "plains",        "savanna"],
            ["birch_forest",  "forest",        "river",         "plains",        "village",       "savanna"],
            ["birch_forest",  "plains",        "river",         "plains",        "plains",        "savanna"],
            ["forest",        "plains",        "river",         "forest",        "forest",        "taiga"],
            ["forest",        "forest",        "river",         "forest",        "taiga",         "taiga"],
        ],
    },
    "red_sun_plateau_6x6": {
        "description": "A dry upland of desert and savanna with a single life-giving river and village corridor",
        "grid": [
            ["extreme_hills", "desert",        "desert",        "savanna",       "savanna",       "plains"],
            ["desert",        "desert",        "river",         "savanna",       "savanna",       "plains"],
            ["desert",        "plains",        "river",         "plains",        "village",       "plains"],
            ["desert",        "plains",        "river",         "plains",        "savanna",       "savanna"],
            ["extreme_hills", "plains",        "river",         "forest",        "forest",        "savanna"],
            ["extreme_hills", "forest",        "river",         "forest",        "taiga",         "taiga"],
        ],
    },
    "crownlands_8x8": {
        "description": "A large inland realm with a mountain north, central plains heartland, and branching river spine",
        "grid": [
            ["extreme_hills", "extreme_hills", "forest",        "forest",        "birch_forest",  "plains",        "plains",        "savanna"],
            ["extreme_hills", "forest",        "forest",        "river",         "birch_forest",  "plains",        "plains",        "savanna"],
            ["forest",        "forest",        "plains",        "river",         "plains",        "plains",        "savanna",       "savanna"],
            ["forest",        "plains",        "plains",        "river",         "plains",        "village",       "savanna",       "savanna"],
            ["birch_forest",  "plains",        "forest",        "river",         "forest",        "plains",        "plains",        "taiga"],
            ["birch_forest",  "forest",        "forest",        "river",         "forest",        "plains",        "taiga",         "taiga"],
            ["forest",        "forest",        "taiga",         "river",         "taiga",         "taiga",         "taiga",         "ice"],
            ["forest",        "taiga",         "taiga",         "river",         "taiga",         "ice",           "ice",           "ice"],
        ],
    },
    "mountain_test_layered": {
        "description": "Minimal layered mountain test with a summit chunk above a 3x3 lower ring",
        "display_grid": [
            ["plains", "plains", "plains"],
            ["plains", "extreme_hills", "plains"],
            ["plains", "plains", "plains"],
        ],
        "levels": [
            {
                "name": "summit",
                "grid": [
                    ["desert"],
                ],
            },
            {
                "name": "base",
                "w_offset": 0,
                "grid": [
                    ["plains", "desert", "desert"],
                    ["desert", None,     "desert"],
                    ["plains", "plains", "plains"],
                ],
            },
        ],
        "generation_order": [
            {"level": "base", "row": 0, "col": 0},
            {"level": "base", "row": 0, "col": 2},
            {"level": "base", "row": 2, "col": 0},
            {"level": "base", "row": 2, "col": 2},
            {"level": "summit", "row": 0, "col": 0},
            {"level": "base", "row": 0, "col": 1},
            {"level": "base", "row": 1, "col": 0},
            {"level": "base", "row": 1, "col": 2},
            {"level": "base", "row": 2, "col": 1},
        ],
    },
}

UNCONDITIONAL_BIOME = "__unconditional__"
DEFAULT_UNCONDITIONAL_GRID_SIZE = 5


@dataclass(frozen=True)
class ChunkPlacement:
    placement_id: str
    biome: str
    level_index: int
    level_name: str
    row: int
    col: int
    h0: int
    w0: int
    d0: int


# =============================================================================
# Biome Resolution
# =============================================================================

def resolve_biome_index(biome_name: str, converter: BlockBiomeConverter) -> int:
    """Resolve a biome name to its model index, with case-insensitive fuzzy matching."""
    if not converter.biome_to_index:
        raise ValueError("Converter has no biome_to_index mapping")

    # Exact match
    if biome_name in converter.biome_to_index:
        return converter.biome_to_index[biome_name]

    # Case-insensitive
    lower = biome_name.lower()
    for name, idx in converter.biome_to_index.items():
        if name.lower() == lower:
            return idx

    # Substring match
    matches = [
        (n, i) for n, i in converter.biome_to_index.items()
        if lower in n.lower() or n.lower() in lower
    ]
    if len(matches) == 1:
        return matches[0][1]
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous biome '{biome_name}', matches: {[m[0] for m in matches]}"
        )

    available = list(converter.biome_to_index.keys())
    raise ValueError(f"Unknown biome '{biome_name}'. Available: {available}")


# =============================================================================
# Layout Normalization
# =============================================================================

def _validate_rectangular_grid(grid: List[List[Any]], *, allow_null: bool = False):
    """Validate a 2D grid and optionally allow empty cells."""
    if not isinstance(grid, list) or not grid or not isinstance(grid[0], list):
        raise ValueError("Grid must be a non-empty 2D list")

    num_cols = len(grid[0])
    if num_cols == 0:
        raise ValueError("Grid rows must not be empty")

    for i, row in enumerate(grid):
        if len(row) != num_cols:
            raise ValueError(f"Row {i} has {len(row)} columns, expected {num_cols}")
        for j, cell in enumerate(row):
            if cell is None and allow_null:
                continue
            if not isinstance(cell, str) or not cell:
                raise ValueError(f"Invalid cell at row {i}, col {j}: {cell!r}")


def _sanitize_level_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(name))
    return safe or "level"


def _build_unconditional_layout_data(grid_size: int = DEFAULT_UNCONDITIONAL_GRID_SIZE) -> Dict[str, Any]:
    return {
        "description": f"{grid_size}x{grid_size} unconditional grid",
        "grid": [[UNCONDITIONAL_BIOME for _ in range(grid_size)] for _ in range(grid_size)],
        "unconditional": True,
    }


def _load_context_grid_layout(
    context_grid_path: str,
    *,
    converter: BlockBiomeConverter,
    cell_size: int,
    overlap: int,
    vertical_overlap: int,
) -> Tuple[str, Dict[str, Any]]:
    """
    Load a 2D JSON grid of seed-context .pt files.

    Each non-null entry points to a notebook-authored seed context payload from
    context_test.ipynb / inpaint.py. Generation stays unconditional for biome,
    but each cell receives these authored fixed voxels in addition to normal
    neighboring chunk overlap context.
    """
    context_grid_file = Path(context_grid_path)
    with open(context_grid_file) as f:
        raw_payload = json.load(f)

    if isinstance(raw_payload, list):
        raw_context_grid = raw_payload
        raw_fill_grid = None
    elif isinstance(raw_payload, dict):
        raw_context_grid = raw_payload.get("context_grid", None)
        raw_fill_grid = raw_payload.get("fill_grid", raw_payload.get("grid", None))
        if raw_context_grid is None:
            raise ValueError(
                f"Context grid file must contain 'context_grid' when using dict format: {context_grid_file}"
            )
    else:
        raise ValueError(
            f"Context grid file must be either a 2D list or a dict payload: {context_grid_file}"
        )

    _validate_rectangular_grid(raw_context_grid, allow_null=True)
    num_rows = len(raw_context_grid)
    num_cols = len(raw_context_grid[0])

    if raw_fill_grid is None:
        raw_fill_grid = [[UNCONDITIONAL_BIOME for _ in range(num_cols)] for _ in range(num_rows)]
    else:
        _validate_rectangular_grid(raw_fill_grid, allow_null=True)
        if len(raw_fill_grid) != num_rows or len(raw_fill_grid[0]) != num_cols:
            raise ValueError(
                f"fill_grid must match context_grid shape in {context_grid_file}: "
                f"context_grid={num_rows}x{num_cols}, "
                f"fill_grid={len(raw_fill_grid)}x{len(raw_fill_grid[0])}"
            )

    unresolved_grid = []
    resolved_grid = []
    display_grid = []
    context_payloads_by_coord: Dict[Tuple[int, int], Dict[str, Any]] = {}
    generation_order = []
    fill_grid = []
    for row, row_data in enumerate(raw_context_grid):
        unresolved_row = []
        resolved_row = []
        display_row = []
        fill_row = []
        for col, entry in enumerate(row_data):
            fill_entry = raw_fill_grid[row][col]
            fill_biome = UNCONDITIONAL_BIOME if fill_entry in (None, "", "__unconditional__", "unconditional") else str(fill_entry)
            fill_row.append(fill_biome)
            if entry is None:
                unresolved_row.append(None)
                resolved_row.append(None)
                display_row.append("uncond" if fill_biome == UNCONDITIONAL_BIOME else fill_biome)
                continue

            entry_path = Path(str(entry))
            if not entry_path.is_absolute():
                entry_path = (context_grid_file.parent / entry_path).resolve()
            payload = _load_seed_context_payload(entry_path, converter)
            voxels = payload["source_indices"]
            context_mask = payload["context_mask"]
            if tuple(voxels.shape) != (cell_size, cell_size, cell_size):
                raise ValueError(
                    f"Context file {entry_path} has shape {tuple(voxels.shape)}, "
                    f"expected {(cell_size, cell_size, cell_size)}"
                )
            if tuple(context_mask.shape) != (cell_size, cell_size, cell_size):
                raise ValueError(
                    f"Context mask in {entry_path} has shape {tuple(context_mask.shape)}, "
                    f"expected {(cell_size, cell_size, cell_size)}"
                )
            context_payloads_by_coord[(row, col)] = payload
            unresolved_row.append(str(entry))
            resolved_row.append(str(entry_path))
            display_row.append(entry_path.stem)
            generation_order.append({"row": row, "col": col})
        unresolved_grid.append(unresolved_row)
        resolved_grid.append(resolved_row)
        display_grid.append(display_row)
        fill_grid.append(fill_row)

    layout_name = f"{context_grid_file.stem}_seed_context"
    layout_data = {
        "description": "Unconditional semantic super-sampling with per-cell seed context files",
        "grid": fill_grid,
        "display_grid": display_grid,
        "context_grid": unresolved_grid,
        "resolved_context_grid": resolved_grid,
        "generation_order": generation_order if generation_order else None,
        "unconditional": all(
            biome == UNCONDITIONAL_BIOME
            for row in fill_grid for biome in row
        ),
        "seed_context_layout": True,
    }
    layout_spec = _normalize_layout_data(
        layout_data=layout_data,
        layout_name=layout_name,
        cell_size=cell_size,
        overlap=overlap,
        vertical_overlap=vertical_overlap,
    )
    layout_spec["context_payloads"] = {
        placement.placement_id: context_payloads_by_coord[(placement.row, placement.col)]
        for placement in layout_spec["placements"]
        if (placement.row, placement.col) in context_payloads_by_coord
    }
    return layout_name, layout_spec


def _unconditional_class_label(
    model,
    converter: BlockBiomeConverter,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    Return a dummy class label tensor for CFG-null unconditional sampling.

    For our class-conditional DiT, unconditional generation is represented by
    the model's learned null conditioning branch, not by omitting the class
    tensor entirely. We therefore pass any valid class index and force
    `cond_scale=0`, which makes `forward_with_cond_scale()` return the null
    logits.
    """
    if not getattr(model, "class_conditional", False):
        return None
    if not hasattr(model, "forward_with_cond_scale"):
        raise ValueError(
            "This class-conditional model does not expose forward_with_cond_scale(), "
            "so unconditional sampling via the learned null-conditioning branch "
            "is not supported by semantic_super_sample.py."
        )
    if converter.biome_to_index:
        dummy_index = min(converter.biome_to_index.values())
    else:
        num_classes = getattr(model, "num_classes", None)
        if num_classes in (None, 0):
            raise ValueError(
                "Class-conditional model is missing num_classes / biome mappings; "
                "cannot construct a dummy class label for unconditional sampling."
            )
        dummy_index = 0
    return torch.tensor([int(dummy_index)], dtype=torch.long, device=device)


def _resolve_generation_order(
    generation_order: Any,
    placements: List[ChunkPlacement],
    levels: List[Dict[str, Any]],
) -> Optional[List[ChunkPlacement]]:
    """Resolve optional user-authored generation order selectors."""
    if generation_order is None:
        return None
    if not isinstance(generation_order, list) or not generation_order:
        raise ValueError("generation_order must be a non-empty list when provided")

    placements_by_id = {placement.placement_id: placement for placement in placements}
    level_by_name = {level["name"]: level for level in levels}
    duplicate_level_names = {
        level_name
        for level_name in level_by_name
        if sum(level["name"] == level_name for level in levels) > 1
    }

    resolved = []
    seen_ids = set()

    for idx, entry in enumerate(generation_order):
        placement = None
        if isinstance(entry, str):
            placement = placements_by_id.get(entry)
            if placement is None:
                raise ValueError(
                    f"generation_order[{idx}] references unknown placement_id '{entry}'"
                )
        elif isinstance(entry, dict):
            placement_id = entry.get("placement_id")
            if placement_id is not None:
                placement = placements_by_id.get(str(placement_id))
                if placement is None:
                    raise ValueError(
                        f"generation_order[{idx}] references unknown placement_id '{placement_id}'"
                    )
            else:
                level_name = _sanitize_level_name(entry.get("level", "base"))
                if level_name in duplicate_level_names:
                    raise ValueError(
                        f"generation_order[{idx}] uses ambiguous level name '{level_name}'; "
                        "use placement_id instead"
                    )
                if level_name not in level_by_name:
                    raise ValueError(
                        f"generation_order[{idx}] references unknown level '{level_name}'"
                    )
                if "row" not in entry or "col" not in entry:
                    raise ValueError(
                        f"generation_order[{idx}] must define either placement_id or row/col selectors"
                    )

                authored_row = int(entry["row"])
                col = int(entry["col"])
                level_grid = level_by_name[level_name]["grid"]
                if not (0 <= authored_row < len(level_grid)):
                    raise ValueError(
                        f"generation_order[{idx}] row {authored_row} is out of bounds "
                        f"for level '{level_name}'"
                    )
                if not (0 <= col < len(level_grid[0])):
                    raise ValueError(
                        f"generation_order[{idx}] col {col} is out of bounds "
                        f"for level '{level_name}'"
                    )
                if level_grid[authored_row][col] is None:
                    raise ValueError(
                        f"generation_order[{idx}] points at an empty cell in level '{level_name}'"
                    )

                internal_row = len(level_grid) - 1 - authored_row
                matches = [
                    candidate for candidate in placements
                    if candidate.level_name == level_name
                    and candidate.row == internal_row
                    and candidate.col == col
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"generation_order[{idx}] could not be resolved uniquely for "
                        f"level='{level_name}', row={authored_row}, col={col}"
                    )
                placement = matches[0]
        else:
            raise ValueError(
                f"generation_order[{idx}] must be a placement_id string or selector dict"
            )

        if placement.placement_id in seen_ids:
            raise ValueError(
                f"generation_order contains duplicate placement '{placement.placement_id}'"
            )
        seen_ids.add(placement.placement_id)
        resolved.append(placement)

    return resolved


def _normalize_layout_data(
    layout_data: Any,
    layout_name: str,
    cell_size: int,
    overlap: int,
    vertical_overlap: int,
) -> Dict[str, Any]:
    """Normalize 2D and layered layouts into explicit chunk placements."""
    stride = cell_size - overlap
    vertical_stride = cell_size - vertical_overlap

    if isinstance(layout_data, list):
        grid = layout_data
        _validate_rectangular_grid(grid)
        grid_bu = list(reversed(grid))
        placements = []
        for row in range(len(grid_bu)):
            for col in range(len(grid_bu[0])):
                biome = grid_bu[row][col]
                placements.append(
                    ChunkPlacement(
                        placement_id=f"r{row}_c{col}",
                        biome=biome,
                        level_index=0,
                        level_name="base",
                        row=row,
                        col=col,
                        h0=col * stride,
                        w0=0,
                        d0=row * stride,
                    )
                )
        return {
            "kind": "2d",
            "name": layout_name,
            "raw": {"grid": grid},
            "display_grid": grid,
            "placements": placements,
            "levels": [{"name": "base", "w_offset": 0, "grid": grid}],
            "generation_order": None,
        }

    if not isinstance(layout_data, dict):
        raise ValueError("Layout data must be a 2D grid list or a layout dict")

    if "levels" not in layout_data:
        if "grid" not in layout_data:
            raise ValueError("Layout dict must contain either 'grid' or 'levels'")
        normalized = _normalize_layout_data(
            layout_data["grid"],
            layout_name,
            cell_size,
            overlap,
            vertical_overlap,
        )
        normalized["raw"] = layout_data
        if "display_grid" in layout_data:
            _validate_rectangular_grid(layout_data["display_grid"], allow_null=True)
            normalized["display_grid"] = [
                [cell if isinstance(cell, str) and cell else " " for cell in row]
                for row in layout_data["display_grid"]
            ]
        normalized["generation_order"] = _resolve_generation_order(
            layout_data.get("generation_order"),
            normalized["placements"],
            normalized["levels"],
        )
        return normalized

    levels = layout_data["levels"]
    if not isinstance(levels, list) or not levels:
        raise ValueError("Layered layout must contain a non-empty 'levels' list")
    if vertical_stride <= 0:
        raise ValueError(
            f"vertical_overlap must be in [0, {cell_size - 1}], got {vertical_overlap}"
        )

    placements = []
    base_display_grid = layout_data.get("display_grid")
    if base_display_grid is not None:
        _validate_rectangular_grid(base_display_grid, allow_null=True)
        display_grid = [
            [cell if isinstance(cell, str) and cell else " " for cell in row]
            for row in base_display_grid
        ]
    else:
        display_grid = None

    if display_grid is not None:
        anchor_rows = len(display_grid)
        anchor_cols = len(display_grid[0])
    else:
        anchor_rows = max(len(level["grid"]) for level in levels)
        anchor_cols = max(len(level["grid"][0]) for level in levels)

    anchor_world_h = (anchor_cols - 1) * stride + cell_size
    anchor_world_d = (anchor_rows - 1) * stride + cell_size

    normalized_levels = []
    for level_index, level in enumerate(levels):
        if not isinstance(level, dict) or "grid" not in level:
            raise ValueError(f"Layer {level_index} must be a dict containing 'grid'")

        level_name = _sanitize_level_name(level.get("name", f"level_{level_index}"))
        level_grid = level["grid"]
        _validate_rectangular_grid(level_grid, allow_null=True)

        if display_grid is None:
            display_grid = [
                [cell if isinstance(cell, str) and cell else " " for cell in row]
                for row in level_grid
            ]

        h_stride = int(level.get("h_stride", stride))
        d_stride = int(level.get("d_stride", stride))
        level_rows = len(level_grid)
        level_cols = len(level_grid[0])
        level_world_h = (level_cols - 1) * h_stride + cell_size
        level_world_d = (level_rows - 1) * d_stride + cell_size
        default_h_offset = max(0, (anchor_world_h - level_world_h) // 2)
        default_d_offset = max(0, (anchor_world_d - level_world_d) // 2)
        default_w_offset = max(0, (len(levels) - 1 - level_index) * vertical_stride)

        w_offset = int(level.get("w_offset", default_w_offset))
        h_offset = int(level.get("h_offset", default_h_offset))
        d_offset = int(level.get("d_offset", default_d_offset))

        if min(w_offset, h_offset, d_offset) < 0:
            raise ValueError("Layer offsets must be non-negative")
        if h_stride <= 0 or d_stride <= 0:
            raise ValueError("Layer strides must be positive")

        normalized_levels.append(
            {
                "name": level_name,
                "w_offset": w_offset,
                "h_offset": h_offset,
                "d_offset": d_offset,
                "h_stride": h_stride,
                "d_stride": d_stride,
                "grid": level_grid,
            }
        )

        grid_bu = list(reversed(level_grid))
        for row in range(len(grid_bu)):
            for col in range(len(grid_bu[0])):
                biome = grid_bu[row][col]
                if biome is None:
                    continue
                placements.append(
                    ChunkPlacement(
                        placement_id=f"l{level_index}_{level_name}_r{row}_c{col}",
                        biome=biome,
                        level_index=level_index,
                        level_name=level_name,
                        row=row,
                        col=col,
                        h0=h_offset + col * h_stride,
                        w0=w_offset,
                        d0=d_offset + row * d_stride,
                    )
                )

    if not placements:
        raise ValueError("Layered layout contains no actual chunk placements")

    return {
        "kind": "layered",
        "name": layout_name,
        "raw": layout_data,
        "display_grid": display_grid,
        "placements": placements,
        "levels": normalized_levels,
        "generation_order": _resolve_generation_order(
            layout_data.get("generation_order"),
            placements,
            normalized_levels,
        ),
    }


# =============================================================================
# Display Utilities
# =============================================================================

def print_grid(grid: List[List[str]], title: str = "World Layout"):
    """Pretty-print a grid layout to the console."""
    max_len = max(len(b) for row in grid for b in row)
    border = "-" * ((max_len + 3) * len(grid[0]) + 1)
    print(f"\n{title}:")
    print(f"  {border}")
    for row in grid:
        cells = " | ".join(f"{b:>{max_len}}" for b in row)
        print(f"  | {cells} |")
    print(f"  {border}")


BIOME_COLORS = {
    "forest": (34, 139, 34), "ocean": (0, 105, 148),
    "beach": (238, 214, 175), "desert": (210, 180, 140),
    "plains": (124, 252, 0), "village": (139, 90, 43),
    "snowy_tundra": (220, 220, 240), "swamp": (47, 79, 47),
    "taiga": (0, 100, 0), "savanna": (189, 183, 107),
    "jungle": (0, 128, 0), "mountains": (128, 128, 128),
    "river": (64, 164, 223), "badlands": (211, 63, 24),
    "ice": (220, 240, 255), "extreme_hills": (128, 128, 128),
}


def _biome_color(biome: str) -> tuple:
    """Get a display color for a biome name."""
    if biome in BIOME_COLORS:
        return BIOME_COLORS[biome]
    for key, val in BIOME_COLORS.items():
        if key in biome.lower() or biome.lower() in key:
            return val
    return (128, 128, 128)


def _load_font(size: int, bold: bool = False):
    """Best-effort font loader with a Pillow fallback."""
    font_names = ["arialbd.ttf", "Arial Bold.ttf"] if bold else ["arial.ttf", "Arial.ttf"]
    for name in font_names:
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _format_biome_label(biome: str) -> str:
    """Format biome labels for compact rendering."""
    parts = biome.replace("_", " ").split()
    if len(parts) <= 1:
        return biome.replace("_", " ")
    midpoint = (len(parts) + 1) // 2
    return " ".join(parts[:midpoint]) + "\n" + " ".join(parts[midpoint:])


def render_grid_overview(grid: List[List[str]], output_path: str, cell_px: int = 120):
    """Create an isometric biome map aligned to the world render orientation."""
    num_rows, num_cols = len(grid), len(grid[0])
    tile_w = max(96, int(cell_px * 1.35))
    tile_h = max(48, int(tile_w * 0.48))
    half_w = tile_w // 2
    half_h = tile_h // 2
    margin = 36

    positions = []
    min_x = float("inf")
    max_x = float("-inf")
    min_y = float("inf")
    max_y = float("-inf")

    for r in range(num_rows):
        north_idx = num_rows - 1 - r
        for c in range(num_cols):
            # This projection mirrors the orientation of the isometric world
            # render: the top row of the layout appears toward the near-right,
            # and the bottom row appears toward the far-left.
            cx = (north_idx + c) * half_w
            cy = (north_idx - c) * half_h
            positions.append((r, c, cx, cy))
            min_x = min(min_x, cx - half_w)
            max_x = max(max_x, cx + half_w)
            min_y = min(min_y, cy - half_h)
            max_y = max(max_y, cy + half_h)

    img_w = int(max_x - min_x + 2 * margin)
    img_h = int(max_y - min_y + 2 * margin)
    img = Image.new("RGBA", (img_w, img_h), (28, 28, 32, 255))
    draw = ImageDraw.Draw(img)
    label_font = _load_font(max(13, tile_h // 4), bold=True)
    shifted_positions = []
    for r, c, cx, cy in positions:
        sx = int(cx - min_x + margin)
        sy = int(cy - min_y + margin)
        shifted_positions.append((r, c, sx, sy))

    # Draw back-to-front for cleaner overlap layering.
    shifted_positions.sort(key=lambda item: (item[3], item[2]))

    for r, c, cx, cy in shifted_positions:
        biome = grid[r][c]
        color = _biome_color(biome)
        diamond = [
            (cx, cy - half_h),
            (cx + half_w, cy),
            (cx, cy + half_h),
            (cx - half_w, cy),
        ]
        shadow = [(x, y + 5) for x, y in diamond]
        draw.polygon(shadow, fill=(0, 0, 0, 70))
        draw.polygon(diamond, fill=color + (255,), outline=(18, 18, 18, 255))

        # Add a subtle inset line so neighboring diamonds remain readable.
        inset = max(6, tile_h // 8)
        inner = [
            (cx, cy - half_h + inset),
            (cx + half_w - inset, cy),
            (cx, cy + half_h - inset),
            (cx - half_w + inset, cy),
        ]
        draw.polygon(inner, outline=(255, 255, 255, 60))

        label = _format_biome_label(biome)
        brightness = sum(color) / 3
        text_col = (12, 12, 12, 255) if brightness > 150 else (248, 248, 248, 255)
        bbox = draw.multiline_textbbox((0, 0), label, font=label_font, spacing=2, align="center")
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.multiline_text(
            (cx - tw / 2, cy - th / 2),
            label,
            fill=text_col,
            font=label_font,
            spacing=2,
            align="center",
            stroke_width=2,
            stroke_fill=(255, 255, 255, 210) if brightness > 150 else (20, 20, 20, 220),
        )

    img.convert("RGB").save(output_path)
    print(f"Grid overview saved to {output_path}")


def render_side_by_side_reference(
    left_path: str,
    right_path: str,
    output_path: str,
    left_title: str = "Biome Layout",
    right_title: str = "Generated World",
):
    """Compose two existing images into a labeled side-by-side comparison."""
    if not os.path.exists(left_path) or not os.path.exists(right_path):
        return

    left = Image.open(left_path).convert("RGB")
    right = Image.open(right_path).convert("RGB")
    target_h = max(left.height, right.height)

    if left.height != target_h:
        left = left.resize((int(left.width * target_h / left.height), target_h), LANCZOS_RESAMPLE)
    if right.height != target_h:
        right = right.resize((int(right.width * target_h / right.height), target_h), LANCZOS_RESAMPLE)

    pad = 24
    gap = 28
    title_h = 42
    canvas = Image.new(
        "RGB",
        (left.width + right.width + gap + 2 * pad, target_h + title_h + 2 * pad),
        (24, 24, 28),
    )
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(22, bold=True)

    left_x = pad
    right_x = pad + left.width + gap
    img_y = pad + title_h
    canvas.paste(left, (left_x, img_y))
    canvas.paste(right, (right_x, img_y))

    draw.text((left_x, pad + 6), left_title, fill=(235, 235, 235), font=title_font)
    draw.text((right_x, pad + 6), right_title, fill=(235, 235, 235), font=title_font)

    canvas.save(output_path)
    print(f"Side-by-side reference saved to {output_path}")


def render_cell_grid(
    cells: Dict[Tuple[int, int], torch.Tensor],
    grid: List[List[str]],
    converter: BlockBiomeConverter,
    visualizer: MinecraftVisualizerPyVista,
    textured: bool,
    output_path: str,
    image_size: int = 256,
):
    """Render all cells arranged in a grid image matching the layout."""
    grid_bu = list(reversed(grid))
    num_rows = len(grid_bu)
    num_cols = len(grid_bu[0])

    cell_images = {}
    for (row, col), cell_data in cells.items():
        blocks = converter.convert_to_original_blocks(cell_data.unsqueeze(0))
        try:
            if textured:
                plotter = visualizer.visualize_chunk_textured(
                    blocks[0], interactive=False, show_axis=False
                )
            else:
                plotter = visualizer.visualize_chunk(
                    blocks[0], interactive=False, show_axis=False
                )
            plotter.screenshot(
                filename="_tmp_cell.png",
                window_size=(image_size, image_size),
                transparent_background=False,
            )
            plotter.close()
            cell_images[(row, col)] = Image.open("_tmp_cell.png").copy()
        except Exception as e:
            print(f"  Warning: cell ({row},{col}) render failed: {e}")

    if os.path.exists("_tmp_cell.png"):
        os.remove("_tmp_cell.png")

    if not cell_images:
        return

    pad = 4
    label_h = 24
    img_w = num_cols * image_size + (num_cols + 1) * pad
    img_h = num_rows * (image_size + label_h) + (num_rows + 1) * pad

    canvas = Image.new("RGB", (img_w, img_h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Display top-to-bottom (reverse row order since grid_bu is bottom-up)
    for row in range(num_rows):
        for col in range(num_cols):
            display_row = num_rows - 1 - row
            x0 = pad + col * (image_size + pad)
            y0 = pad + display_row * (image_size + label_h + pad)

            if (row, col) in cell_images:
                canvas.paste(cell_images[(row, col)], (x0, y0))

            biome = grid_bu[row][col]
            draw.text((x0, y0 + image_size + 2), biome, fill=(200, 200, 200), font=font)

    canvas.save(output_path)
    print(f"Cell grid saved to {output_path}")


# =============================================================================
# Core Generation
# =============================================================================

def _raster_scan_order(num_rows: int, num_cols: int) -> List[Tuple[int, int]]:
    """Bottom-to-top, left-to-right raster scan."""
    order = []
    for row in range(num_rows):
        for col in range(num_cols):
            order.append((row, col))
    return order


def _spiral_inward_scan_order(num_rows: int, num_cols: int) -> List[Tuple[int, int]]:
    """Spiral inward: outer ring first, then next ring, etc.

    This ensures inner cells have context from all surrounding sides.
    """
    visited = set()
    order = []
    top, bottom, left, right = 0, num_rows - 1, 0, num_cols - 1

    while top <= bottom and left <= right:
        # Bottom edge, left to right
        for col in range(left, right + 1):
            if (top, col) not in visited:
                order.append((top, col))
                visited.add((top, col))
        top += 1

        # Right edge, bottom to top
        for row in range(top, bottom + 1):
            if (row, right) not in visited:
                order.append((row, right))
                visited.add((row, right))
        right -= 1

        # Top edge, right to left
        if top <= bottom:
            for col in range(right, left - 1, -1):
                if (bottom, col) not in visited:
                    order.append((bottom, col))
                    visited.add((bottom, col))
            bottom -= 1

        # Left edge, top to bottom
        if left <= right:
            for row in range(bottom, top - 1, -1):
                if (row, left) not in visited:
                    order.append((row, left))
                    visited.add((row, left))
            left += 1

    return order


def _frontier_scan_order(num_rows: int, num_cols: int) -> List[Tuple[int, int]]:
    """Greedily pick the next cell with the most generated neighbors.

    Ties are broken toward the grid center, then by raster order. This keeps
    the active generation region compact so future cells are more likely to
    receive context from multiple sides.
    """
    all_cells = [(row, col) for row in range(num_rows) for col in range(num_cols)]
    generated = set()
    order = []
    center_row = (num_rows - 1) / 2.0
    center_col = (num_cols - 1) / 2.0

    def neighbor_count(row: int, col: int) -> int:
        count = 0
        if col > 0 and (row, col - 1) in generated:
            count += 1
        if col < num_cols - 1 and (row, col + 1) in generated:
            count += 1
        if row > 0 and (row - 1, col) in generated:
            count += 1
        if row < num_rows - 1 and (row + 1, col) in generated:
            count += 1
        return count

    while len(order) < len(all_cells):
        remaining = [cell for cell in all_cells if cell not in generated]
        next_cell = min(
            remaining,
            key=lambda cell: (
                -neighbor_count(cell[0], cell[1]),
                abs(cell[0] - center_row) + abs(cell[1] - center_col),
                cell[0],
                cell[1],
            ),
        )
        order.append(next_cell)
        generated.add(next_cell)

    return order


def _checkerboard_scan_order(num_rows: int, num_cols: int) -> List[Tuple[int, int]]:
    """Two-pass checkerboard order.

    First pass generates one parity of cells; second pass fills the gaps. The
    second pass therefore tends to have stronger context from multiple sides.
    """
    raster = _raster_scan_order(num_rows, num_cols)
    first_pass = [cell for cell in raster if (cell[0] + cell[1]) % 2 == 0]
    second_pass = [cell for cell in raster if (cell[0] + cell[1]) % 2 == 1]
    return first_pass + second_pass


def _biome_component_scan_order(grid_bu: List[List[str]]) -> List[Tuple[int, int]]:
    """Generate one connected same-biome component at a time.

    Components are discovered in bottom-up raster order on the processing grid,
    and each component is fully generated before moving to the next. Within a
    component, cells are emitted in raster order for simplicity.
    """
    num_rows = len(grid_bu)
    num_cols = len(grid_bu[0])
    visited = set()
    cell_order = []

    for row in range(num_rows):
        for col in range(num_cols):
            if (row, col) in visited:
                continue

            biome = grid_bu[row][col]
            stack = [(row, col)]
            component = []
            visited.add((row, col))

            while stack:
                cur_row, cur_col = stack.pop()
                component.append((cur_row, cur_col))

                neighbors = [
                    (cur_row, cur_col - 1),
                    (cur_row, cur_col + 1),
                    (cur_row - 1, cur_col),
                    (cur_row + 1, cur_col),
                ]
                for n_row, n_col in neighbors:
                    if not (0 <= n_row < num_rows and 0 <= n_col < num_cols):
                        continue
                    if (n_row, n_col) in visited:
                        continue
                    if grid_bu[n_row][n_col] != biome:
                        continue
                    visited.add((n_row, n_col))
                    stack.append((n_row, n_col))

            component.sort()
            cell_order.extend(component)

    return cell_order


def _frontier_order_for_sparse_cells(cells: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Frontier order for an arbitrary occupied subset of a 2D grid."""
    if not cells:
        return []

    remaining = set(cells)
    generated = set()
    order = []
    center_row = sum(cell[0] for cell in cells) / len(cells)
    center_col = sum(cell[1] for cell in cells) / len(cells)

    def neighbor_count(row: int, col: int) -> int:
        return sum(
            (n_row, n_col) in generated
            for n_row, n_col in (
                (row, col - 1),
                (row, col + 1),
                (row - 1, col),
                (row + 1, col),
            )
        )

    while remaining:
        next_cell = min(
            remaining,
            key=lambda cell: (
                -neighbor_count(cell[0], cell[1]),
                abs(cell[0] - center_row) + abs(cell[1] - center_col),
                cell[0],
                cell[1],
            ),
        )
        order.append(next_cell)
        generated.add(next_cell)
        remaining.remove(next_cell)

    return order


def _build_layered_cell_order(
    placements: List[ChunkPlacement],
    scan_order: str,
) -> List[ChunkPlacement]:
    """Order layered placements top-down, then by a simple within-level scan."""
    if scan_order not in {"raster", "frontier"}:
        raise ValueError(
            f"Layered layouts currently support only 'raster' and 'frontier', got '{scan_order}'"
        )

    placements_by_level: Dict[int, List[ChunkPlacement]] = {}
    for placement in placements:
        placements_by_level.setdefault(placement.level_index, []).append(placement)

    ordered = []
    for level_index in sorted(
        placements_by_level.keys(),
        key=lambda idx: (
            -placements_by_level[idx][0].w0,
            idx,
        ),
    ):
        level_placements = placements_by_level[level_index]
        coord_to_placement = {(p.row, p.col): p for p in level_placements}
        coords = list(coord_to_placement.keys())
        if scan_order == "frontier":
            ordered_coords = _frontier_order_for_sparse_cells(coords)
        else:
            ordered_coords = sorted(coords)
        ordered.extend(coord_to_placement[coord] for coord in ordered_coords)

    return ordered


def _generated_context_labels(
    row: int,
    col: int,
    num_rows: int,
    num_cols: int,
    generated: set,
) -> List[str]:
    """Describe already-generated neighboring cells around a target location."""
    labels = []
    if col > 0 and (row, col - 1) in generated:
        labels.append("left")
    if col < num_cols - 1 and (row, col + 1) in generated:
        labels.append("right")
    if row > 0 and (row - 1, col) in generated:
        labels.append("below")
    if row < num_rows - 1 and (row + 1, col) in generated:
        labels.append("above")
    if row > 0 and col > 0 and (row - 1, col - 1) in generated:
        labels.append("below-left")
    if row > 0 and col < num_cols - 1 and (row - 1, col + 1) in generated:
        labels.append("below-right")
    if row < num_rows - 1 and col > 0 and (row + 1, col - 1) in generated:
        labels.append("above-left")
    if row < num_rows - 1 and col < num_cols - 1 and (row + 1, col + 1) in generated:
        labels.append("above-right")
    return labels


def _boxes_overlap(start_a: int, size_a: int, start_b: int, size_b: int) -> bool:
    end_a = start_a + size_a
    end_b = start_b + size_b
    return start_a < end_b and start_b < end_a


def _generated_context_labels_for_placement(
    placement: ChunkPlacement,
    generated: List[ChunkPlacement],
    cell_size: int,
) -> List[str]:
    """Describe overlapping context directions for arbitrary chunk placements."""
    labels = set()
    for other in generated:
        if not _boxes_overlap(placement.h0, cell_size, other.h0, cell_size):
            continue
        if not _boxes_overlap(placement.w0, cell_size, other.w0, cell_size):
            continue
        if not _boxes_overlap(placement.d0, cell_size, other.d0, cell_size):
            continue

        if other.w0 > placement.w0:
            labels.add("upper")
        elif other.w0 < placement.w0:
            labels.add("lower")

        if other.h0 < placement.h0:
            labels.add("left")
        elif other.h0 > placement.h0:
            labels.add("right")

        if other.d0 < placement.d0:
            labels.add("below")
        elif other.d0 > placement.d0:
            labels.add("above")

    ordered_labels = [
        label for label in ["left", "right", "below", "above", "upper", "lower"]
        if label in labels
    ]
    return ordered_labels


def _clone_seed_context_payload(seed_payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Clone a normalized seed payload so it can be embedded in saved cell files."""
    if seed_payload is None:
        return None

    source_indices = seed_payload.get("source_indices")
    context_mask = seed_payload.get("context_mask")
    if source_indices is None or context_mask is None:
        raise ValueError("Seed payload must contain source_indices and context_mask")
    if not torch.is_tensor(source_indices) or not torch.is_tensor(context_mask):
        raise ValueError("Seed payload tensors must be torch.Tensor instances")

    cloned = {
        "source_indices": source_indices.long().cpu().clone(),
        "context_mask": context_mask.bool().cpu().clone(),
    }

    inpaint_mask = seed_payload.get("inpaint_mask")
    if inpaint_mask is not None:
        if not torch.is_tensor(inpaint_mask):
            raise ValueError("Seed payload inpaint_mask must be a torch.Tensor")
        cloned["inpaint_mask"] = inpaint_mask.bool().cpu().clone()

    context_path = seed_payload.get("context_path")
    if context_path is not None:
        cloned["context_path"] = str(context_path)

    for key in ("format", "metadata", "notes"):
        value = seed_payload.get(key)
        if value is not None:
            cloned[key] = value

    return cloned


def _build_saved_cell_payload(
    cell_tensor: torch.Tensor,
    *,
    placement: ChunkPlacement,
    seed_payload: Optional[Dict[str, Any]],
):
    """Persist generated cells plus any authored seed context used to create them."""
    cloned_seed = _clone_seed_context_payload(seed_payload)
    if cloned_seed is None:
        return cell_tensor.cpu()
    return {
        "tensor": cell_tensor.cpu(),
        "placement_id": placement.placement_id,
        "biome": placement.biome,
        "seed_context": cloned_seed,
    }


def _extract_saved_cell_tensor(saved_cell) -> torch.Tensor:
    """Support both legacy tensor-only cells and dict payloads with metadata."""
    if torch.is_tensor(saved_cell):
        return saved_cell.cpu()
    if isinstance(saved_cell, dict):
        tensor = saved_cell.get("tensor", saved_cell.get("cell_tensor"))
        if torch.is_tensor(tensor):
            return tensor.cpu()
    raise ValueError(
        f"Saved cell payload must be a tensor or dict containing 'tensor', got {type(saved_cell).__name__}"
    )


def _sanitize_output_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in str(name))
    return safe.strip("._") or "artifact"


def _resample_intermediates_exact(
    intermediates: torch.Tensor,
    num_frames: int,
) -> torch.Tensor:
    """Nearest-neighbor resample to an exact frame count, allowing repeats."""
    if intermediates.dim() != 4:
        raise ValueError(
            f"Expected [T,H,W,D] intermediates, got {tuple(intermediates.shape)}"
        )
    target = max(1, int(num_frames))
    total = int(intermediates.shape[0])
    if total <= 0:
        raise ValueError("Intermediates tensor must contain at least one frame")
    if total == target:
        return intermediates
    idx = torch.linspace(0, total - 1, steps=target).round().long()
    return intermediates[idx]


def _save_cell_diffusion_trace(
    *,
    trace_dir: Path,
    placement: ChunkPlacement,
    biome: str,
    diffusion_trace: Optional[Dict[str, Any]],
) -> Optional[str]:
    """Persist one placement's logit trace for later whole-world stitching."""
    if diffusion_trace is None:
        return None

    trace_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{placement.placement_id}_{_sanitize_output_name(biome)}"
    trace_path = trace_dir / f"{stem}_logit_intermediates.pt"
    intermediates = diffusion_trace["intermediates"]
    if intermediates.dim() != 5 or intermediates.shape[0] != 1:
        raise ValueError(
            f"Expected single-sample intermediates [1,T,H,W,D], got {tuple(intermediates.shape)}"
        )

    torch.save(
        {
            "placement": asdict(placement),
            "biome": biome,
            "trace_mode": diffusion_trace.get("trace_mode", "full_volume_logit_samples"),
            "fraction_known": float(diffusion_trace["fraction_known"]),
            "t_start": float(diffusion_trace["t_start"]),
            "effective_steps": int(diffusion_trace["effective_steps"]),
            "save_step_indices": [int(idx) for idx in diffusion_trace.get("save_step_indices", [])],
            "logit_save_bias_power": float(diffusion_trace.get("logit_save_bias_power", 1.0)),
            "intermediates": intermediates[0].cpu().to(torch.int16),
        },
        trace_path,
    )
    return str(trace_path)


def _build_world_diffusion_intermediates(
    *,
    placement_order: List[ChunkPlacement],
    trace_paths_by_id: Dict[str, str],
    world_shape: Tuple[int, int, int],
    cell_size: int,
    num_frames: int,
    air_idx: int,
) -> torch.Tensor:
    """Stitch per-cell traces into full-world frames using final overwrite order."""
    world_h, world_w, world_d = world_shape
    world_intermediates = torch.full(
        (int(num_frames), world_h, world_w, world_d),
        int(air_idx),
        dtype=torch.int16,
    )

    for placement in placement_order:
        trace_path = trace_paths_by_id.get(placement.placement_id)
        if trace_path is None:
            raise ValueError(
                f"Missing saved diffusion trace for placement {placement.placement_id}"
            )
        trace_payload = torch.load(trace_path, map_location="cpu", weights_only=False)
        cell_intermediates = trace_payload["intermediates"].cpu()
        cell_intermediates = _resample_intermediates_exact(cell_intermediates, num_frames).to(torch.int16)
        world_intermediates[
            :,
            placement.h0:placement.h0 + cell_size,
            placement.w0:placement.w0 + cell_size,
            placement.d0:placement.d0 + cell_size,
        ] = cell_intermediates

    return world_intermediates


def _save_world_diffusion_artifacts(
    *,
    output_dir: Path,
    placement_order: List[ChunkPlacement],
    trace_paths_by_id: Dict[str, str],
    world_shape: Tuple[int, int, int],
    cell_size: int,
    num_frames: int,
    air_idx: int,
    converter: BlockBiomeConverter,
    image_size: int,
    fps: int,
    textured: bool,
    textures_dir: Optional[str],
) -> Optional[Dict[str, str]]:
    """Create and save the whole-world diffusion tensor plus animated GIF."""
    if not trace_paths_by_id:
        return None

    print("\nStitching whole-world diffusion intermediates...")
    world_intermediates = _build_world_diffusion_intermediates(
        placement_order=placement_order,
        trace_paths_by_id=trace_paths_by_id,
        world_shape=world_shape,
        cell_size=cell_size,
        num_frames=num_frames,
        air_idx=air_idx,
    )

    tensor_path = output_dir / "world_logit_intermediates.pt"
    gif_path = output_dir / "world_logit.gif"
    torch.save(
        {
            "intermediates": world_intermediates,
            "world_shape": list(world_shape),
            "cell_size": int(cell_size),
            "num_frames": int(num_frames),
            "placement_order": [asdict(placement) for placement in placement_order],
            "cell_trace_files": trace_paths_by_id,
            "trace_mode": "stitched_full_canvas_logit_samples",
        },
        tensor_path,
    )
    print(f"  Saved world diffusion tensor to {tensor_path}")

    try:
        render_diffusion_gif(
            world_intermediates,
            str(gif_path),
            converter,
            num_frames=int(world_intermediates.shape[0]),
            image_size=min(image_size * 2, 1024),
            fps=fps,
            textured=textured,
            include_initial_empty=False,
            textures_dir=(textures_dir if textured else None),
        )
        print(f"  Saved whole-world diffusion GIF to {gif_path}")
    except Exception as e:
        print(f"  Whole-world diffusion GIF warning: {e}")

    return {
        "intermediates_path": str(tensor_path),
        "gif_path": str(gif_path),
    }


def generate_world(
    model,
    inpainter: MD4Inpainter,
    layout_spec: Dict[str, Any],
    converter: BlockBiomeConverter,
    device: torch.device,
    cell_size: int = 32,
    overlap: int = 4,
    reverse_steps: int = 1000,
    cond_scale: float = 4.0,
    air_idx: int = 0,
    output_dir: Optional[Path] = None,
    visualizer=None,
    textured: bool = False,
    image_size: int = 512,
    render_progress: bool = False,
    scan_order: str = "raster",
    save_gif: bool = False,
    gif_timesteps: int = 40,
    gif_fps: int = 10,
    gif_early_bias: float = 2.0,
    textures_dir: Optional[str] = None,
) -> torch.Tensor:
    """
    Generate a world by inpainting chunk placements in a configurable scan order.

    Each new chunk receives context from whatever voxels are already known in
    its overlapping region with previously generated chunks, including edge,
    corner, and vertical overlap.

    Memory strategy:
    - The world tensor is assembled incrementally on CPU.
    - A CPU boolean mask tracks which voxels are already fixed and can be used
      as context for future cells.
    - Individual cell .pt files are saved to disk.
    - Nothing is kept on GPU between cells.

    Axis convention for tensors (B, H, W, D):
        H (dim 1) = horizontal ground axis
        W (dim 2) = Y = vertical height (rendered as up)
        D (dim 3) = horizontal ground axis
    Ground plane is H x D (dims 1 and 3). W (dim 2) is vertical.

    Args:
        scan_order: "raster" (left-to-right, bottom-to-top),
                    "spiral" (outer ring first, spiraling inward), or
                    "frontier" (pick the next cell with the most generated
                    neighbors), or "checkerboard" (alternating cells first,
                    then fill the gaps), or "biome" (finish each connected
                    same-biome component before moving to the next)
    """
    placements: List[ChunkPlacement] = layout_spec["placements"]
    context_payloads_by_id: Dict[str, Dict[str, Any]] = layout_spec.get("context_payloads", {})
    if not placements:
        raise ValueError("Layout spec contains no chunk placements")

    custom_generation_order = layout_spec.get("generation_order")
    if layout_spec["kind"] == "2d":
        display_grid = layout_spec["display_grid"]
        grid_bu = list(reversed(display_grid))
        num_rows = len(grid_bu)
        num_cols = len(grid_bu[0])
        placement_lookup = {(p.row, p.col): p for p in placements}
        if scan_order == "spiral":
            ordered_coords = _spiral_inward_scan_order(num_rows, num_cols)
        elif scan_order == "frontier":
            ordered_coords = _frontier_scan_order(num_rows, num_cols)
        elif scan_order == "checkerboard":
            ordered_coords = _checkerboard_scan_order(num_rows, num_cols)
        elif scan_order == "biome":
            ordered_coords = _biome_component_scan_order(grid_bu)
        else:
            ordered_coords = _raster_scan_order(num_rows, num_cols)
        default_order = [placement_lookup[coord] for coord in ordered_coords]
    else:
        if custom_generation_order is not None and len(custom_generation_order) == len(placements):
            default_order = []
        else:
            default_order = _build_layered_cell_order(placements, scan_order)
        num_rows = num_cols = None

    if custom_generation_order is not None:
        explicit_ids = {placement.placement_id for placement in custom_generation_order}
        placement_order = list(custom_generation_order) + [
            placement for placement in default_order
            if placement.placement_id not in explicit_ids
        ]
    else:
        placement_order = default_order

    print(f"\nGeneration plan:")
    if layout_spec["kind"] == "2d":
        stride = cell_size - overlap
        print(f"  Grid: {num_cols} cols x {num_rows} rows")
        print(f"  Cell size: {cell_size}^3, Overlap: {overlap} blocks, Stride: {stride}")
    else:
        print(f"  Layered layout: {len(layout_spec['levels'])} levels")
        print(f"  Placements: {len(placements)} chunks")
        print(f"  Cell size: {cell_size}^3, Overlap: {overlap} blocks")
    if custom_generation_order is not None:
        print(
            f"  Explicit generation order: {len(custom_generation_order)} placements"
        )
        if len(custom_generation_order) < len(placements):
            print(f"  Remaining placements fall back to scan order: {scan_order}")

    world_h = max(p.h0 + cell_size for p in placements)
    world_w = max(p.w0 + cell_size for p in placements)
    world_d = max(p.d0 + cell_size for p in placements)
    print(f"  World dimensions: ground={world_h}x{world_d}, height={world_w}")
    print(f"  Scan order: {scan_order}")
    print(f"  Total chunks to generate: {len(placement_order)}")

    biome_idx_lookup = {
        biome: resolve_biome_index(biome, converter)
        for biome in sorted({p.biome for p in placements if p.biome != UNCONDITIONAL_BIOME})
    }

    # Assembled world on CPU (H, W, D). Fill with air_idx so ungenerated
    # regions render as air in progress images.
    world = torch.full((world_h, world_w, world_d), air_idx, dtype=torch.long)
    world_known = torch.zeros((world_h, world_w, world_d), dtype=torch.bool)

    # Build a stitched seed-context preview volume for visualization only.
    # Seed contexts themselves remain local per-cell constraints on a fixed
    # 32-block stride; overlap should affect only neighboring chunk context.
    seeded_placements = 0
    seeded_voxels = 0
    if context_payloads_by_id:
        preview_h = max(p.h0 + cell_size for p in placements)
        preview_w = max(p.w0 + cell_size for p in placements)
        preview_d = max(p.d0 + cell_size for p in placements)

        stitched_seed_world = torch.full((preview_h, preview_w, preview_d), air_idx, dtype=torch.long)
        stitched_seed_known = torch.zeros((preview_h, preview_w, preview_d), dtype=torch.bool)

        for placement in placements:
            payload = context_payloads_by_id.get(placement.placement_id)
            if payload is None:
                continue

            seed_source = payload["source_indices"].cpu()
            seed_mask = payload["context_mask"].bool().cpu()
            if tuple(seed_source.shape) != (cell_size, cell_size, cell_size):
                raise ValueError(
                    f"Seed context for {placement.placement_id} has shape {tuple(seed_source.shape)}, "
                    f"expected {(cell_size, cell_size, cell_size)}"
                )
            if tuple(seed_mask.shape) != (cell_size, cell_size, cell_size):
                raise ValueError(
                    f"Seed context mask for {placement.placement_id} has shape {tuple(seed_mask.shape)}, "
                    f"expected {(cell_size, cell_size, cell_size)}"
                )

            seed_h0 = placement.h0
            seed_w0 = placement.w0
            seed_d0 = placement.d0

            preview_slice = stitched_seed_world[
                seed_h0:seed_h0 + cell_size,
                seed_w0:seed_w0 + cell_size,
                seed_d0:seed_d0 + cell_size,
            ]
            preview_known = stitched_seed_known[
                seed_h0:seed_h0 + cell_size,
                seed_w0:seed_w0 + cell_size,
                seed_d0:seed_d0 + cell_size,
            ]
            preview_slice[seed_mask] = seed_source[seed_mask]
            preview_known[seed_mask] = True
            seeded_placements += 1
            seeded_voxels += int(seed_mask.sum().item())

        print(
            f"  Built stitched seed context preview: {seeded_placements} cells, "
            f"{seeded_voxels} fixed voxels"
        )
        if output_dir is not None:
            torch.save(
                {
                    "world": stitched_seed_world,
                    "world_known": stitched_seed_known,
                    "seeded_placements": seeded_placements,
                    "seeded_voxels": seeded_voxels,
                },
                output_dir / "stitched_seed_context.pt",
            )
            print(f"  Saved stitched seed context tensor to {output_dir / 'stitched_seed_context.pt'}")

            if visualizer is not None:
                try:
                    stitched_blocks = converter.convert_to_original_blocks(stitched_seed_world.unsqueeze(0))
                    render_chunk_to_file_fitted_iso(
                        stitched_blocks[0],
                        str(output_dir / "stitched_seed_context.png"),
                        visualizer,
                        textured,
                        min(image_size * 3, 2048),
                    )
                    print(f"  Saved stitched seed context render to {output_dir / 'stitched_seed_context.png'}")
                except Exception as e:
                    print(f"  Stitched seed context render warning: {e}")

    generated_coords = set()
    generated_placements: List[ChunkPlacement] = []

    total = len(placement_order)
    count = 0

    cell_dir = None
    trace_dir = None
    trace_paths_by_id: Dict[str, str] = {}
    if output_dir:
        cell_dir = output_dir / "cells"
        cell_dir.mkdir(parents=True, exist_ok=True)
        if save_gif:
            trace_dir = output_dir / "cell_intermediates"
            trace_dir.mkdir(parents=True, exist_ok=True)

    for placement in placement_order:
        count += 1
        biome = placement.biome
        biome_idx = biome_idx_lookup.get(biome)
        h0 = placement.h0
        w0 = placement.w0
        d0 = placement.d0

        # Build input tensor (B, H, W, D) and mask
        inp = torch.zeros(
            1, cell_size, cell_size, cell_size,
            dtype=torch.long, device=device,
        )
        mask = torch.ones(
            1, cell_size, cell_size, cell_size,
            dtype=torch.bool, device=device,
        )

        seed_payload = context_payloads_by_id.get(placement.placement_id)
        seed_known = 0
        if seed_payload is not None:
            seed_source = seed_payload["source_indices"]
            seed_mask = seed_payload["context_mask"]
            seed_mask_device = seed_mask.to(device)
            inp[0][seed_mask_device] = seed_source[seed_mask].to(device)
            mask[0][seed_mask_device] = False
            seed_known = int(seed_mask.sum().item())

        context_slice = world[h0:h0 + cell_size, w0:w0 + cell_size, d0:d0 + cell_size]
        known_slice = world_known[h0:h0 + cell_size, w0:w0 + cell_size, d0:d0 + cell_size]
        known_device = known_slice.to(device)
        if known_slice.any():
            inp[0][known_device] = context_slice[known_slice].to(device)
            mask[0][known_device] = False

        if layout_spec["kind"] == "2d":
            ctx_parts = _generated_context_labels(
                placement.row, placement.col, num_rows, num_cols, generated_coords
            )
        else:
            ctx_parts = _generated_context_labels_for_placement(
                placement, generated_placements, cell_size
            )

        frac = (~mask).float().mean().item()
        if ctx_parts:
            status = f"context: {', '.join(ctx_parts)} ({frac*100:.1f}% known)"
        elif seed_known > 0:
            status = f"seed context only ({frac*100:.1f}% known)"
        else:
            status = "generating from scratch"

        if layout_spec["kind"] == "2d":
            location = f"col={placement.col} row={placement.row}"
        else:
            location = (
                f"level={placement.level_name} z={placement.w0} "
                f"col={placement.col} row={placement.row}"
            )
        effective_cond_scale = cond_scale
        if biome == UNCONDITIONAL_BIOME:
            print(f"\n[{count}/{total}] {location} | unconditional | {status}")
            label = _unconditional_class_label(model, converter, device)
            if label is not None:
                effective_cond_scale = 0.0
        else:
            print(f"\n[{count}/{total}] {location} | {biome} (idx {biome_idx}) | {status}")
            label = torch.tensor([biome_idx], dtype=torch.long, device=device)
        diffusion_trace = None
        if save_gif:
            result, diffusion_trace = inpainter.inpaint_time_aligned(
                model, inp, mask,
                reverse_steps=reverse_steps,
                progress=True,
                air_index_fallback=air_idx,
                class_cond=label,
                cond_scale=effective_cond_scale,
                return_logit_intermediates=True,
                num_logit_intermediates=gif_timesteps,
                logit_save_bias_power=gif_early_bias,
            )
        else:
            result = inpainter.inpaint_time_aligned(
                model, inp, mask,
                reverse_steps=reverse_steps,
                progress=True,
                air_index_fallback=air_idx,
                class_cond=label,
                cond_scale=effective_cond_scale,
            )

        cell_cpu = result[0].cpu()  # (H, W, D)

        # ---- Write to world tensor immediately ----
        world[h0:h0 + cell_size, w0:w0 + cell_size, d0:d0 + cell_size] = cell_cpu
        world_known[h0:h0 + cell_size, w0:w0 + cell_size, d0:d0 + cell_size] = True

        generated_coords.add((placement.row, placement.col))
        generated_placements.append(placement)

        # ---- Save cell to disk ----
        if cell_dir:
            if layout_spec["kind"] == "2d":
                filename = f"r{placement.row}_c{placement.col}_{biome}.pt"
            else:
                filename = f"{placement.placement_id}_{biome}.pt"
            torch.save(
                _build_saved_cell_payload(
                    cell_cpu,
                    placement=placement,
                    seed_payload=seed_payload,
                ),
                cell_dir / filename,
            )
            if save_gif and trace_dir is not None:
                trace_path = _save_cell_diffusion_trace(
                    trace_dir=trace_dir,
                    placement=placement,
                    biome=biome,
                    diffusion_trace=diffusion_trace,
                )
                if trace_path is not None:
                    trace_paths_by_id[placement.placement_id] = trace_path

        # ---- Context + infill renders (always when visualizer available) ----
        if output_dir and visualizer:
            ctx_dir = output_dir / "context_chunk_imgs"
            ctx_dir.mkdir(parents=True, exist_ok=True)

            cell_blocks = converter.convert_to_original_blocks(
                cell_cpu.unsqueeze(0)
            )

            if not ctx_parts:
                try:
                    render_chunk_to_file(
                        cell_blocks[0],
                        str(ctx_dir / f"{count:03d}_{placement.placement_id}_{biome}.png"),
                        visualizer, textured, image_size,
                    )
                except Exception as e:
                    print(f"  Render warning: {e}")
            else:
                AIR_BLOCK_ID = 5
                context_blocks = cell_blocks.clone()
                context_mask_3d = mask[0].cpu()
                context_blocks[0][context_mask_3d] = AIR_BLOCK_ID

                try:
                    render_side_by_side(
                        chunks=[context_blocks[0], cell_blocks[0]],
                        labels=["Context", f"Infilled ({biome})"],
                        output_path=str(
                            ctx_dir / f"{count:03d}_{placement.placement_id}_{biome}.png"
                        ),
                        visualizer=visualizer,
                        textured=textured,
                        image_size=image_size,
                    )
                except Exception as e:
                    print(f"  Render warning: {e}")

        # ---- Optional: render cumulative world progress ----
        if render_progress and output_dir and visualizer:
            progress_dir = output_dir / "progress"
            progress_dir.mkdir(parents=True, exist_ok=True)
            try:
                world_so_far = converter.convert_to_original_blocks(
                    world.unsqueeze(0)
                )
                render_chunk_to_file(
                    world_so_far[0],
                    str(progress_dir / f"{count:03d}_{placement.placement_id}.png"),
                    visualizer, textured,
                    min(image_size * 2, 2048),
                )
            except Exception as e:
                print(f"  Progress render warning: {e}")

        del result

    if save_gif and output_dir is not None:
        _save_world_diffusion_artifacts(
            output_dir=output_dir,
            placement_order=placement_order,
            trace_paths_by_id=trace_paths_by_id,
            world_shape=(world_h, world_w, world_d),
            cell_size=cell_size,
            num_frames=gif_timesteps,
            air_idx=air_idx,
            converter=converter,
            image_size=image_size,
            fps=gif_fps,
            textured=textured,
            textures_dir=textures_dir,
        )

    return world


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Semantic Super Sampling — generate large worlds with MD4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available layouts: {', '.join(WORLD_LAYOUTS.keys())}",
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--mappings", type=str, required=True,
                        help="Path to block/biome mappings file")
    parser.add_argument("--layout", type=str, default=None,
                        help="Predefined layout name (see --list_layouts). "
                             "If omitted and --custom_grid is not provided, runs "
                             "an unconditional 5x5 generation.")
    parser.add_argument("--custom_grid", type=str, default=None,
                        help="Path to JSON file with custom grid (overrides --layout). "
                             "JSON can be a 2D list of biome name strings, "
                             "a dict with a 'grid' key, or a layered dict with "
                             "a 'levels' list.")
    parser.add_argument("--context_grid", type=str, default=None,
                        help="Path to a JSON 2D grid of seed-context .pt files, "
                             "or a dict with 'context_grid' plus optional "
                             "'fill_grid'. Seeded cells generate first, then the "
                             "remaining cells are filled using unconditional or "
                             "biome-conditioned inpainting from fill_grid.")
    parser.add_argument("--output_dir", type=str, default="./super_sample_results",
                        help="Output directory for results")
    parser.add_argument("--overlap", type=int, default=4,
                        help="Context overlap between adjacent cells in blocks (default: 4)")
    parser.add_argument("--vertical_overlap", type=int, default=None,
                        help="Vertical overlap between stacked chunks in blocks "
                             "(default: same as --overlap)")
    parser.add_argument("--config", type=str, default=None,
                        help="Optional path to config.json")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--reverse_steps", type=int, default=None,
                        help="Override number of reverse diffusion steps")
    parser.add_argument("--cond_scale", type=float, default=None,
                        help="Classifier-free guidance scale")
    parser.add_argument("--textures_dir", type=str, default="block_textures/",
                        help="Path to block textures for rendering")
    parser.add_argument("--image_size", type=int, default=512,
                        help="Per-cell render size in pixels")
    parser.add_argument("--save_gif", action="store_true",
                        help="Save per-cell logit traces and render a stitched whole-world diffusion GIF")
    parser.add_argument("--gif_timesteps", type=int, default=40,
                        help="Number of denoising snapshots per cell when --save_gif is enabled")
    parser.add_argument("--gif_fps", type=int, default=10,
                        help="Frames per second for the stitched whole-world diffusion GIF")
    parser.add_argument("--gif_early_bias", type=float, default=2.0,
                        help="Bias GIF snapshots toward earlier denoising steps. 1.0 = uniform, >1 = earlier-heavy, <1 = later-heavy")
    parser.add_argument("--list_biomes", action="store_true",
                        help="List available biomes from the model and exit")
    parser.add_argument("--list_layouts", action="store_true",
                        help="List predefined layouts and exit")
    parser.add_argument("--render_cell_grid", action="store_true",
                        help="Render a composite grid image of all cells")
    parser.add_argument("--render_progress", action="store_true",
                        help="Render cumulative world image after each cell is added")
    parser.add_argument("--scan_order", type=str, default="raster",
                        choices=["raster", "spiral", "frontier", "checkerboard", "biome"],
                        help="Cell generation order: 'raster' (left-to-right, "
                             "bottom-to-top) or 'spiral' (outer ring inward, "
                             "so inner cells get context from all sides), or "
                             "'frontier' (always fill the cell with the most "
                             "already-generated neighbors), or "
                             "'checkerboard' (alternating cells first, then "
                             "fill the gaps with stronger context), or "
                             "'biome' (finish one connected same-biome region "
                             "at a time)")

    args = parser.parse_args()
    if args.save_gif and args.gif_timesteps < 1:
        raise ValueError(f"--gif_timesteps must be >= 1 when --save_gif is enabled, got {args.gif_timesteps}")
    if args.gif_early_bias <= 0:
        raise ValueError(f"--gif_early_bias must be > 0, got {args.gif_early_bias}")

    # --list_layouts doesn't need model
    if args.list_layouts:
        print("Available layouts:\n")
        for name, info in WORLD_LAYOUTS.items():
            print(f"  {name}: {info['description']}")
            if "grid" in info:
                for row in info["grid"]:
                    print(f"    {row}")
            elif "levels" in info:
                for level in info["levels"]:
                    print(f"    [{level.get('name', 'level')}] w_offset={level.get('w_offset', 0)}")
                    for row in level["grid"]:
                        print(f"      {row}")
            print()
        return

    # -------------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------------
    print(f"Loading model from {args.checkpoint}...")
    loaded = load_model_from_file(
        args.checkpoint,
        config_path=args.config,
        mappings_path=args.mappings,
        device=args.device,
        use_ema_if_available=True,
    )

    model = loaded["model"]
    converter = loaded["converter"]
    device = loaded["device"]
    cell_size = loaded["image_size"]

    if args.list_biomes:
        print("Available biomes:")
        for name, idx in sorted(
            (converter.biome_to_index or {}).items(), key=lambda x: x[1]
        ):
            print(f"  {idx}: {name}")
        return

    reverse_steps = args.reverse_steps or loaded.get("reverse_steps", 1000)
    cond_scale = args.cond_scale or loaded.get("default_cond_scale", 4.0)

    print(f"Model loaded: {loaded['num_blocks']} blocks, "
          f"{loaded.get('num_classes', 'N/A')} classes, "
          f"cell {cell_size}^3")
    print(f"Reverse steps: {reverse_steps}, CFG scale: {cond_scale}")

    # -------------------------------------------------------------------------
    # Inpainter setup
    # -------------------------------------------------------------------------
    accelerator = Accelerator(mixed_precision="no")
    inpainter = MD4Inpainter(
        accelerator=accelerator,
        num_classes=loaded["num_blocks"],
        device=device,
    )

    try:
        air_idx = converter.get_air_block_index()
    except Exception:
        air_idx = 0

    # Validate overlap
    if args.overlap < 0 or args.overlap >= cell_size // 2:
        raise ValueError(
            f"Overlap must be in [0, {cell_size // 2 - 1}], got {args.overlap}"
        )
    vertical_overlap = args.overlap if args.vertical_overlap is None else args.vertical_overlap
    if vertical_overlap < 0 or vertical_overlap >= cell_size:
        raise ValueError(
            f"vertical_overlap must be in [0, {cell_size - 1}], got {vertical_overlap}"
        )

    # -------------------------------------------------------------------------
    # Resolve grid layout
    # -------------------------------------------------------------------------
    unconditional_mode = False
    if args.context_grid:
        layout_name, layout_spec = _load_context_grid_layout(
            args.context_grid,
            converter=converter,
            cell_size=cell_size,
            overlap=args.overlap,
            vertical_overlap=vertical_overlap,
        )
        layout_data = layout_spec["raw"]
        unconditional_mode = bool(layout_data.get("unconditional", False))
        print(f"\nLoaded seed context grid from {args.context_grid}")
    elif args.custom_grid:
        with open(args.custom_grid) as f:
            data = json.load(f)
        layout_data = data
        layout_name = Path(args.custom_grid).stem
    elif args.layout is None:
        unconditional_mode = True
        layout_data = _build_unconditional_layout_data()
        layout_name = f"unconditional_{DEFAULT_UNCONDITIONAL_GRID_SIZE}x{DEFAULT_UNCONDITIONAL_GRID_SIZE}"
        print(
            f"\nNo layout specified. Falling back to unconditional "
            f"{DEFAULT_UNCONDITIONAL_GRID_SIZE}x{DEFAULT_UNCONDITIONAL_GRID_SIZE} generation."
        )
    else:
        if args.layout not in WORLD_LAYOUTS:
            print(f"Unknown layout '{args.layout}'. "
                  f"Available: {list(WORLD_LAYOUTS.keys())}")
            return
        layout_data = WORLD_LAYOUTS[args.layout]
        layout_name = args.layout

    if not args.context_grid:
        layout_spec = _normalize_layout_data(
            layout_data=layout_data,
            layout_name=layout_name,
            cell_size=cell_size,
            overlap=args.overlap,
            vertical_overlap=vertical_overlap,
        )

    # Validate biome names
    print("\nValidating biome names...")
    unique_biomes = {placement.biome for placement in layout_spec["placements"]}
    resolved_biomes = sorted(biome for biome in unique_biomes if biome != UNCONDITIONAL_BIOME)
    if resolved_biomes:
        for biome in resolved_biomes:
            idx = resolve_biome_index(biome, converter)
            resolved = converter.index_to_biome.get(idx, biome)
            print(f"  {biome} -> idx {idx} ({resolved})")
    if UNCONDITIONAL_BIOME in unique_biomes:
        print("  unconditional -> no biome conditioning")

    if layout_spec["display_grid"] is not None:
        print_grid(layout_spec["display_grid"], f"Layout: {layout_name}")

    # -------------------------------------------------------------------------
    # Output setup
    # -------------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "checkpoint": args.checkpoint,
        "layout": layout_name,
        "layout_data": layout_spec["raw"],
        "display_grid": layout_spec["display_grid"],
        "overlap": args.overlap,
        "vertical_overlap": vertical_overlap,
        "cell_size": cell_size,
        "reverse_steps": reverse_steps,
        "cond_scale": cond_scale,
        "scan_order": args.scan_order,
        "unconditional": unconditional_mode,
        "save_gif": args.save_gif,
        "gif_timesteps": args.gif_timesteps,
        "gif_fps": args.gif_fps,
        "gif_early_bias": args.gif_early_bias,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    with open(output_dir / "chunk_manifest.json", "w") as f:
        json.dump({
            "layout_kind": layout_spec["kind"],
            "layout_data": layout_spec["raw"],
            "display_grid": layout_spec["display_grid"],
            "overlap": args.overlap,
            "vertical_overlap": vertical_overlap,
            "cell_size": cell_size,
            "placements": [asdict(placement) for placement in layout_spec["placements"]],
            "save_gif": args.save_gif,
            "gif_timesteps": args.gif_timesteps,
            "gif_fps": args.gif_fps,
            "gif_early_bias": args.gif_early_bias,
        }, f, indent=2)

    # -------------------------------------------------------------------------
    # Visualizer
    # -------------------------------------------------------------------------
    if args.textures_dir and os.path.exists(args.textures_dir):
        visualizer = MinecraftVisualizerPyVista(
            textures_dir=args.textures_dir, build_textures=True,
        )
        textured = True
    else:
        visualizer = MinecraftVisualizerPyVista()
        textured = False

    if layout_spec["display_grid"] is not None:
        render_grid_overview(layout_spec["display_grid"], str(output_dir / "grid_overview.png"))

    # -------------------------------------------------------------------------
    # Generate
    # -------------------------------------------------------------------------
    # Cell saves/renders go to output_dir; pass None to skip
    world = generate_world(
        model=model,
        inpainter=inpainter,
        layout_spec=layout_spec,
        converter=converter,
        device=device,
        cell_size=cell_size,
        overlap=args.overlap,
        reverse_steps=reverse_steps,
        cond_scale=cond_scale,
        air_idx=air_idx,
        output_dir=output_dir,
        visualizer=visualizer,
        textured=textured,
        image_size=args.image_size,
        render_progress=args.render_progress,
        scan_order=args.scan_order,
        save_gif=args.save_gif,
        gif_timesteps=args.gif_timesteps,
        gif_fps=args.gif_fps,
        gif_early_bias=args.gif_early_bias,
        textures_dir=args.textures_dir,
    )

    # -------------------------------------------------------------------------
    # Save and render
    # -------------------------------------------------------------------------
    print(f"\nSaving world data...")
    torch.save({
        "world": world,
        "grid": layout_spec["raw"].get("grid", layout_spec["display_grid"]),
        "layout_data": layout_spec["raw"],
        "overlap": args.overlap,
        "vertical_overlap": vertical_overlap,
        "cell_size": cell_size,
    }, output_dir / "world.pt")

    # Render assembled world at larger resolution.
    # convert_to_original_blocks and the PyVista visualizer both operate
    # entirely on CPU/numpy — no VRAM is used for rendering.
    print("Rendering assembled world...")
    world_blocks = converter.convert_to_original_blocks(world.unsqueeze(0))
    world_render_size = min(args.image_size * 3, 2048)
    try:
        render_chunk_to_file_fitted_iso(
            world_blocks[0],
            str(output_dir / "world_render.png"),
            visualizer, textured, world_render_size,
        )
        print(f"World render saved to {output_dir / 'world_render.png'}")
        if layout_spec["display_grid"] is not None:
            render_side_by_side_reference(
                str(output_dir / "grid_overview.png"),
                str(output_dir / "world_render.png"),
                str(output_dir / "layout_vs_world.png"),
                left_title="Biome Layout (Aligned Overview)",
                right_title="Generated World Render",
            )
    except Exception as e:
        print(f"Warning: World render failed: {e}")

    # Optional cell grid composite — reloads saved .pt files from disk
    if args.render_cell_grid:
        if layout_spec["kind"] != "2d":
            print("Cell grid composite is only supported for 2D layouts right now.")
        else:
            print("Rendering cell grid composite...")
            try:
                cell_dir = output_dir / "cells"
                saved_cells = {}
                generation_grid = layout_spec["raw"].get("grid", layout_spec["display_grid"])
                grid_bu = list(reversed(generation_grid))
                for row in range(len(grid_bu)):
                    for col in range(len(grid_bu[0])):
                        biome = grid_bu[row][col]
                        pt_path = cell_dir / f"r{row}_c{col}_{biome}.pt"
                        if pt_path.exists():
                            saved_cells[(row, col)] = _extract_saved_cell_tensor(
                                torch.load(pt_path, map_location="cpu", weights_only=False)
                            )
                render_cell_grid(
                    saved_cells, generation_grid, converter, visualizer, textured,
                    str(output_dir / "cell_grid.png"),
                    image_size=args.image_size,
                )
            except Exception as e:
                print(f"Warning: Cell grid render failed: {e}")

    print(f"\n{'='*60}")
    print(f"SUPER SAMPLING COMPLETE")
    if layout_spec["kind"] == "2d":
        print(f"  Layout: {layout_name} ({len(layout_spec['display_grid'][0])}x{len(layout_spec['display_grid'])} cells)")
    else:
        print(f"  Layout: {layout_name} ({len(layout_spec['placements'])} placements across {len(layout_spec['levels'])} levels)")
    print(f"  World: ground={world.shape[0]}x{world.shape[2]}, height={world.shape[1]}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
