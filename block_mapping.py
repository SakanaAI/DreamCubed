#!/usr/bin/env python3
"""
Block remapping / compression utilities.

Step 1: Load block frequency data and report basic stats:
- how many blocks never appear (count == 0)  [only counts explicit zeros present in the report]
- how many blocks appear but under a threshold (0 < count < threshold)
- how many blocks meet/exceed the threshold (count >= threshold)

Also supports writing a `user_data_block_types.json` file (formatted like `block_types_updated.json`)
but derived from the frequency report, containing only the block IDs and names observed there.

Step 2: Generate a "compression mapping" to reduce vocabulary size.
This is intended to be a practical, iterative starting point:
- by default, every block maps to itself (identity)
- some blocks can be dropped by mapping to "AIR" (decorative / non-structural micro-blocks)
- some blocks can be normalized (variants -> base, colors -> base)

Outputs:
- an id->id JSON mapping in the canonical `assets/block_types_updated.json` ID space
  (so deletions map to AIR=5 and 3000+ renderer IDs are representable)
- a name->name JSON mapping (human-readable)
- a meta JSON mapping including category + reason + steps
- a Markdown table documenting every mapping
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_frequency_rows(freq_json_path: Path) -> List[Dict[str, Any]]:
    """
    Loads assets/block_frequency_report.json which is shaped like a list of:
      { "block_id": 1, "block_name": "STONE", "count": 314490943, "percent": 31.003 }

    Returns: the raw list of row dicts.
    """
    with freq_json_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError(f"Expected list in {freq_json_path}, got {type(raw).__name__}")

    return [r for r in raw if isinstance(r, dict)]


def _row_int(row: Dict[str, Any], key: str) -> int:
    v = row.get(key)
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    raise ValueError(f"Row is missing int-like '{key}': {row}")


def _row_str(row: Dict[str, Any], key: str) -> str:
    v = row.get(key)
    if isinstance(v, str):
        return v
    raise ValueError(f"Row is missing string '{key}': {row}")


def rows_to_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Returns: { "STONE": 314490943, ... }
    If duplicate names exist, sums counts (shouldn't happen, but harmless).
    """
    counts: Dict[str, int] = {}
    for row in rows:
        name = _row_str(row, "block_name")
        count = _row_int(row, "count")
        counts[name] = counts.get(name, 0) + count
    if not counts:
        raise ValueError("No usable rows found in frequency JSON")
    return counts


def write_user_data_block_types(rows: List[Dict[str, Any]], out_path: Path) -> None:
    """
    Writes a JSON mapping shaped like `assets/block_types_updated.json`:
      { "<block_id>": "<block_name>", ... }
    Built strictly from the frequency table rows (includes rows even if count==0).
    """
    mapping: Dict[str, str] = {}
    for row in rows:
        block_id = _row_int(row, "block_id")
        block_name = _row_str(row, "block_name")
        k = str(block_id)
        if k in mapping and mapping[k] != block_name:
            raise ValueError(f"Conflicting names for block_id={block_id}: {mapping[k]!r} vs {block_name!r}")
        mapping[k] = block_name

    if not mapping:
        raise ValueError("No block_id/block_name pairs found to write user_data_block_types.json")

    # Write in ascending numeric block_id order (as strings, like block_types_updated.json)
    ordered = {k: mapping[k] for k in sorted(mapping.keys(), key=lambda s: int(s))}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(ordered, f, indent=2, sort_keys=False)
        f.write("\n")


def bucket_counts(counts_by_name: Dict[str, int], threshold: int) -> Tuple[int, int, int, int]:
    """
    Returns (n_total, n_zero, n_under, n_overeq) where:
      - n_zero: count == 0
      - n_under: 0 < count < threshold
      - n_overeq: count >= threshold
    """
    n_zero = n_under = n_overeq = 0
    for c in counts_by_name.values():
        if c == 0:
            n_zero += 1
        elif c < threshold:
            n_under += 1
        else:
            n_overeq += 1
    return len(counts_by_name), n_zero, n_under, n_overeq


def load_user_block_types_map(path: Path) -> Dict[int, str]:
    """
    Load a { "<id>": "<NAME>" } JSON and return {id:int -> name:str}.
    """
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected dict in {path}, got {type(raw).__name__}")
    out: Dict[int, str] = {}
    for k, v in raw.items():
        if not isinstance(v, str):
            raise ValueError(f"Expected string block name for key={k!r} in {path}")
        out[int(k)] = v
    if not out:
        raise ValueError(f"No block types found in {path}")
    return out


_MC_COLOR_PREFIXES = (
    "WHITE_",
    "ORANGE_",
    "MAGENTA_",
    "LIGHT_BLUE_",
    "YELLOW_",
    "LIME_",
    "PINK_",
    "GRAY_",
    "LIGHT_GRAY_",
    "CYAN_",
    "PURPLE_",
    "BLUE_",
    "BROWN_",
    "GREEN_",
    "RED_",
    "BLACK_",
)


def _is_stair_or_slab(name: str) -> bool:
    # Keep geometry-critical blocks; we never map these to AIR in the baseline mapping.
    # NOTE: This is conservative: it also treats "STONE_SLAB" and "SMOOTH_STONE_SLAB" as slabs.
    return name.endswith("_STAIRS") or ("SLAB" in name)


def _target_if_present(candidates: List[str], names: set[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in names:
            return candidate
    return None


def _build_compression_mapping_for_user_blocks(
    id_to_name: Dict[int, str],
    counts_by_name: Optional[Dict[str, int]] = None,
    *,
    rare_count_threshold: int = 100,
) -> Tuple[Dict[str, str], Dict[str, Dict[str, str]]]:
    """
    Returns:
      - simple mapping: {source_name: target_name}
      - meta mapping:   {source_name: {"to": ..., "category": ..., "reason": ...}}

    Categories:
      - identity
      - useless_to_air
      - variant_to_base
      - color_to_base
      - wood_to_oak
      - rare_to_air
    """
    names = set(id_to_name.values())

    # ---- Rule 1: "useless"/micro blocks -> AIR (conservative baseline) ----
    # We intentionally avoid mapping slabs/stairs to AIR to preserve sub-block geometry.
    delete_exact = {
        # light / invisible / dev-only
        "BARRIER",
        "LIGHT",

        # small decorative or non-structural
        "TORCH",
        "WALL_TORCH",
        "REDSTONE_TORCH",
        "REDSTONE_WALL_TORCH",
        "END_ROD",
        "CHAIN",
        "LADDER",
        "TRIPWIRE",
        "TRIPWIRE_HOOK",
        "LEVER",
        "FLOWER_POT",
        "PAINTING",
        # thin decorative blocks we don't want to preserve
        "TRAPDOOR",
        # rail family (all are sub-block details for our purposes)
        "RAIL",
        # sub-block / attachment / redstone-internals / misc decoration
        "DETECTOR_RAIL",
        "GOLDEN_RAIL",
        "ACTIVATOR_RAIL",
        "POWERED_RAIL",
        "MELON_STEM",
        "PUMPKIN_STEM",
        "ATTACHED_MELON_STEM",
        "ATTACHED_PUMPKIN_STEM",
        "BIG_DRIPLEAF_STEM",
        "MONSTER_EGG",
        "BUDDING_AMETHYST",
        "CANDLE",
        "LANTERN",
        "MOVING_PISTON",
        "DECORATED_POT",
        "BED",
        "SIGN",
        "SKULL",
        "CANDLE_CAKE",
        "CAVE_VINES",
        "CAVE_VINES_PLANT",
        "MOSS_CARPET",
        "PALE_MOSS_CARPET",
        "BIG_DRIPLEAF",
        "SMALL_DRIPLEAF",
        "GLOW_LICHEN",
        "HANGING_ROOTS",
        "SPORE_BLOSSOM",
        "PINK_PETALS",
        "LEAF_LITTER",
        "SEA_PICKLE",
        "SEAGRASS",
        "TALL_SEAGRASS",
        "BUSH",
        "FIREFLY_BUSH",
        "KELP",
        "KELP_PLANT",
        "CAMPFIRE",
        "LECTERN",
        "MANGROVE_ROOTS",
        "CRIMSON_ROOTS",
        "WARPED_ROOTS",
        "NETHER_SPROUTS",
        "CAULDRON",
        "SCAFFOLDING",
        "POINTED_DRIPSTONE",
        "BELL",
        "LIGHTNING_ROD",
        "CHORUS_FLOWER",
        "CHORUS_PLANT",
        "PALE_HANGING_MOSS",
        "PALE_MOSS_BLOCK",
        "PITCHER_CROP",
        "PITCHER_PLANT",
        "CAKE",
        # containers we don't want to preserve
        "SHULKER_BOX",
        # NOTE: Many potted plants are handled via prefix rule below.
    }

    delete_prefixes = (
        "POTTED_",
    )
    # These are thin/decoration/signage variants with many material prefixes.
    delete_suffixes = (
        "_BUTTON",
        "_PRESSURE_PLATE",
        "_WALL_SIGN",
        "_SIGN",
        "_WALL_HANGING_SIGN",
        "_HANGING_SIGN",
        "_BANNER",
        "_WALL_BANNER",
        "_HEAD",
        "_WALL_HEAD",
        "_SKULL",
        "_WALL_SKULL",
        "_TRAPDOOR",
        "_SHELF",
        "_RAIL",
        "_LANTERN",
        "_CANDLE",
        "_CANDLE_CAKE",
        "_SHULKER_BOX",
        "_AMETHYST_BUD",
        "_TORCH",
        "_CHAIN",
        "_CORAL_FAN",
        "_WALL_FAN",
        "_LIGHTNING_ROD",
        "_CORAL",
        "_CORAL_BLOCK",
    )

    def rule_useless_to_air(name: str) -> Optional[Tuple[str, str]]:
        if _is_stair_or_slab(name):
            return None
        if name in delete_exact or any(name.startswith(pfx) for pfx in delete_prefixes) or any(name.endswith(sfx) for sfx in delete_suffixes):
            return "AIR", "Decorative/micro block removed to reduce vocabulary; mapped to AIR."
        return None

    # ---- Rule 2: variants -> base ----
    # We do not aggressively collapse all build-material variants; this is a first pass.
    variant_pairs = {
        # mossy/cracked/infested are mostly texture variants
        "MOSSY_COBBLESTONE": "COBBLESTONE",
        "MOSSY_STONE_BRICKS": "STONE_BRICKS",
        "CRACKED_STONE_BRICKS": "STONE_BRICKS",
        "CHISELED_STONE_BRICKS": "STONE_BRICKS",
        "INFESTED_STONE": "STONE",
        "INFESTED_COBBLESTONE": "COBBLESTONE",
        "INFESTED_STONE_BRICKS": "STONE_BRICKS",
        "INFESTED_MOSSY_STONE_BRICKS": "STONE_BRICKS",
        "INFESTED_CRACKED_STONE_BRICKS": "STONE_BRICKS",
        "INFESTED_CHISELED_STONE_BRICKS": "STONE_BRICKS",

        # stone family polish -> base
        "POLISHED_ANDESITE": "ANDESITE",
        "POLISHED_DIORITE": "DIORITE",
        "POLISHED_GRANITE": "GRANITE",

        # smooth/cut/chiseled (sand/quartz)
        "SMOOTH_STONE": "STONE",
        "SMOOTH_SANDSTONE": "SANDSTONE",
        "SMOOTH_RED_SANDSTONE": "RED_SANDSTONE",
        "CUT_SANDSTONE": "SANDSTONE",
        "CUT_RED_SANDSTONE": "RED_SANDSTONE",
        "CHISELED_SANDSTONE": "SANDSTONE",
        "CHISELED_RED_SANDSTONE": "RED_SANDSTONE",
        "SMOOTH_QUARTZ": "QUARTZ_BLOCK",
        "CHISELED_QUARTZ_BLOCK": "QUARTZ_BLOCK",
        "QUARTZ_PILLAR": "QUARTZ_BLOCK",
        "QUARTZ_BRICKS": "QUARTZ_BLOCK",

        # Canonicalize newer explicit-name blocks to older canonical blocks (helps reduce vocab + reuse textures):
        "STONE_BRICKS": "STONEBRICK",
        "TERRACOTTA": "HARDENED_CLAY",
        "OAK_PLANKS": "PLANKS",
        "OAK_LOG": "LOG",
        "OAK_WOOD": "LOG",
        "OAK_LEAVES": "LEAVES",
        "OAK_SAPLING": "SAPLING",
        "OAK_DOOR": "WOODEN_DOOR",
        "OAK_SLAB": "WOODEN_SLAB",
        "OAK_FENCE": "FENCE",
        "OAK_FENCE_GATE": "FENCE_GATE",
        "NOTE_BLOCK": "NOTEBLOCK",
        "COBWEB": "WEB",
        "NETHER_PORTAL": "PORTAL",
        "LILY_PAD": "WATERLILY",
        "MAGMA_BLOCK": "MAGMA",
        "NETHER_QUARTZ_ORE": "QUARTZ_ORE",
        "END_STONE_BRICKS": "END_BRICKS",
        "LOG2": "LOG",
        "LEAVES2": "LEAVES",
        "TALL_GRASS": "TALLGRASS",
        "STAINED_GLASS": "GLASS",
        "STAINED_GLASS_PANE": "GLASS_PANE",
        "STAINED_HARDENED_CLAY": "HARDENED_CLAY",
        "WOOL": "WHITE_WOOL",
        "CARPET": "WHITE_CARPET",

        # More legacy-name normalization
        "BRICKS": "BRICK_BLOCK",
        "NETHER_BRICKS": "NETHER_BRICK",
        "RED_NETHER_BRICKS": "RED_NETHER_BRICK",
        "GRASS_BLOCK": "GRASS",
        "DIRT_PATH": "GRASS_PATH",
        "CAVE_AIR": "AIR",
        "VOID_AIR": "AIR",

        # User-requested variant collapses
        "TRAPPED_CHEST": "CHEST",
        "CARVED_PUMPKIN": "PUMPKIN",
        "CHISELED_BOOKSHELF": "BOOKSHELF",
        "CHISELED_NETHER_BRICKS": "NETHER_BRICKS",
        "COARSE_DIRT": "DIRT",
        "COBBLED_DEEPSLATE": "DEEPSLATE",
        "INFESTED_DEEPSLATE": "DEEPSLATE",
        "CUT_COPPER": "COPPER_BLOCK",
        "OXIDIZED_CUT_COPPER": "CUT_COPPER",
        "WEATHERED_CUT_COPPER": "CUT_COPPER",
        "EXPOSED_CUT_COPPER": "CUT_COPPER",
        "DAMAGED_ANVIL": "ANVIL",
        "CHIPPED_ANVIL": "ANVIL",
        "PACKED_MUD": "MUD",
        "ROOTED_DIRT": "DIRT",
        "POLISHED_BASALT": "BASALT",
        "SILVER_GLAZED_TERRACOTTA": "TERRACOTTA",
        "RED_NETHER_BRICK": "NETHER_BRICK",
        "RED_SANDSTONE": "SANDSTONE",
        "FLOWERING_AZALEA": "AZALEA",
        "FLOWERING_AZALEA_LEAVES": "AZALEA_LEAVES",
        "CRACKED_NETHER_BRICKS": "NETHER_BRICKS",
        "CRACKED_POLISHED_BLACKSTONE_BRICKS": "POLISHED_BLACKSTONE_BRICKS",
    }

    flower_to_red = {
        "POPPY",
        "BLUE_ORCHID",
        "ALLIUM",
        "AZURE_BLUET",
        "RED_TULIP",
        "ORANGE_TULIP",
        "WHITE_TULIP",
        "PINK_TULIP",
        "OXEYE_DAISY",
        "CORNFLOWER",
        "LILY_OF_THE_VALLEY",
        "WITHER_ROSE",
        "TORCHFLOWER",
        "OPEN_EYEBLOSSOM",
        "CLOSED_EYEBLOSSOM",
        "WILDFLOWERS",
        "CACTUS_FLOWER",
    }
    double_plant_like = {
        "SUNFLOWER",
        "LILAC",
        "ROSE_BUSH",
        "PEONY",
        "LARGE_FERN",
    }
    grass_like = {
        "TALL_GRASS",
        "TALLGRASS",
        "SHORT_DRY_GRASS",
        "TALL_DRY_GRASS",
        "FERN",
    }

    def rule_variant_to_base(name: str) -> Optional[Tuple[str, str]]:
        if not _is_stair_or_slab(name):
            base = variant_pairs.get(name)
            if base is not None and base in names:
                return base, "Minor texture/material variant collapsed to its base block."

        # RED_* material families -> base family (reduce vocab; keep geometry)
        if name.startswith("RED_NETHER_BRICK"):
            # e.g. RED_NETHER_BRICK_WALL -> NETHER_BRICK_WALL
            target = "NETHER_BRICK" + name.removeprefix("RED_NETHER_BRICK")
            if target in names:
                return target, "Red nether brick variants are collapsed to the base nether brick family."
        if name.startswith("RED_SANDSTONE"):
            # e.g. RED_SANDSTONE_WALL -> SANDSTONE_WALL
            target = "SANDSTONE" + name.removeprefix("RED_SANDSTONE")
            if target in names:
                return target, "Red sandstone variants are collapsed to the base sandstone family."

        # deepslate ore variants -> normal ore
        if name.startswith("DEEPSLATE_") and name.endswith("_ORE"):
            base = name.removeprefix("DEEPSLATE_")
            if base in names:
                return base, "Deepslate ore variants are treated as the same ore to reduce vocabulary."

        # dead coral variants -> live coral base (limit to coral only)
        if name.startswith("DEAD_") and ("CORAL" in name):
            base = name.removeprefix("DEAD_")
            if base in names:
                return base, "Dead coral variants are treated as the corresponding live coral block."

        # cracked variants -> base (only if base exists)
        if name.startswith("CRACKED_"):
            base = name.removeprefix("CRACKED_")
            if base in names:
                return base, "Cracked variants are treated as the corresponding base block."

        # polished variants -> base (only if base exists)
        if name.startswith("POLISHED_"):
            base = name.removeprefix("POLISHED_")
            if base in names:
                return base, "Polished variants are treated as the corresponding base block."

        # stripped wood/hyphae -> unstripped counterpart
        if name.startswith("STRIPPED_"):
            base = name.removeprefix("STRIPPED_")
            if base in names:
                return base, "Stripped wood/hyphae is a texture variant; map to the unstripped base."

        # cracked deepslate variants -> base (if base exists)
        if name.startswith("CRACKED_DEEPSLATE_"):
            base = name.removeprefix("CRACKED_")
            if base in names:
                return base, "Cracked variants are texture variants; map to the uncracked base."

        # waxed copper -> non-waxed copper (if present)
        if name.startswith("WAXED_"):
            base = name.removeprefix("WAXED_")
            if base in names:
                return base, "Waxed copper is behaviorally similar for our purposes; map to non-waxed base."

        # legacy/new-name flora aliases -> compact old canonical families
        if name == "DANDELION" and "YELLOW_FLOWER" in names:
            return "YELLOW_FLOWER", "Modern flower alias collapsed to the legacy flower family."
        if name in flower_to_red and "RED_FLOWER" in names:
            return "RED_FLOWER", "Flower variants are collapsed to the legacy red-flower family."
        if name in double_plant_like and "DOUBLE_PLANT" in names:
            return "DOUBLE_PLANT", "Tall flora variants are collapsed to the legacy double-plant family."
        if name in grass_like and "TALLGRASS" in names:
            return "TALLGRASS", "Grass/fern variants are collapsed to the legacy tallgrass family."

        return None

    # ---- Rule 3: structural families -> generic base ----
    def rule_structural_to_base(name: str) -> Optional[Tuple[str, str]]:
        if name != "OAK_STAIRS" and name.endswith("_STAIRS") and "OAK_STAIRS" in names:
            return "OAK_STAIRS", "All stair variants are collapsed to OAK_STAIRS."

        if name != "FENCE_GATE" and name.endswith("_FENCE_GATE") and "FENCE_GATE" in names:
            return "FENCE_GATE", "Fence gate variants are collapsed to the base FENCE_GATE class."

        if name not in {"FENCE", "NETHER_BRICK_FENCE"} and name.endswith("_FENCE") and "FENCE" in names:
            return "FENCE", "Fence variants are collapsed to the base FENCE class."

        if name != "WOODEN_DOOR" and name.endswith("_DOOR") and "WOODEN_DOOR" in names:
            return "WOODEN_DOOR", "Door variants are collapsed to the base WOODEN_DOOR class."

        if name != "CHEST" and name.endswith("_CHEST") and "CHEST" in names:
            return "CHEST", "Chest variants are collapsed to CHEST."

        if name != "IRON_BARS" and (name.endswith("_BARS") or name.endswith("_GRATE")) and "IRON_BARS" in names:
            return "IRON_BARS", "Bars/grate variants are collapsed to IRON_BARS."

        if name != "SAPLING" and name.endswith("_SAPLING") and "SAPLING" in names:
            return "SAPLING", "Sapling variants are collapsed to SAPLING."

        if name == "MANGROVE_PROPAGULE" and "SAPLING" in names:
            return "SAPLING", "Mangrove propagules are treated as generic saplings."

        return None

    # ---- Rule 4: copper family -> generic material/base classes ----
    def rule_copper_to_base(name: str) -> Optional[Tuple[str, str]]:
        if "COPPER" not in name:
            return None

        if name.endswith("_ORE"):
            return None

        if name.endswith("_DOOR") and "WOODEN_DOOR" in names:
            return "WOODEN_DOOR", "Copper doors are collapsed to the base WOODEN_DOOR class."

        if name.endswith("_CHEST") and "CHEST" in names:
            return "CHEST", "Copper chests are collapsed to CHEST."

        if (name.endswith("_BARS") or name.endswith("_GRATE")) and "IRON_BARS" in names:
            return "IRON_BARS", "Copper bars/grates are collapsed to IRON_BARS."

        if name.endswith("_BULB") and "REDSTONE_LAMP" in names:
            return "REDSTONE_LAMP", "Copper bulbs are collapsed to REDSTONE_LAMP."

        if any(
            name.endswith(sfx)
            for sfx in ("_CHAIN", "_TORCH", "_WALL_TORCH", "_LANTERN", "_GOLEM_STATUE", "_TRAPDOOR")
        ):
            return "AIR", "Decorative/non-structural copper block removed to AIR."

        if name.endswith("_SLAB"):
            target = _target_if_present(["STONE_SLAB", "WOODEN_SLAB"], names)
            if target is not None:
                return target, "Copper slab variants are collapsed to a generic slab family."

        if name.endswith("_LIGHTNING_ROD"):
            return "AIR", "Lightning rods are treated as decorative details and removed to AIR."

        if "COPPER_BLOCK" in names:
            return "COPPER_BLOCK", "Copper material/state variants are collapsed to COPPER_BLOCK."

        return None

    # ---- Rule 5: wood species -> OAK ----
    _WOOD_SPECIES_PREFIXES = (
        "ACACIA",
        "BIRCH",
        "SPRUCE",
        "JUNGLE",
        "DARK_OAK",
        "MANGROVE",
        "CHERRY",
        "PALE_OAK",
        "BAMBOO",
        "CRIMSON",
        "WARPED",
    )

    def rule_wood_to_oak(name: str) -> Optional[Tuple[str, str]]:
        # Collapse any "<SPECIES>_<THING>" to "OAK_<THING>" if that target exists.
        # This is intentionally broad (planks/logs/wood/stairs/slabs/fences/doors/trapdoors/leaves/etc).
        if name.startswith("OAK_"):
            return None

        # Nether wood uses STEM/HYPHAE terminology; treat those like LOG/WOOD.
        if name.endswith("_STEM") and "OAK_LOG" in names and (name.startswith("CRIMSON_") or name.startswith("WARPED_")):
            return "OAK_LOG", "Nether wood STEM treated like LOG; compressed to OAK_LOG."
        if name.endswith("_HYPHAE") and "OAK_WOOD" in names and (name.startswith("CRIMSON_") or name.startswith("WARPED_")):
            return "OAK_WOOD", "Nether wood HYPHAE treated like WOOD; compressed to OAK_WOOD."

        # Bamboo mosaic doesn't have an OAK_MOSAIC counterpart; treat as planks family.
        if name == "BAMBOO_MOSAIC" and "OAK_PLANKS" in names:
            return "OAK_PLANKS", "Bamboo mosaic treated as planks-family; compressed to OAK_PLANKS."
        if name == "BAMBOO_MOSAIC_SLAB" and "OAK_SLAB" in names:
            return "OAK_SLAB", "Bamboo mosaic slab treated as slab-family; compressed to OAK_SLAB."
        if name == "BAMBOO_MOSAIC_STAIRS" and "OAK_STAIRS" in names:
            return "OAK_STAIRS", "Bamboo mosaic stairs treated as stairs-family; compressed to OAK_STAIRS."

        for pfx in _WOOD_SPECIES_PREFIXES:
            key = pfx + "_"
            if name.startswith(key):
                rest = name[len(key) :]
                target = "OAK_" + rest
                if target in names:
                    return target, f"Wood species compressed to OAK: {pfx} -> OAK."
        return None

    # ---- Rule 6: colors -> base ----
    def rule_color_to_base(name: str) -> Optional[Tuple[str, str]]:
        # NOTE: We intentionally skip slabs/stairs here for the baseline mapping,
        # to avoid accidentally changing sub-block geometry.
        if _is_stair_or_slab(name):
            return None

        # stained glass blocks/panes -> clear glass / glass pane
        for p in _MC_COLOR_PREFIXES:
            if name.startswith(p) and name.endswith("_STAINED_GLASS"):
                if "GLASS" in names:
                    return "GLASS", "Colored stained glass is collapsed to clear GLASS."
            if name.startswith(p) and name.endswith("_STAINED_GLASS_PANE"):
                if "GLASS_PANE" in names:
                    return "GLASS_PANE", "Colored stained glass panes are collapsed to GLASS_PANE."

        # wool -> white wool
        for p in _MC_COLOR_PREFIXES:
            if name.startswith(p) and name.endswith("_WOOL") and "WHITE_WOOL" in names:
                return "WHITE_WOOL", "Colored wool is collapsed to WHITE_WOOL."

        # carpet -> white carpet
        for p in _MC_COLOR_PREFIXES:
            if name.startswith(p) and name.endswith("_CARPET") and "WHITE_CARPET" in names:
                return "WHITE_CARPET", "Colored carpet is collapsed to WHITE_CARPET."

        # concrete / concrete powder -> white variants
        for p in _MC_COLOR_PREFIXES:
            if name.startswith(p) and name.endswith("_CONCRETE") and "WHITE_CONCRETE" in names:
                return "WHITE_CONCRETE", "Colored concrete is collapsed to WHITE_CONCRETE."
            if name.startswith(p) and name.endswith("_CONCRETE_POWDER") and "WHITE_CONCRETE_POWDER" in names:
                return "WHITE_CONCRETE_POWDER", "Colored concrete powder is collapsed to WHITE_CONCRETE_POWDER."

        # terracotta / glazed terracotta -> plain terracotta
        for p in _MC_COLOR_PREFIXES:
            if name.startswith(p) and name.endswith("_TERRACOTTA") and "TERRACOTTA" in names:
                return "TERRACOTTA", "Colored terracotta is collapsed to TERRACOTTA."
            if name.startswith(p) and name.endswith("_GLAZED_TERRACOTTA") and "TERRACOTTA" in names:
                return "TERRACOTTA", "Glazed terracotta is collapsed to plain TERRACOTTA to drop patterns/colors."

        # candles -> base candle (keep presence, drop color)
        for p in _MC_COLOR_PREFIXES:
            if name.startswith(p) and name.endswith("_CANDLE") and "CANDLE" in names:
                return "CANDLE", "Colored candles are collapsed to CANDLE."

        # beds are non-full blocks and mostly decorative for our rendering; drop them by default
        for p in _MC_COLOR_PREFIXES:
            if name.startswith(p) and name.endswith("_BED"):
                return "AIR", "Beds are treated as non-structural decoration; map to AIR."

        # shulker boxes -> chest (keep 'container' vibe, drop color/type)
        for p in _MC_COLOR_PREFIXES:
            if name.startswith(p) and name.endswith("_SHULKER_BOX") and "CHEST" in names:
                return "CHEST", "Colored shulker boxes are collapsed to CHEST."

        return None

    # ---- Apply rules with precedence, allowing chaining ----
    simple: Dict[str, str] = {}
    meta: Dict[str, Dict[str, str]] = {}

    for _bid in sorted(id_to_name.keys()):
        src = id_to_name[_bid]
        cur = src
        steps: List[Dict[str, str]] = []

        # Hard cap to avoid accidental loops
        for _iter in range(10):
            changed = False

            r = rule_useless_to_air(cur)
            if r is not None:
                nxt, why = r
                steps.append({"from": cur, "to": nxt, "category": "useless_to_air", "reason": why})
                cur = nxt
                changed = True
                # Deleting to AIR is terminal
                break

            r = rule_variant_to_base(cur)
            if r is not None:
                nxt, why = r
                steps.append({"from": cur, "to": nxt, "category": "variant_to_base", "reason": why})
                cur = nxt
                changed = True
                continue

            r = rule_structural_to_base(cur)
            if r is not None:
                nxt, why = r
                steps.append({"from": cur, "to": nxt, "category": "variant_to_base", "reason": why})
                cur = nxt
                changed = True
                continue

            r = rule_copper_to_base(cur)
            if r is not None:
                nxt, why = r
                cat = "variant_to_base" if nxt != "AIR" else "useless_to_air"
                steps.append({"from": cur, "to": nxt, "category": cat, "reason": why})
                cur = nxt
                changed = True
                if nxt == "AIR":
                    break
                continue

            r = rule_wood_to_oak(cur)
            if r is not None:
                nxt, why = r
                steps.append({"from": cur, "to": nxt, "category": "wood_to_oak", "reason": why})
                cur = nxt
                changed = True
                continue

            r = rule_color_to_base(cur)
            if r is not None:
                nxt, why = r
                cat = "color_to_base" if nxt != "AIR" else "useless_to_air"
                steps.append({"from": cur, "to": nxt, "category": cat, "reason": why})
                cur = nxt
                changed = True
                if nxt == "AIR":
                    break
                continue

            if not changed:
                break

        dst = cur
        src_count = None if counts_by_name is None else counts_by_name.get(src)
        if (
            dst != "AIR"
            and src_count is not None
            and int(rare_count_threshold) > 0
            and int(src_count) < int(rare_count_threshold)
        ):
            steps.append(
                {
                    "from": cur,
                    "to": "AIR",
                    "category": "rare_to_air",
                    "reason": f"Observed only {int(src_count)} times, below rare-count threshold {int(rare_count_threshold)}.",
                }
            )
            dst = "AIR"
        if len(steps) == 0:
            category = "identity"
            reason = "No compression applied (identity mapping)."
        else:
            category = steps[-1]["category"]
            reason = " ; ".join([f'{s["category"]}: {s["reason"]}' for s in steps])

        simple[src] = dst
        meta[src] = {
            "to": dst,
            "category": category,
            "reason": reason,
            "steps": steps,
        }

    return simple, meta


def write_json_file(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
        f.write("\n")

def build_id_compression_mapping(
    user_id_to_name: Dict[int, str],
    name_to_name: Dict[str, str],
    *,
    target_name_to_id: Optional[Dict[str, int]] = None,
    air_id_fallback: int = 5,
) -> Tuple[Dict[str, int], Dict[str, int], int]:
    """
    Convert a name->name compression mapping into an id->id mapping.

    - Source IDs come from `user_id_to_name` (your dataset's ID space).
    - Target IDs come from `target_name_to_id` if provided (recommended: `block_types_updated.json`,
      where AIR is ID 5). If not provided, targets are resolved within the user ID space.

    Returns:
      - id_to_id: { "<src_id>": <dst_id>, ... }    (keys are strings for JSON consistency)
      - name_to_id: { "<NAME>": <id>, ... }
      - air_id: numeric AIR id used for deletions

    AIR handling:
      - If "AIR" exists in the *target* mapping, we use that ID.
      - Otherwise, we use `air_id_fallback` (default: 5).
    """
    name_to_id = {name: int(bid) for bid, name in user_id_to_name.items()}
    if target_name_to_id is None:
        target_name_to_id = name_to_id
    air_id = int(target_name_to_id.get("AIR", int(air_id_fallback)))

    id_to_id: Dict[str, int] = {}
    for bid, src_name in user_id_to_name.items():
        if src_name not in name_to_name:
            raise ValueError(f"Missing compression mapping entry for source block name: {src_name}")
        dst_name = name_to_name[src_name]
        if dst_name == "AIR":
            dst_id = air_id
        else:
            if dst_name not in target_name_to_id:
                raise ValueError(
                    f"Compression mapping targets '{dst_name}', but it is not present in target block types. "
                    "Either add it to the block list or change the mapping."
                )
            dst_id = int(target_name_to_id[dst_name])
        id_to_id[str(int(bid))] = int(dst_id)

    return id_to_id, name_to_id, air_id


def write_compression_mapping_md(
    out_path: Path,
    id_to_name: Dict[int, str],
    name_to_target: Dict[str, str],
    meta: Dict[str, Dict[str, str]],
    *,
    target_name_to_id: Optional[Dict[str, int]] = None,
    air_id: int = 5,
) -> None:
    src_name_to_id = {v: int(k) for k, v in id_to_name.items()}
    if target_name_to_id is None:
        target_name_to_id = src_name_to_id
    rows = []
    for bid in sorted(id_to_name.keys()):
        src = id_to_name[bid]
        dst = name_to_target[src]
        dst_id = air_id if dst == "AIR" else target_name_to_id.get(dst)
        m = meta.get(src, {})
        rows.append((bid, src, dst, dst_id, m.get("category", ""), m.get("reason", "")))

    # Summary counts
    cat_counts: Dict[str, int] = {}
    for _bid, _src, _dst, _dst_id, cat, _reason in rows:
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# Block compression mapping (user data)\n\n")
        f.write("This document is auto-generated from `assets/user_data_block_types.json`.\n\n")
        f.write("## Notes\n\n")
        f.write("- The mapping is intentionally conservative and intended as a starting point.\n")
        f.write("- **Stairs and slabs are never deleted to AIR** to preserve sub-block geometry.\n")
        f.write(f"- Deletions map to canonical **AIR_ID={air_id}** (from `block_types_updated.json`).\n\n")

        f.write("## Category summary\n\n")
        for k in sorted(cat_counts.keys()):
            f.write(f"- **{k}**: {cat_counts[k]}\n")
        f.write("\n")

        f.write("## Full mapping table\n\n")
        f.write("| source_id | source_name | target_name | target_id | category | rationale |\n")
        f.write("| ---: | --- | --- | ---: | --- | --- |\n")
        for bid, src, dst, dst_id, cat, reason in rows:
            dst_id_s = "" if dst_id is None else str(dst_id)
            # escape pipes to keep markdown table intact
            reason_s = (reason or "").replace("|", "\\|")
            f.write(f"| {bid} | {src} | {dst} | {dst_id_s} | {cat} | {reason_s} |\n")


def write_surviving_block_types_json(
    out_path: Path,
    id_to_name: Dict[int, str],
    name_to_target: Dict[str, str],
    *,
    target_name_to_id: Optional[Dict[str, int]] = None,
) -> None:
    target_to_sources: Dict[str, List[str]] = {}
    for src_name in id_to_name.values():
        dst_name = name_to_target[src_name]
        if dst_name == "AIR":
            continue
        target_to_sources.setdefault(dst_name, []).append(src_name)

    entries: List[Dict[str, Any]] = []
    for target_name in sorted(target_to_sources.keys()):
        source_names = sorted(target_to_sources[target_name])
        target_id = None if target_name_to_id is None else target_name_to_id.get(target_name)
        entries.append(
            {
                "target_name": target_name,
                "target_id": None if target_id is None else int(target_id),
                "source_count": len(source_names),
                "source_names": source_names,
            }
        )

    payload = {
        "surviving_block_count": len(entries),
        "surviving_block_types": entries,
    }
    write_json_file(out_path, payload)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze block frequency report for remapping/compression.")
    ap.add_argument(
        "--freq-json",
        type=Path,
        default=Path("assets/block_frequency_report.json"),
        help="Path to the parsed block frequency JSON",
    )
    ap.add_argument(
        "--threshold",
        type=int,
        default=1000,
        help="Count threshold for 'rare' blocks (default: 1000)",
    )
    ap.add_argument(
        "--write-user-block-types",
        action="store_true",
        help="If set, writes user_data_block_types.json derived from the frequency report",
    )
    ap.add_argument(
        "--user-block-types-out",
        type=Path,
        default=Path("assets/block_types_updated.json"),
        help="Output path for the generated user block types JSON",
    )

    ap.add_argument(
        "--user-block-types-in",
        type=Path,
        default=Path("assets/block_types_updated.json"),
        help="Input path for user block types JSON (id->name) to generate a compression mapping from",
    )
    ap.add_argument(
        "--write-compression-mapping",
        action="store_true",
        help="If set, writes block compression mapping JSON + Markdown for the provided user block types",
    )
    ap.add_argument(
        "--compression-json-out",
        type=Path,
        default=Path("assets/block_compression_mapping.json"),
        help="Output path for the compression mapping JSON (id->id)",
    )
    ap.add_argument(
        "--compression-name-json-out",
        type=Path,
        default=Path("assets/block_compression_mapping_by_name.json"),
        help="Output path for the human-readable compression mapping JSON (name->name)",
    )
    ap.add_argument(
        "--compression-meta-json-out",
        type=Path,
        default=Path("assets/block_compression_mapping_meta.json"),
        help="Output path for the meta compression mapping JSON (keyed by source_id; includes names, ids, category, reasons)",
    )
    ap.add_argument(
        "--compression-md-out",
        type=Path,
        default=Path("assets/block_compression_mapping.md"),
        help="Output path for the Markdown documentation of the compression mapping",
    )
    ap.add_argument(
        "--surviving-block-types-out",
        type=Path,
        default=Path("assets/block_surviving_types.json"),
        help="Output path for the surviving post-remap block types JSON",
    )
    ap.add_argument(
        "--map-below-count-to-air",
        type=int,
        default=100,
        help="Map source block types observed fewer than this count to AIR after remapping (default: 100, use 0 to disable).",
    )
    ap.add_argument(
        "--target-block-types-in",
        type=Path,
        default=Path("assets/block_types_updated.json"),
        help="Target block types JSON (id->name) used to resolve target IDs (canonical, AIR=5).",
    )
    ap.add_argument(
        "--air-id-fallback",
        type=int,
        default=5,
        help="If 'AIR' is missing from target block types, use this numeric ID for deletions-to-air (default: 5).",
    )
    args = ap.parse_args()

    rows = load_frequency_rows(args.freq_json)
    counts_by_name = rows_to_counts(rows)
    n_total, n_zero, n_under, n_overeq = bucket_counts(counts_by_name, args.threshold)

    print("Block frequency threshold summary (from frequency report only)")
    print("------------------------------------------------------------")
    print(f"Total blocks in report:        {n_total}")
    print(f"count == 0:                    {n_zero}")
    print(f"0 < count < {args.threshold}:              {n_under}")
    print(f"count >= {args.threshold}:                 {n_overeq}")

    if args.write_user_block_types:
        write_user_data_block_types(rows, args.user_block_types_out)
        print("")
        print(f"Wrote user block types JSON:   {args.user_block_types_out}")

    if args.write_compression_mapping:
        id_to_name = load_user_block_types_map(args.user_block_types_in)
        name_to_name, meta_by_name = _build_compression_mapping_for_user_blocks(
            id_to_name,
            counts_by_name,
            rare_count_threshold=int(args.map_below_count_to_air),
        )

        target_id_to_name = load_user_block_types_map(args.target_block_types_in)
        target_name_to_id = {name: int(bid) for bid, name in target_id_to_name.items()}

        id_to_id, _src_name_to_id, air_id = build_id_compression_mapping(
            id_to_name,
            name_to_name,
            target_name_to_id=target_name_to_id,
            air_id_fallback=int(args.air_id_fallback),
        )

        # Write primary mapping as id->id for applying directly to voxel IDs
        write_json_file(args.compression_json_out, id_to_id)
        # Keep the name->name mapping as a separate convenience artifact
        write_json_file(args.compression_name_json_out, name_to_name)

        # Build a meta mapping keyed by source_id for easy programmatic use
        meta_by_id: Dict[str, Dict[str, Any]] = {}
        for bid, src_name in id_to_name.items():
            bid_s = str(int(bid))
            dst_name = name_to_name[src_name]
            dst_id = air_id if dst_name == "AIR" else target_name_to_id[dst_name]
            mbn = meta_by_name.get(src_name, {})
            meta_by_id[bid_s] = {
                "from": src_name,
                "to": dst_name,
                "to_id": int(dst_id),
                "category": mbn.get("category", "identity"),
                "reason": mbn.get("reason", ""),
                "steps": mbn.get("steps", []),
            }
        write_json_file(args.compression_meta_json_out, meta_by_id)

        write_compression_mapping_md(
            args.compression_md_out,
            id_to_name,
            name_to_name,
            meta_by_name,
            target_name_to_id=target_name_to_id,
            air_id=air_id,
        )
        write_surviving_block_types_json(
            args.surviving_block_types_out,
            id_to_name,
            name_to_name,
            target_name_to_id=target_name_to_id,
        )
        print("")
        print(f"Wrote compression mapping JSON (id->id):      {args.compression_json_out}")
        print(f"Wrote compression mapping JSON (name->name):  {args.compression_name_json_out}")
        print(f"Wrote compression mapping meta JSON: {args.compression_meta_json_out}")
        print(f"Wrote compression mapping MD:        {args.compression_md_out}")
        print(f"Wrote surviving block types JSON:   {args.surviving_block_types_out}")
        if int(args.map_below_count_to_air) > 0:
            print(f"Applied rare-to-AIR threshold:      < {int(args.map_below_count_to_air)}")


if __name__ == "__main__":
    main()


