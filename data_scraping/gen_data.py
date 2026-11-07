from functools import partial
import glob
from math import cos, sin
import json
import math
import os
from pdb import set_trace as TT
import time

from einops import rearrange
from fire import Fire
import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from tqdm import tqdm

from utils import get_view_rot, square_spiral, get_vox_xz_from_view, patch_grpc_evocraft_imports
from seed_utils import find_villages_near_spawn, find_biome_locations

patch_grpc_evocraft_imports()

import clients.python.src.main.proto.minecraft_pb2_grpc
from clients.python.src.main.proto.minecraft_pb2 import *



SCREENSHOT_SIZE = (512, 512)
BBOX_OFFSET = (0, 66)  # For 16" M1 MacBook Pro
BBOX = (BBOX_OFFSET[0], BBOX_OFFSET[1], BBOX_OFFSET[0] + SCREENSHOT_SIZE[0], BBOX_OFFSET[1] + SCREENSHOT_SIZE[1])

EVOCRAFT = 0
MINEDOJO = 1
MC_API = EVOCRAFT

if MC_API == MINEDOJO:
    import minedojo
if MC_API == EVOCRAFT:
    import grpc


class MinedojoClient:
    def __init__(self):
        env = minedojo.make(
            task_id="open-ended", 
            image_size=(512, 512), 
            world_seed=420
        )
        _ = env.reset()
        for _ in range(20):
            # warm up env
            env.step(env.action_space.no_op())


def top_left_corner_screenshot(name: str, bbox: tuple, save_dir: str):
    im = ImageGrab.grab(bbox)
    # Save image to file
    if np.array(im).shape != (512, 512, 4):
        raise Exception(f"Screenshot is not 512x512, size is {np.array(im).shape}. Has the display gone to sleep?")
    im.save(os.path.join(save_dir, f"{name}.png"))
        


def cube_to_voxels(cube: Cube, shape: tuple, min_xyz: tuple):
    voxels = np.zeros(shape, dtype=np.uint8)
    for block in cube.blocks:
        # if block.type != AIR:
        #     voxels[block.position.x - min_xyz[0], block.position.y - min_xyz[1], block.position.z - min_xyz[2]] = 1
        voxels[block.position.x - min_xyz[0], block.position.y - min_xyz[1], block.position.z - min_xyz[2]] = block.type
    return voxels

SIMPLE_BIOME_MAPPING = {
    'ocean': 'ocean',
    'deep_ocean': 'ocean',
    'deep_warm_ocean': 'ocean',
    'desert': 'desert',
    'desert_hills': 'desert',
    'mutated_desert': 'desert',
    'beaches': 'beaches',
    'stone_beach': 'beaches',
    'cave': 'cave',
    'extreme_hills': 'extreme_hills',
    'extreme_hills_with_trees': 'extreme_hills',
    'mutated_extreme_hills_with_trees': 'extreme_hills',
    'mutated_extreme_hills': 'extreme_hills',
    'forest': 'forest',
    'forest_hills': 'forest',
    'mutated_forest': 'forest',
    'mutated_roofed_forest': 'forest',
    'roofed_forest': 'forest',
    'birch_forest': 'birch_forest',
    'birch_forest_hills': 'birch_forest',
    'mutated_birch_forest': 'birch_forest',
    'mutated_birch_forest_hills': 'birch_forest',
    'plains': 'plains',
    'mutated_plains': 'plains',
    'river': 'river',
    'savanna': 'savanna',
    'savanna_rock': 'savanna',
    'mutated_savanna_rock': 'savanna',
    'mutated_savanna': 'savanna',
    'swampland': 'swampland',
    'mutated_swampland': 'swampland',
    'taiga': 'taiga',
    'taiga_hills': 'taiga',
    'mutated_redwood_taiga': 'taiga',
    'mutated_redwood_taiga_hills': 'taiga',
    'redwood_taiga_hills': 'taiga',
    'mutated_taiga': 'taiga',
    'redwood_taiga': 'taiga',
    'ice_flats': "ice",
    'ice_mountain': "ice",
    'cold_beach': "ice",
    'frozen_river': "ice",
    'mutated_ice_flats': "ice",
    'mutated_taiga_cold': "ice",
    'taiga_cold': "ice",
    'taiga_cold_hills': "ice",
    'ice_mountains': "ice",
    'jungle': 'jungle',
    'jungle_edge': 'jungle',
    'jungle_hills': 'jungle',
    'mutated_jungle': 'jungle',
    'mesa_rock': 'mesa',
    'mesa': 'mesa',
    'mesa_clear_rock': 'mesa',
    'mutated_mesa': 'mesa',
    'mutated_mesa_rock': 'mesa',
    'mushroom_island': 'mushroom_island',
    'mushroom_island_shore': 'mushroom_island',
}

# Mapping from MC 1.12 biome string names (used in SIMPLE_BIOME_MAPPING keys) to Pyubiomes biome IDs
# This bridges the naming conventions between MC 1.12 and the Pyubiomes library
MC112_BIOME_TO_PYUBIOMES_ID = {
    # Ocean variants
    'ocean': 0,
    'deep_ocean': 24,
    'deep_warm_ocean': 47,
    # Desert variants
    'desert': 2,
    'desert_hills': 17,
    'mutated_desert': 130,  # desert_lakes in Pyubiomes
    # Beaches
    'beaches': 16,  # beach in Pyubiomes
    'stone_beach': 25,  # stone_shore in Pyubiomes
    # Extreme hills / Mountains
    'extreme_hills': 3,  # mountains in Pyubiomes
    'extreme_hills_with_trees': 34,  # wooded_mountains in Pyubiomes
    'mutated_extreme_hills_with_trees': 162,  # modified_gravelly_mountains in Pyubiomes
    'mutated_extreme_hills': 131,  # gravelly_mountains in Pyubiomes
    # Forest variants
    'forest': 4,
    'forest_hills': 18,  # wooded_hills in Pyubiomes
    'mutated_forest': 132,  # flower_forest in Pyubiomes
    'mutated_roofed_forest': 157,  # dark_forest_hills in Pyubiomes
    'roofed_forest': 29,  # dark_forest in Pyubiomes
    # Birch forest variants
    'birch_forest': 27,
    'birch_forest_hills': 28,
    'mutated_birch_forest': 155,  # tall_birch_forest in Pyubiomes
    'mutated_birch_forest_hills': 156,  # tall_birch_hills in Pyubiomes
    # Plains
    'plains': 1,
    'mutated_plains': 129,  # sunflower_plains in Pyubiomes
    # River
    'river': 7,
    # Savanna variants
    'savanna': 35,
    'savanna_rock': 36,  # savanna_plateau in Pyubiomes
    'mutated_savanna_rock': 164,  # shattered_savanna_plateau in Pyubiomes
    'mutated_savanna': 163,  # shattered_savanna in Pyubiomes
    # Swampland
    'swampland': 6,  # swamp in Pyubiomes
    'mutated_swampland': 134,  # swamp_hills in Pyubiomes
    # Taiga variants
    'taiga': 5,
    'taiga_hills': 19,
    'mutated_redwood_taiga': 160,  # giant_spruce_taiga in Pyubiomes
    'mutated_redwood_taiga_hills': 161,  # giant_spruce_taiga_hills in Pyubiomes
    'redwood_taiga_hills': 33,  # giant_tree_taiga_hills in Pyubiomes
    'mutated_taiga': 133,  # taiga_mountains in Pyubiomes
    'redwood_taiga': 32,  # giant_tree_taiga in Pyubiomes
    # Ice / Cold biomes
    'ice_flats': 12,  # snowy_tundra in Pyubiomes
    'ice_mountain': 13,  # snowy_mountains in Pyubiomes
    'ice_mountains': 13,  # duplicate spelling in SIMPLE_BIOME_MAPPING
    'cold_beach': 26,  # snowy_beach in Pyubiomes
    'frozen_river': 11,
    'mutated_ice_flats': 140,  # ice_spikes in Pyubiomes
    'mutated_taiga_cold': 158,  # snowy_taiga_mountains in Pyubiomes
    'taiga_cold': 30,  # snowy_taiga in Pyubiomes
    'taiga_cold_hills': 31,  # snowy_taiga_hills in Pyubiomes
    # Jungle variants
    'jungle': 21,
    'jungle_edge': 23,
    'jungle_hills': 22,
    # Mesa / Badlands variants
    'mesa_rock': 38,  # wooded_badlands_plateau in Pyubiomes
    'mesa': 37,  # badlands in Pyubiomes
    'mesa_clear_rock': 39,  # badlands_plateau in Pyubiomes
    'mutated_mesa': 165,  # eroded_badlands in Pyubiomes
}


def get_pyubiomes_ids_for_simple_biome(simple_biome_label: str) -> list:
    """
    Given a simple biome label (e.g., 'plains'), return all Pyubiomes biome IDs
    that map to that label according to SIMPLE_BIOME_MAPPING.
    """
    biome_ids = []
    for mc_biome_name, simple_label in SIMPLE_BIOME_MAPPING.items():
        if simple_label == simple_biome_label:
            pyubiomes_id = MC112_BIOME_TO_PYUBIOMES_ID.get(mc_biome_name)
            if pyubiomes_id is not None:
                biome_ids.append(pyubiomes_id)
    return biome_ids


## Removed: indicator blocks are no longer used

VILLAGE_STRUCTURE_BLOCKS = {
    # Core structural blocks for edge checks and structure thresholds
    BlockType.PLANKS,
    BlockType.COBBLESTONE,
    # Slabs commonly used in roofs/floors
    BlockType.WOODEN_SLAB,
    BlockType.STONE_SLAB,
    BlockType.DOUBLE_STONE_SLAB,
    BlockType.STONE_BRICK_STAIRS,
    BlockType.STONE_STAIRS,
    BlockType.BRICK_STAIRS,
    BlockType.SANDSTONE_STAIRS,
    BlockType.RED_SANDSTONE_STAIRS,
    # BlockType.QUARTZ_STAIRS,
    # BlockType.NETHER_BRICK_STAIRS,
    # BlockType.PURPUR_STAIRS,
    BlockType.OAK_STAIRS,
    BlockType.BIRCH_STAIRS,
    BlockType.SPRUCE_STAIRS,
    BlockType.JUNGLE_STAIRS,
    BlockType.DARK_OAK_STAIRS,
    BlockType.ACACIA_STAIRS,
    # BlockType.COBBLESTONE_STAIRS,
    BlockType.GLASS_PANE,
}


# Only structure blocks are used for counting and edge checks
VILLAGE_BLOCKS_FOR_COUNT = VILLAGE_STRUCTURE_BLOCKS

def cube_to_voxels_and_biomes(cube, shape, min_xyz):
    voxels = np.zeros(shape, dtype=np.uint8)
    biomes = np.zeros(shape, dtype=object)
    metadata = np.empty(shape, dtype=object)
    metadata[:] = None

    for block in cube.blocks:
        x_idx = block.position.x - min_xyz[0]
        y_idx = block.position.y - min_xyz[1]
        z_idx = block.position.z - min_xyz[2]
        voxels[x_idx, y_idx, z_idx] = block.type
        biome_label = block.biome
        biomes[x_idx, y_idx, z_idx] = biome_label

        # capture full metadata map (store dict per block)
        if hasattr(block, "metadata"):
            try:
                md = dict(block.metadata)
            except Exception:
                md = {}
            metadata[x_idx, y_idx, z_idx] = md

    return voxels, biomes, metadata


# ----------------------- IO Utilities (atomic + combine) ----------------------- #
def _atomic_savez_compressed(file_path: str, **arrays):
    """Safely write npz by writing to a temp file then atomic replace."""
    try:
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
    except Exception:
        pass

    pid = os.getpid()
    ts = int(time.time() * 1000)
    tmp_path = f"{file_path}.tmp_{pid}_{ts}.npz"
    try:
        np.savez_compressed(tmp_path, **arrays)
        os.replace(tmp_path, file_path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _atomic_savez(file_path: str, compress: bool, **arrays):
    """Atomic save with optional compression."""
    try:
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
    except Exception:
        pass
    pid = os.getpid()
    ts = int(time.time() * 1000)
    tmp_path = f"{file_path}.tmp_{pid}_{ts}.npz"
    try:
        if compress:
            np.savez_compressed(tmp_path, **arrays)
        else:
            np.savez(tmp_path, **arrays)
        os.replace(tmp_path, file_path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

def _combine_npz_files(npz_paths, out_path: str):
    """Combine multiple chunk npz files into one.

    Supports legacy per-voxel biomes under key 'biomes' and the newer
    compact per-sample labels under key 'biome_labels'. If any input file
    contains 'biome_labels', the combined output will write 'biome_labels'.
    Otherwise, it preserves legacy 'biomes'.
    """
    if not npz_paths:
        return False
    all_labels, all_voxels, all_biomes, all_biome_labels, all_metadata = [], [], [], [], []
    any_metadata = False
    for p in npz_paths:
        with np.load(p, allow_pickle=True) as data:
            if 'labels' in data.files:
                all_labels.append(data['labels'])
            if 'voxels' in data.files:
                all_voxels.append(data['voxels'])
            if 'biome_labels' in data.files:
                all_biome_labels.append(data['biome_labels'])
            elif 'biomes' in data.files:
                # Legacy per-voxel biomes; collapse each sample to a single 'village' label
                try:
                    n_samples = int(data['biomes'].shape[0])
                    all_biome_labels.append(np.full((n_samples,), 'village', dtype=object))
                except Exception:
                    all_biomes.append(data['biomes'])
            if 'metadata' in data.files:
                all_metadata.append(data['metadata'])
                any_metadata = True
    if not all_labels:
        return False
    out_labels = np.concatenate(all_labels, axis=0) if all_labels else None
    out_voxels = np.concatenate(all_voxels, axis=0) if all_voxels else None

    # Prefer compact labels if available; otherwise keep legacy biomes
    out_biome_labels = np.concatenate(all_biome_labels, axis=0) if all_biome_labels else None
    out_biomes = np.concatenate(all_biomes, axis=0) if (out_biome_labels is None and all_biomes) else None

    if any_metadata and all_metadata:
        out_metadata = np.concatenate(all_metadata, axis=0)
        if out_biome_labels is not None:
            _atomic_savez_compressed(out_path, labels=out_labels, voxels=out_voxels, biome_labels=out_biome_labels, metadata=out_metadata)
        else:
            _atomic_savez_compressed(out_path, labels=out_labels, voxels=out_voxels, biomes=out_biomes, metadata=out_metadata)
    else:
        if out_biome_labels is not None:
            _atomic_savez_compressed(out_path, labels=out_labels, voxels=out_voxels, biome_labels=out_biome_labels)
        else:
            _atomic_savez_compressed(out_path, labels=out_labels, voxels=out_voxels, biomes=out_biomes)
    return True


def _append_npz_part_to_out(out_path: str, part_path: str, *, include_biomes: bool = True, include_metadata: bool = False, compress: bool = False):
    """Append a single part npz into the combined out npz sequentially (lower peak memory).

    Returns number of samples appended (labels length) or 0 on failure.
    """
    try:
        with np.load(part_path, allow_pickle=True) as p:
            pl = p['labels'] if 'labels' in p.files else None
            pv = p['voxels'] if 'voxels' in p.files else None
            pb = p['biomes'] if include_biomes and 'biomes' in p.files else None
            pm = p['metadata'] if include_metadata and 'metadata' in p.files else None
        if pl is None:
            return 0
        if os.path.isfile(out_path):
            with np.load(out_path, allow_pickle=True) as ex:
                el = ex['labels'] if 'labels' in ex.files else None
                ev = ex['voxels'] if 'voxels' in ex.files else None
                eb = ex['biomes'] if include_biomes and 'biomes' in ex.files else None
                em = ex['metadata'] if include_metadata and 'metadata' in ex.files else None
            out_labels = np.concatenate([el, pl], axis=0) if el is not None else pl
            out_voxels = np.concatenate([ev, pv], axis=0) if ev is not None else pv
            out_biomes = np.concatenate([eb, pb], axis=0) if include_biomes and (eb is not None or pb is not None) else None
            if include_metadata and (em is not None or pm is not None):
                out_metadata = np.concatenate([em, pm], axis=0) if em is not None else pm
                _atomic_savez(out_path, compress=compress, labels=out_labels, voxels=out_voxels, biomes=(out_biomes if include_biomes else None), metadata=out_metadata)
            else:
                if include_biomes and out_biomes is not None:
                    _atomic_savez(out_path, compress=compress, labels=out_labels, voxels=out_voxels, biomes=out_biomes)
                else:
                    _atomic_savez(out_path, compress=compress, labels=out_labels, voxels=out_voxels)
        else:
            if include_metadata and pm is not None:
                _atomic_savez(out_path, compress=compress, labels=pl, voxels=pv, biomes=(pb if include_biomes else None), metadata=pm)
            else:
                if include_biomes and pb is not None:
                    _atomic_savez(out_path, compress=compress, labels=pl, voxels=pv, biomes=pb)
                else:
                    _atomic_savez(out_path, compress=compress, labels=pl, voxels=pv)
        return int(pl.shape[0])
    except Exception:
        return 0


def read_cube(client, min_xyz: tuple, max_xyz: tuple, preload_chunks: bool = True):
    return client.readCube(Cube(min=Point(x=min_xyz[0], y=min_xyz[1], z=min_xyz[2]),
                                max=Point(x=max_xyz[0], y=max_xyz[1], z=max_xyz[2]),
                                preload_chunks=preload_chunks))


def get_vox_chunk(loc, shp, client):
    x, y, z = loc
    min = (x - math.floor(shp[0] // 2), 
           y - math.floor(shp[1] // 2),
           z - math.floor(shp[2] // 2))
    max = (x + math.ceil(shp[0] // 2) - 1, 
        y + math.ceil(shp[1] // 2) - 1, 
        z + math.ceil(shp[2] // 2) - 1)
    cube = client.readCubeAndBiomeMetadata(Cube(min=Point(x=min[0], y=min[1], z=min[2]),
                                      max=Point(x=max[0], y=max[1], z=max[2]),
                                      preload_chunks=True))
    return cube_to_voxels_and_biomes(cube, shp, min)


def read_macro_and_iter_subchunks(client, center_xyz: tuple, chunk_shape: tuple, factor_xz: int):
    """Read a larger cube at once and yield subchunks of size chunk_shape.

    center_xyz: (x, y, z) center position at which we normally read a chunk.
    chunk_shape: (sx, sy, sz) desired subchunk shape.
    factor_xz: expand factor in x and z (1, 2, or 4). y unchanged.

    Yields: tuples (labels, vox, biomes, metadata) for each subchunk.
    """
    if factor_xz <= 1:
        vox, bio, md = get_vox_chunk(center_xyz, chunk_shape, client)
        yield np.array(center_xyz), vox, bio, md
        return

    sx, sy, sz = chunk_shape
    cx, cy, cz = center_xyz

    total_x = sx * factor_xz
    total_z = sz * factor_xz

    xmin = cx - (total_x // 2)
    xmax = xmin + total_x - 1
    ymin = cy - (sy // 2)
    ymax = ymin + sy - 1
    zmin = cz - (total_z // 2)
    zmax = zmin + total_z - 1

    cube = client.readCubeAndBiomeMetadata(Cube(min=Point(x=xmin, y=ymin, z=zmin), max=Point(x=xmax, y=ymax, z=zmax)))
    grid_vox, grid_bio, grid_md = cube_to_voxels_and_biomes(cube, (xmax-xmin+1, ymax-ymin+1, zmax-zmin+1), (xmin, ymin, zmin))

    for fx in range(factor_xz):
        for fz in range(factor_xz):
            xs = fx * sx
            zs = fz * sz
            sub_vox = grid_vox[xs:xs+sx, 0:sy, zs:zs+sz]
            sub_bio = grid_bio[xs:xs+sx, 0:sy, zs:zs+sz]
            sub_md = grid_md[xs:xs+sx, 0:sy, zs:zs+sz]
            sub_cx = xmin + xs + sx // 2
            sub_cy = ymin + sy // 2
            sub_cz = zmin + zs + sz // 2
            yield np.array([sub_cx, sub_cy, sub_cz]), sub_vox, sub_bio, sub_md


def get_biome_array(client, x_start, y, z_start, chunk_shape, stride=3):
    """
    Creates a biome array for a chunk by sampling biomes in the x-z plane and extending vertically.
    
    Args:
        client: Minecraft client connection
        x_start, y, z_start: Starting coordinates of the chunk
        chunk_shape: (x, y, z) shape of the chunk
        stride: Size of the sampling window (stride x stride blocks will share same biome)
    
    Returns:
        numpy array of biomes with shape (x, y, z) matching voxel data
    """
    x_size, y_size, z_size = chunk_shape
    
    # Sample the x-z plane
    x_windows = (x_size + stride - 1) // stride
    z_windows = (z_size + stride - 1) // stride
    
    # Initialize the x-z plane
    biome_plane = np.zeros((x_size, z_size), dtype=object)
    
    # Sample biomes in x-z plane
    for x_idx in range(x_windows):
        for z_idx in range(z_windows):
            sample_x = x_start + (x_idx * stride + stride // 2)
            sample_z = z_start + (z_idx * stride + stride // 2)
            
            biome_response = client.getBiomeAt(Point(x=sample_x, y=y, z=sample_z)).biome
            biome = biome_response.split(':')[1]
            
            x_start_idx = x_idx * stride
            x_end_idx = min((x_idx + 1) * stride, x_size)
            z_start_idx = z_idx * stride
            z_end_idx = min((z_idx + 1) * stride, z_size)
            
            biome_plane[x_start_idx:x_end_idx, z_start_idx:z_end_idx] = biome
    
    # Extend vertically by adding the y dimension in the middle (x, y, z)
    full_biome_array = np.zeros((x_size, y_size, z_size), dtype=object)
    for y_idx in range(y_size):
        full_biome_array[:, y_idx, :] = biome_plane
    
    return full_biome_array


def get_voxels(client, cfg, data_dir):
    """Collect chunks from a user-created map and save into a single map dataset.

    - Reads one chunk per spiral coordinate with small Y-shift around surface.
    - Applies an air-ratio filter to bias toward above-ground content.
    - Writes per-run part files, then combines into one compressed file named
      "{map_name}_chunks.npz" inside Voxels/.
    - Each sample's biome label is replaced with the configured map name.
    """
    map_name = str(getattr(cfg.data, 'map_name', 'unnamed_map'))
    print(f'Collecting {cfg.data.num_samples} chunks for map: {map_name}')

    # Step increments (no macro subchunking)
    x_increment = int(cfg.data.chunk_shape[0])
    z_increment = int(cfg.data.chunk_shape[2])
    print(f'Increments:\n x_increment: {x_increment}\nz_increment: {z_increment}')

    # Output directories
    base_dir = os.path.join(cfg.data.data_dir, "Voxels")
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)
    part_dir = os.path.join(base_dir, f"{map_name}_parts")
    if not os.path.isdir(part_dir):
        os.makedirs(part_dir)

    # Start index (optional override via cfg)
    col_i = int(cfg.data_gen.i) if getattr(cfg.data_gen, 'i', None) not in (None, "",) else 0

    # Per-run tag to avoid part collisions across seeds/restarts
    run_tag = None
    try:
        props_path = os.path.join("server", "server.properties")
        if os.path.isfile(props_path):
            with open(props_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("level-seed="):
                        val = line.split("=", 1)[1].strip()
                        run_tag = f"seed_{val}"
                        break
    except Exception:
        run_tag = None
    if not run_tag:
        run_tag = time.strftime("run_%Y%m%d_%H%M%S")

    # Filters and shifts
    min_air_ratio = float(getattr(cfg.data, 'map_min_air_ratio', 0.20))
    try:
        y_shift_abs = abs(int(getattr(cfg.data, 'map_y_shift_abs', 6)))
    except Exception:
        y_shift_abs = 6

    BATCH_SIZE = int(getattr(cfg.data, 'map_batch_size', 1000))
    chunk_buffer = []
    part_index = 0
    batch_start_time = time.time()

    while col_i < cfg.data.num_samples:
        x, z = square_spiral(col_i)
        x *= x_increment
        z *= z_increment

        if abs(x) >= 3e7 or abs(z) >= 3e7:
            print(f"Terminating at coordinates {x}, {z}")
            return

        highest_point = client.getHighestYAt(Point(x=x, y=0, z=z))
        y = highest_point.y
        if highest_point.y == 0:
            read_cube(client, (x, 0, z), (x, 0, z))
            highest_point = client.getHighestYAt(Point(x=x, y=0, z=z))
            if highest_point.y == 0:
                print(f"Skipping location (x={x}, z={z}): No valid height found")
                col_i += 1
                continue
            y = highest_point.y

        # Single-chunk read with random Y shift
        try:
            y_shift = int(np.random.randint(-y_shift_abs, y_shift_abs + 1))
        except Exception:
            y_shift = 0
        cy = max(0, min(255, int(y + y_shift)))
        labels_xyz = np.array([x, cy, z])
        voxels_arr, biomes_arr, metadata_arr = get_vox_chunk((x, cy, z), tuple(cfg.data.chunk_shape), client)

        # Apply air-ratio filter
        air_ratio = float((voxels_arr == BlockType.AIR).sum()) / float(voxels_arr.size)
        if air_ratio >= min_air_ratio:
            # Replace biome with the map name (store per-sample label for compactness)
            chunk_buffer.append({
                "labels": labels_xyz,
                "voxels": voxels_arr,
                "biome_label": map_name,
                "metadata": metadata_arr,
            })

        if len(chunk_buffer) >= BATCH_SIZE:
            # Write a part file for this run
            labels_batch = np.array([item['labels'] for item in chunk_buffer])
            voxels_batch = np.array([item['voxels'] for item in chunk_buffer])
            biome_labels_batch = np.array([item['biome_label'] for item in chunk_buffer], dtype=object)
            metadata_batch = np.array([item['metadata'] for item in chunk_buffer], dtype=object)
            part_path = os.path.join(part_dir, f"{map_name}_chunks_part_{run_tag}_{part_index}.npz")
            np.savez_compressed(part_path, labels=labels_batch, voxels=voxels_batch, biome_labels=biome_labels_batch, metadata=metadata_batch)

            print(f"Wrote part {part_index} with {len(chunk_buffer)} chunks -> {os.path.basename(part_path)}")
            chunk_buffer = []
            part_index += 1
            batch_start_time = time.time()

        if (col_i + 1) % 100 == 0:
            print(f"{col_i+1} chunks collected.")
        col_i += 1

    # Final partial flush
    if chunk_buffer:
        labels_batch = np.array([item['labels'] for item in chunk_buffer])
        voxels_batch = np.array([item['voxels'] for item in chunk_buffer])
        biome_labels_batch = np.array([item['biome_label'] for item in chunk_buffer], dtype=object)
        metadata_batch = np.array([item['metadata'] for item in chunk_buffer], dtype=object)
        part_path = os.path.join(part_dir, f"{map_name}_chunks_part_{run_tag}_{part_index}.npz")
        np.savez_compressed(part_path, labels=labels_batch, voxels=voxels_batch, biome_labels=biome_labels_batch, metadata=metadata_batch)
        print(f"Wrote final part {part_index} with {len(chunk_buffer)} chunks -> {os.path.basename(part_path)}")
        part_index += 1

    # Combine parts into a single file (append to existing if present)
    print("Combining map parts into final file...")
    final_path = os.path.join(base_dir, f"{map_name}_chunks.npz")
    try:
        part_files = [
            os.path.join(part_dir, f) for f in os.listdir(part_dir)
            if f.startswith(f"{map_name}_chunks_part_{run_tag}_") and f.endswith('.npz')
        ]
        try:
            paths_sorted = sorted(part_files, key=lambda p: int(os.path.basename(p).split('_')[-1].split('.')[0]))
        except Exception:
            paths_sorted = sorted(part_files)

        labels_list, voxels_list, biome_labels_list, metadata_list = [], [], [], []
        pre_count = 0
        if os.path.isfile(final_path):
            with np.load(final_path, allow_pickle=True) as ex:
                if 'labels' in ex.files:
                    labels_list.append(ex['labels'])
                    pre_count = int(ex['labels'].shape[0])
                if 'voxels' in ex.files:
                    voxels_list.append(ex['voxels'])
                if 'biome_labels' in ex.files:
                    biome_labels_list.append(ex['biome_labels'])
                if 'metadata' in ex.files:
                    metadata_list.append(ex['metadata'])

        part_count = 0
        for p in paths_sorted:
            with np.load(p, allow_pickle=True) as data:
                if 'labels' in data.files:
                    labels_list.append(data['labels'])
                    part_count += int(data['labels'].shape[0])
                if 'voxels' in data.files:
                    voxels_list.append(data['voxels'])
                if 'biome_labels' in data.files:
                    biome_labels_list.append(data['biome_labels'])
                if 'metadata' in data.files:
                    metadata_list.append(data['metadata'])
            try:
                os.remove(p)
            except Exception:
                pass

        if labels_list:
            out_labels = np.concatenate(labels_list, axis=0)
            out_voxels = np.concatenate(voxels_list, axis=0) if voxels_list else None
            out_biome_labels = np.concatenate(biome_labels_list, axis=0) if biome_labels_list else None
            out_metadata = np.concatenate(metadata_list, axis=0) if metadata_list else None

            np.savez_compressed(
                final_path,
                labels=out_labels,
                voxels=out_voxels,
                biome_labels=out_biome_labels,
                metadata=out_metadata,
            )
            print(f"  Appended {part_count} samples (prev {pre_count}) -> {os.path.basename(final_path)}")
        else:
            print("  No parts to combine.")
    except Exception as e:
        print(f"Could not combine map files: {e}")


def get_nether(client, cfg, data_dir):
    """Collect chunks from the Nether and label each sample as 'nether'.
    Scans the full vertical column by striding a window along the Y axis.
    """
    print(f"Collecting {cfg.data.num_samples} chunks for Nether (Vertical Stride)")

    # Ensure server switches to Nether dimension
    try:
        print("[nether] Switching dimension -> nether")
        client.setActiveDimension(DimensionRequest(dimension="nether", teleport_player=False), timeout=5.0)
        print("[nether] Dimension switch RPC ok")
    except Exception as e:
        print(f"[nether] setActiveDimension unavailable/failed: {e}")

    # Preload a rectangular area of Nether chunks
    try:
        radius_blocks = int(getattr(cfg.data, 'nether_preload_radius_blocks', 512))
        min_x = -radius_blocks
        max_x = radius_blocks
        min_z = -radius_blocks
        max_z = radius_blocks
        print(f"[nether] Preloading chunks in [{min_x},{min_z}]..[{max_x},{max_z}]")
        client.preloadChunks(PreloadChunksRequest(min_x=min_x, max_x=max_x, min_z=min_z, max_z=max_z), timeout=30.0)
        print("[nether] Preload complete")
    except Exception as e:
        print(f"[nether] preloadChunks unavailable/failed: {e}")

    # Step increments
    x_increment = int(cfg.data.chunk_shape[0])
    z_increment = int(cfg.data.chunk_shape[2])
    print(f'Increments (nether): x_increment: {x_increment}, z_increment: {z_increment}')

    # Output directories
    base_dir = os.path.join(cfg.data.data_dir, "Voxels")
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)
    part_dir = os.path.join(base_dir, "nether_parts")
    if not os.path.isdir(part_dir):
        os.makedirs(part_dir)

    # Start index (optional override via cfg)
    col_i = int(cfg.data_gen.i) if getattr(cfg.data_gen, 'i', None) not in (None, "",) else 0

    # Per-run tag
    run_tag = None
    try:
        props_path = os.path.join("server", "server.properties")
        if os.path.isfile(props_path):
            with open(props_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("level-seed="):
                        val = line.split("=", 1)[1].strip()
                        run_tag = f"seed_{val}"
                        break
    except Exception:
        run_tag = None
    if not run_tag:
        run_tag = time.strftime("run_%Y%m%d_%H%M%S")

    # Filters and params
    min_air_ratio = float(getattr(cfg.data, 'map_min_air_ratio', 0.20))
    BATCH_SIZE = int(getattr(cfg.data, 'map_batch_size', 1000))
    chunk_buffer = []
    part_index = 0

    sx, sy, sz = tuple(cfg.data.chunk_shape)
    
    # Define vertical striding parameters
    # Stride by half the chunk height to ensure good coverage without excessive overlap
    y_stride = max(1, sy // 2)
    
    # Valid center Y range: ensure window stays within 0..127
    y_min_cy = sy // 2
    y_max_cy = 127 - (sy - (sy // 2))

    # Track collected count
    total_samples = 0

    while total_samples < cfg.data.num_samples:
        x, z = square_spiral(col_i)
        x *= x_increment
        z *= z_increment

        if abs(x) >= 3e7 or abs(z) >= 3e7:
            print(f"Terminating at coordinates {x}, {z}")
            return

        if (col_i % 50) == 0:
            print(f"[nether] spiral #{col_i} at ({x},{z}) | collected: {total_samples}")

        # Iterate vertically through the column
        # Start from bottom (just above bedrock) and move up to roof
        for cy in range(y_min_cy, y_max_cy + 1, y_stride):
            if total_samples >= cfg.data.num_samples:
                break

            labels_xyz = np.array([x, cy, z])
            try:
                minx = x - (sx // 2)
                miny = cy - (sy // 2)
                minz = z - (sz // 2)
                maxx = minx + sx - 1
                maxy = miny + sy - 1
                maxz = minz + sz - 1
                
                # Full read with metadata
                cube = client.readCubeAndBiomeMetadata(Cube(
                    min=Point(x=minx, y=miny, z=minz), 
                    max=Point(x=maxx, y=maxy, z=maxz), 
                    preload_chunks=True
                ), timeout=10.0)
                
                voxels_arr, biomes_arr, metadata_arr = cube_to_voxels_and_biomes(cube, (sx, sy, sz), (minx, miny, minz))
            except Exception as e:
                print(f"[nether] chunk read error at ({x},{cy},{z}): {e}. Skipping.")
                continue

            air_ratio = float((voxels_arr == BlockType.AIR).sum()) / float(voxels_arr.size)
            
            # Keep chunk if it has enough air (not solid netherrack) but not too much (empty space)
            if air_ratio >= min_air_ratio and air_ratio < 0.8:
                chunk_buffer.append({
                    "labels": labels_xyz,
                    "voxels": voxels_arr,
                    "biome_label": 'nether',
                    "metadata": metadata_arr,
                })
                total_samples += 1
                
                # Check buffer flush
                if len(chunk_buffer) >= BATCH_SIZE:
                    labels_batch = np.array([item['labels'] for item in chunk_buffer])
                    voxels_batch = np.array([item['voxels'] for item in chunk_buffer])
                    biome_labels_batch = np.array([item['biome_label'] for item in chunk_buffer], dtype=object)
                    metadata_batch = np.array([item['metadata'] for item in chunk_buffer], dtype=object)
                    part_path = os.path.join(part_dir, f"nether_chunks_part_{run_tag}_{part_index}.npz")
                    np.savez_compressed(part_path, labels=labels_batch, voxels=voxels_batch, biome_labels=biome_labels_batch, metadata=metadata_batch)
                    print(f"Wrote part {part_index} with {len(chunk_buffer)} chunks -> {os.path.basename(part_path)}")
                    chunk_buffer = []
                    part_index += 1

        col_i += 1

    # Final partial flush
    if chunk_buffer:
        labels_batch = np.array([item['labels'] for item in chunk_buffer])
        voxels_batch = np.array([item['voxels'] for item in chunk_buffer])
        biome_labels_batch = np.array([item['biome_label'] for item in chunk_buffer], dtype=object)
        metadata_batch = np.array([item['metadata'] for item in chunk_buffer], dtype=object)
        part_path = os.path.join(part_dir, f"nether_chunks_part_{run_tag}_{part_index}.npz")
        np.savez_compressed(part_path, labels=labels_batch, voxels=voxels_batch, biome_labels=biome_labels_batch, metadata=metadata_batch)
        print(f"Wrote final part {part_index} with {len(chunk_buffer)} chunks -> {os.path.basename(part_path)}")
        part_index += 1

    # Combine parts into a single file (append to existing if present)
    print("Combining nether parts into final file...")
    # final_path = os.path.join(base_dir, "nether_chunks.npz")
    # try:
    #     part_files = [
    #         os.path.join(part_dir, f) for f in os.listdir(part_dir)
    #         if f.startswith(f"nether_chunks_part_{run_tag}_") and f.endswith('.npz')
    #     ]
    #     try:
    #         paths_sorted = sorted(part_files, key=lambda p: int(os.path.basename(p).split('_')[-1].split('.')[0]))
    #     except Exception:
    #         paths_sorted = sorted(part_files)

    #     labels_list, voxels_list, biome_labels_list, metadata_list = [], [], [], []
    #     pre_count = 0
    #     if os.path.isfile(final_path):
    #         with np.load(final_path, allow_pickle=True) as ex:
    #             if 'labels' in ex.files:
    #                 labels_list.append(ex['labels'])
    #                 pre_count = int(ex['labels'].shape[0])
    #             if 'voxels' in ex.files:
    #                 voxels_list.append(ex['voxels'])
    #             if 'biome_labels' in ex.files:
    #                 biome_labels_list.append(ex['biome_labels'])
    #             if 'metadata' in ex.files:
    #                 metadata_list.append(ex['metadata'])

    #     part_count = 0
    #     for p in paths_sorted:
    #         with np.load(p, allow_pickle=True) as data:
    #             if 'labels' in data.files:
    #                 labels_list.append(data['labels'])
    #                 part_count += int(data['labels'].shape[0])
    #             if 'voxels' in data.files:
    #                 voxels_list.append(data['voxels'])
    #             if 'biome_labels' in data.files:
    #                 biome_labels_list.append(data['biome_labels'])
    #             if 'metadata' in data.files:
    #                 metadata_list.append(data['metadata'])
    #         try:
    #             os.remove(p)
    #         except Exception:
    #             pass

    #     if labels_list:
    #         out_labels = np.concatenate(labels_list, axis=0)
    #         out_voxels = np.concatenate(voxels_list, axis=0) if voxels_list else None
    #         out_biome_labels = np.concatenate(biome_labels_list, axis=0) if biome_labels_list else None
    #         out_metadata = np.concatenate(metadata_list, axis=0) if metadata_list else None

    #         np.savez_compressed(
    #             final_path,
    #             labels=out_labels,
    #             voxels=out_voxels,
    #             biome_labels=out_biome_labels,
    #             metadata=out_metadata,
    #         )
    #         print(f"  Appended {part_count} samples (prev {pre_count}) -> {os.path.basename(final_path)}")
    #     else:
    #         print("  No parts to combine.")
    # except Exception as e:
    #     print(f"Could not combine nether files: {e}")

def get_caves(client, cfg, data_dir):
    """Collect cave chunks using a simple air-threshold heuristic.

    Traversal:
    - Move across X/Z in a square spiral with step equal to chunk size in X/Z.
    - For each (x,z) center, start just below the surface and slide a window
      downward along Y with stride 8 (configurable) evaluating many windows.

    Selection:
    - Keep windows with air ratio above a configurable threshold.
    - Label each kept sample with a compact biome label 'cave'.
    - Persist in parts and combine into a single compressed npz at the end.
    """
    print(f"Collecting up to {cfg.data.num_samples} cave chunks (air-threshold)")

    # Window and traversal steps
    sx, sy, sz = tuple(cfg.data.chunk_shape)
    x_increment = sx
    z_increment = sz
    y_stride = int(getattr(cfg.data, 'cave_y_stride', 8))
    if y_stride <= 0:
        y_stride = 8
    print(f"Increments (caves):\n x_increment: {x_increment}\n z_increment: {z_increment}\n y_stride: {y_stride}")
    print('wtf')

    # Output directories
    base_dir = os.path.join(cfg.data.data_dir, "Voxels")
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)
    part_dir = os.path.join(base_dir, "caves_parts")
    if not os.path.isdir(part_dir):
        os.makedirs(part_dir)

    # Start index (optional override via cfg)
    col_i = int(cfg.data_gen.i) if getattr(cfg.data_gen, 'i', None) not in (None, "",) else 0

    # Per-run tag to avoid part collisions across seeds/restarts
    run_tag = None
    try:
        props_path = os.path.join("server", "server.properties")
        if os.path.isfile(props_path):
            with open(props_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("level-seed="):
                        val = line.split("=", 1)[1].strip()
                        run_tag = f"seed_{val}"
                        break
    except Exception:
        run_tag = None
    if not run_tag:
        run_tag = time.strftime("run_%Y%m%d_%H%M%S")

    # Thresholds and batching
    min_air_ratio = float(getattr(cfg.data, 'cave_min_air_ratio', 0.10))
    BATCH_SIZE = int(getattr(cfg.data, 'map_batch_size', 2000))
    chunk_buffer = []
    part_index = 0
    collected = 0
    batch_start_time = time.time()

    # Helper: compute window center from desired top-of-window Y
    top_offset = int(math.ceil(sy / 2.0) - 1)  # for sy=32 -> 15
    bottom_span = sy - 1                        # for sy=32 -> 31

    while collected < int(getattr(cfg.data, 'num_samples', 0)):
        x, z = square_spiral(col_i)
        x *= x_increment
        z *= z_increment

        if abs(x) >= 3e7 or abs(z) >= 3e7:
            print(f"Terminating at coordinates {x}, {z}")
            break

        # Resolve surface Y and ensure chunk generation
        highest_point = client.getHighestYAt(Point(x=x, y=0, z=z))
        y_top = highest_point.y
        if y_top == 0:
            read_cube(client, (x, 0, z), (x, 0, z))
            highest_point = client.getHighestYAt(Point(x=x, y=0, z=z))
            y_top = highest_point.y
            if y_top == 0:
                # Skip if still invalid
                col_i += 1
                continue

        # Start just below surface; slide window downward by y_stride
        top_of_window = max(0, min(255, int(y_top - 1)))

        while collected < int(getattr(cfg.data, 'num_samples', 0)):
            # Stop if the bottom of the window would go below bedrock
            if (top_of_window - bottom_span) < 0:
                break

            # Compute center Y from top-of-window, clamp to valid range so [min,max] is within [0,255]
            cy = top_of_window - top_offset
            cy = max(top_offset, min(255 - (sy - 1 - top_offset), cy))

            labels_xyz = np.array([x, cy, z])
            voxels_arr, biomes_arr, metadata_arr = get_vox_chunk((x, cy, z), (sx, sy, sz), client)

            # Air threshold selection
            air_ratio = float((voxels_arr == BlockType.AIR).sum()) / float(voxels_arr.size)
            if air_ratio >= min_air_ratio:
                chunk_buffer.append({
                    "labels": labels_xyz,
                    "voxels": voxels_arr,
                    "biome_label": 'cave',
                    "metadata": metadata_arr,
                })
                collected += 1
                if collected % 100 == 0:
                    print(f"Collected {collected} chunks")

            if len(chunk_buffer) >= BATCH_SIZE:
                labels_batch = np.array([item['labels'] for item in chunk_buffer])
                voxels_batch = np.array([item['voxels'] for item in chunk_buffer])
                biome_labels_batch = np.array([item['biome_label'] for item in chunk_buffer], dtype=object)
                metadata_batch = np.array([item['metadata'] for item in chunk_buffer], dtype=object)
                part_path = os.path.join(part_dir, f"cave_chunks_part_{run_tag}_{part_index}.npz")
                np.savez_compressed(part_path, labels=labels_batch, voxels=voxels_batch, biome_labels=biome_labels_batch, metadata=metadata_batch)
                print(f"Wrote part {part_index} with {len(chunk_buffer)} chunks -> {os.path.basename(part_path)}")
                chunk_buffer = []
                part_index += 1
                batch_start_time = time.time()

            # Next window lower
            next_top = top_of_window - y_stride
            if next_top < 0:
                break
            top_of_window = next_top

        # Next (x,z) center in spiral
        col_i += 1

    # Final partial flush
    if chunk_buffer:
        labels_batch = np.array([item['labels'] for item in chunk_buffer])
        voxels_batch = np.array([item['voxels'] for item in chunk_buffer])
        biome_labels_batch = np.array([item['biome_label'] for item in chunk_buffer], dtype=object)
        metadata_batch = np.array([item['metadata'] for item in chunk_buffer], dtype=object)
        part_path = os.path.join(part_dir, f"cave_chunks_part_{run_tag}_{part_index}.npz")
        np.savez_compressed(part_path, labels=labels_batch, voxels=voxels_batch, biome_labels=biome_labels_batch, metadata=metadata_batch)
        print(f"Wrote final part {part_index} with {len(chunk_buffer)} chunks -> {os.path.basename(part_path)}")
        part_index += 1

    # Combine parts into single file
    # # print("Combining cave parts into final file...")
    # # final_path = os.path.join(base_dir, "cave_chunks.npz")
    # # try:
    # #     part_files = [
    # #         os.path.join(part_dir, f) for f in os.listdir(part_dir)
    # #         if f.startswith(f"cave_chunks_part_{run_tag}_") and f.endswith('.npz')
    # #     ]
    # #     try:
    # #         paths_sorted = sorted(part_files, key=lambda p: int(os.path.basename(p).split('_')[-1].split('.')[0]))
    # #     except Exception:
    # #         paths_sorted = sorted(part_files)

    # #     labels_list, voxels_list, biome_labels_list, metadata_list = [], [], [], []
    # #     pre_count = 0
    # #     if os.path.isfile(final_path):
    # #         with np.load(final_path, allow_pickle=True) as ex:
    # #             if 'labels' in ex.files:
    # #                 labels_list.append(ex['labels'])
    # #                 pre_count = int(ex['labels'].shape[0])
    # #             if 'voxels' in ex.files:
    # #                 voxels_list.append(ex['voxels'])
    # #             if 'biome_labels' in ex.files:
    # #                 biome_labels_list.append(ex['biome_labels'])
    # #             if 'metadata' in ex.files:
    # #                 metadata_list.append(ex['metadata'])

    # #     part_count = 0
    # #     for p in paths_sorted:
    # #         with np.load(p, allow_pickle=True) as data:
    # #             if 'labels' in data.files:
    # #                 labels_list.append(data['labels'])
    # #                 part_count += int(data['labels'].shape[0])
    # #             if 'voxels' in data.files:
    # #                 voxels_list.append(data['voxels'])
    # #             if 'biome_labels' in data.files:
    # #                 biome_labels_list.append(data['biome_labels'])
    # #             if 'metadata' in data.files:
    # #                 metadata_list.append(data['metadata'])
    # #         try:
    # #             os.remove(p)
    # #         except Exception:
    # #             pass

    # #     if labels_list:
    # #         out_labels = np.concatenate(labels_list, axis=0)
    # #         out_voxels = np.concatenate(voxels_list, axis=0) if voxels_list else None
    # #         out_biome_labels = np.concatenate(biome_labels_list, axis=0) if biome_labels_list else None
    # #         out_metadata = np.concatenate(metadata_list, axis=0) if metadata_list else None

    # #         np.savez_compressed(
    # #             final_path,
    # #             labels=out_labels,
    # #             voxels=out_voxels,
    # #             biome_labels=out_biome_labels,
    # #             metadata=out_metadata,
    # #         )
    # #         print(f"  Appended {part_count} samples (prev {pre_count}) -> {os.path.basename(final_path)}")
    # #     else:
    # #         print("  No parts to combine.")
    # except Exception as e:
    #     print(f"Could not combine cave files: {e}")


def get_voxels_in_area(client, cfg, data_dir):
    """Collect chunks within a user-specified X/Z square area.

    Behavior mirrors get_voxels, but traversal is bounded to a provided area:
    - Provide two corners via cfg (either data.area_corners or min/max keys)
    - Traverse centers on a grid with stride equal to chunk_shape in X/Z
    - Apply the same air-ratio filter and small random Y shift
    - Write part files and combine into a single compressed npz
    - Replace biome label with the configured map name per sample
    """
    map_name = str(getattr(cfg.data, 'map_name', 'unnamed_map'))
    print(f"Collecting up to {cfg.data.num_samples} chunks for map: {map_name} within configured area")

    # Strides (allow custom X/Z/Y strides; default to chunk_shape)
    x_increment = int(getattr(cfg.data, 'area_stride_x', cfg.data.chunk_shape[0]))
    z_increment = int(getattr(cfg.data, 'area_stride_z', cfg.data.chunk_shape[2]))
    y_increment = int(getattr(cfg.data, 'area_stride_y', cfg.data.chunk_shape[1]))
    if x_increment <= 0:
        x_increment = int(cfg.data.chunk_shape[0])
    if z_increment <= 0:
        z_increment = int(cfg.data.chunk_shape[2])
    if y_increment <= 0:
        y_increment = int(cfg.data.chunk_shape[1])
    print(f'Increments (area mode):\n x_increment: {x_increment}\n z_increment: {z_increment}\n y_increment: {y_increment}')

    # Output directories
    base_dir = os.path.join(cfg.data.data_dir, "Voxels")
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)
    part_dir = os.path.join(base_dir, f"{map_name}_parts")
    if not os.path.isdir(part_dir):
        os.makedirs(part_dir)

    # Resolve area bounds from config
    def _parse_area_from_cfg(dc):
        corners = getattr(dc, 'area_corners', None)
        if corners is not None:
            try:
                # Normalize OmegaConf containers to plain Python lists
                try:
                    from omegaconf import ListConfig
                    if isinstance(corners, ListConfig):
                        corners = list(corners)
                except Exception:
                    pass
                # Support [[x1,z1],[x2,z2]] or [x1,z1,x2,z2]
                if isinstance(corners, (list, tuple)) and len(corners) == 2:
                    p0, p1 = corners[0], corners[1]
                    # Normalize nested ListConfig
                    try:
                        from omegaconf import ListConfig
                        if isinstance(p0, ListConfig):
                            p0 = list(p0)
                        if isinstance(p1, ListConfig):
                            p1 = list(p1)
                    except Exception:
                        pass
                    if isinstance(p0, (list, tuple)) and isinstance(p1, (list, tuple)) and len(p0) == 2 and len(p1) == 2:
                        (x1, z1), (x2, z2) = p0, p1
                    else:
                        raise ValueError("area_corners must be [[x1,z1],[x2,z2]] or [x1,z1,x2,z2]")
                elif isinstance(corners, (list, tuple)) and len(corners) == 4:
                    x1, z1, x2, z2 = corners
                else:
                    raise ValueError("area_corners must be [[x1,z1],[x2,z2]] or [x1,z1,x2,z2]")
                return int(x1), int(z1), int(x2), int(z2)
            except Exception as e:
                raise ValueError(f"Invalid area_corners in cfg: {corners} ({e})")

        # Fallback explicit keys
        try:
            x1 = getattr(dc, 'area_min_x')
            z1 = getattr(dc, 'area_min_z')
            x2 = getattr(dc, 'area_max_x')
            z2 = getattr(dc, 'area_max_z')
            if None in (x1, z1, x2, z2):
                raise ValueError
            return int(x1), int(z1), int(x2), int(z2)
        except Exception:
            raise ValueError("Must provide area via data.area_corners or data.area_min_x/area_min_z/area_max_x/area_max_z")

    x1, z1, x2, z2 = _parse_area_from_cfg(cfg.data)
    xmin_raw, xmax_raw = (x1, x2) if x1 <= x2 else (x2, x1)
    zmin_raw, zmax_raw = (z1, z2) if z1 <= z2 else (z2, z1)

    # Align bounds to our grid of centers so we only visit centers inside the area
    def _align_up(val, step):
        # First grid center >= val
        return int(step * math.ceil(val / float(step)))

    def _align_down(val, step):
        # Last grid center <= val
        return int(step * math.floor(val / float(step)))

    xmin = _align_up(xmin_raw, x_increment)
    xmax = _align_down(xmax_raw, x_increment)
    zmin = _align_up(zmin_raw, z_increment)
    zmax = _align_down(zmax_raw, z_increment)

    if xmin > xmax or zmin > zmax:
        print(f"Configured area too small after alignment: x:[{xmin_raw},{xmax_raw}] z:[{zmin_raw},{zmax_raw}] -> centers x:[{xmin},{xmax}] z:[{zmin},{zmax}]")
        return

    print(f"Area centers x:[{xmin},{xmax}] step={x_increment}, z:[{zmin},{zmax}] step={z_increment}")

    # Per-run tag to avoid part collisions across seeds/restarts
    run_tag = None
    try:
        props_path = os.path.join("server", "server.properties")
        if os.path.isfile(props_path):
            with open(props_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("level-seed="):
                        val = line.split("=", 1)[1].strip()
                        run_tag = f"seed_{val}"
                        break
    except Exception:
        run_tag = None
    if not run_tag:
        run_tag = time.strftime("run_%Y%m%d_%H%M%S")

    # Filters and stop condition
    min_air_ratio = float(getattr(cfg.data, 'map_min_air_ratio', 0.20))
    stop_below_air_ratio = float(getattr(cfg.data, 'area_stop_air_ratio', 0.20))

    BATCH_SIZE = int(getattr(cfg.data, 'map_batch_size', 1000))
    chunk_buffer = []
    part_index = 0
    batch_start_time = time.time()

    collected = 0
    max_samples = int(getattr(cfg.data, 'num_samples', 0))

    # Progress counters
    centers_processed = 0
    progress_every = int(getattr(cfg.data, 'area_progress_every_centers', 50))

    # Iterate over bounded grid
    for x in range(xmin, xmax + 1, x_increment):
        if max_samples and collected >= max_samples:
            break
        if abs(x) >= 3e7:
            print(f"Terminating at x coordinate {x} (out of world bounds)")
            break
        for z in range(zmin, zmax + 1, z_increment):
            if max_samples and collected >= max_samples:
                break
            if abs(z) >= 3e7:
                print(f"Terminating at z coordinate {z} (out of world bounds)")
                break

            highest_point = client.getHighestYAt(Point(x=x, y=0, z=z))
            y_top = highest_point.y
            if highest_point.y == 0:
                read_cube(client, (x, 0, z), (x, 0, z))
                highest_point = client.getHighestYAt(Point(x=x, y=0, z=z))
                if highest_point.y == 0:
                    # Skip if no valid height found even after nudge
                    continue
                y_top = highest_point.y

            # Estimate terrain surface from a sampled local heightmap over the chunk footprint
            sx, sy, sz = tuple(cfg.data.chunk_shape)
            height_stride = int(getattr(cfg.data, 'area_height_sample_stride', 8))
            max_height_samples = int(getattr(cfg.data, 'area_max_height_samples', 64))
            heights = []
            xmin_c = x - (sx // 2)
            xmax_c = xmin_c + sx - 1
            zmin_c = z - (sz // 2)
            zmax_c = zmin_c + sz - 1
            try:
                step_x = max(1, height_stride)
                step_z = max(1, height_stride)
                for gx in range(xmin_c, xmax_c + 1, step_x):
                    for gz in range(zmin_c, zmax_c + 1, step_z):
                        hp = client.getHighestYAt(Point(x=gx, y=0, z=gz))
                        try:
                            yv = int(getattr(hp, 'y', 0))
                            if yv > 0:
                                heights.append(yv)
                                if len(heights) >= max_height_samples:
                                    raise StopIteration
                        except Exception:
                            pass
                        # Continue until cap reached
                
            except Exception:
                pass

            if heights:
                # Configurable surface estimator: 'min' | 'median' | 'percentile'
                surface_method = str(getattr(cfg.data, 'area_surface_method', 'percentile')).lower()
                h_arr = np.array(heights, dtype=np.int32)
                if surface_method == 'min':
                    surface_y_est = int(h_arr.min())
                elif surface_method == 'median':
                    surface_y_est = int(np.median(h_arr))
                else:
                    try:
                        perc = float(getattr(cfg.data, 'area_surface_percentile', 50.0))
                    except Exception:
                        perc = 60.0
                    perc = max(0.0, min(100.0, perc))
                    try:
                        surface_y_est = int(np.percentile(h_arr, perc))
                    except Exception:
                        surface_y_est = int(np.median(h_arr))
            else:
                surface_y_est = int(y_top)

            surface_buffer = int(getattr(cfg.data, 'area_surface_buffer', 0))

            # Iterate downward along Y without random shifts; stop when window top dips below estimated surface
            y_current = max(0, min(255, int(y_top)))
            while True:
                if max_samples and collected >= max_samples:
                    break
                cy = max(0, min(255, int(y_current)))

                # Stop if the top of the window is below the estimated surface for this area
                top_of_window = cy + (sy // 2)
                # Also stop if the remaining margin above surface is too thin
                min_top_margin = int(getattr(cfg.data, 'area_min_top_margin', 8))
                if top_of_window < (surface_y_est - surface_buffer):
                    break
                if top_of_window - surface_y_est < min_top_margin:
                    break
                labels_xyz = np.array([x, cy, z])
                voxels_arr, biomes_arr, metadata_arr = get_vox_chunk((x, cy, z), tuple(cfg.data.chunk_shape), client)

                air_ratio = float((voxels_arr == BlockType.AIR).sum()) / float(voxels_arr.size)
                # Optional additional stop: if extremely solid (very low air), we consider underground and stop
                if air_ratio < stop_below_air_ratio:
                    break

                if air_ratio >= min_air_ratio:
                    chunk_buffer.append({
                        "labels": labels_xyz,
                        "voxels": voxels_arr,
                        "biome_label": map_name,
                        "metadata": metadata_arr,
                    })
                    collected += 1

                if len(chunk_buffer) >= BATCH_SIZE:
                    labels_batch = np.array([item['labels'] for item in chunk_buffer])
                    voxels_batch = np.array([item['voxels'] for item in chunk_buffer])
                    biome_labels_batch = np.array([item['biome_label'] for item in chunk_buffer], dtype=object)
                    metadata_batch = np.array([item['metadata'] for item in chunk_buffer], dtype=object)
                    part_path = os.path.join(part_dir, f"{map_name}_chunks_part_{run_tag}_{part_index}.npz")
                    np.savez_compressed(part_path, labels=labels_batch, voxels=voxels_batch, biome_labels=biome_labels_batch, metadata=metadata_batch)
                    print(f"Wrote part {part_index} with {len(chunk_buffer)} chunks -> {os.path.basename(part_path)}")
                    chunk_buffer = []
                    part_index += 1
                    batch_start_time = time.time()

                # Next Y center downwards
                next_y = cy - y_increment
                if next_y < 0:
                    break
                y_current = next_y

            centers_processed += 1
            if progress_every > 0 and (centers_processed % progress_every == 0):
                print(f"Processed centers: {centers_processed} | collected so far: {collected}")

    # Final partial flush
    if chunk_buffer:
        labels_batch = np.array([item['labels'] for item in chunk_buffer])
        voxels_batch = np.array([item['voxels'] for item in chunk_buffer])
        biome_labels_batch = np.array([item['biome_label'] for item in chunk_buffer], dtype=object)
        metadata_batch = np.array([item['metadata'] for item in chunk_buffer], dtype=object)
        part_path = os.path.join(part_dir, f"{map_name}_chunks_part_{run_tag}_{part_index}.npz")
        np.savez_compressed(part_path, labels=labels_batch, voxels=voxels_batch, biome_labels=biome_labels_batch, metadata=metadata_batch)
        print(f"Wrote final part {part_index} with {len(chunk_buffer)} chunks -> {os.path.basename(part_path)}")
        part_index += 1

    # Combine parts into a single file (append to existing if present)
    print("Combining map parts into final file (area mode)...")
    final_path = os.path.join(base_dir, f"{map_name}_chunks.npz")
    try:
        part_files = [
            os.path.join(part_dir, f) for f in os.listdir(part_dir)
            if f.startswith(f"{map_name}_chunks_part_{run_tag}_") and f.endswith('.npz')
        ]
        try:
            paths_sorted = sorted(part_files, key=lambda p: int(os.path.basename(p).split('_')[-1].split('.')[0]))
        except Exception:
            paths_sorted = sorted(part_files)

        labels_list, voxels_list, biome_labels_list, metadata_list = [], [], [], []
        pre_count = 0
        if os.path.isfile(final_path):
            with np.load(final_path, allow_pickle=True) as ex:
                if 'labels' in ex.files:
                    labels_list.append(ex['labels'])
                    pre_count = int(ex['labels'].shape[0])
                if 'voxels' in ex.files:
                    voxels_list.append(ex['voxels'])
                if 'biome_labels' in ex.files:
                    biome_labels_list.append(ex['biome_labels'])
                if 'metadata' in ex.files:
                    metadata_list.append(ex['metadata'])

        part_count = 0
        for p in paths_sorted:
            with np.load(p, allow_pickle=True) as data:
                if 'labels' in data.files:
                    labels_list.append(data['labels'])
                    part_count += int(data['labels'].shape[0])
                if 'voxels' in data.files:
                    voxels_list.append(data['voxels'])
                if 'biome_labels' in data.files:
                    biome_labels_list.append(data['biome_labels'])
                if 'metadata' in data.files:
                    metadata_list.append(data['metadata'])
            try:
                os.remove(p)
            except Exception:
                pass

        if labels_list:
            out_labels = np.concatenate(labels_list, axis=0)
            out_voxels = np.concatenate(voxels_list, axis=0) if voxels_list else None
            out_biome_labels = np.concatenate(biome_labels_list, axis=0) if biome_labels_list else None
            out_metadata = np.concatenate(metadata_list, axis=0) if metadata_list else None

            np.savez_compressed(
                final_path,
                labels=out_labels,
                voxels=out_voxels,
                biome_labels=out_biome_labels,
                metadata=out_metadata,
            )
            print(f"  Appended {part_count} samples (prev {pre_count}) -> {os.path.basename(final_path)}")
        else:
            print("  No parts to combine.")
    except Exception as e:
        print(f"Could not combine map files: {e}")

def get_voxels_by_biome(client, cfg):
    """Collect chunks, group by majority biome, and write compressed per-biome datasets.

    Simplifications:
    - Always uses metadata-rich reads via readCubeAndBiomeMetadata (through helpers).
    - Always groups by simplified majority biome (see SIMPLE_BIOME_MAPPING).
    - Writes per-biome parts with: labels, voxels, biome_labels (1 string per sample), metadata.
    - No legacy feature flags, no alternate dense RPC branches, no Y-shift modes.
    """
    print(f'Getting {cfg.data.num_samples} voxels (by biome)')

    # Traversal step scales with optional macro factor for subchunk iteration
    macro_factor_xz = int(getattr(cfg.data, 'macro_factor_xz', 1))
    eff_macro = max(1, macro_factor_xz)
    x_increment = (cfg.data.chunk_shape[0] * eff_macro)
    z_increment = (cfg.data.chunk_shape[2] * eff_macro)
    print(f'Increments:\n x_increment: {x_increment}\nz_increment: {z_increment}')

    # Output directories
    base_dir = os.path.join(cfg.data.data_dir, "Voxels")
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)
    biome_dir = os.path.join(base_dir, "biome_chunks_32_moreparts")
    if not os.path.isdir(biome_dir):
        os.makedirs(biome_dir)

    # Start index (optional override via cfg)
    col_i = int(cfg.data_gen.i) if getattr(cfg.data_gen, 'i', None) not in (None, "",) else 0

    # Create a run tag to avoid temporary-part filename collisions across runs.
    # Prefer the current world seed if available; otherwise, use a timestamp.
    run_tag = None
    try:
        props_path = os.path.join("server", "server.properties")
        if os.path.isfile(props_path):
            with open(props_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("level-seed="):
                        val = line.split("=", 1)[1].strip()
                        # Keep as raw string to preserve Java's exact seed format
                        run_tag = f"seed_{val}"
                        break
    except Exception:
        run_tag = None
    if not run_tag:
        run_tag = time.strftime("run_%Y%m%d_%H%M%S")

    BATCH_SIZE = 2500
    chunk_buffer = []
    part_index = 0
    batch_start_time = time.time()

    # Filter: drop very dense underground windows
    min_air_ratio = float(getattr(cfg.data, 'biome_min_air_ratio', 0.20))
    # Random Y shift range (symmetric around surface height)
    try:
        y_shift_abs = abs(int(getattr(cfg.data, 'biome_y_shift_abs', 6)))
    except Exception:
        y_shift_abs = 6

    def normalize_biome_label(label: str):
        if label is None:
            return None
        if isinstance(label, bytes):
            try:
                label = label.decode("utf-8")
            except Exception:
                label = str(label)
        label = str(label)
        if ":" in label:
            label = label.split(":")[-1]
        return label.lower()

    def get_majority_simple_biome(biome_array: np.ndarray) -> str:
        flat = biome_array.ravel()
        labels = [normalize_biome_label(b) for b in flat if b is not None]
        if not labels:
            return "unknown"
        unique, counts = np.unique(np.array(labels, dtype=object), return_counts=True)
        majority = unique[int(np.argmax(counts))]
        return SIMPLE_BIOME_MAPPING.get(majority, majority)

    # Hard-coded ignore list of simplified biomes to skip (e.g., oversampled ones)
    IGNORE_BIOMES = []#["ocean", 'forest']

    def write_group_batch(simple_biome: str, labels_list, voxels_list, biome_labels_list, metadata_list, part_idx: int):
        """Write a per-batch file for one biome.

        Persists only the majority label per chunk to keep files compact.
        """
        file_path = os.path.join(biome_dir, f"{simple_biome}_chunks_part_{run_tag}_{part_idx}.npz")
        new_labels = np.array(labels_list)
        new_voxels = np.array(voxels_list)
        biome_labels = np.array(biome_labels_list, dtype=object)
        new_metadata = np.array(metadata_list, dtype=object)
        _atomic_savez_compressed(
            file_path,
            labels=new_labels,
            voxels=new_voxels,
            biome_labels=biome_labels,
            metadata=new_metadata,
        )

    print("Collecting voxels and writing to per-biome files")
    t0_all = time.time()

    while col_i < cfg.data.num_samples:
        x, z = square_spiral(col_i)
        x *= x_increment
        z *= z_increment

        if abs(x) >= 3e7 or abs(z) >= 3e7:
            print(f"Terminating at coordinates {x}, {z}")
            return

        # Resolve surface Y and ensure chunk generation
        highest_point = client.getHighestYAt(Point(x=x, y=0, z=z))
        # highest_point2 = client.getHighestYAt(Point(x=(x + x_increment) // 2, y=0, z=(z + z_increment) // 2))
        y = highest_point.y
        if highest_point.y == 0:
            read_cube(client, (x, 0, z), (x, 0, z))
            highest_point = client.getHighestYAt(Point(x=x, y=0, z=z))
            if highest_point.y == 0:
                print(f"Skipping location (x={x}, z={z}): No valid height found")
                col_i += 1
                continue
            y = highest_point.y

        # Read a single chunk with a small random Y shift around surface
        try:
            y_shift = int(np.random.randint(-y_shift_abs, y_shift_abs + 1))
        except Exception:
            y_shift = 0
        cy = max(0, min(255, int(y + y_shift)))
        labels_xyz = np.array([x, cy, z])
        # Read a slightly larger area (+16 in X and Z) then center-crop back to window size
        sx, sy, sz = tuple(cfg.data.chunk_shape)
        padded_shape = (sx + 16, sy, sz + 16)
        vox_padded, bio_padded, md_padded = get_vox_chunk((x, cy, z), padded_shape, client)
        x_start = (padded_shape[0] - sx) // 2
        z_start = (padded_shape[2] - sz) // 2
        x_end = x_start + sx
        z_end = z_start + sz
        voxels_arr = vox_padded[x_start:x_end, :, z_start:z_end]
        biomes_arr = bio_padded[x_start:x_end, :, z_start:z_end]
        metadata_arr = md_padded[x_start:x_end, :, z_start:z_end]

        # Filter low-air chunks to bias toward above-ground content
        air_ratio = float((voxels_arr == BlockType.AIR).sum()) / float(voxels_arr.size)
        if air_ratio >= min_air_ratio:
            simple_biome = get_majority_simple_biome(biomes_arr)
            if simple_biome not in IGNORE_BIOMES:
                chunk_buffer.append({
                    "labels": labels_xyz,
                    "voxels": voxels_arr,
                    "metadata": metadata_arr,
                    "simple_biome": simple_biome,
                })

        if len(chunk_buffer) >= BATCH_SIZE:
            # Batch timing (collection)
            t_collect = time.time() - batch_start_time
            t_write_start = time.time()

            # Group and write by biome
            grouped = {}
            for item in chunk_buffer:
                grouped.setdefault(item['simple_biome'], []).append(item)

            for simple_biome, data_grp in grouped.items():
                labels_list = [it['labels'] for it in data_grp]
                vox_list = [it['voxels'] for it in data_grp]
                bio_labels = [it['simple_biome'] for it in data_grp]
                md_list = [it['metadata'] for it in data_grp]
                write_group_batch(simple_biome, labels_list, vox_list, bio_labels, md_list, part_index)
                print(f"Wrote batch {part_index} with {len(labels_list)} chunks to {simple_biome}_chunks_part_{part_index}.npz")

            t_write = time.time() - t_write_start
            t_total = t_collect + t_write
            n = BATCH_SIZE
            if n > 0:
                print(f"Batch {part_index}: collect={t_collect:.2f}s, write={t_write:.2f}s, total={t_total:.2f}s, throughput={n/max(t_total,1e-6):.1f} chunks/s")

            chunk_buffer = []
            part_index += 1
            batch_start_time = time.time()

        if (col_i + 1) % 100 == 0:
            print(f"{col_i+1} chunks collected.")
        col_i += 1

    # Final partial flush
    if chunk_buffer:
        t_collect = time.time() - batch_start_time
        t_write_start = time.time()

        grouped = {}
        for item in chunk_buffer:
            grouped.setdefault(item['simple_biome'], []).append(item)

        for simple_biome, data_grp in grouped.items():
            labels_list = [it['labels'] for it in data_grp]
            vox_list = [it['voxels'] for it in data_grp]
            bio_labels = [it['simple_biome'] for it in data_grp]
            md_list = [it['metadata'] for it in data_grp]
            write_group_batch(simple_biome, labels_list, vox_list, bio_labels, md_list, part_index)
            print(f"Wrote batch {part_index} with {len(labels_list)} chunks to {simple_biome}_chunks_part_{part_index}.npz (final flush)")

        t_write = time.time() - t_write_start
        t_total = t_collect + t_write
        n = len(chunk_buffer)
        if n > 0:
            print(f"Batch {part_index} (final): collect={t_collect:.2f}s, write={t_write:.2f}s, total={t_total:.2f}s, throughput={n/max(t_total,1e-6):.1f} chunks/s")
        part_index += 1

    # Combine parts -> final per-biome files
    # print("Combining per-batch biome files into final per-biome files...")
    # t0_combine = time.time()
    # try:
    #     part_files = [f for f in os.listdir(biome_dir) if f.endswith('.npz') and '_chunks_part_' in f]
    #     biome_to_parts = {}
    #     for fname in part_files:
    #         biome_name = fname.split('_chunks_part_')[0]
    #         biome_to_parts.setdefault(biome_name, []).append(os.path.join(biome_dir, fname))

    #     for biome_name, paths in biome_to_parts.items():
    #         try:
    #             paths_sorted = sorted(paths, key=lambda p: int(os.path.basename(p).split('_chunks_part_')[1].split('.')[0]))
    #         except Exception:
    #             paths_sorted = sorted(paths)

    #         out_path = os.path.join(biome_dir, f"{biome_name}_chunks.npz")
    #         print(f"  Combining biome: {biome_name} -> {os.path.basename(out_path)} ({len(paths_sorted)} parts)")

    #         labels_list, voxels_list, biome_labels_list, metadata_list = [], [], [], []
    #         pre_count = 0
    #         if os.path.isfile(out_path):
    #             try:
    #                 with np.load(out_path, allow_pickle=True) as ex:
    #                     if 'labels' in ex.files:
    #                         labels_list.append(ex['labels'])
    #                         pre_count = int(ex['labels'].shape[0])
    #                     if 'voxels' in ex.files:
    #                         voxels_list.append(ex['voxels'])
    #                     if 'biome_labels' in ex.files:
    #                         biome_labels_list.append(ex['biome_labels'])
    #                     if 'metadata' in ex.files:
    #                         metadata_list.append(ex['metadata'])
    #             except Exception:
    #                 pass

    #         part_count = 0
    #         for p in paths_sorted:
    #             try:
    #                 with np.load(p, allow_pickle=True) as data:
    #                     if 'labels' in data.files:
    #                         labels_list.append(data['labels'])
    #                         part_count += int(data['labels'].shape[0])
    #                     if 'voxels' in data.files:
    #                         voxels_list.append(data['voxels'])
    #                     if 'biome_labels' in data.files:
    #                         biome_labels_list.append(data['biome_labels'])
    #                     if 'metadata' in data.files:
    #                         metadata_list.append(data['metadata'])
    #             except Exception:
    #                 pass
    #             try:
    #                 os.remove(p)
    #             except Exception:
    #                 pass

    #         if labels_list:
    #             out_labels = np.concatenate(labels_list, axis=0)
    #             out_voxels = np.concatenate(voxels_list, axis=0) if voxels_list else None
    #             out_biome_labels = np.concatenate(biome_labels_list, axis=0) if biome_labels_list else None
    #             out_metadata = np.concatenate(metadata_list, axis=0) if metadata_list else None

    #             np.savez_compressed(
    #                 out_path,
    #                 labels=out_labels,
    #                 voxels=out_voxels,
    #                 biome_labels=out_biome_labels,
    #                 metadata=out_metadata,
    #             )

    #         dt_biome = time.time() - t0_combine
    #         print(f"    Done {biome_name}: appended {part_count} samples (prev {pre_count}) in {dt_biome:.2f}s")

    #     # Print counts per combined biome file
    #     biome_files = sorted([f for f in os.listdir(biome_dir) if f.endswith('_chunks.npz') and '_part_' not in f])
    #     print("Per-biome sample counts (combined):")
    #     total = 0
    #     for fname in biome_files:
    #         path = os.path.join(biome_dir, fname)
    #         try:
    #             with np.load(path, allow_pickle=True) as data:
    #                 n = data['labels'].shape[0] if 'labels' in data.files else 0
    #             biome_name = fname[:-len('_chunks.npz')]
    #             print(f"  {biome_name}: {n}")
    #             total += n
    #         except Exception as e:
    #             print(f"  {fname}: error reading ({e})")
    #     print(f"  total: {total}")
    #     dt_combine = time.time() - t0_combine
    #     print(f"Combine phase took {dt_combine:.2f}s")
    # except Exception as e:
    #     print(f"Could not combine per-biome files: {e}")
    dt_all = time.time() - t0_all
    print(f"get_voxels_by_biome total wall time: {dt_all:.2f}s")



# ========================= Village Sweep Helpers ========================= #

def _append_village_npz(file_path: str, labels_list, voxels_list, biome_labels_list, metadata_list):
    if not labels_list:
        return
    new_labels = np.array(labels_list)
    new_voxels = np.array(voxels_list)
    new_biome_labels = np.array(biome_labels_list, dtype=object)
    new_metadata = np.array(metadata_list, dtype=object)

    if os.path.isfile(file_path):
        with np.load(file_path, allow_pickle=True) as data:
            ex_labels = data['labels']
            ex_voxels = data['voxels']
            if 'biome_labels' in data.files:
                ex_biome_labels = data['biome_labels']
            elif 'biomes' in data.files:
                # Legacy per-voxel biomes; collapse to 'village' label
                ex_biome_labels = np.full((data['biomes'].shape[0],), 'village', dtype=object)
            else:
                ex_biome_labels = None
            ex_metadata = data['metadata'] if 'metadata' in data.files else None
        labels_out = np.concatenate([ex_labels, new_labels], axis=0)
        voxels_out = np.concatenate([ex_voxels, new_voxels], axis=0)
        biome_labels_out = np.concatenate([ex_biome_labels, new_biome_labels], axis=0) if ex_biome_labels is not None else new_biome_labels
        if ex_metadata is not None:
            metadata_out = np.concatenate([ex_metadata, new_metadata], axis=0)
            np.savez_compressed(file_path, labels=labels_out, voxels=voxels_out, biome_labels=biome_labels_out, metadata=metadata_out)
        else:
            np.savez_compressed(file_path, labels=labels_out, voxels=voxels_out, biome_labels=biome_labels_out)
    else:
        np.savez_compressed(file_path, labels=new_labels, voxels=new_voxels, biome_labels=new_biome_labels, metadata=new_metadata)


def _is_inside_recent_sweep(swept_regions, x, y, z) -> bool:
    """Return True if (x,z) lies inside any previously swept x/z rectangle.

    Note: We intentionally ignore Y to avoid accidentally skipping distinct
    villages at different elevations within the same x/z footprint.
    """
    for (mn, mx) in swept_regions:
        if mn[0] <= x <= mx[0] and mn[2] <= z <= mx[2]:
            return True
    return False


def _has_edge_structure(window_vox, edge_margin: int) -> bool:
    # Use a conservative set for edge checks to avoid over-chopping due to
    # peripheral elements like fences/doors touching borders.
    mask = np.isin(window_vox, np.array(list(VILLAGE_STRUCTURE_BLOCKS), dtype=np.uint8))
    if mask[0:edge_margin, :, :].any() or mask[-edge_margin:, :, :].any():
        return True
    if mask[:, :, 0:edge_margin].any() or mask[:, :, -edge_margin:].any():
        return True
    # Also treat top/bottom faces as chopped to avoid vertically cut samples
    if mask[:, 0:edge_margin, :].any() or mask[:, -edge_margin:, :].any():
        return True
    return False


def _run_village_sweep(client, cfg, center_xyz, chunk_shape,
                       structure_block_threshold, edge_margin, save_noisy, max_noisy_keep,
                       preload_grid_stride, preload_sleep_ms, village_dir, village_tag: str = None):
    """Read a large cube around a village center and slide a window across it.

    Steps:
    - Compute x/z bounds and a moderate vertical range around center.
    - Preload chunks to ensure the server has generated terrain.
    - Read a large cube (double-read optional) and choose the denser grid.
    - Slide a (chunk_shape) window with small stride over the cube.
    - For each window apply thresholds (indicator, structure, air) and edge check.
    - Save clean windows; optionally keep top noisy windows.
    - Append results to on-disk npz files.
    """
    cx, cy, cz = center_xyz
    sx, sy, sz = chunk_shape
    # Fixed-size region: (4*sx, 2*sy, 4*sz), centered at (cx, avgY, cz)
    # big_x = 8 * sx
    big_x = 256
    # big_y = 3 * sy
    big_y = 64
    # big_z = 8 * sz
    big_z= 256
    half_x = big_x // 2
    half_z = big_z // 2
    xmin = cx - half_x
    xmax = xmin + big_x - 1
    zmin = cz - half_z
    zmax = zmin + big_z - 1
    # Temporary vertical bounds (will recenter after preload avgY)
    ymin = max(0, cy - (big_y // 2))
    ymax = min(255, ymin + big_y - 1)
    print(f"[Village sweep] Center=({cx},{cy},{cz}) initial bounds x:[{xmin},{xmax}] y:[{ymin},{ymax}] z:[{zmin},{zmax}] (region={big_x}x{big_y}x{big_z})")

    # Preload around bounds to encourage chunk generation and stable reads
    try:
        ping_count = 0
        y_samples = []
        # Use chunk-aligned pings (step=16) to force neighbor chunk generation
        for gx in range(xmin, xmax + 1, 16):
            for gz in range(zmin, zmax + 1, 16):
                hp = client.getHighestYAt(Point(x=gx, y=0, z=gz))
                try:
                    y_val = int(getattr(hp, 'y', 0))
                    if y_val > 0:
                        y_samples.append(y_val)
                except Exception:
                    pass
                ping_count += 1
        print(f"[Village sweep] Preloaded chunks with {ping_count} getHighestYAt pings (stride=16)")
        if preload_sleep_ms > 0:
            time.sleep(preload_sleep_ms / 1000.0)
    except Exception as e:
        print(f"[Village sweep] Preload pings failed or partially failed: {e}")

    # Recenter vertical sweep around the average surface height from preloads
    try:
        if 'y_samples' in locals() and len(y_samples) > 0:
            avg_y = int(sum(y_samples) / len(y_samples))
            old_ymin, old_ymax = ymin, ymax
            # Use the fixed big_y height when recentering
            ymin = max(0, avg_y - (big_y // 2))
            ymax = min(255, ymin + big_y - 1)
            print(f"[Village sweep] Recentering vertical sweep: avg_highestY={avg_y} (n={len(y_samples)}) y:[{old_ymin},{old_ymax}] -> y:[{ymin},{ymax}]")
    except Exception as e:
        print(f"[Village sweep] Failed to recenter by average surface height: {e}")

    # Tiled metadata reads to stay under gRPC size limits; still do a stabilized double-read
    def _read_large_cube_tiled(xmin_, ymin_, zmin_, xmax_, ymax_, zmax_, tx=64, ty=64, tz=64):
        X = xmax_ - xmin_ + 1
        Y = ymax_ - ymin_ + 1
        Z = zmax_ - zmin_ + 1
        vox = np.zeros((X, Y, Z), dtype=np.uint8)
        bio = np.zeros((X, Y, Z), dtype=object)
        md = np.empty((X, Y, Z), dtype=object)
        md[:] = None
        for x0 in range(xmin_, xmax_ + 1, max(1, tx)):
            for y0 in range(ymin_, ymax_ + 1, max(1, ty)):
                for z0 in range(zmin_, zmax_ + 1, max(1, tz)):
                    x1 = min(x0 + tx - 1, xmax_)
                    y1 = min(y0 + ty - 1, ymax_)
                    z1 = min(z0 + tz - 1, zmax_)
                    cube = client.readCubeAndBiomeMetadata(Cube(min=Point(x=x0, y=y0, z=z0), max=Point(x=x1, y=y1, z=z1)))
                    tv, tb, tm = cube_to_voxels_and_biomes(cube, (x1-x0+1, y1-y0+1, z1-z0+1), (x0, y0, z0))
                    ex = x0 - xmin_
                    ey = y0 - ymin_
                    ez = z0 - zmin_
                    vox[ex:ex+tv.shape[0], ey:ey+tv.shape[1], ez:ez+tv.shape[2]] = tv
                    bio[ex:ex+tv.shape[0], ey:ey+tv.shape[1], ez:ez+tv.shape[2]] = tb
                    md[ex:ex+tv.shape[0], ey:ey+tv.shape[1], ez:ez+tv.shape[2]] = tm
        return vox, bio, md

    print("[Village sweep] Reading large cube from server (stabilized, tiled)...")
    tile_y = min(64, big_y)
    grid1_vox, grid1_bio, grid1_md = _read_large_cube_tiled(xmin, ymin, zmin, xmax, ymax, zmax, tx=64, ty=tile_y, tz=64)
    struct_count1 = int(np.isin(grid1_vox, np.array(list(VILLAGE_STRUCTURE_BLOCKS), dtype=np.uint8)).sum())
    delay_ms = int(getattr(cfg.data, 'double_read_delay_ms', 400))
    if delay_ms > 0:
        time.sleep(delay_ms / 1000.0)
    grid2_vox, grid2_bio, grid2_md = _read_large_cube_tiled(xmin, ymin, zmin, xmax, ymax, zmax, tx=64, ty=tile_y, tz=64)
    struct_count2 = int(np.isin(grid2_vox, np.array(list(VILLAGE_STRUCTURE_BLOCKS), dtype=np.uint8)).sum())
    pick_second = struct_count2 > struct_count1
    grid_vox, grid_bio, grid_md = (grid2_vox, grid2_bio, grid2_md) if pick_second else (grid1_vox, grid1_bio, grid1_md)
    print(f"[Village sweep] Read grid shaped vox={grid_vox.shape}, biomes={grid_bio.shape} (picked={'second' if pick_second else 'first'})")

    # Optional metadata sanity check
    debug_md = bool(getattr(cfg.data, 'debug_village_metadata', True))
    if debug_md:
        try:
            nonempty_mask = np.vectorize(lambda m: isinstance(m, dict) and len(m) > 0)(grid_md)
            md_count = int(nonempty_mask.sum())
            print(f"[Village sweep] Metadata non-empty blocks in large cube: {md_count}")
            if md_count > 0:
                coords = np.argwhere(nonempty_mask)
                for i in range(min(5, coords.shape[0])):
                    xi, yi, zi = coords[i]
                    wx, wy, wz = xmin + int(xi), ymin + int(yi), zmin + int(zi)
                    print(f"  md at world [{wx},{wy},{wz}]: {grid_md[xi, yi, zi]}")
        except Exception as e:
            print(f"[Village sweep] Metadata debug probe failed: {e}")

    # Slide a (chunk_shape) window with small stride
    small_stride = int(getattr(cfg.data, 'village_sweep_stride', 4))
    x_max_start = grid_vox.shape[0] - chunk_shape[0]
    y_max_start = grid_vox.shape[1] - chunk_shape[1]
    z_max_start = grid_vox.shape[2] - chunk_shape[2]
    def _steps(limit, step):
        return 0 if limit < 0 else (limit // step) + 1
    total_windows = _steps(max(0, x_max_start), small_stride) * _steps(max(0, y_max_start), max(1, small_stride)) * _steps(max(0, z_max_start), small_stride)
    print(f"[Village sweep] Sweeping {total_windows} windows with stride {small_stride}")

    clean_labels, clean_vox, clean_bio, clean_md = [], [], [], []
    noisy_labels, noisy_vox, noisy_bio, noisy_md = [], [], [], []
    scanned_windows = rejected_structure = rejected_air = filtered_as_chopped = 0
    max_structure = 0
    # Track structure counts for reporting
    sum_struct_all = 0
    cnt_struct_all = 0
    sum_struct_clean = 0
    cnt_struct_clean = 0
    sum_struct_noisy = 0
    cnt_struct_noisy = 0

    for xs in range(0, max(0, x_max_start + 1), small_stride):
        for ys in range(0, max(0, y_max_start + 1), max(1, small_stride)):
            for zs in range(0, max(0, z_max_start + 1), small_stride):
                scanned_windows += 1
                wv = grid_vox[xs:xs+chunk_shape[0], ys:ys+chunk_shape[1], zs:zs+chunk_shape[2]]
                wb = grid_bio[xs:xs+chunk_shape[0], ys:ys+chunk_shape[1], zs:zs+chunk_shape[2]]
                wm = grid_md[xs:xs+chunk_shape[0], ys:ys+chunk_shape[1], zs:zs+chunk_shape[2]]

                mask_all = np.isin(wv, np.array(list(VILLAGE_BLOCKS_FOR_COUNT), dtype=np.uint8))
                structure_count = int(mask_all.sum())
                max_structure = max(max_structure, structure_count)

                # Threshold on total structure presence
                if structure_count < structure_block_threshold:
                    rejected_structure += 1
                    continue

                # Air ratio filter to avoid underground windows
                min_air_ratio = float(getattr(cfg.data, 'min_air_ratio', 0.20))
                air_ratio = float((wv == BlockType.AIR).sum()) / float(wv.size)
                if air_ratio < min_air_ratio:
                    rejected_air += 1
                    continue

                # Face-based chop check on all faces using structure blocks
                chopped = _has_edge_structure(wv, edge_margin)
                if chopped:
                    filtered_as_chopped += 1
                wx = xmin + xs + chunk_shape[0] // 2
                wy = ymin + ys + chunk_shape[1] // 2
                wz = zmin + zs + chunk_shape[2] // 2
                if not chopped:
                    # Always store a single biome label 'village' per sample
                    clean_labels.append(np.array([wx, wy, wz]))
                    clean_vox.append(wv)
                    clean_bio.append('village')
                    clean_md.append(wm)
                    sum_struct_clean += structure_count
                    cnt_struct_clean += 1
                elif save_noisy:
                    noisy_labels.append(np.array([wx, wy, wz]))
                    noisy_vox.append(wv)
                    noisy_bio.append('village')
                    noisy_md.append(wm)
                    sum_struct_noisy += structure_count
                    cnt_struct_noisy += 1

                sum_struct_all += structure_count
                cnt_struct_all += 1

                if scanned_windows % 2000 == 0:
                    print(
                        f"[Village sweep] Scanned {scanned_windows}/{total_windows} | kept={len(clean_labels)+len(noisy_labels)} "
                        f"(rejects_structure={rejected_structure}, rejects_air={rejected_air}, edge_chopped={filtered_as_chopped}; max_struct={max_structure})"
                    )

    kept_noisy = len(noisy_labels)
    if save_noisy and len(noisy_labels) > max_noisy_keep:
        sel = np.random.choice(len(noisy_labels), size=max_noisy_keep, replace=False)
        noisy_labels = [noisy_labels[i] for i in sel]
        noisy_vox = [noisy_vox[i] for i in sel]
        noisy_bio = [noisy_bio[i] for i in sel]
        noisy_md = [noisy_md[i] for i in sel]
        kept_noisy = len(noisy_labels)

    avg_struct_all = (sum_struct_all / cnt_struct_all) if cnt_struct_all > 0 else 0.0
    avg_struct_clean = (sum_struct_clean / cnt_struct_clean) if cnt_struct_clean > 0 else 0.0
    avg_struct_noisy = (sum_struct_noisy / cnt_struct_noisy) if cnt_struct_noisy > 0 else 0.0

    print(
        f"[Village sweep] Summary: scanned={scanned_windows} kept_clean={len(clean_labels)} kept_noisy={kept_noisy} "
        f"rejects_structure={rejected_structure} rejects_air={rejected_air} edge_chopped={filtered_as_chopped} max_struct={max_structure} "
        f"avg_struct_all={avg_struct_all:.1f} avg_struct_clean={avg_struct_clean:.1f} avg_struct_noisy={avg_struct_noisy:.1f}"
    )

    # Persist results to per-village files
    clean_name = f"village_clean_chunks_{village_tag}.npz" if village_tag else 'village_clean_chunks.npz'
    noisy_name = f"village_chunks_{village_tag}.npz" if village_tag else 'village_chunks.npz'
    # Write compact per-sample biome labels ('village')
    _append_village_npz(os.path.join(village_dir, clean_name), clean_labels, clean_vox, clean_bio, clean_md)
    if save_noisy:
        _append_village_npz(os.path.join(village_dir, noisy_name), noisy_labels, noisy_vox, noisy_bio, noisy_md)

    return ((xmin, ymin, zmin), (xmax, ymax, zmax))



def get_villages(client, cfg, data_dir):
    """Scrape village chunks only, using an efficient in-memory sweep.

    High-level steps:
    1) Traverse in a square spiral with a coarse stride until a village is detected.
    2) On detection, read one large cube around the center (expand x/z heavily, y moderately).
    3) Slide a window of size chunk_shape across the cube in-memory with small stride.
    4) For each window: compute thresholds and edge checks to classify as clean vs noisy.
    5) Append clean windows to village_clean_chunks.npz, noisy to village_chunks.npz.
    6) Record swept area to avoid redundant sweeps; resume coarse traversal.
    """

    # Configs and thresholds
    chunk_shape = tuple(cfg.data.chunk_shape)
    coarse_xz_step = int(getattr(cfg.data, 'coarse_stride', max(12, chunk_shape[0])))
    small_stride = int(getattr(cfg.data, 'village_sweep_stride', 8))
    sweep_radius_chunks = int(getattr(cfg.data, 'village_sweep_radius_chunks', 3))
    village_block_threshold = int(getattr(cfg.data, 'village_block_threshold', 60))
    structure_block_threshold = int(getattr(cfg.data, 'structure_block_threshold', 60))
    edge_margin = int(getattr(cfg.data, 'edge_margin', 1))
    save_noisy = bool(getattr(cfg.data, 'save_noisy_village', False))
    max_noisy_keep = int(getattr(cfg.data, 'max_noisy_village_per_sweep', 100))
    preload_grid_stride = int(getattr(cfg.data, 'preload_grid_stride', chunk_shape[0]))
    preload_sleep_ms = int(getattr(cfg.data, 'preload_sleep_ms', 250))

    print(
        "Village scrape config: "
        f"chunk_shape={chunk_shape}, coarse_stride={coarse_xz_step}, "
        f"sweep_stride={small_stride}, sweep_radius_chunks={sweep_radius_chunks}, "
        f"village_thresh={village_block_threshold}, structure_thresh={structure_block_threshold}, "
        f"edge_margin={edge_margin}, save_noisy={save_noisy}"
    )

    base_dir = os.path.join(cfg.data.data_dir, "Voxels")
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)
    village_dir = os.path.join(base_dir, "villages_32_metadata_new")
    if not os.path.isdir(village_dir):
        os.makedirs(village_dir)

    def append_to_file(file_path, labels_list, voxels_list, biomes_list, metadata_list):
        _append_village_npz(file_path, labels_list, voxels_list, biomes_list, metadata_list)

    # Recent swept areas to avoid redundant sweeps
    swept_regions = []  # list of ((xmin, ymin, zmin), (xmax, ymax, zmax))
    def inside_recent_sweep(x, y, z):
        return _is_inside_recent_sweep(swept_regions, x, y, z)

    def run_village_sweep(cx, cy, cz, village_tag):
        bounds = _run_village_sweep(
            client, cfg, (cx, cy, cz), chunk_shape, structure_block_threshold,
            edge_margin, save_noisy, max_noisy_keep,
            preload_grid_stride, preload_sleep_ms,
            village_dir, village_tag
        )
        swept_regions.append(bounds)
        print("[Village sweep] Completed and recorded swept region.")

    # Use Pyubiomes to locate villages near spawn, then sweep around those centers
    # Read world seed from server properties
    seed = None
    try:
        props_path = os.path.join("server", "server.properties")
        if os.path.isfile(props_path):
            with open(props_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("level-seed="):
                        val = line.split("=", 1)[1].strip()
                        seed = int(val)
                        break
    except Exception as e:
        print(f"[Villages] Failed to read seed from server.properties: {e}")

    if seed is None:
        raise RuntimeError("Could not determine world seed for village search. Ensure server/server.properties has a numeric level-seed.")

    radius_blocks = int(getattr(cfg.data, 'village_search_radius_blocks', 10024))
    max_sites = getattr(cfg.data, 'max_village_sites', None)
    try:
        max_sites_int = int(max_sites) if max_sites is not None else None
    except Exception:
        max_sites_int = None

    print(f"[Villages] Finding villages for seed={seed} near spawn with radius={radius_blocks} blocks...")
    village_centers = find_villages_near_spawn(seed=seed, radius_blocks=radius_blocks, max_results=max_sites_int)
    print(f"[Villages] Found {len(village_centers)} village center(s).")

    sweeps = 0
    for (vx, vz) in village_centers:
        # Determine Y at center and validate chunk generation
        highest_point = client.getHighestYAt(Point(x=vx, y=0, z=vz))
        vy = highest_point.y
        if vy == 0:
            # Nudge server to generate; retry once
            read_cube(client, (vx, 0, vz), (vx, 0, vz))
            highest_point = client.getHighestYAt(Point(x=vx, y=0, z=vz))
            vy = highest_point.y
            if vy == 0:
                print(f"[Villages] Skipping village at ({vx},{vz}) — could not resolve surface Y.")
                continue

        if inside_recent_sweep(vx, vy, vz):
            print(f"[Villages] Skip center inside recent sweep at ({vx},{vz}).")
            continue

        print(f"[Villages] Sweeping around village center at ({vx},{vy},{vz})")
        village_tag = f"{vx}_{vy}_{vz}"
        run_village_sweep(vx, vy, vz, village_tag)
        sweeps += 1

    # Combine per-village files into final combined files
    # try:
    #     clean_parts = sorted([
    #         os.path.join(village_dir, f) for f in os.listdir(village_dir)
    #         if f.startswith('village_clean_chunks_') and f.endswith('.npz')
    #     ])
    #     noisy_parts = sorted([
    #         os.path.join(village_dir, f) for f in os.listdir(village_dir)
    #         if f.startswith('village_chunks_') and f.endswith('.npz')
    #     ])

    #     combined_clean = os.path.join(village_dir, 'village_clean_chunks.npz')
    #     combined_noisy = os.path.join(village_dir, 'village_chunks.npz')

    #     if _combine_npz_files(clean_parts, combined_clean):
    #         print(f"Combined clean village chunks -> {combined_clean}")
    #     if save_noisy and _combine_npz_files(noisy_parts, combined_noisy):
    #         print(f"Combined noisy village chunks -> {combined_noisy}")

    #     # Print final counts
    #     files = [p for p in [combined_clean, combined_noisy] if os.path.isfile(p)]
    #     print('Village sample counts (combined):')
    #     total = 0
    #     for path in files:
    #         with np.load(path, allow_pickle=True) as data:
    #             n = data['labels'].shape[0] if 'labels' in data.files else 0
    #         print(f"  {os.path.basename(path)}: {n}")
    #         total += n
    #     print(f"  total: {total}")
    # except Exception as e:
    #     print(f"Could not combine village files: {e}")


def get_targeted_biome(client, cfg):
    """Collect chunks specifically targeting a biome label using Pyubiomes to find locations.
    
    This function:
    1. Reads target_biome_label from config
    2. Finds all MC biome IDs that map to that label via SIMPLE_BIOME_MAPPING
    3. Uses Pyubiomes to efficiently find locations where those biomes exist
    4. Scrapes voxel data at those locations
    5. Only saves chunks where the majority biome matches the target label
    """
    target_biome_label = str(getattr(cfg.data, 'targeted_biome_label', None))
    if not target_biome_label:
        raise ValueError("target_biome_label must be set in config for targeted_biome mode")
    
    print(f'[Targeted Biome] Collecting chunks for biome label: {target_biome_label}')
    
    # Get all Pyubiomes biome IDs that map to this simple label
    biome_ids = get_pyubiomes_ids_for_simple_biome(target_biome_label)
    if not biome_ids:
        raise ValueError(f"No biome IDs found for label '{target_biome_label}'. Check SIMPLE_BIOME_MAPPING.")
    print(f'[Targeted Biome] Searching for Pyubiomes biome IDs: {biome_ids}')
    
    # Read world seed from server properties
    seed = None
    try:
        props_path = os.path.join("server", "server.properties")
        if os.path.isfile(props_path):
            with open(props_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("level-seed="):
                        val = line.split("=", 1)[1].strip()
                        seed = int(val)
                        break
    except Exception as e:
        print(f"[Targeted Biome] Failed to read seed from server.properties: {e}")
    
    if seed is None:
        raise RuntimeError("Could not determine world seed. Ensure server/server.properties has a numeric level-seed.")
    
    print(f'[Targeted Biome] Using seed: {seed}')
    
    # Search parameters
    radius_blocks = int(getattr(cfg.data, 'targeted_biome_radius', 16384))
    stride = int(getattr(cfg.data, 'targeted_biome_stride', 32))
    num_samples = int(getattr(cfg.data, 'num_samples', 1000))
    # NOTE: Candidate location finding is pluggable.
    # Default to the fast path (seed-based locator). In this repo, `seed_utils`
    # will prefer a locally-compiled cubiomes shared library when available.
    locator = str(getattr(cfg.data, 'targeted_biome_locator', 'pyubiomes')).lower().strip()
    # Cap how many candidate points we will consider (prevents huge scans when
    # the target biome is rare or additional filters are strict).
    max_candidates = int(getattr(cfg.data, 'targeted_biome_max_candidates', max(5000, num_samples * 50)))
    
    def normalize_biome_label(label):
        if label is None:
            return None
        if isinstance(label, bytes):
            try:
                label = label.decode("utf-8")
            except Exception:
                label = str(label)
        label = str(label)
        if ":" in label:
            label = label.split(":")[-1]
        return label.lower()

    def simple_biome_from_raw(raw_label: str) -> str:
        raw = normalize_biome_label(raw_label)
        if raw is None:
            return "unknown"
        return SIMPLE_BIOME_MAPPING.get(raw, raw)

    def find_biome_locations_via_server(spawn_x: int, spawn_z: int) -> list:
        """
        Slow but authoritative: stride over the world and ask the running server
        what biome is at each (x,z). This avoids Pyubiomes/cubiomes mismatch.
        """
        # Use a fixed Y for 1.12 (biomes are 2D); any reasonable Y works.
        yq = 64
        found = []
        x1b, z1b = spawn_x - radius_blocks, spawn_z - radius_blocks
        x2b, z2b = spawn_x + radius_blocks, spawn_z + radius_blocks
        total = max(1, ((x2b - x1b) // stride) * ((z2b - z1b) // stride))
        pbar = tqdm(total=total, desc=f"[Targeted Biome] Server-scan for {target_biome_label}", unit="pt")
        try:
            for x in range(x1b, x2b, stride):
                for z in range(z1b, z2b, stride):
                    pbar.update(1)
                    try:
                        raw = client.getBiomeAt(Point(x=x, y=yq, z=z)).biome
                    except Exception:
                        continue
                    if simple_biome_from_raw(raw) == target_biome_label:
                        found.append((x, z))
                        if len(found) >= max_candidates:
                            return found
        finally:
            pbar.close()
        return found

    # Find biome locations
    print(f'[Targeted Biome] Searching radius={radius_blocks} blocks with stride={stride} (locator={locator})...')
    if locator == "server":
        # Center the scan on Pyubiomes spawn estimate (close enough; server is authoritative).
        try:
            from Pyubiomes import get_spawn as pyu_get_spawn, Versions as PyuVersions
            spawn_x, spawn_z = pyu_get_spawn(seed, PyuVersions.MC_1_12)
        except Exception:
            spawn_x, spawn_z = (0, 0)
        biome_locations = find_biome_locations_via_server(int(spawn_x), int(spawn_z))
    else:
        biome_locations = find_biome_locations(
            seed=seed,
            biome_ids=biome_ids,
            version=None,  # Use default (1.12)
            radius_blocks=radius_blocks,
            stride=stride,
        )
    print(f'[Targeted Biome] Found {len(biome_locations)} candidate locations')
    
    if not biome_locations:
        print(f'[Targeted Biome] No locations found for biome {target_biome_label}. Exiting.')
        return
    
    # Debug: Show first few locations
    print(f'[Targeted Biome] First 5 candidate locations: {biome_locations[:5]}')
    
    # Setup output directories
    base_dir = os.path.join(cfg.data.data_dir, "Voxels")
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)
    biome_dir = os.path.join(base_dir, f"targeted_{target_biome_label}")
    if not os.path.isdir(biome_dir):
        os.makedirs(biome_dir)
    
    # Run tag for part file naming
    run_tag = f"seed_{seed}"
    
    # Chunk shape and collection parameters
    chunk_shape = tuple(cfg.data.chunk_shape)
    min_air_ratio = float(getattr(cfg.data, 'map_min_air_ratio', 0.20))
    try:
        y_shift_abs = abs(int(getattr(cfg.data, 'biome_y_shift_abs', 12)))
    except Exception:
        y_shift_abs = 12
    
    BATCH_SIZE = int(getattr(cfg.data, 'map_batch_size', 1000))
    chunk_buffer = []
    part_index = 0
    batch_start_time = time.time()
    
    def get_majority_simple_biome(biome_array):
        flat = biome_array.ravel()
        labels = [normalize_biome_label(b) for b in flat if b is not None]
        if not labels:
            return "unknown"
        unique, counts = np.unique(np.array(labels, dtype=object), return_counts=True)
        majority = unique[int(np.argmax(counts))]
        return SIMPLE_BIOME_MAPPING.get(majority, majority)
    
    def write_batch(labels_list, voxels_list, biome_labels_list, metadata_list, part_idx):
        file_path = os.path.join(biome_dir, f"{target_biome_label}_chunks_part_{run_tag}_{part_idx}.npz")
        new_labels = np.array(labels_list)
        new_voxels = np.array(voxels_list)
        biome_labels = np.array(biome_labels_list, dtype=object)
        new_metadata = np.array(metadata_list, dtype=object)
        _atomic_savez_compressed(
            file_path,
            labels=new_labels,
            voxels=new_voxels,
            biome_labels=biome_labels,
            metadata=new_metadata,
        )
    
    collected = 0
    skipped_biome_mismatch = 0
    skipped_air_ratio = 0
    skipped_height = 0
    biome_mismatch_counts = {}  # Track what biomes we're actually finding
    debug_mismatch_examples = []  # Store first few mismatch examples for debugging
    MAX_DEBUG_EXAMPLES = 10
    
    t0_all = time.time()
    
    # Use tqdm for progress bar
    pbar = tqdm(biome_locations, desc=f"[Targeted Biome] Scanning for {target_biome_label}", unit="loc")
    
    for loc_idx, (x, z) in enumerate(pbar):
        if collected >= num_samples:
            break
        
        # Update progress bar with current stats
        pbar.set_postfix({
            'collected': collected,
            'mismatch': skipped_biome_mismatch,
            'air': skipped_air_ratio,
            'height': skipped_height
        })
        
        # Resolve surface Y
        highest_point = client.getHighestYAt(Point(x=x, y=0, z=z))
        y = highest_point.y
        if highest_point.y == 0:
            # Force chunk generation
            read_cube(client, (x, 0, z), (x, 0, z))
            highest_point = client.getHighestYAt(Point(x=x, y=0, z=z))
            if highest_point.y == 0:
                skipped_height += 1
                continue
            y = highest_point.y
        
        # Apply random Y shift
        try:
            y_shift = int(np.random.randint(-y_shift_abs, y_shift_abs + 1))
        except Exception:
            y_shift = 0
        cy = max(0, min(255, int(y + y_shift)))
        labels_xyz = np.array([x, cy, z])
        
        # Read chunk with padding for center-crop (same as get_voxels_by_biome)
        sx, sy, sz = chunk_shape
        padded_shape = (sx + 16, sy, sz + 16)
        vox_padded, bio_padded, md_padded = get_vox_chunk((x, cy, z), padded_shape, client)
        x_start = (padded_shape[0] - sx) // 2
        z_start = (padded_shape[2] - sz) // 2
        x_end = x_start + sx
        z_end = z_start + sz
        voxels_arr = vox_padded[x_start:x_end, :, z_start:z_end]
        biomes_arr = bio_padded[x_start:x_end, :, z_start:z_end]
        metadata_arr = md_padded[x_start:x_end, :, z_start:z_end]
        
        # Air ratio filter
        air_ratio = float((voxels_arr == BlockType.AIR).sum()) / float(voxels_arr.size)
        if air_ratio < min_air_ratio:
            skipped_air_ratio += 1
            continue
        
        # Get majority simple biome and filter for target
        simple_biome = get_majority_simple_biome(biomes_arr)
        if simple_biome != target_biome_label:
            skipped_biome_mismatch += 1
            # Track what biome we actually found for debugging
            biome_mismatch_counts[simple_biome] = biome_mismatch_counts.get(simple_biome, 0) + 1
            
            # Store detailed debug examples
            if len(debug_mismatch_examples) < MAX_DEBUG_EXAMPLES:
                # Get all unique biomes in this chunk for context
                flat_biomes = biomes_arr.ravel()
                raw_labels = [normalize_biome_label(b) for b in flat_biomes if b is not None]
                if raw_labels:
                    unique_raw, counts_raw = np.unique(np.array(raw_labels, dtype=object), return_counts=True)
                    biome_breakdown = dict(zip(unique_raw, counts_raw))
                else:
                    biome_breakdown = {}
                debug_mismatch_examples.append({
                    'pos': (x, z),
                    'expected': target_biome_label,
                    'majority_found': simple_biome,
                    'breakdown': biome_breakdown,
                })
            continue
        
        # Accept this chunk
        chunk_buffer.append({
            "labels": labels_xyz,
            "voxels": voxels_arr,
            "metadata": metadata_arr,
            "simple_biome": simple_biome,
        })
        collected += 1
        
        # Write batch if buffer is full
        if len(chunk_buffer) >= BATCH_SIZE:
            t_collect = time.time() - batch_start_time
            t_write_start = time.time()
            
            labels_list = [it['labels'] for it in chunk_buffer]
            vox_list = [it['voxels'] for it in chunk_buffer]
            bio_labels = [it['simple_biome'] for it in chunk_buffer]
            md_list = [it['metadata'] for it in chunk_buffer]
            write_batch(labels_list, vox_list, bio_labels, md_list, part_index)
            
            t_write = time.time() - t_write_start
            t_total = t_collect + t_write
            n = len(chunk_buffer)
            print(f"[Targeted Biome] Batch {part_index}: {n} chunks, collect={t_collect:.2f}s, write={t_write:.2f}s, total={t_total:.2f}s")
            
            chunk_buffer = []
            part_index += 1
            batch_start_time = time.time()
        
        if collected > 0 and collected % 100 == 0:
            tqdm.write(f"[Targeted Biome] Milestone: Collected {collected}/{num_samples} chunks")
    
    # Final partial flush
    if chunk_buffer:
        t_collect = time.time() - batch_start_time
        t_write_start = time.time()
        
        labels_list = [it['labels'] for it in chunk_buffer]
        vox_list = [it['voxels'] for it in chunk_buffer]
        bio_labels = [it['simple_biome'] for it in chunk_buffer]
        md_list = [it['metadata'] for it in chunk_buffer]
        write_batch(labels_list, vox_list, bio_labels, md_list, part_index)
        
        t_write = time.time() - t_write_start
        n = len(chunk_buffer)
        print(f"[Targeted Biome] Final batch {part_index}: {n} chunks")
    
    pbar.close()
    
    dt_all = time.time() - t0_all
    print(f"\n[Targeted Biome] Collection complete:")
    print(f"  Total collected: {collected}")
    print(f"  Skipped (biome mismatch): {skipped_biome_mismatch}")
    print(f"  Skipped (air ratio): {skipped_air_ratio}")
    print(f"  Skipped (no height): {skipped_height}")
    print(f"  Wall time: {dt_all:.2f}s")
    
    # Show what biomes we actually found instead of the target
    if biome_mismatch_counts:
        print(f"\n[Targeted Biome] Biome mismatch breakdown (what we found instead of '{target_biome_label}'):")
        sorted_mismatches = sorted(biome_mismatch_counts.items(), key=lambda x: -x[1])
        for biome, count in sorted_mismatches[:10]:  # Top 10
            print(f"    {biome}: {count}")
    
    # Show detailed debug examples
    if debug_mismatch_examples:
        print(f"\n[Targeted Biome] First {len(debug_mismatch_examples)} mismatch examples (detailed):")
        for i, ex in enumerate(debug_mismatch_examples):
            print(f"  Example {i+1}: pos={ex['pos']}")
            print(f"    Expected (target): {ex['expected']}")
            print(f"    Majority found: {ex['majority_found']}")
            print(f"    Raw biome breakdown in chunk:")
            # Sort by count descending
            sorted_breakdown = sorted(ex['breakdown'].items(), key=lambda x: -x[1])
            for biome_name, cnt in sorted_breakdown[:5]:
                pct = 100.0 * cnt / sum(ex['breakdown'].values())
                print(f"      {biome_name}: {cnt} ({pct:.1f}%)")


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
# def main(mode: str = "screenies", num_samples: int = 1000, load: bool = True):
def main(cfg: DictConfig) -> None:
    if MC_API == EVOCRAFT:
        channel = grpc.insecure_channel(
            'localhost:5001',
            options=[
                ('grpc.max_send_message_length', 500 * 1024 * 1024),
                ('grpc.max_receive_message_length', 500 * 1024 * 1024),
            ],
        )
        client = clients.python.src.main.proto.minecraft_pb2_grpc.MinecraftServiceStub(channel)
    elif MC_API == MINEDOJO:
        client = MinedojoClient()
    # NOTE: num_samples is an upper bound on number of screenshots (since we don't take screenshots from on top of water).
    #  We do take all voxel samples, though.
    data_dir = str(getattr(cfg.data, 'data_dir', 'data'))
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if cfg.data_gen.mode == "voxels":
        get_voxels(client, cfg=cfg, data_dir=data_dir)
    elif cfg.data_gen.mode == "voxels_by_biome":
        get_voxels_by_biome(client, cfg=cfg)
    elif cfg.data_gen.mode == "voxels_in_area":
        get_voxels_in_area(client, cfg=cfg, data_dir=data_dir)
    elif cfg.data_gen.mode == "villages":
        get_villages(client, cfg=cfg, data_dir=data_dir)
    elif cfg.data_gen.mode == "caves":
        get_caves(client, cfg=cfg, data_dir=data_dir)
    elif cfg.data_gen.mode == "nether":
        get_nether(client, cfg=cfg, data_dir=data_dir)
    elif cfg.data_gen.mode == "targeted_biome":
        get_targeted_biome(client, cfg=cfg)
    else:
        raise ValueError("Invalid mode")




if __name__ == "__main__":
    main()