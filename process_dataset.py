from data_utils import process_class_conditional_dataset, process_class_conditional_dataset_memmap, BlockBiomeConverter, build_biome_dataset_from_parts_memmap_balanced, build_biome_dataset_from_parts_memmap_distributed
from data_utils import build_biome_dataset_from_parts, build_biome_dataset_from_parts_memmap, build_balanced_biome_dataset, chop_to_16
from visualization_utils import MinecraftVisualizerPyVista, save_chunks
import numpy as np
from pathlib import Path
import os
import time
import wandb
import psutil

# ---- comand for downloading biome chunk files ----
# hf download SakanaAI/dreamcubed --repo-type dataset --local-dir data/ --include "biome_chunks_32_combined/*"

def log_memory_usage(stage_name):
    """Log current memory usage with wandb."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_gb = mem_info.rss / (1024 ** 3)
    
    vm = psutil.virtual_memory()
    system_mem_used_gb = vm.used / (1024 ** 3)
    system_mem_total_gb = vm.total / (1024 ** 3)
    system_mem_percent = vm.percent
    
    print(f"\n[MEMORY] {stage_name}")
    print(f"  Process Memory: {mem_gb:.2f} GB")
    print(f"  System Memory: {system_mem_used_gb:.2f} / {system_mem_total_gb:.2f} GB ({system_mem_percent:.1f}%)")
    
    wandb.log({
        f"memory/process_gb_{stage_name}": mem_gb,
        f"memory/system_used_gb_{stage_name}": system_mem_used_gb,
        f"memory/system_percent_{stage_name}": system_mem_percent,
    })
    
    return mem_gb

if __name__ == "__main__":
    # Initialize wandb
    wandb.init(
        project="dream-cubed-data-processing",
        name="process_biome_data",
        config={
            "biome_dir": "data/32_biome_parts/",
            "method": "memmap",
        }
    )
    
    log_memory_usage("start")
    # =============================================================
    #               Natural + Human processing
    # =============================================================
    # natural_biome_dir = "data/<natural_biome_chunks>/"
    # human_clean_biome_dir = "data/<human_map_chunks>"
    # compression_mapping_path = "assets/human_block_compression_mapping_all.json"
    # compression_mapping_by_name_path = "assets/human_block_compression_mapping_all_names.json"
    # compression_block_types_path = "assets/block_types_updated.json"
    # natural_biomes =  [
    #     'village_clean',
    #     "beaches",
    #     "jungle",
    #     "birch_forest",
    #     "desert",
    #     "extreme_hills",
    #     "forest",
    #     "ocean",
    #     "plains",
    #     "river",
    #     "savanna",
    #     "swampland",
    #     "taiga",
    #     'ice',
    #     'cave',
    # ]
    # human_biomes = [
    #     "ancient_empire",
    #     "kingdom_of_sarano",
    #     "marethea",
    #     "osirion",
    #     "sora_kingdom",
    #     "the_5_bridges",
    # ]
    # biome_names = natural_biomes + human_biomes
    # human_single_file_overrides = {
    #     biome: os.path.join(human_clean_biome_dir, biome)
    #     for biome in human_biomes
    # }

    # # Define structural blocks for village filtering (example block IDs)
    # village_structural_blocks = [160, 224, 3, 196, 88, 41, 15, 210, 114, 149, 53, 24, 249, 219, 190, 222, 63]
    
    # output_path = os.path.join("data", "natural_human_balanced_dataset_cleaned")

    # log_memory_usage("before_build")
    
    # t_start = time.perf_counter()
    # t_build_start = t_start
    # memmap_dir = build_biome_dataset_from_parts_memmap_balanced(
    #     biome_dir=natural_biome_dir,
    #     biome_names=biome_names,
    #     output_path=output_path,
    #     total_size=1_000_000,
    #     single_file_overrides=human_single_file_overrides,
    #     apply_block_compression=True,
    #     compression_mapping_path=compression_mapping_path,
    #     compression_mapping_by_name_path=compression_mapping_by_name_path,
    #     compression_block_types_path=compression_block_types_path,
    #     compression_strict=True,
    #     apply_metadata_remap=True,
    # )
    
    # log_memory_usage("after_build")
    
    # process_class_conditional_dataset_memmap(memmap_dir, val_split=0.1)
    # t_end = time.perf_counter()
    
    # log_memory_usage("after_process")

    # total_time_s = t_end - t_start



    # =============================================================
    #               Natural Occurence Dataset
    # =============================================================
    # biome_dir = "data/32_biome_parts_combined/"  # directory containing files like '{biome}_chunks.npz'
    # natural_biome_distribution = {'ocean': 35.65,
    # 'forest': 11.76,
    # 'plains': 7.64,
    # 'ice': 7.39,
    # 'extreme_hills': 7.18,
    # 'desert': 5.89,
    # 'taiga': 5.52,
    # 'river': 4.02,
    # 'beaches': 3.9,
    # 'savanna': 3.59,
    # 'birch_forest': 2.81,
    # 'swampland': 2.81,
    # 'jungle': 1.83}
    # biome_names = list(natural_biome_distribution.keys())

    # # Define structural blocks for village filtering (example block IDs)
    # village_structural_blocks = [160, 224, 3, 196, 88, 41, 15, 210, 114, 149, 53, 24, 249, 219, 190, 222, 63]
    
    # output_path = os.path.join("data", "natural_occurrence_dataset")

    # log_memory_usage("before_build")
    
    # t_start = time.perf_counter()
    # t_build_start = t_start

    # memmap_dir = build_biome_dataset_from_parts_memmap_distributed(
    #     biome_dir=biome_dir,
    #     biome_names=biome_names,
    #     output_path=output_path,
    #     biome_distribution=natural_biome_distribution,
    #     total_size=1_000_000,
    #     village_relabel_blocks=village_structural_blocks,
    #     village_relabel_threshold=60,
    #     village_relabel_label="village",
    #     village_relabel_source_biomes=biome_names,
    #     match_cave_to_village=True,
    #     cave_biome_name="cave",
    #     cave_label="cave",
    # )
    
    # log_memory_usage("after_build")
    
    # process_class_conditional_dataset_memmap(memmap_dir, val_split=0.1)
    # t_end = time.perf_counter()
    
    # log_memory_usage("after_process")

    # total_time_s = t_end - t_start



    # =============================================================
    #               Village Boosted Dataset
    # =============================================================
    # biome_dir = "data/<natural_biome_chunks>/"  # directory containing files like '{biome}_chunks.npz'
    # biome_names = [
    #     'village_clean',
    #     "beaches",
    #     "jungle",
    #     "birch_forest",
    #     "desert",
    #     "extreme_hills",
    #     "forest",
    #     "ocean",
    #     "plains",
    #     "river",
    #     "savanna",
    #     "swampland",
    #     "taiga",
    #     'ice',
    #     'cave',
    # ]

    # # Define structural blocks for village filtering (example block IDs)
    # village_structural_blocks = [160, 224, 3, 196, 88, 41, 15, 210, 114, 149, 53, 24, 249, 219, 190, 222, 63]
    
    # output_path = os.path.join("data", "village_boosted_dataset")

    # log_memory_usage("before_build")
    
    # t_start = time.perf_counter()
    # t_build_start = t_start
    
    # memmap_dir = build_biome_dataset_from_parts_memmap_balanced(
    #     biome_dir=biome_dir,
    #     biome_names=biome_names,
    #     output_path=output_path,
    #     total_size=1000000,
    #     boosted_biome='village_clean',
    #     boost_factor=2.0

    # )
    
    # log_memory_usage("after_build")
    
    # process_class_conditional_dataset_memmap(memmap_dir, val_split=0.1)
    # t_end = time.perf_counter()
    
    # log_memory_usage("after_process")

    # total_time_s = t_end - t_start



    # =============================================================
    #               Balanced Dataset
    # =============================================================
    biome_dir = "data/<natural_biome_chunks>/"  # directory containing files like '{biome}_chunks.npz'
    biome_names = [
        'village_clean',
        "beaches",
        "jungle",
        "birch_forest",
        "desert",
        "extreme_hills",
        "forest",
        "ocean",
        "plains",
        "river",
        "savanna",
        "swampland",
        "taiga",
        'ice',
        'cave',
    ]

    # Define structural blocks for village filtering (example block IDs)
    village_structural_blocks = [160, 224, 3, 196, 88, 41, 15, 210, 114, 149, 53, 24, 249, 219, 190, 222, 63]
    
    output_path = os.path.join("data", "balanced_dataset")

    log_memory_usage("before_build")
    
    t_start = time.perf_counter()
    t_build_start = t_start
    
    memmap_dir = build_biome_dataset_from_parts_memmap_balanced(
        biome_dir=biome_dir,
        biome_names=biome_names,
        output_path=output_path,
        total_size=1000000,
    )
    
    log_memory_usage("after_build")
    
    process_class_conditional_dataset_memmap(memmap_dir, val_split=0.1)
    t_end = time.perf_counter()
    
    log_memory_usage("after_process")

    total_time_s = t_end - t_start

    wandb.finish()


