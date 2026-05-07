import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import json
import os
import time
from typing import Any, Dict, Tuple

from utils import square_spiral


DEFAULT_HOST = "localhost:5001"
DEFAULT_DIMENSION = "overworld"
DEFAULT_MAX_RADIUS_BLOCKS = 10_000
DEFAULT_WINDOW_SIZE = 32
DEFAULT_SAMPLE_Y = 0
DEFAULT_PROGRESS_EVERY = 100
DEFAULT_SAVE_EVERY = 500


def load_grpc_modules():
    try:
        import grpc  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "This script requires grpcio in the active Python environment. "
            "Install it in the env you use to run Minecraft scraping, then rerun."
        ) from exc

    try:
        from clients.python.src.main.proto import minecraft_pb2
        from clients.python.src.main.proto import minecraft_pb2_grpc
    except ImportError:
        from utils import patch_grpc_evocraft_imports

        patch_grpc_evocraft_imports()
        from clients.python.src.main.proto import minecraft_pb2
        from clients.python.src.main.proto import minecraft_pb2_grpc

    return grpc, minecraft_pb2, minecraft_pb2_grpc


def load_simple_biome_mapping(repo_root: str) -> Dict[str, str]:
    """Load SIMPLE_BIOME_MAPPING directly from gen_data.py."""
    gen_data_path = os.path.join(repo_root, "gen_data.py")
    with open(gen_data_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=gen_data_path)

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "SIMPLE_BIOME_MAPPING":
                value = ast.literal_eval(node.value)
                if not isinstance(value, dict):
                    raise ValueError("SIMPLE_BIOME_MAPPING is not a dict")
                return value

    raise ValueError("Could not find SIMPLE_BIOME_MAPPING in gen_data.py")


def read_world_seed(repo_root: str) -> str:
    props_path = os.path.join(repo_root, "server", "server.properties")
    try:
        with open(props_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("level-seed="):
                    return line.split("=", 1)[1].strip() or "unknown"
    except FileNotFoundError:
        pass
    return "unknown"


def normalize_biome_label(label: str) -> str:
    if not label:
        return "unknown"
    if ":" in label:
        return label.split(":", 1)[1]
    return label


def map_biome_label(raw_biome: str, simple_biome_mapping: Dict[str, str]) -> str:
    return simple_biome_mapping.get(raw_biome, raw_biome)


def get_window_sample_positions(center_x: int, center_z: int, window_size: int) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    quarter = max(1, window_size // 4)
    return (
        (center_x - quarter, center_z - quarter),
        (center_x - quarter, center_z + quarter),
        (center_x + quarter, center_z - quarter),
        (center_x + quarter, center_z + quarter),
    )


def read_window_biome_counts(
    client: Any,
    minecraft_pb2: Any,
    center_x: int,
    center_z: int,
    sample_y: int,
    window_size: int,
) -> Counter:
    counts = Counter()
    for sample_x, sample_z in get_window_sample_positions(center_x, center_z, window_size):
        response = client.getBiomeAt(
            minecraft_pb2.Point(
                x=sample_x,
                y=sample_y,
                z=sample_z,
            )
        )
        counts[normalize_biome_label(response.biome)] += 1
    return counts


def estimate_num_windows(max_radius_blocks: int, window_size: int) -> int:
    count = 0
    spiral_idx = 0
    while True:
        x, z = square_spiral(spiral_idx)
        center_x = x * window_size
        center_z = z * window_size
        if max(abs(center_x), abs(center_z)) > max_radius_blocks:
            break
        count += 1
        spiral_idx += 1
    return count


def sorted_counter_dict(counter: Counter) -> Dict[str, int]:
    return {k: int(v) for k, v in sorted(counter.items(), key=lambda item: (-item[1], item[0]))}


def distribution_dict(counter: Counter) -> Dict[str, float]:
    total = sum(counter.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in sorted(counter.items(), key=lambda item: (-item[1], item[0]))}


def make_default_out_path(repo_root: str, world_seed: str, dimension: str) -> str:
    safe_seed = world_seed if world_seed else "unknown"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join(repo_root, "analysis", "biome_distributions")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"biome_dist_{dimension}_seed_{safe_seed}_{timestamp}.json")


def write_summary(
    out_path: str,
    *,
    host: str,
    dimension: str,
    world_seed: str,
    max_radius_blocks: int,
    num_windows_processed: int,
    start_index: int,
    window_size: int,
    sample_y: int,
    samples_per_window: int,
    raw_counts: Counter,
    mapped_counts: Counter,
    unmapped_counts: Counter,
    started_at: float,
) -> None:
    payload = {
        "host": host,
        "dimension": dimension,
        "world_seed": world_seed,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started_at,
        "max_radius_blocks": max_radius_blocks,
        "num_windows_processed": num_windows_processed,
        "start_index": start_index,
        "window_size": window_size,
        "sample_y": sample_y,
        "samples_per_window": samples_per_window,
        "total_biome_samples": int(sum(mapped_counts.values())),
        "mapped_biome_counts": sorted_counter_dict(mapped_counts),
        "mapped_biome_distribution": distribution_dict(mapped_counts),
        "raw_biome_counts": sorted_counter_dict(raw_counts),
        "unmapped_raw_biomes": sorted_counter_dict(unmapped_counts),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)


def print_progress(processed_windows: int, estimated_total_windows: int, mapped_counts: Counter, started_at: float) -> None:
    elapsed = max(time.time() - started_at, 1e-6)
    windows_per_sec = processed_windows / elapsed
    print(f"[{processed_windows}/{estimated_total_windows}] {windows_per_sec:.2f} windows/s")
    for biome, count in mapped_counts.most_common(10):
        pct = 100.0 * count / max(sum(mapped_counts.values()), 1)
        print(f"  {biome}: {count} ({pct:.2f}%)")


def run(
    host: str,
    dimension: str,
    max_radius_blocks: int,
    window_size: int,
    sample_y: int,
    start_index: int,
    progress_every: int,
    save_every: int,
    out_path: str,
) -> str:
    if max_radius_blocks <= 0:
        raise ValueError("--max-radius-blocks must be positive")
    if window_size <= 0:
        raise ValueError("--window-size must be positive")
    if progress_every <= 0:
        raise ValueError("--progress-every must be positive")
    if save_every <= 0:
        raise ValueError("--save-every must be positive")

    repo_root = os.path.dirname(os.path.abspath(__file__))
    simple_biome_mapping = load_simple_biome_mapping(repo_root)
    world_seed = read_world_seed(repo_root)

    if not out_path:
        out_path = make_default_out_path(repo_root, world_seed, dimension)
    else:
        out_path = os.path.abspath(out_path)
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    raw_counts = Counter()
    mapped_counts = Counter()
    unmapped_counts = Counter()
    started_at = time.time()
    samples_per_window = 4
    estimated_total_windows = estimate_num_windows(max_radius_blocks=max_radius_blocks, window_size=window_size)

    grpc, minecraft_pb2, minecraft_pb2_grpc = load_grpc_modules()
    channel = grpc.insecure_channel(
        host,
        options=[
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
        ],
    )
    client = minecraft_pb2_grpc.MinecraftServiceStub(channel)

    client.setActiveDimension(
        minecraft_pb2.DimensionRequest(
            dimension=dimension,
            teleport_player=False,
        )
    )

    print(f"Sampling biome distribution from {host} in dimension '{dimension}'")
    print(f"World seed: {world_seed}")
    print(f"Max radius from origin: {max_radius_blocks} blocks")
    print(f"Window size: {window_size} x {window_size}")
    print(f"Samples per window: {samples_per_window}")
    print(f"Estimated windows: {estimated_total_windows}")
    print(f"Estimated biome samples: {estimated_total_windows * samples_per_window}")
    print(f"Sample y: {sample_y}")
    print(f"Output: {out_path}")
    if estimated_total_windows > 100_000:
        print("Warning: this radius/window combination is large and may still take a long time.")

    try:
        processed_windows = 0
        spiral_idx = start_index
        while True:
            x, z = square_spiral(spiral_idx)
            center_x = x * window_size
            center_z = z * window_size
            if max(abs(center_x), abs(center_z)) > max_radius_blocks:
                break

            window_counts = read_window_biome_counts(
                client=client,
                minecraft_pb2=minecraft_pb2,
                center_x=center_x,
                center_z=center_z,
                sample_y=sample_y,
                window_size=window_size,
            )

            raw_counts.update(window_counts)
            for raw_biome, count in window_counts.items():
                mapped_biome = map_biome_label(raw_biome, simple_biome_mapping)
                mapped_counts[mapped_biome] += count
                if raw_biome not in simple_biome_mapping:
                    unmapped_counts[raw_biome] += count

            processed_windows += 1
            spiral_idx += 1
            if processed_windows % progress_every == 0 or processed_windows == estimated_total_windows:
                print_progress(processed_windows, estimated_total_windows, mapped_counts, started_at)

            if processed_windows % save_every == 0 or processed_windows == estimated_total_windows:
                write_summary(
                    out_path,
                    host=host,
                    dimension=dimension,
                    world_seed=world_seed,
                    max_radius_blocks=max_radius_blocks,
                    num_windows_processed=processed_windows,
                    start_index=start_index,
                    window_size=window_size,
                    sample_y=sample_y,
                    samples_per_window=samples_per_window,
                    raw_counts=raw_counts,
                    mapped_counts=mapped_counts,
                    unmapped_counts=unmapped_counts,
                    started_at=started_at,
                )
    finally:
        channel.close()

    print("\nFinal mapped biome distribution:")
    for biome, frac in distribution_dict(mapped_counts).items():
        print(f"  {biome}: {frac:.6f}")

    if unmapped_counts:
        print("\nUnmapped raw biomes encountered:")
        for biome, count in unmapped_counts.most_common():
            print(f"  {biome}: {count}")

    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate biome distribution in a Minecraft world by square-spiral sampling chunk footprints."
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="gRPC host, default: localhost:5001")
    parser.add_argument("--dimension", default=DEFAULT_DIMENSION, help="Dimension to sample: overworld, nether, or the_end")
    parser.add_argument("--max-radius-blocks", type=int, default=DEFAULT_MAX_RADIUS_BLOCKS, help="Stop once the spiral reaches this block radius from origin")
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE, help="Window width/depth in blocks")
    parser.add_argument("--sample-y", type=int, default=DEFAULT_SAMPLE_Y, help="Y used for the sparse biome queries")
    parser.add_argument("--start-index", type=int, default=0, help="Starting square-spiral index")
    parser.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY, help="Print progress every N window samples")
    parser.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY, help="Checkpoint the output JSON every N window samples")
    parser.add_argument("--out", default="", help="Optional output JSON path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        host=args.host,
        dimension=args.dimension,
        max_radius_blocks=args.max_radius_blocks,
        window_size=args.window_size,
        sample_y=args.sample_y,
        start_index=args.start_index,
        progress_every=args.progress_every,
        save_every=args.save_every,
        out_path=args.out,
    )
