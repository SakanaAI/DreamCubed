import argparse
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime


SERVER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "server"))
SERVER_PROPERTIES = os.path.join(SERVER_DIR, "server.properties")
WORLD_DIR = os.path.join(SERVER_DIR, "structure_enabled_world")
JAR_NAME = "spongevanilla-1.12.2-7.3.0.jar"

# Match the readiness signals used by the existing automation.
READY_PATTERNS = [
    re.compile(r"Query running on .*:25565"),
    re.compile(r"No rcon password set .* rcon disabled!"),
    re.compile(r"Done \([0-9.]+s\)"),
    re.compile(r"Starting Minecraft server on .*:25565"),
]


def build_java_cmd() -> str:
    env_java_cmd = os.environ.get("JAVA_CMD")
    if env_java_cmd:
        if "-jar" in env_java_cmd:
            return env_java_cmd
        return f"{env_java_cmd} -jar {JAR_NAME} nogui"

    if sys.platform == "darwin":
        base = "/usr/libexec/java_home -v 1.8 --exec java"
    else:
        java_home = os.environ.get("JAVA_HOME")
        if java_home:
            base = f"\"{os.path.join(java_home, 'bin', 'java')}\""
        else:
            base = "java"
    return f"{base} -jar {JAR_NAME} nogui"


def update_level_seed(seed: int) -> None:
    with open(SERVER_PROPERTIES, "r", encoding="utf-8") as f:
        content = f.read()

    if "level-seed=" in content:
        content = re.sub(r"^level-seed=.*$", f"level-seed={seed}", content, flags=re.MULTILINE)
    else:
        content = content.rstrip("\n") + f"\nlevel-seed={seed}\n"

    with open(SERVER_PROPERTIES, "w", encoding="utf-8") as f:
        f.write(content)


def delete_world_dir() -> None:
    if os.path.isdir(WORLD_DIR):
        shutil.rmtree(WORLD_DIR)


def start_server(log_path: str):
    env = os.environ.copy()
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["preexec_fn"] = os.setsid

    with open(log_path, "ab", buffering=0) as logf:
        proc = subprocess.Popen(
            build_java_cmd(),
            cwd=SERVER_DIR,
            shell=True,
            stdout=logf,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE,
            env=env,
            **kwargs,
        )
    return proc


def wait_for_server_ready(log_path: str, timeout_sec: int = 600, proc=None) -> bool:
    start = time.time()
    last_size = 0
    while time.time() - start < timeout_sec:
        if os.path.isfile(log_path):
            with open(log_path, "rb") as f:
                f.seek(last_size)
                chunk = f.read()
                if not chunk:
                    time.sleep(1.0)
                    continue
                last_size = f.tell()
                text = chunk.decode(errors="ignore")
                for line in text.splitlines():
                    if any(pattern.search(line) for pattern in READY_PATTERNS):
                        return True
        else:
            if proc is not None and proc.poll() is not None:
                return False

        if proc is not None and proc.poll() is not None:
            return False
        time.sleep(0.5)
    return False


def stop_server(proc, grace_timeout: int = 30) -> None:
    if proc is None:
        return

    try:
        if proc.stdin:
            try:
                proc.stdin.write(b"stop\n")
                proc.stdin.flush()
            except Exception:
                pass

        try:
            proc.wait(timeout=grace_timeout)
            return
        except subprocess.TimeoutExpired:
            pass

        if sys.platform == "win32":
            try:
                subprocess.call(["taskkill", "/F", "/T", "/PID", str(proc.pid)])
            except Exception:
                pass
        else:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            except Exception:
                pass
            try:
                proc.wait(timeout=10)
                return
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
    finally:
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass


def resolve_seeds(seeds, iterations: int):
    if seeds:
        return [int(seed) for seed in seeds]

    rng = random.Random()
    return [rng.randrange(-2**63, 2**63) for _ in range(iterations)]


def build_biome_dist_cmd(args, out_path: str):
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "get_biome_dist.py"),
        "--host",
        args.host,
        "--dimension",
        args.dimension,
        "--max-radius-blocks",
        str(args.max_radius_blocks),
        "--window-size",
        str(args.window_size),
        "--sample-y",
        str(args.sample_y),
        "--start-index",
        str(args.start_index),
        "--progress-every",
        str(args.progress_every),
        "--save-every",
        str(args.save_every),
        "--out",
        out_path,
    ]
    return cmd


def iterate_collection(args) -> None:
    seeds_to_run = resolve_seeds(args.seeds, args.iterations)
    repo_root = os.path.dirname(__file__)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for idx, seed in enumerate(seeds_to_run, 1):
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = os.path.join(repo_root, "outputs", ts)
        os.makedirs(run_dir, exist_ok=True)

        log_path = os.path.join(run_dir, f"server_{seed}.log")
        out_path = os.path.join(output_dir, f"biome_dist_{args.dimension}_seed_{seed}.json")

        print(f"[Run {idx}/{len(seeds_to_run)}] Using seed {seed}")
        print(f"Biome distribution output: {out_path}")

        proc = None
        try:
            update_level_seed(seed)
            delete_world_dir()

            proc = start_server(log_path)
            print(f"Server starting (pid={proc.pid}). Waiting for readiness via {log_path}...")
            ready = wait_for_server_ready(log_path, timeout_sec=args.server_ready_timeout, proc=proc)
            if not ready:
                rc = proc.poll()
                if rc is not None:
                    print(f"Server exited early with code {rc}. Check log at {log_path}.")
                else:
                    print("Server did not signal readiness within timeout. Skipping this seed.")
                continue

            print("Server is ready. Running biome distribution sweep...")
            cmd = build_biome_dist_cmd(args, out_path)
            rc = subprocess.call(cmd, cwd=repo_root)
            print(f"get_biome_dist.py finished with code {rc}")
        finally:
            print("Stopping server...")
            stop_server(proc)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automate Minecraft world regeneration and biome-distribution collection."
    )
    parser.add_argument("--seeds", nargs="*", help="List of seeds to run. If omitted, uses --iterations random seeds.")
    parser.add_argument("--iterations", type=int, default=1, help="Number of random-seed runs when --seeds is omitted")
    parser.add_argument("--host", type=str, default="localhost:5001", help="Minecraft gRPC host for get_biome_dist.py")
    parser.add_argument("--dimension", type=str, default="overworld", help="Dimension to sample: overworld, nether, or the_end")
    parser.add_argument("--max-radius-blocks", type=int, default=10_000, help="Stop once the spiral reaches this block radius from origin")
    parser.add_argument("--window-size", type=int, default=32, help="Window width/depth in blocks")
    parser.add_argument("--sample-y", type=int, default=0, help="Y level used for the biome-plane query")
    parser.add_argument("--start-index", type=int, default=0, help="Starting square-spiral index")
    parser.add_argument("--progress-every", type=int, default=100, help="Print progress every N window samples")
    parser.add_argument("--save-every", type=int, default=500, help="Checkpoint the biome JSON every N window samples")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "analysis", "biome_distributions"),
        help="Directory where one biome-distribution JSON will be written per seed",
    )
    parser.add_argument("--server-ready-timeout", type=int, default=600, help="Seconds to wait for server readiness")
    return parser.parse_args()


def main():
    args = parse_args()
    iterate_collection(args)


if __name__ == "__main__":
    main()
