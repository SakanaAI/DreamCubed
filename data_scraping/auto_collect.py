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

# Recognize multiple signals of readiness across vanilla/sponge versions
READY_PATTERNS = [
    re.compile(r"Query running on .*:25565"),
    re.compile(r"No rcon password set .* rcon disabled!"),
    re.compile(r"Done \([0-9.]+s\)"),
    re.compile(r"Starting Minecraft server on .*:25565"),
]


def build_java_cmd() -> str:
    """Construct a cross-platform Java launch command for the server.

    Priority:
      1) Respect env JAVA_CMD if provided (should include everything necessary)
      2) On macOS, prefer /usr/libexec/java_home -v 1.8 to ensure Java 8
      3) Else, prefer JAVA_HOME if set; fallback to plain 'java'
    Always append ' -jar <jar> nogui'.
    """
    env_java_cmd = os.environ.get("JAVA_CMD")
    jar_path = JAR_NAME
    base = None
    if env_java_cmd:
        # Assume user provided full command; just ensure jar + nogui present if omitted
        if "-jar" in env_java_cmd:
            cmd = env_java_cmd
        else:
            cmd = f"{env_java_cmd} -jar {jar_path} nogui"
        return cmd

    if sys.platform == "darwin":
        base = "/usr/libexec/java_home -v 1.8 --exec java"
    else:
        java_home = os.environ.get("JAVA_HOME")
        if java_home:
            base = f"\"{os.path.join(java_home, 'bin', 'java')}\""
        else:
            base = "java"
    return f"{base} -jar {jar_path} nogui"


def update_level_seed(seed: int) -> None:
    with open(SERVER_PROPERTIES, "r") as f:
        content = f.read()
    if "level-seed=" in content:
        content = re.sub(r"^level-seed=.*$", f"level-seed={seed}", content, flags=re.MULTILINE)
    else:
        content = content.rstrip("\n") + f"\nlevel-seed={seed}\n"
    with open(SERVER_PROPERTIES, "w") as f:
        f.write(content)


def delete_world_dir() -> None:
    if os.path.isdir(WORLD_DIR):
        shutil.rmtree(WORLD_DIR)


def start_server(log_path: str):
    env = os.environ.copy()
    # Ensure we run from server directory
    
    # Platform-specific arguments
    kwargs = {}
    if sys.platform == "win32":
        # Windows: Create new process group
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        # Unix: Create new session/process group
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
            **kwargs
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
                    if any(p.search(line) for p in READY_PATTERNS):
                        return True
        else:
            # If no log yet and process died, fail fast
            if proc is not None and proc.poll() is not None:
                return False
        # If process died during wait, fail fast
        if proc is not None and proc.poll() is not None:
            return False
        time.sleep(0.5)
    return False


def stop_server(proc, grace_timeout: int = 30) -> None:
    if proc is None:
        return
    try:
        # Try to send 'stop' to stdin
        if proc.stdin:
            try:
                proc.stdin.write(b"stop\n")
                proc.stdin.flush()
            except Exception:
                pass
        # Give it a moment to exit cleanly
        try:
            proc.wait(timeout=grace_timeout)
            return
        except subprocess.TimeoutExpired:
            pass
        
        # Force kill logic
        if sys.platform == "win32":
            # Windows: Kill process tree
            try:
                subprocess.call(["taskkill", "/F", "/T", "/PID", str(proc.pid)])
            except Exception:
                pass
        else:
            # Unix: Send signals to process group
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


def run_gen_data(mode: str = "villages", overrides=None) -> int:
    # Hydra script at repo root
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "gen_data.py"), f"data_gen.mode={mode}"]
    if overrides:
        cmd.extend(list(overrides))
    return subprocess.call(cmd)


def get_data_part_dir(mode: str, targeted_biome_label: str = None) -> str:
    """Determine the output directory for part files based on mode.
    Matches logic in gen_data.py.
    """
    base_dir = os.path.join("data", "Voxels")
    
    if mode == "nether":
        return os.path.join(base_dir, "nether_parts")
    elif mode == "caves":
        return os.path.join(base_dir, "caves_parts")
    elif mode == "villages":
        return os.path.join(base_dir, "villages_32_metadata_new")
    elif mode == "voxels":
        # Default map name in config is kargeth
        return os.path.join(base_dir, "kargeth_parts")
    elif mode == "targeted_biome":
        # gen_data writes part files directly into Voxels/targeted_{label}/
        # (not a *_parts directory).
        if not targeted_biome_label:
            targeted_biome_label = "unknown"
        return os.path.join(base_dir, f"targeted_{targeted_biome_label}")
    
    return base_dir


def iterate_collection(
    seeds,
    iterations: int,
    mode: str,
    server_name: str = None,
    targeted_biome_label: str = None,
    targeted_biome_radius: int = None,
    targeted_biome_stride: int = None,
    targeted_biome_locator: str = None,
    targeted_biome_max_candidates: int = None,
) -> None:
    seeds_to_run = []
    if seeds:
        seeds_to_run = [int(s) for s in seeds]
    else:
        rng = random.Random()
        for _ in range(iterations):
            # Use full 64-bit signed Java seed range
            seeds_to_run.append(rng.randrange(-2**63, 2**63))

    for idx, seed in enumerate(seeds_to_run, 1):
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = os.path.join(os.path.dirname(__file__), "outputs", ts)
        os.makedirs(run_dir, exist_ok=True)
        log_path = os.path.join(run_dir, f"server_{seed}.log")

        print(f"[Run {idx}/{len(seeds_to_run)}] Using seed {seed}")
        print("Stopping any previous server instance (if running) will be handled per-run.")

        # Ensure old world removed and seed set
        update_level_seed(seed)
        delete_world_dir()

        # Start server and wait ready
        proc = start_server(log_path)
        print(f"Server starting (pid={proc.pid}). Tailing {log_path} for readiness...")
        ready = wait_for_server_ready(log_path, proc=proc)
        if not ready:
            rc = proc.poll()
            if rc is not None:
                print(f"Server process exited early with code {rc}. Check log at {log_path}.")
            else:
                print("Server did not signal readiness within timeout. Stopping and skipping this seed.")
            stop_server(proc)
            continue
        print("Server is ready. Running gen_data...")

        overrides = []
        if mode == "targeted_biome":
            if targeted_biome_label:
                overrides.append(f"data.targeted_biome_label={targeted_biome_label}")
            if targeted_biome_radius is not None:
                overrides.append(f"data.targeted_biome_radius={int(targeted_biome_radius)}")
            if targeted_biome_stride is not None:
                overrides.append(f"data.targeted_biome_stride={int(targeted_biome_stride)}")
            if targeted_biome_locator:
                overrides.append(f"data.targeted_biome_locator={targeted_biome_locator}")
            if targeted_biome_max_candidates is not None:
                overrides.append(f"data.targeted_biome_max_candidates={int(targeted_biome_max_candidates)}")

        rc = run_gen_data(mode=mode, overrides=overrides)
        print(f"gen_data finished with code {rc}")

        # if rc == 0:
        #     print("Data collection successful. Running combination step...")
        #     target_dir = get_data_part_dir(mode, targeted_biome_label=targeted_biome_label)
        #     if os.path.isdir(target_dir):
        #         print(f"Combining files in {target_dir}...")
        #         try:
        #             combined_part_files.run(
        #                 input_dir=target_dir,
        #                 output_dir=None,  # Default to input_dir + "_combined"
        #                 group_size=5,
        #                 seed=None,
        #                 server_name=server_name
        #             )
        #         except Exception as e:
        #             print(f"Error running combination step: {e}")
        #     else:
        #         print(f"Warning: Expected output directory {target_dir} does not exist. Skipping combination.")

        print("Stopping server...")
        stop_server(proc)


def main():
    parser = argparse.ArgumentParser(description="Automate Minecraft world regen and data collection")
    parser.add_argument("--seeds", nargs="*", help="List of seeds to run. If omitted, uses --iterations random seeds.")
    parser.add_argument("--iterations", type=int, default=1, help="Number of iterations with random seeds when no seeds provided")
    parser.add_argument("--mode", type=str, default="villages", help="gen_data data_gen.mode value (default: villages)")
    parser.add_argument("--server-name", type=str, default=None, help="Server name identifier for combined files")
    # targeted_biome mode overrides (optional; if omitted, gen_data uses conf/config.yaml)
    parser.add_argument("--targeted-biome-label", type=str, default=None, help="Override data.targeted_biome_label (targeted_biome mode)")
    parser.add_argument("--targeted-biome-radius", type=int, default=None, help="Override data.targeted_biome_radius (blocks)")
    parser.add_argument("--targeted-biome-stride", type=int, default=None, help="Override data.targeted_biome_stride (blocks)")
    parser.add_argument("--targeted-biome-locator", type=str, default=None, help="Override data.targeted_biome_locator (e.g. pyubiomes/server)")
    parser.add_argument("--targeted-biome-max-candidates", type=int, default=None, help="Override data.targeted_biome_max_candidates")
    args = parser.parse_args()

    iterate_collection(
        args.seeds,
        args.iterations,
        args.mode,
        args.server_name,
        targeted_biome_label=args.targeted_biome_label,
        targeted_biome_radius=args.targeted_biome_radius,
        targeted_biome_stride=args.targeted_biome_stride,
        targeted_biome_locator=args.targeted_biome_locator,
        targeted_biome_max_candidates=args.targeted_biome_max_candidates,
    )


if __name__ == "__main__":
    main()


