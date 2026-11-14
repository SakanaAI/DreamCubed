import os
import typing as T


import ctypes
import subprocess

import Pyubiomes
from Pyubiomes import Versions, Structures, get_spawn, structure_in_area, is_valid_structure_pos, PyuMap, biome_at_pos


_DREAM_CUBIOMES_LIB: T.Optional[ctypes.CDLL] = None


def _dream_lib_path() -> str:
	# Keep build products inside the repo so we don't depend on site-packages wheels.
	return os.path.join(os.path.dirname(__file__), "native", "libdream_cubiomes.so")


def _ensure_dream_cubiomes_loaded() -> T.Optional[ctypes.CDLL]:
	"""
	Build and load our local cubiomes-based shared library if possible.
	This avoids relying on the (often wrong) Pyubiomes wheel's bundled cubiomes.
	"""
	global _DREAM_CUBIOMES_LIB
	if _DREAM_CUBIOMES_LIB is not None:
		return _DREAM_CUBIOMES_LIB

	so_path = _dream_lib_path()
	if not os.path.isfile(so_path):
		# Best-effort compile. If this fails, callers will fall back to Pyubiomes.
		src_path = os.path.join(os.path.dirname(__file__), "native", "dream_cubiomes.c")
		try:
			os.makedirs(os.path.dirname(so_path), exist_ok=True)
			cmd = [
				"gcc",
				# NOTE: cubiomes has UB that can miscompile under -O3 in practice.
				# We intentionally stay at -O2 and disable strict-aliasing/wrapv to
				# preserve correct worldgen outputs.
				"-O2",
				"-std=c99",
				"-D_DEFAULT_SOURCE",
				"-fno-strict-aliasing",
				"-fwrapv",
				"-fPIC",
				"-shared",
				"-o",
				so_path,
				src_path,
				"-lm",
			]
			subprocess.check_call(cmd)
		except Exception:
			return None

	try:
		lib = ctypes.CDLL(so_path)
		# int dream_biome_id_at(int mc, int64_t seed, int x, int z)
		lib.dream_biome_id_at.argtypes = [ctypes.c_int, ctypes.c_longlong, ctypes.c_int, ctypes.c_int]
		lib.dream_biome_id_at.restype = ctypes.c_int

		# int dream_find_biome_locations(int mc, int64_t seed, int x1, int z1, int x2, int z2, int stride,
		#     const int* wanted, int wanted_len, int* out_xz, int max_out)
		lib.dream_find_biome_locations.argtypes = [
			ctypes.c_int,
			ctypes.c_longlong,
			ctypes.c_int,
			ctypes.c_int,
			ctypes.c_int,
			ctypes.c_int,
			ctypes.c_int,
			ctypes.POINTER(ctypes.c_int),
			ctypes.c_int,
			ctypes.POINTER(ctypes.c_int),
			ctypes.c_int,
		]
		lib.dream_find_biome_locations.restype = ctypes.c_int

		_DREAM_CUBIOMES_LIB = lib
		print(f"[seed_utils] Loaded fast cubiomes backend: {so_path}")
		return lib
	except Exception:
		return None


def find_villages_near_spawn(
	seed: int,
	version: T.Optional[object] = None,
	radius_blocks: int = 4096,
	max_results: T.Optional[int] = None,
) -> T.List[T.Tuple[int, int]]:
	"""
	Return village center (x, z) coordinates near the world spawn using Pyubiomes.

	Args:
		seed: Java world seed (signed 64-bit integer)
		version: Pyubiomes Versions constant. If None, pick a default matching 1.12.x.
		radius_blocks: search square radius around spawn in blocks.
		max_results: optional cap on number of villages returned.

	Returns:
		List of (x, z) for village centers.
	"""
	if Pyubiomes is None:
		raise ImportError("Pyubiomes is not installed or failed to import.")

	# Default to 1.12 if not provided (server is 1.12.2)
	mc_version = version
	if mc_version is None:
		# Fallbacks by common naming across forks of Pyubiomes
		mc_version = getattr(Versions, "MC_1_12", None) or getattr(Versions, "MC_1_12_2", None) or getattr(Versions, "MC_1_13", None)
	spawn = get_spawn(seed, mc_version)
	# Some wrappers return tuple (x, z); others return (x, y, z) or an object
	if isinstance(spawn, (tuple, list)) and len(spawn) >= 2:
		spawn_x, spawn_z = int(spawn[0]), int(spawn[-1])
	else:
		# Last resort: assume attributes
		spawn_x = int(getattr(spawn, "x", 0))
		spawn_z = int(getattr(spawn, "z", 0))

	print(f"[pyubiomes] Seed={seed} Spawn=({spawn_x}, {spawn_z})")
	# Use BLOCK coordinates for search bounds (matches this Pyubiomes build)
	x1b, z1b = spawn_x - radius_blocks, spawn_z - radius_blocks
	x2b, z2b = spawn_x + radius_blocks, spawn_z + radius_blocks
	print(f"[pyubiomes] Search bounds (blocks) x:[{x1b},{x2b}] z:[{z1b},{z2b}] (radius {radius_blocks} blocks)")

	# Use structure_in_area recursively to find ALL attempted spawns in the rectangle
	villages: T.List[T.Tuple[int, int]] = []
	seen: T.Set[T.Tuple[int, int]] = set()

	# Work in half-open BLOCK bounds [x1, x2) x [z1, z2) to satisfy x1<x2, z1<z2
	stack: T.List[T.Tuple[int, int, int, int]] = [(x1b, z1b, x2b + 1, z2b + 1)]
	while stack:
		x1h, z1h, x2h, z2h = stack.pop()
		# Require strictly increasing bounds
		if not (x1h < x2h and z1h < z2h):
			continue
		# Query this rectangle (convert to closed interval by subtracting 1 from max)
		pos = structure_in_area(Structures.Village, seed, x1h, z1h, x2h - 1, z2h - 1, mc_version)
		if not pos:
			continue
		print(f"[pyubiomes] Found village")
		# Normalize position → BLOCK coords
		if isinstance(pos, (tuple, list)) and len(pos) >= 2:
			bx, bz = int(pos[0]), int(pos[1])
		else:
			bx = int(getattr(pos, "x", 0))
			bz = int(getattr(pos, "z", 0))
		if (bx, bz) not in seen:
			seen.add((bx, bz))
			# Validate the structure position before accepting
			valid = is_valid_structure_pos(Structures.Village, seed, bx, bz, mc_version)
			if valid:
				print(f"[pyubiomes] Valid village at coords ({bx},{bz})")
				villages.append((bx, bz))
			else:
				print(f"[pyubiomes] Invalid village at coords ({bx},{bz}) — skipping")
			if max_results is not None and len(villages) >= max_results:
				return villages
		# Subdivide into four quadrants that exclude the found cell (block coords)
		stack.append((x1h, z1h, bx,     bz))      # left-top
		stack.append((bx + 1, z1h, x2h,  bz))      # right-top
		stack.append((x1h, bz + 1, bx,   z2h))     # left-bot
		stack.append((bx + 1, bz + 1, x2h, z2h))   # right-bot

	return villages


def find_biome_locations(
	seed: int,
	biome_ids: T.Union[int, T.List[int]],
	version: T.Optional[object] = None,
	radius_blocks: int = 4096,
	stride: int = 32,
) -> T.List[T.Tuple[int, int]]:
	"""
	Return (x, z) coordinates where any of the specified biomes are found near the world spawn.
	
	Args:
		seed: Java world seed (signed 64-bit integer)
		biome_ids: Single Pyubiomes Biomes constant (int) or list of biome IDs
		version: Pyubiomes Versions constant. If None, pick a default.
		radius_blocks: search square radius around spawn in blocks.
		stride: step size in blocks for sampling.
		
	Returns:
		List of (x, z) coordinates where any of the biomes was detected.
	"""
	if Pyubiomes is None:
		raise ImportError("Pyubiomes is not installed or failed to import.")
	
	# Normalize biome_ids to a list
	if isinstance(biome_ids, int):
		biome_ids = [biome_ids]
		
	# Default to 1.12 if not provided (server is 1.12.2)
	mc_version = version
	if mc_version is None:
		# Fallbacks by common naming across forks of Pyubiomes
		mc_version = getattr(Versions, "MC_1_12", None) or getattr(Versions, "MC_1_12_2", None) or getattr(Versions, "MC_1_13", None)
		
	spawn = get_spawn(seed, mc_version)
	# Some wrappers return tuple (x, z); others return (x, y, z) or an object
	if isinstance(spawn, (tuple, list)) and len(spawn) >= 2:
		spawn_x, spawn_z = int(spawn[0]), int(spawn[-1])
	else:
		# Last resort: assume attributes
		spawn_x = int(getattr(spawn, "x", 0))
		spawn_z = int(getattr(spawn, "z", 0))
		
	print(f"[pyubiomes] Seed={seed} Spawn=({spawn_x}, {spawn_z}) Search for Biomes={biome_ids}")
	
	x1b, z1b = spawn_x - radius_blocks, spawn_z - radius_blocks
	x2b, z2b = spawn_x + radius_blocks, spawn_z + radius_blocks
	
	# Prefer our repo-local cubiomes implementation (fast + matches the actual world).
	lib = _ensure_dream_cubiomes_loaded()
	if lib is not None:
		# Keep this print extremely lightweight (this function can be called a lot).
		# If you want to silence it, remove or gate behind a config flag.
		# print("[seed_utils] Using repo-local cubiomes shared library for biome scan.")
		wanted_arr = (ctypes.c_int * len(biome_ids))(*[int(b) for b in biome_ids])
		# Upper bound on results: scanning is cheap, but keep output bounded.
		# We can tune later, but this avoids pathological memory usage.
		max_out = max(1, ((x2b - x1b) // max(1, stride)) * ((z2b - z1b) // max(1, stride)))
		max_out = min(max_out, 2_000_000)  # hard cap
		out_buf = (ctypes.c_int * (max_out * 2))()
		n = int(
			lib.dream_find_biome_locations(
				int(mc_version),
				ctypes.c_longlong(int(seed)),
				int(x1b),
				int(z1b),
				int(x2b),
				int(z2b),
				int(stride),
				wanted_arr,
				int(len(biome_ids)),
				out_buf,
				int(max_out),
			)
		)
		return [(int(out_buf[i * 2]), int(out_buf[i * 2 + 1])) for i in range(n)]

	# Fallback: pure Pyubiomes API (may be wrong depending on the installed wheel).
	locations: T.List[T.Tuple[int, int]] = []
	for x in range(x1b, x2b, stride):
		for z in range(z1b, z2b, stride):
			for biome_id in biome_ids:
				if biome_at_pos(biome_id, seed, x, z, mc_version):
					locations.append((x, z))
					break
	return locations
