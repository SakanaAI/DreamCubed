# Pyubiomes

This is a vendored local wrapper around the Cubiomes library used by the Minecraft scraping utilities.

Install it from the repository root with:

```bash
python -m pip install -e Pyubiomes
```

The package builds a C extension, so Python headers and a C compiler must be available. On Linux, this is typically enough:

```bash
sudo apt-get install -y build-essential python3-dev
```

The scraping code uses these APIs:

```python
biome_at_pos(biome: int, seed: int, xpos: int, zpos: int, version: int)
biomes_in_area(biomes: list, seed: int, x1: int, z1: int, x2: int, z2: int, version: int)
structure_in_area(struct_type: int, lower48: int, x1: int, z1: int, x2: int, z2: int, version: int)
is_valid_structure_pos(struct_type: int, seed: int, structx: int, structz: int, version: int)
get_spawn(seed: int, version: int)
```
