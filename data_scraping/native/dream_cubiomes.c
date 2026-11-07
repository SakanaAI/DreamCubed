// Fast cubiomes-based biome queries for this repo.
//
// This is intentionally NOT dependent on the Pyubiomes wheel installed in your env.
// We compile against the vendored cubiomes sources in /Pyubiomes/cubiomes, which we
// validated match the actual world biomes stored in region files.
//
// Build (example):
//   # NOTE: do NOT use -O3 here; cubiomes has UB that can miscompile under -O3.
//   gcc -O2 -std=c99 -D_DEFAULT_SOURCE -fno-strict-aliasing -fwrapv -fPIC -shared -o native/libdream_cubiomes.so \
//     native/dream_cubiomes.c -lm
//
// Exposed C API:
//   int dream_biome_id_at(int mc, int64_t seed, int x, int z);
//   int dream_find_biome_locations(...);

#include <stdint.h>
#include <stdlib.h>

// Pull cubiomes implementation directly.
#include "../Pyubiomes/cubiomes/finders.c"
#include "../Pyubiomes/cubiomes/generator.c"
#include "../Pyubiomes/cubiomes/layers.c"
#include "../Pyubiomes/cubiomes/util.c"

int dream_biome_id_at(int mc, int64_t seed, int x, int z)
{
    LayerStack g;
    setupGenerator(&g, mc);
    applySeed(&g, seed);

    Layer *l = g.entry_1;
    int *cache = allocCache(l, 1, 1);
    genArea(l, cache, x, z, 1, 1);
    int id = cache[0];
    free(cache);
    return id;
}

static inline int _is_in_list(int id, const int *wanted, int wanted_len)
{
    for (int i = 0; i < wanted_len; i++)
    {
        if (wanted[i] == id)
            return 1;
    }
    return 0;
}

// Returns number of found locations written to out_xz (pairs of ints).
// Scans x in [x1, x2) stepping stride, z in [z1, z2) stepping stride.
int dream_find_biome_locations(
        int mc,
        int64_t seed,
        int x1,
        int z1,
        int x2,
        int z2,
        int stride,
        const int *wanted,
        int wanted_len,
        int *out_xz,
        int max_out)
{
    if (stride <= 0 || max_out <= 0 || wanted_len <= 0)
        return 0;

    LayerStack g;
    setupGenerator(&g, mc);
    applySeed(&g, seed);

    Layer *l = g.entry_1;
    int *cache = allocCache(l, 1, 1);

    int out_n = 0;
    for (int x = x1; x < x2; x += stride)
    {
        for (int z = z1; z < z2; z += stride)
        {
            genArea(l, cache, x, z, 1, 1);
            int id = cache[0];
            if (_is_in_list(id, wanted, wanted_len))
            {
                out_xz[out_n * 2 + 0] = x;
                out_xz[out_n * 2 + 1] = z;
                out_n++;
                if (out_n >= max_out)
                    goto done;
            }
        }
    }

done:
    free(cache);
    return out_n;
}


