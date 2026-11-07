// check the biome at a block position
#include "../cubiomes/finders.c"
#include "../cubiomes/generator.c"
#include "../cubiomes/layers.c"
#include "../cubiomes/util.c"
#include <stdlib.h>
#include <stdio.h>


int posToRegionx(int x1, int z1, int structType, int version){
	
	StructureConfig sconf;
    getConfig(structType, version, &sconf);

	int rx1;
	if (sconf.regionSize == 32)
        {
            rx1 = floor(x1/512);
        }
    else
        {
            rx1 = (x1 * 16) - (x1 < 0);
        }
		return rx1;
}
int posToRegionz(int x1, int z1, int structType, int version){
	
	StructureConfig sconf;
    getConfig(structType, version, &sconf);

	int rz1;
	if (sconf.regionSize == 32)
        {

            rz1 = floor(z1/512);

        }
    else
        {
            rz1 = (z1 * 16) - (z1 < 0);
        }
		return rz1;
}
int cBiomeAtPos(int biome, int64_t seed, int xpos, int zpos, int version)
{
    LayerStack g;
    setupGenerator(&g, version);
    
    // Scale 1:4 to avoid Voronoi noise mismatch
    int bx = xpos >> 2;
    int bz = zpos >> 2;
    
    applySeed(&g, seed);

    // Use entry_4 (1:4 scale)
    Layer *l = g.entry_4;
    int *ids = allocCache(l, 1, 1);
    genArea(l, ids, bx, bz, 1, 1);
    int biomeID = ids[0];
    free(ids);
    
    if (biomeID == biome)
        return 1;
	else return 0;
}

int cBiomesInArea(int* biomes, int biomes_len, int64_t seed, int x1, int z1, int x2, int z2, int version)
{
    LayerStack g;
    BiomeFilter filter;

    setupGenerator(&g, version);

    // Define the required biomes.
    int* wanted=biomes;

    filter = setupBiomeFilter(wanted, biomes_len);

    // Scale to 1:4
    int sx1 = x1 >> 2;
    int sz1 = z1 >> 2;
    int sx2 = x2 >> 2;
    int sz2 = z2 >> 2;
    int w = sx2 - sx1;
    int h = sz2 - sz1;
    if (w <= 0) w = 1;
    if (h <= 0) h = 1;

    // Use L_RIVER_MIX_4 (55) for 1.12, L_OCEAN_MIX_4 (63) for 1.13+
    // These constants are from generator.h
    int entry = (version <= MC_1_12) ? L_RIVER_MIX_4 : L_OCEAN_MIX_4;

    int *area = allocCache(&g.layers[entry], w, h);

    if (checkForBiomes(&g, entry, area, seed, sx1, sz1, w, h, filter, 1) > 0){
		free(area);
		return 1;
	}
    free(area);
    return 0;
}

Pos cGetSpawn(int64_t seed, int version){
	LayerStack g;
    setupGenerator(&g, version);
    applySeed(&g, seed);
	Pos pos;
	pos=getSpawn(version, &g, NULL, seed);
	return pos;
}
long findStructure(int structType, int x1, int z1, int x2, int z2, int version) //This function should not be used and was only here for testing purposes
{

    LayerStack g;
    setupGenerator(&g, version);

    int64_t lower48;
    for (lower48 = 0; ; lower48++)
    {
        // The structure position depends only on the region coordinates and
        // the lower 48-bits of the world seed.
        Pos p;
        if (!getStructurePos(structType, version, lower48, 0, 0, &p))
            continue;

        // Look for a seed with the structure at the origin chunk.
        if (p.x >= x2 || p.z >= z2 || p.x<=x1 ||p.z<=z1)
            continue;

        // Look for a full 64-bit seed with viable biomes.
        int64_t upper16;
        for (upper16 = 0; upper16 < 0x10000; upper16++)
        {
            int64_t seed = lower48 | (upper16 << 48);
            if (isViableStructurePos(structType, version, &g, seed, p.x, p.z))
                return seed;
            }
        }
    }


Pos bruh;
Pos structureLower48(int structType, long lower48, int x1, int z1, int x2, int z2, int version){

	LayerStack g;
    setupGenerator(&g, version);
	
	for(int regx=posToRegionx(x1, z1, structType, version); regx<=posToRegionx(x2, z2, structType, version); regx++){
		for(int regz=posToRegionz(x1, z1, structType, version); regz<=posToRegionz(x2, z2, structType, version); regz++){
			if (!getStructurePos(structType, version, lower48, regx, regz, &bruh)) break;

			if (bruh.x >= x2 || bruh.z >= z2 || bruh.x<=x1 ||bruh.z<=z1) break;
	
			return bruh;
			}
	}
bruh.x=30000000;
return bruh;

}
long cIsViableStructurePos(int structType, int64_t seed,int structx, int structz, int version){
	LayerStack g;
    setupGenerator(&g, version);

	return (long)isViableStructurePos(structType, version, &g, seed, structx, structz);

		
	}

// find spawn and the first N strongholds
Pos* cgetStrongholds(int64_t seed, int N, int version)
{
    // Only the first stronghold has a position which can be estimated
    // (+/-112 blocks) without biome check.
    Pos* Strongholds = (Pos*)malloc(N * sizeof(Pos));
    StrongholdIter sh;
    Pos pos = initFirstStronghold(&sh, version, seed);
    Strongholds[0]=pos;

    LayerStack g;
    setupGenerator(&g, version);
    applySeed(&g, seed);



    for (int i = 0; i < N; i++)
    {
        if (nextStronghold(&sh, &g, NULL) <= 0)
            break;
        Strongholds[i]=sh.pos;
    }
    return Strongholds;
}


int cSaveMap(int64_t seed, int areaX, int areaZ, int areaWidth, int areaHeight, int scale, int version, char* filename)
{
    unsigned char biomeColours[256][3];

    // Initialize a colour map for biomes.
    initBiomeColours(biomeColours);

    // Initialize a stack of biome layers.
    LayerStack g;
    setupGenerator(&g, version);
    // Extract the desired layer.
    Layer *layer = &g.layers[L_SHORE_16];

    unsigned int imgWidth = areaWidth*scale, imgHeight = areaHeight*scale;

    // Allocate a sufficient buffer for the biomes and for the image pixels.
    int *biomeIds = allocCache(layer, areaWidth, areaHeight);
    unsigned char *rgb = (unsigned char *) malloc(3*imgWidth*imgHeight);

    // Apply the seed only for the required layers and generate the area.
    setLayerSeed(layer, seed);
    genArea(layer, biomeIds, areaX, areaZ, areaWidth, areaHeight);

    // Map the biomes to a color buffer and save to an image.
    biomesToImage(rgb, biomeColours, biomeIds, areaWidth, areaHeight, scale, 2);
    savePPM(strcat(filename, ".ppm"), rgb, imgWidth, imgHeight);

    // Clean up.
    free(biomeIds);
    free(rgb);

    return 0;
}