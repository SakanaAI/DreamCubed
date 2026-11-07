#
#simple example program
#that can find biomes and structures

import Pyubiomes
import random
from Pyubiomes import Versions, Biomes, Structures

import concurrent.futures
from Pyubiomes import structure_in_area, biomes_in_area, is_valid_structure_pos, get_spawn, get_strongholds, PyuMap

wanted=[Biomes.plains, Biomes.savanna] #program does not like it if there's only 1 argument

structure=Structures.Village
nether_biomes=[Biomes.nether_wastes, Biomes.crimson_forest, Biomes.soul_sand_valley, Biomes.warped_forest]



def finder(start, increment):
	seed=0
	while True:

		structurecheck=structure_in_area(structure, seed, -128, -128, 128, 128, 16) #check to see if the structure is in the area
		print(seed)
		if structurecheck:
			if is_valid_structure_pos(structure, seed, *structurecheck, 16): #check if structure can spawn
				biomecheck=biomes_in_area(wanted, seed, -128, -128, 128, 128, 16)
				if biomecheck: 
					print(f"found seed {seed}")
					exit()
					break

		
		seed=random.randint(-2**63, 2**63-1) #generate "random" seed

			
				

def start(n):
		with concurrent.futures.ThreadPoolExecutor(max_workers=n) as executor:
			jobs = {executor.submit(finder, *(i,n)) for i in range(n)}
		for fut in concurrent.futures.as_completed(jobs):
			print(fut.result())
print("printing map:")
PyuMap(-5332437328450840283, -16, -16, 32, 32, 4, 16, "mushroom_village").save().toPNG() #create a biome map of this cool seed I found

start(1)


#Biomes.badlands, Biomes.badlands_plateau, Biomes.bamboo_jungle, Biomes.bamboo_jungle_hills, Biomes.beach, Biomes.birch_forest, Biomes.birch_forest_hills, Biomes.cold_ocean, Biomes.dark_forest, Biomes.deep_cold_ocean, Biomes.deep_lukewarm_ocean, Biomes.deep_frozen_ocean, Biomes.desert, Biomes.desert_hills, Biomes.forest, Biomes.frozen_river, Biomes.giant_tree_taiga, Biomes.giant_tree_taiga_hills, Biomes.jungle, Biomes.jungle_edge, Biomes.jungle_hills, Biomes.lukewarm_ocean, Biomes.mountains, Biomes.mushroom_field_shore, Biomes.mushroom_fields, Biomes.plains, Biomes.river, Biomes.savanna, Biomes.savanna_plateau, Biomes.snowy_beach, Biomes.snowy_mountains, Biomes.snowy_taiga, Biomes.snowy_taiga_hills, Biomes.snowy_tundra, Biomes.stone_shore, Biomes.swamp, Biomes.taiga, Biomes.taiga_hills, Biomes.warm_ocean, Biomes.wooded_badlands_plateau, Biomes.wooded_hills, Biomes.wooded_mountains

