from minecraft_nether_gen_rs import NetherGen,create_new_nether,get_biome,NetherBiomes,delete,get_biome_structure,get_biome_decorator
import ctypes
import  minecraft_nether_gen_rs


def  nether_biome_at_pos(seed: int, x:int, z:int):
		'''
		Returns
		Parameters:

		x: 
			The x-coordinate of the point the searcher will search 
		z: 
			The z-coordinate of the point the searcher will search 

		'''
		nether_gen: ctypes.POINTER(NetherGen)=create_new_nether(seed)
		return get_biome(nether_gen, x, 64, z)
	
def nether_biomes_in_area(seed: int, biomes:list, x1:int, z1:int, x2: int, z2: int, incrementor: int=8):
	'''
Finds if all of biomes in the list of biomes is in the area. The incrementor is the spacing between the blocks the searcher should check (default=8)
	nether_gen,
	Returns: bool
	'''
	if seed<0:
		seed+=1<<64
	nether_gen: ctypes.POINTER(NetherGen)=create_new_nether(seed)
	searchx=x1
	searchz=z1
	biomes_in_area=[]
	biomes_set=set(biomes)
	while searchx<=x2:
		while searchz<=z2:
			biome=get_biome(nether_gen, searchx, 64, searchz)
			if not biome in biomes:
				biomes_in_area.append(biome)
				if biomes_set==set(biomes_in_area):
					return True
			else: searchz+=incrementor
			
			searchx+=incrementor
		return False
	
