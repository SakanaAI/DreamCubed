class Versions:
	'''
	The constants for minecraft versions used by Pyubiomes. It is probably easier to just use the value, but whatever
	'''
	MC_1_0,MC_1_1,MC_1_2,MC_1_3,MC_1_4,MC_1_5,MC_1_6,MC_1_7,MC_1_8,MC_1_9,MC_1_10,MC_1_11,MC_1_12,MC_1_13,MC_1_14,MC_1_15,MC_1_16=tuple(range(17))

class Biomes:
	'''
	Converts the namespaced id of a biome to the numerical id
	'''
	none=-1,
	ocean=0
	plains=1
	desert=2
	mountains=3
	forest=4
	taiga=5
	swamp=6
	river=7
	nether_wastes=8
	the_end=9
	frozen_ocean=10
	frozen_river=11
	snowy_tundra=12
	snowy_mountains=13
	mushroom_fields=14
	mushroom_field_shore=15
	beach=16
	desert_hills=17
	wooded_hills=18
	taiga_hills=19
	mountain_edge=20
	jungle=21
	jungle_hills=22
	jungle_edge=23
	deep_ocean=24
	stone_shore=25
	snowy_beach=26
	birch_forest=27
	birch_forest_hills=28
	dark_forest=29
	snowy_taiga=30
	snowy_taiga_hills=31
	giant_tree_taiga=32
	giant_tree_taiga_hills=33
	wooded_mountains=34
	savanna=35
	savanna_plateau=36
	badlands=37
	wooded_badlands_plateau=38
	badlands_plateau=39
	small_end_islands=40
	end_midlands=41
	end_highlands=42
	end_barrens=43
	warm_ocean=44
	lukewarm_ocean=45
	cold_ocean=46
	deep_warm_ocean=47
	deep_lukewarm_ocean=48
	deep_cold_ocean=49
	deep_frozen_ocean=50
	the_void=127
	sunflower_plains=129
	desert_lakes=130
	gravelly_mountains=131
	flower_forest=132
	taiga_mountains=133
	swamp_hills=134
	ice_spikes=140
	modified_jungle=149
	modified_jungle_edge=151
	tall_birch_forest=155
	tall_birch_hills=156
	dark_forest_hills=157
	snowy_taiga_mountains=158
	giant_spruce_taiga=160
	giant_spruce_taiga_hills=161
	modified_gravelly_mountains=162
	shattered_savanna=163
	shattered_savanna_plateau=164
	eroded_badlands=165
	modified_wooded_badlands_plateau=166
	modified_badlands_plateau=167
	bamboo_jungle=168
	bamboo_jungle_hills=169
	soul_sand_valley=170
	crimson_forest=171
	warped_forest=172
	basalt_deltas=173

class Structures:
	'''
	Converts the namespaced ids of structures into their numerical id
	'''
	Feature=0#for locations of temple generation attempts pre1.13
	Desert_Pyramid=1
	Jungle_Pyramid=2
	Swamp_Hut=3
	Igloo=4
	Village=5
	Ocean_Ruin=6
	Shipwreck=7
	Monument=8
	Mansion=9
	Outpost=10
	Ruined_Portal=11
	Treasure=12
	Fortress=13
	Bastion=14
	End_City=15



