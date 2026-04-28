import torch
import numpy as np
import pyvista as pv
import matplotlib.cm as cm 
import colorsys
from matplotlib import pyplot as plt
import os
from PIL import Image
from math import ceil, sqrt


class MissingTextureError(Exception):
    """Raised when required texture files are missing."""
    pass


class MissingBlockMappingError(Exception):
    """Raised when block IDs are encountered that have no texture mapping."""
    pass


# Hardcoded texture directories (must exist in the project root)
# Order matters: primary is checked first, then fallbacks in order
TEXTURE_DIRS = [
    "block_textures",       # Primary texture directory
    "block_textures_1.17",  # 1.17 textures (deepslate, crimson, warped, etc.)
    "block_textures_1.21",  # 1.21 textures (additional fallbacks)
]


# pyvista based visualizer (much more performant that matplotlib for interactive voxel plotting)
class MinecraftVisualizerPyVista:
    def __init__(self, textures_dir=None, block_texture_map=None, build_textures=True,
                 strict_mode=True):
        """
        Initialize the Minecraft visualizer.
        
        Args:
            textures_dir: Primary directory for block textures (default: uses TEXTURE_DIRS)
            block_texture_map: Optional custom block ID to texture mapping
            build_textures: Whether to build the texture atlas on init
            strict_mode: If True (default), raise exceptions for missing textures/mappings
                        instead of using placeholders. Set to False for lenient mode.
        """
        self.strict_mode = strict_mode
        # Mapping from minecraft block IDs to voxel colors
        self.blocks_to_cols = {
            0: (0.5, 0.25, 0.0),    # light brown
            10: 'black', # bedrock x
            29: "#006400", # cacutus
            38: "#B8860B",  # clay x
            60: "brown",  # dirt x
            92: "gold",  # gold ore x
            93: "green",  # grass x
            115: "brown",  # ladder...?
            119: (.02, .28, .16, 0.9),  # transparent forest green (RGBA) for leaves  x
            120: (.02, .28, .16, 0.9),  # leaves2  x
            194: "yellow",  # sand x
            217: "gray",  # stone x 
            240: (0.0, 0.0, 1.0, 0.4),  # water x
            227: (0.0, 1.0, 0.0, .3), # tall grass x
            237: (0.33, 0.7, 0.33, 0.3), # vine x
            40: "#2F4F4F",  # coal ore x
            62: "#228B22",  # double plant
            108: "#BEBEBE",  # iron ore x
            131: "saddlebrown",  # log1 x
            132: "saddlebrown",  #log2 x
            95: "lightgray",  # gravel x
            243: "wheat",  # wheat x
            197: "limegreen",  # sapling x
            166: "orange",  #pumpkin  x
            167: "#FF8C00",  # pumpkin stem
            184: "#FFA07A",  # red flower  x
            195: "tan",  # sandstone x
            250: "white",  #wool 
            251: "gold",   #yellow flower x
            204: "white", # snow  x
            8103: 'blue' # solid blue for rendering
        }
        try:
            import panel as pn
            pn.extension('vtk')
            pv.set_jupyter_backend('static')
        except ImportError:
            print("Please install panel with: pip install panel")
        
        # texture-related caches - use hardcoded TEXTURE_DIRS
        self.texture_dirs = TEXTURE_DIRS if textures_dir is None else [textures_dir] + TEXTURE_DIRS[1:]
        self.block_texture_map = block_texture_map or self._default_block_texture_map()
        self.atlas_rgba = None
        self.uv_rects = None
        self.atlas_texture = None

        if build_textures:
            self._load_and_build_atlas()
        
        # Cache per-block render modes (extend as needed)
        self.block_render_modes = {
            1: 'sprite_cross', # fence
            11: 'sprite_cross', # beetroots
            13: 'sprite_cross',   # birch fence - approximate as cross
            14: 'sprite_cross',   # birch fence gate - approximate as cross
            26: 'sprite_cross', # brown mushroom
            32: 'sprite_cross',   # carrots (crop)
            43: 'sprite_cross',   # cocoa (attached crop) - approximate
            51: 'sprite_cross',   # dark oak fence - approximate as cross
            52: 'sprite_cross',   # dark oak fence gate - approximate as cross
            56: 'sprite_cross', # dead bush
            62: 'sprite_cross',    # double plant
            
            79: 'sprite_cross',   # fence (generic) - approximate as cross
            80: 'sprite_cross',   # fence gate - approximate as cross
            81: 'sprite_cross',   # fire - approximate as cross planes
            82: 'sprite_cross', # flower pot
            112: 'sprite_cross',  # jungle fence - approximate as cross
            113: 'sprite_cross',  # jungle fence gate - approximate as cross
            115: 'sprite_cross',  # ladder - approximate as cross
            121: 'sprite_cross',  # lever - approximate as cross
            139: 'sprite_cross', # monster egg
            144: 'sprite_cross',  # nether brick fence - approximate as cross
            162: 'sprite_cross', # potatoes
            167: 'sprite_cross',  # pumpkin stem (plant)
            227: 'sprite_cross',  # tall grass
            197: 'sprite_cross',  # sapling
            237: 'sprite_cross',  # vine (simple sprite version)
            243: 'sprite_cross',  # wheat
            184: 'sprite_cross',  # red flower
            251: 'sprite_cross',  # yellow flower
            192: 'sprite_cross', # sugar cane
            241: 'sprite_cross', # waterlily
            208: 'sprite_cross',  # spruce fence - approximate as cross
            209: 'sprite_cross',  # spruce fence gate - approximate as cross
            182: 'sprite_cross',  # redstone torch (on)
            229: 'sprite_cross',  # torch - approximate as cross
            234: 'sprite_cross',  # redstone torch (off)
            242: 'sprite_cross',  # cobweb
            214: 'sprite_cross',  # standing banner - approximate
            215: 'sprite_cross',  # standing sign - approximate
            238: 'sprite_cross',  # wall banner - approximate
            239: 'sprite_cross',  # wall sign - approximate
            146: 'sprite_cross',  # nether wart
            186: 'sprite_cross',  # red mushroom
            # Coral fans
            314: 'sprite_cross',  # brain coral fan
            315: 'sprite_cross',  # brain coral wall fan
            324: 'sprite_cross',  # bubble coral fan
            325: 'sprite_cross',  # bubble coral wall fan
            511: 'sprite_cross',  # fire coral fan
            512: 'sprite_cross',  # fire coral wall fan
            535: 'sprite_cross',  # horn coral fan
            536: 'sprite_cross',  # horn coral wall fan
            852: 'sprite_cross',  # tube coral fan
            853: 'sprite_cross',  # tube coral wall fan
            # Seagrass
            774: 'sprite_cross',  # seagrass
            776: 'sprite_cross',  # short dry grass
            841: 'sprite_cross',  # tall dry grass
            842: 'sprite_cross',  # tall seagrass
            # Bush / Cactus flower / Potted plants / Chain
            327: 'sprite_cross',  # bush
            328: 'sprite_cross',  # cactus flower
            339: 'sprite_cross',  # chain
            508: 'sprite_cross',  # firefly bush
            703: 'sprite_cross',  # potted cactus
            # 250: 'sprite_cross', # wool

            # default is 'cube' for others
        }

        # Expanded render modes: slabs, straight stairs, and corner stairs
        self.block_render_modes.update({
            # slabs
            248: {'mode': 'slab', 'half': 'bottom'}, # wooden pressure plate
            3000: {'mode': 'slab', 'half': 'bottom'},
            3001: {'mode': 'slab', 'half': 'top'},

            # straight stairs (bottom)
            3010: {'mode': 'stair', 'shape': 'straight', 'half': 'bottom', 'facing': 'px'},
            3011: {'mode': 'stair', 'shape': 'straight', 'half': 'bottom', 'facing': 'nx'},
            3012: {'mode': 'stair', 'shape': 'straight', 'half': 'bottom', 'facing': 'py'},
            3013: {'mode': 'stair', 'shape': 'straight', 'half': 'bottom', 'facing': 'ny'},

            # straight stairs (top)
            3020: {'mode': 'stair', 'shape': 'straight', 'half': 'top', 'facing': 'px'},
            3021: {'mode': 'stair', 'shape': 'straight', 'half': 'top', 'facing': 'nx'},
            3022: {'mode': 'stair', 'shape': 'straight', 'half': 'top', 'facing': 'py'},
            3023: {'mode': 'stair', 'shape': 'straight', 'half': 'top', 'facing': 'ny'},

            # outer corners (bottom)
            3030: {'mode': 'stair', 'shape': 'outer', 'turn': 'left',  'half': 'bottom', 'facing': 'px'},
            3031: {'mode': 'stair', 'shape': 'outer', 'turn': 'right', 'half': 'bottom', 'facing': 'px'},
            3032: {'mode': 'stair', 'shape': 'outer', 'turn': 'left',  'half': 'bottom', 'facing': 'nx'},
            3033: {'mode': 'stair', 'shape': 'outer', 'turn': 'right', 'half': 'bottom', 'facing': 'nx'},
            3034: {'mode': 'stair', 'shape': 'outer', 'turn': 'left',  'half': 'bottom', 'facing': 'py'},
            3035: {'mode': 'stair', 'shape': 'outer', 'turn': 'right', 'half': 'bottom', 'facing': 'py'},
            3036: {'mode': 'stair', 'shape': 'outer', 'turn': 'left',  'half': 'bottom', 'facing': 'ny'},
            3037: {'mode': 'stair', 'shape': 'outer', 'turn': 'right', 'half': 'bottom', 'facing': 'ny'},

            # outer corners (top)
            3040: {'mode': 'stair', 'shape': 'outer', 'turn': 'left',  'half': 'top', 'facing': 'px'},
            3041: {'mode': 'stair', 'shape': 'outer', 'turn': 'right', 'half': 'top', 'facing': 'px'},
            3042: {'mode': 'stair', 'shape': 'outer', 'turn': 'left',  'half': 'top', 'facing': 'nx'},
            3043: {'mode': 'stair', 'shape': 'outer', 'turn': 'right', 'half': 'top', 'facing': 'nx'},
            3044: {'mode': 'stair', 'shape': 'outer', 'turn': 'left',  'half': 'top', 'facing': 'py'},
            3045: {'mode': 'stair', 'shape': 'outer', 'turn': 'right', 'half': 'top', 'facing': 'py'},
            3046: {'mode': 'stair', 'shape': 'outer', 'turn': 'left',  'half': 'top', 'facing': 'ny'},
            3047: {'mode': 'stair', 'shape': 'outer', 'turn': 'right', 'half': 'top', 'facing': 'ny'},

            # inner corners (bottom)
            3050: {'mode': 'stair', 'shape': 'inner', 'turn': 'left',  'half': 'bottom', 'facing': 'px'},
            3051: {'mode': 'stair', 'shape': 'inner', 'turn': 'right', 'half': 'bottom', 'facing': 'px'},
            3052: {'mode': 'stair', 'shape': 'inner', 'turn': 'left',  'half': 'bottom', 'facing': 'nx'},
            3053: {'mode': 'stair', 'shape': 'inner', 'turn': 'right', 'half': 'bottom', 'facing': 'nx'},
            3054: {'mode': 'stair', 'shape': 'inner', 'turn': 'left',  'half': 'bottom', 'facing': 'py'},
            3055: {'mode': 'stair', 'shape': 'inner', 'turn': 'right', 'half': 'bottom', 'facing': 'py'},
            3056: {'mode': 'stair', 'shape': 'inner', 'turn': 'left',  'half': 'bottom', 'facing': 'ny'},
            3057: {'mode': 'stair', 'shape': 'inner', 'turn': 'right', 'half': 'bottom', 'facing': 'ny'},

            # inner corners (top)
            3060: {'mode': 'stair', 'shape': 'inner', 'turn': 'left',  'half': 'top', 'facing': 'px'},
            3061: {'mode': 'stair', 'shape': 'inner', 'turn': 'right', 'half': 'top', 'facing': 'px'},
            3062: {'mode': 'stair', 'shape': 'inner', 'turn': 'left',  'half': 'top', 'facing': 'nx'},
            3063: {'mode': 'stair', 'shape': 'inner', 'turn': 'right', 'half': 'top', 'facing': 'nx'},
            3064: {'mode': 'stair', 'shape': 'inner', 'turn': 'left',  'half': 'top', 'facing': 'py'},
            3065: {'mode': 'stair', 'shape': 'inner', 'turn': 'right', 'half': 'top', 'facing': 'py'},
            3066: {'mode': 'stair', 'shape': 'inner', 'turn': 'left',  'half': 'top', 'facing': 'ny'},
            3067: {'mode': 'stair', 'shape': 'inner', 'turn': 'right', 'half': 'top', 'facing': 'ny'},
        })

        # Additional approximate render modes for new dataset blocks
        # NOTE: These are placeholders to get reasonable visualization; adjust as needed when proper models are added.
        self.block_render_modes.update({
            
        })

    def _default_block_texture_map(self):
        return {
            0: {'all': 'oak_door_top.png'}, # door
            1: {'all': 'fence.png'}, # fence
            4: {'all': 'empty.png'}, # activator rail
            6: {'all': 'anvil.png'},   # anvil 
            7: {'all': 'empty.png'},   # BARRIER (invisible)
            8: {'all': 'beacon.png'},   # beacon 
            9: {'all': 'red_wool.png'},   # bed
            10: {'all': 'bedrock.png'}, # bedrock
            11: {'all': 'beetroots.png'}, # beetroots
            12: {'all': 'oak_door_top.png'}, #  birch door
            13: {'all': 'fence.png'},  # birch fence (use 'birch_fence.png')
            14: {'all': 'fence.png'},  # birch fence gate
            20: {'top': 'bone_block_top.png', 'side': 'bone_block_side.png', 'bottom': 'bone_block_top.png'},  # bone block
            21: {'top': 'oak_planks.png', 'side': 'bookshelf.png', 'bottom': 'oak_planks.png'}, # bookshelf
            22: {'all': 'brewing_stand.png'},  # brewing stand 
            23: {'all': 'bricks.png'},  # brick block 
            26: {'all': 'brown_mushroom.png'}, # brown mushroom
            27: {'all': 'brown_mushroom_block.png'}, # brown mushroom block
            29: {'top': 'cactus_top.png', 'side': 'cactus_side.png', 'bottom': 'cactus_top.png'}, # cactus
            30: {'all': 'cake_top.png'}, # cake
            31: {'top': 'empty.png', 'side': 'empty.png', 'bottom': 'red_wool.png'},  #  carpet
            32: {'all': 'carrots.png'}, # carrots
            33: {'top': 'cauldron_top.png', 'side': 'cauldron_side.png', 'bottom': 'cauldron_bottom.png'}, # cauldron
            35: {'all': 'oak_planks.png'}, # chest 
            38: {'all': 'clay.png'},  # clay x
            39: {'all': 'coal_block.png'}, # coal block 
            40: {'all': 'coal_ore.png'},  # coal ore x
            41: {'all': 'cobblestone.png'}, # cobblestone x
            42: {'all': 'cobblestone.png'},  # obblestone wall 
            43: {'all': 'cocoa.png'},  # cocoa (use 'cocoa.png')
            44: {'all': 'chain_command_block_conditional.png'},  # command block ('command_block.png')
            45: {'all': 'light_gray_concrete.png'},  # concrete
            47: {'top': 'crafting_table_top.png', 'side': 'crafting_table_front.png', 'bottom': 'crafting_table_top.png'}, # crafting table
            50: {'all': 'oak_door_top.png'},  #  dark oak door ('dark_oak_door.png')
            51: {'all': 'fence.png'},  # dark oak fence
            52: {'all': 'fence.png'},  # dark oak fence gate
            54: {'top': 'daylight_detector_top.png', 'side': 'daylight_detector_top.png', 'bottom': 'daylight_detector_top.png'},  # daylight detector
            56: {'all': 'dead_bush.png'}, # dead bush
            57: {'all': 'empty.png'}, # detector rail
            58: {'all': 'diamond_block.png'},  # diamond block 
            59: {'all': 'diamond_ore.png'},  # diamond ore 
            60: {'all': 'dirt.png'},  # dirt x
            61: {'all': 'dispenser_front.png'},  # dispenser
            62: {'all': 'double_plant.png'},  # double plant
            63: {'all': 'smooth_stone_slab_side.png'},  # double stone slab (full cube)
            64: {'all': 'smooth_stone_slab_side.png'}, # double stone slab2
            65: {'all': 'oak_planks.png'},  # double wooden slab (full cube)
            66: {'all': 'dragon_egg.png'},  #  dragon egg ('dragon_egg.png')
            67: {'all': 'dropper_front.png'},  # dropper
            68: {'all': 'emerald_block.png'},  # Temerald block
            69: {'all': 'emerald_ore.png'},  # emerald ore
            70: {'top': 'enchanting_table_top.png', 'side': 'enchanting_table_side.png', 'bottom': 'enchanting_table_top.png'},  # enchanting table
            71: {'all': 'oak_planks.png'}, # ender chest
            72: {'all': 'end_stone_bricks.png'}, # end bricks
            74: {'all': 'end_portal_frame_top.png'},  # end portal (special rendering)
            75: {'top': 'end_portal_frame_top.png', 'side': 'end_portal_frame_side.png', 'bottom': 'end_portal_frame_top.png'}, # end portal frame
            76: {'all': 'end_rod.png'}, # end rod
            77: {'all': 'end_stone.png'},  # end stone
            78: {'top': 'farmland_moist.png', 'side': 'dirt.png', 'bottom': 'dirt.png'}, # farmland
            79: {'all': 'fence.png'},  #  fence (generic)
            80: {'all': 'fence.png'},  # fence gate
            81: {'all': 'fire.png'},  # fire (sprite)
            82: {'all': 'flower_pot.png'}, # flower pot
            83: {'all': 'lava.png'}, # flowing lava x
            84: {'all': 'water.png'}, # flowing water
            86: {'top': 'furnace_top.png', 'side': 'furnace_front.png', 'bottom': 'furnace_top.png'}, # furnace
            87: {'all': 'glass.png'},  # glass
            88: {'all': 'glass.png'}, # glass pane 
            89: {'all': 'glowstone.png'},  # glowstone
            90: {'all': 'empty.png'}, # golden rail
            91: {'all': 'gold_block.png'},  # gold block
            92: {'all': 'gold_ore.png'},  # gold ore x
            93: {'top': 'grass_top.png', 'side': 'grass_side.png', 'bottom': 'dirt.png'},  # grass x
            94: {'top': 'dirt_path_top.png', 'side': 'dirt_path_side.png', 'bottom': 'dirt_path_top.png'},  # grass path
            95: {'all': 'gravel.png'},  # gravel x
            96: {'all': 'gray_glazed_terracotta.png'}, # gray glazed terracotta
            100: {'all': 'clay.png'}, # hardened clay
            101: {'top': 'hay_block_top.png', 'side': 'hay_block_side.png', 'bottom': 'hay_block_top.png'},  # hay block
            102: {'all': 'empty.png'}, # TODO: heavy weighted pressure plate
            103: {'all': 'hopper_outside.png'}, # hopper (model)
            104: {'all': 'ice.png'}, # ice
            105: {'all': 'iron_bars.png'}, # iron bars
            106: {'all': 'iron_block.png'}, # iron block
            107: {'all': 'iron_door_top.png'}, # iron door
            108: {'all': 'iron_ore.png'},  # iron ore x
            109: {'all': 'iron_trapdoor.png'}, # iron trapdoor
            110: {'top': 'jukebox_top.png', 'side': 'jukebox_side.png', 'bottom': 'jukebox_top.png'}, # jukebox
            111: {'all': 'oak_door_top.png'}, #  jungle door
            112: {'all': 'fence.png'}, # jungle fence
            113: {'all': 'fence.png'}, # jungle fence gate
            115: {'all': 'ladder.png'}, # ladder (needs side-only/flat plane)
            116: {'all': 'lapis_block.png'}, #  lapis block
            117: {'all': 'lapis_ore.png'}, # lapis ore
            118: {'all': 'lava.png'}, # lava x
            119: {'all': 'leaves_green.png'},  # transparent forest green (RGBA) for leaves  x
            120: {'all': 'leaves_birch.png'},  # leaves2  x
            121: {'all': 'lever.png'}, #  lever (flat model)
            124: {'all': 'empty.png'}, # TODO: light weighted pressure plate
            127: {'top': 'furnace_top.png', 'side': 'furnace_front.png', 'bottom': 'furnace_top.png'}, # lit furnace
            128: {'top': 'pumpkin_top.png', 'side': 'pumpkin_side.png', 'bottom': 'pumpkin_side.png'}, # lit pumpkin
            129: {'all': 'redstone_lamp_on.png'}, #  lit redstone lamp
            130: {'all': 'redstone_ore.png'}, # lit redstone ore
            131: {'all': 'log_big_oak.png'},  # log1 x
            132: {'all': 'log_birch.png'},  #log2 x
            135: {'all': 'magma.png'}, # magma
            136: {'top': 'melon_top.png', 'side': 'melon_side.png', 'bottom': 'melon_top.png'}, # melon block
            137: {'all': 'empty.png'}, # melon stem
            138: {'all': 'spawner.png'}, # mob spawner x
            139: {'all': 'dragon_egg.png'}, # monster egg (note: 66 is dragon egg; fix mapping)
            140: {'all': 'mossy_cobblestone.png'}, # mossy cobblestone x
            141: {'top': 'mycelium_top.png', 'side': 'mycelium_side.png', 'bottom': 'mycelium_top.png'}, # mycelium
            142: {'all': 'netherrack.png'}, # netherrack
            143: {'all': 'nether_bricks.png'}, # nether brick
            144: {'all': 'fence.png'}, # nether brick fence
            146: {'all': 'nether_wart.png'}, # nether wart
            147: {'all': 'nether_wart_block.png'}, # nether wart block
            148: {'all': 'note_block.png'}, # noteblock
            151: {'all': 'obsidian.png'}, #  obsidian
            154: {'all': 'packed_ice.png'}, # packed ice
            157: {'top': 'piston_top.png', 'side': 'piston_side.png', 'bottom': 'piston_top.png'}, # piston
            158: {'top': 'piston_top.png', 'side': 'piston_side.png', 'bottom': 'piston_top.png'}, # piston extension
            159: {'all': 'piston_top.png'}, #  piston head
            160: {'all': 'oak_planks.png'}, # planks
            161: {'all': 'empty.png'}, # TODO: portal (special rendering)
            162: {'all': 'potatoes.png'}, # potatoes
            164: {'all': 'empty.png'}, # TODO: powered repeater (flat)
            165: {'all': 'prismarine.png'}, # prismarine
            166: {'top': 'pumpkin_top.png', 'side': 'pumpkin_side.png', 'bottom': 'pumpkin_side.png'},  #pumpkin  x
            167: {'all': 'pumpkin_stem.png'},  # pumpkin stem
            170: {'all': 'purpur_block.png'}, # purpur block
            175: {'all': 'quartz_block_side.png'}, # quartz block
            176: {'all': 'nether_quartz_ore.png'}, # quartz ore
            178: {'all': 'empty.png'}, # TODO: rail (flat)
            179: {'all': 'redstone_block.png'}, # redstone block
            180: {'all': 'redstone_lamp_on.png'}, # redstone lamp
            181: {'all': 'redstone_ore.png'}, #  redstone ore
            182: {'all': 'redstone_torch.png'}, #  redstone torch (on)
            183: {'all': 'empty.png'}, #  redstone wire (flat)
            184: {'all': 'flower_red.png'},  # red flower  x
            185: {'all': 'red_glazed_terracotta.png'}, # red glazed terracotta
            186: {'all': 'red_mushroom.png'}, # red mushroom
            187: {'all': 'red_mushroom_block.png'}, # red mushroom block x
            188: {'all': 'red_nether_bricks.png'}, # red nether brick
            189: {'all': 'red_sandstone.png'}, # red sandstone
            192: {'all': 'sugar_cane.png'}, # reeds (believe this is sugar cane)
            194: {'all': 'sand.png'},  # sand x
            195: {'top': 'sandstone_top.png', 'side': 'sandstone_side.png', 'bottom': 'sandstone_side.png'},  # sandstone x
            197: {'all': 'sapling_oak.png'},  # sapling x
            201: {'all': 'empty.png'}, # TODO: skull (model)
            202: {'all': 'slime_block.png'}, # slime
            203: {'all': 'snow.png'}, # snow  x
            204: {'all': 'snow.png'}, # snow layer x
            205: {'all': 'soul_sand.png'}, #  soul sand
            206: {'all': 'sponge.png'}, # sponge
            207: {'all': 'oak_door_top.png'}, #  spruce door
            208: {'all': 'fence.png'}, # pruce fence 
            209: {'all': 'fence.png'}, # spruce fence gate
            211: {'all': 'glass.png'}, # stained glass (variants)
            212: {'all': 'glass.png'}, # glass pane (variants)
            213: {'all': 'clay.png'}, # stained hardened clay 
            214: {'all': 'empty.png'}, #  standing banner (sprite/billboard)
            215: {'all': 'empty.png'}, #  standing sign (sprite/billboard)
            216: {'top': 'piston_top.png', 'side': 'piston_side.png', 'bottom': 'piston_top.png'}, # sticky piston
            217: {'all': 'stone.png'},  # stone x 
            218: {'all': 'stone_bricks.png'}, # stonebrick
            220: {'all': 'empty.png'}, # stone button (flat)
            221: {'all': 'empty.png'}, #  stone pressure plate (flat)
            229: {'all': 'torch.png'}, #  torch 
            240: {'all': 'water.png'},  # water x
            241: {'all': 'lily_of_the_valley.png'}, #  waterlily (use 'waterlily.png')
            227: {'all': 'tall_grass.png'}, # tall grass x
            228: {'top': 'tnt_top.png', 'side': 'tnt_side.png', 'bottom': 'tnt_top.png'}, # TNT
            230: {'all': 'empty.png'}, # TODO: trapdoor (flat)
            231: {'all': 'oak_planks.png'}, #  trapped chest
            232: {'all': 'empty.png'}, # tripwire
            233: {'all': 'empty.png'}, # tripwire hook
            234: {'all': 'redstone_torch.png'}, # TODO: unlit redstone torch
            235: {'all': 'comparator.png'}, # unpowered comparator
            236: {'all': 'empty.png'}, # TODO: unpowered repeater (flat)
            237: {'all': 'vine.png'}, # vine x
            238: {'all': 'empty.png'}, # TODO: wall banner
            239: {'all': 'empty.png'}, # TODO: wall sign
            242: {'all': 'cobweb.png'}, #  web (cobweb)
            243: {'all': 'wheat.png'},  # wheat x
            246: {'all': 'empty.png'}, # TODO: wooden button (flat)
           
            
            251: {'all': 'flower_yellow.png'},   #yellow flower x
            247: {'all': 'oak_door_top.png'}, # wooden door 
            248: {'all': 'oak_planks.png'}, #  wooden pressure plate (render as slab)
            250: {'all': 'white_wool.png'}, # wool
            990: {'all': 'white_wool.png'}, # wool
            # slabs and stairs (use oak planks texture for all faces)
            196: {'all': 'sandstone_side.png'}, # sandstone stairs (use sandstone textures)
            198: {'all': 'sea_lantern.png'}, #  sea lantern
            
            # ===== NEW BLOCKS (IDs 254+) =====
            # Acacia variants
            2: {'all': 'fence.png'},  # ACACIA_FENCE_GATE
            3: {'all': 'oak_planks.png'},  # ACACIA_STAIRS
            254: {'all': 'empty.png'},  # ACACIA_HANGING_SIGN
            255: {'all': 'acacia_shelf.png'},  # ACACIA_SHELF
            256: {'all': 'empty.png'},  # ACACIA_WALL_HANGING_SIGN
            257: {'all': 'empty.png'},  # ACACIA_WALL_SIGN
            258: {'all': 'log_acacia.png'},  # ACACIA_WOOD
            
            # Amethyst
            259: {'all': 'amethyst_block.png'},  # AMETHYST_BLOCK
            260: {'all': 'amethyst_cluster.png'},  # AMETHYST_CLUSTER
            326: {'all': 'budding_amethyst.png'},  # BUDDING_AMETHYST
            548: {'all': 'large_amethyst_bud.png'},  # LARGE_AMETHYST_BUD
            585: {'all': 'medium_amethyst_bud.png'},  # MEDIUM_AMETHYST_BUD
            782: {'all': 'amethyst_cluster.png'},  # SMALL_AMETHYST_BUD
            
            # Ancient Debris / Netherite
            261: {'top': 'ancient_debris_top.png', 'side': 'ancient_debris_side.png', 'bottom': 'ancient_debris_top.png'},  # ANCIENT_DEBRIS
            603: {'all': 'netherite_block.png'},  # NETHERITE_BLOCK
            
            # Andesite
            262: {'all': 'stone_andesite.png'},  # ANDESITE
            263: {'all': 'stone_andesite.png'},  # ANDESITE_SLAB
            264: {'all': 'stone_andesite.png'},  # ANDESITE_STAIRS
            265: {'all': 'stone_andesite.png'},  # ANDESITE_WALL
            667: {'all': 'stone_andesite_smooth.png'},  # POLISHED_ANDESITE
            668: {'all': 'stone_andesite_smooth.png'},  # POLISHED_ANDESITE_SLAB
            669: {'all': 'stone_andesite_smooth.png'},  # POLISHED_ANDESITE_STAIRS
            
            # Azalea
            268: {'all': 'azalea_side.png'},  # AZALEA
            269: {'all': 'azalea_leaves.png'},  # AZALEA_LEAVES
            514: {'all': 'flowering_azalea_side.png'},  # FLOWERING_AZALEA
            515: {'all': 'azalea_leaves_flowers.png'},  # FLOWERING_AZALEA_LEAVES
            
            # Bamboo
            270: {'all': 'bamboo_stem.png'},  # BAMBOO
            271: {'top': 'bamboo_block_top.png', 'side': 'bamboo_block.png', 'bottom': 'bamboo_block_top.png'},  # BAMBOO_BLOCK
            272: {'all': 'empty.png'},  # BAMBOO_BUTTON
            273: {'all': 'bamboo_door_top.png'},  # BAMBOO_DOOR
            274: {'all': 'bamboo_fence.png'},  # BAMBOO_FENCE
            275: {'all': 'bamboo_fence_gate.png'},  # BAMBOO_FENCE_GATE
            276: {'all': 'empty.png'},  # BAMBOO_HANGING_SIGN
            277: {'all': 'bamboo_mosaic.png'},  # BAMBOO_MOSAIC
            278: {'all': 'bamboo_mosaic.png'},  # BAMBOO_MOSAIC_SLAB
            279: {'all': 'bamboo_mosaic.png'},  # BAMBOO_MOSAIC_STAIRS
            280: {'all': 'bamboo_planks.png'},  # BAMBOO_PLANKS
            281: {'all': 'bamboo_planks.png'},  # BAMBOO_PRESSURE_PLATE
            282: {'all': 'bamboo_sapling.png'},  # BAMBOO_SAPLING
            283: {'all': 'bamboo_shelf.png'},  # BAMBOO_SHELF
            284: {'all': 'empty.png'},  # BAMBOO_SIGN
            285: {'all': 'bamboo_planks.png'},  # BAMBOO_SLAB
            286: {'all': 'bamboo_planks.png'},  # BAMBOO_STAIRS
            287: {'all': 'bamboo_trapdoor.png'},  # BAMBOO_TRAPDOOR
            288: {'all': 'empty.png'},  # BAMBOO_WALL_HANGING_SIGN
            289: {'all': 'empty.png'},  # BAMBOO_WALL_SIGN
            817: {'all': 'bamboo_block.png'},  # STRIPPED_BAMBOO_BLOCK
            
            # Barrel
            290: {'top': 'barrel_top.png', 'side': 'barrel_side.png', 'bottom': 'barrel_bottom.png'},  # BARREL
            
            # Basalt
            291: {'top': 'basalt_top.png', 'side': 'basalt_side.png', 'bottom': 'basalt_top.png'},  # BASALT
            670: {'all': 'polished_basalt_side.png'},  # POLISHED_BASALT
            786: {'all': 'smooth_basalt.png'},  # SMOOTH_BASALT
            
            # Bee blocks
            292: {'top': 'beehive_top.png', 'side': 'beehive_front.png', 'bottom': 'beehive_top.png'},  # BEEHIVE
            293: {'top': 'bee_nest_top.png', 'side': 'bee_nest_front.png', 'bottom': 'bee_nest_bottom.png'},  # BEE_NEST
            294: {'all': 'bell_side.png'},  # BELL
            
            # Birch variants
            15: {'all': 'oak_planks.png'},  # BIRCH_STAIRS
            297: {'all': 'empty.png'},  # BIRCH_HANGING_SIGN
            298: {'all': 'birch_shelf.png'},  # BIRCH_SHELF
            299: {'all': 'empty.png'},  # BIRCH_WALL_HANGING_SIGN
            300: {'all': 'empty.png'},  # BIRCH_WALL_SIGN
            301: {'all': 'log_birch.png'},  # BIRCH_WOOD
            
            # Big Dripleaf
            295: {'all': 'big_dripleaf_top.png'},  # BIG_DRIPLEAF
            296: {'all': 'big_dripleaf_stem.png'},  # BIG_DRIPLEAF_STEM
            
            # Blackstone
            302: {'all': 'blackstone.png'},  # BLACKSTONE
            303: {'all': 'blackstone.png'},  # BLACKSTONE_SLAB
            304: {'all': 'blackstone.png'},  # BLACKSTONE_STAIRS
            305: {'all': 'blackstone.png'},  # BLACKSTONE_WALL
            363: {'all': 'chiseled_polished_blackstone.png'},  # CHISELED_POLISHED_BLACKSTONE
            517: {'all': 'gilded_blackstone.png'},  # GILDED_BLACKSTONE
            671: {'all': 'polished_blackstone.png'},  # POLISHED_BLACKSTONE
            672: {'all': 'polished_blackstone_bricks.png'},  # POLISHED_BLACKSTONE_BRICKS
            673: {'all': 'polished_blackstone_bricks.png'},  # POLISHED_BLACKSTONE_BRICK_SLAB
            674: {'all': 'polished_blackstone_bricks.png'},  # POLISHED_BLACKSTONE_BRICK_STAIRS
            675: {'all': 'polished_blackstone_bricks.png'},  # POLISHED_BLACKSTONE_BRICK_WALL
            676: {'all': 'empty.png'},  # POLISHED_BLACKSTONE_BUTTON
            680: {'all': 'polished_blackstone.png'},  # POLISHED_BLACKSTONE_WALL
            397: {'all': 'cracked_polished_blackstone_bricks.png'},  # CRACKED_POLISHED_BLACKSTONE_BRICKS
            
            # Glazed Terracottas (missing ones)
            16: {'all': 'glazed_terracotta_black.png'},  # BLACK_GLAZED_TERRACOTTA
            18: {'all': 'glazed_terracotta_blue.png'},  # BLUE_GLAZED_TERRACOTTA
            25: {'all': 'glazed_terracotta_brown.png'},  # BROWN_GLAZED_TERRACOTTA
            48: {'all': 'glazed_terracotta_cyan.png'},  # CYAN_GLAZED_TERRACOTTA
            98: {'all': 'glazed_terracotta_green.png'},  # GREEN_GLAZED_TERRACOTTA
            122: {'all': 'glazed_terracotta_light_blue.png'},  # LIGHT_BLUE_GLAZED_TERRACOTTA
            125: {'all': 'glazed_terracotta_lime.png'},  # LIME_GLAZED_TERRACOTTA
            133: {'all': 'glazed_terracotta_magenta.png'},  # MAGENTA_GLAZED_TERRACOTTA
            152: {'all': 'glazed_terracotta_orange.png'},  # ORANGE_GLAZED_TERRACOTTA
            155: {'all': 'glazed_terracotta_pink.png'},  # PINK_GLAZED_TERRACOTTA
            168: {'all': 'glazed_terracotta_purple.png'},  # PURPLE_GLAZED_TERRACOTTA
            199: {'all': 'glazed_terracotta_silver.png'},  # SILVER_GLAZED_TERRACOTTA
            244: {'all': 'glazed_terracotta_white.png'},  # WHITE_GLAZED_TERRACOTTA
            252: {'all': 'glazed_terracotta_yellow.png'},  # YELLOW_GLAZED_TERRACOTTA
            
            # Shulker boxes
            17: {'all': 'shulker_top_black.png'},  # BLACK_SHULKER_BOX
            19: {'all': 'shulker_top_blue.png'},  # BLUE_SHULKER_BOX
            28: {'all': 'shulker_top_brown.png'},  # BROWN_SHULKER_BOX
            49: {'all': 'shulker_top_cyan.png'},  # CYAN_SHULKER_BOX
            97: {'all': 'shulker_top_gray.png'},  # GRAY_SHULKER_BOX
            99: {'all': 'shulker_top_green.png'},  # GREEN_SHULKER_BOX
            123: {'all': 'shulker_top_light_blue.png'},  # LIGHT_BLUE_SHULKER_BOX
            126: {'all': 'shulker_top_lime.png'},  # LIME_SHULKER_BOX
            134: {'all': 'shulker_top_magenta.png'},  # MAGENTA_SHULKER_BOX
            153: {'all': 'shulker_top_orange.png'},  # ORANGE_SHULKER_BOX
            156: {'all': 'shulker_top_pink.png'},  # PINK_SHULKER_BOX
            169: {'all': 'shulker_top_purple.png'},  # PURPLE_SHULKER_BOX
            191: {'all': 'shulker_top_red.png'},  # RED_SHULKER_BOX
            200: {'all': 'shulker_top_silver.png'},  # SILVER_SHULKER_BOX
            245: {'all': 'shulker_top_white.png'},  # WHITE_SHULKER_BOX
            253: {'all': 'shulker_top_yellow.png'},  # YELLOW_SHULKER_BOX
            778: {'all': 'shulker_top_undyed.png'},  # SHULKER_BOX
            
            # Blue Ice
            311: {'all': 'blue_ice.png'},  # BLUE_ICE
            85: {'all': 'frosted_ice_0.png'},  # FROSTED_ICE
            
            # Bricks
            316: {'all': 'brick.png'},  # BRICKS (alternate ID)
            24: {'all': 'brick.png'},  # BRICK_STAIRS
            317: {'all': 'brick.png'},  # BRICK_SLAB
            318: {'all': 'brick.png'},  # BRICK_WALL
            
            # Calcite
            329: {'all': 'calcite.png'},  # CALCITE
            
            # Campfire
            331: {'all': 'campfire.png'},  # CAMPFIRE
            800: {'all': 'soul_campfire.png'},  # SOUL_CAMPFIRE
            
            # Candles (render as empty/sprite)
            306: {'all': 'empty.png'},  # BLACK_CANDLE
            309: {'all': 'empty.png'},  # BLUE_CANDLE
            319: {'all': 'empty.png'},  # BROWN_CANDLE
            332: {'all': 'empty.png'},  # CANDLE
            430: {'all': 'empty.png'},  # CYAN_CANDLE
            524: {'all': 'empty.png'},  # GRAY_CANDLE
            526: {'all': 'empty.png'},  # GREEN_CANDLE
            554: {'all': 'empty.png'},  # LIGHT_BLUE_CANDLE
            556: {'all': 'empty.png'},  # LIGHT_GRAY_CANDLE
            560: {'all': 'empty.png'},  # LIME_CANDLE
            564: {'all': 'empty.png'},  # MAGENTA_CANDLE
            615: {'all': 'empty.png'},  # ORANGE_CANDLE
            658: {'all': 'empty.png'},  # PINK_CANDLE
            740: {'all': 'empty.png'},  # PURPLE_CANDLE
            749: {'all': 'empty.png'},  # RED_CANDLE
            968: {'all': 'empty.png'},  # WHITE_CANDLE
            974: {'all': 'empty.png'},  # YELLOW_CANDLE
            
            # Cartography/Crafting tables
            334: {'top': 'cartography_table_top.png', 'side': 'cartography_table_side1.png', 'bottom': 'cartography_table_top.png'},  # CARTOGRAPHY_TABLE
            513: {'top': 'fletcher_table_top.png', 'side': 'fletcher_table_side1.png', 'bottom': 'fletcher_table_top.png'},  # FLETCHING_TABLE
            784: {'all': 'smithing_table_side.png'},  # SMITHING_TABLE
            
            # Cherry
            340: {'all': 'empty.png'},  # CHERRY_BUTTON
            341: {'all': 'cherry_door_top.png'},  # CHERRY_DOOR
            342: {'all': 'fence.png'},  # CHERRY_FENCE
            343: {'all': 'fence.png'},  # CHERRY_FENCE_GATE
            344: {'all': 'empty.png'},  # CHERRY_HANGING_SIGN
            345: {'all': 'cherry_leaves.png'},  # CHERRY_LEAVES
            346: {'top': 'cherry_log_top.png', 'side': 'cherry_log_side.png', 'bottom': 'cherry_log_top.png'},  # CHERRY_LOG
            347: {'all': 'cherry_planks.png'},  # CHERRY_PLANKS
            348: {'all': 'cherry_planks.png'},  # CHERRY_PRESSURE_PLATE
            349: {'all': 'cherry_sapling.png'},  # CHERRY_SAPLING
            350: {'all': 'cherry_shelf.png'},  # CHERRY_SHELF
            352: {'all': 'cherry_planks.png'},  # CHERRY_SLAB
            353: {'all': 'cherry_planks.png'},  # CHERRY_STAIRS
            354: {'all': 'cherry_trapdoor.png'},  # CHERRY_TRAPDOOR
            357: {'all': 'cherry_log_side.png'},  # CHERRY_WOOD
            820: {'all': 'stripped_cherry_log_side.png'},  # STRIPPED_CHERRY_LOG
            821: {'all': 'stripped_cherry_log_side.png'},  # STRIPPED_CHERRY_WOOD
            
            # Chiseled blocks
            359: {'all': 'chiseled_bookshelf_empty.png'},  # CHISELED_BOOKSHELF
            360: {'all': 'chiseled_copper.png'},  # CHISELED_COPPER
            362: {'all': 'chiseled_nether_bricks.png'},  # CHISELED_NETHER_BRICKS
            364: {'all': 'quartz_block_chiseled.png'},  # CHISELED_QUARTZ_BLOCK
            367: {'all': 'sandstone_carved.png'},  # CHISELED_SANDSTONE
            368: {'all': 'stonebrick_carved.png'},  # CHISELED_STONE_BRICKS
            369: {'all': 'chiseled_tuff.png'},  # CHISELED_TUFF
            370: {'all': 'chiseled_tuff_bricks.png'},  # CHISELED_TUFF_BRICKS
            
            # Chorus
            36: {'all': 'chorus_flower.png'},  # CHORUS_FLOWER
            37: {'all': 'chorus_plant.png'},  # CHORUS_PLANT
            
            # Coarse Dirt / Rooted Dirt
            372: {'all': 'coarse_dirt.png'},  # COARSE_DIRT
            765: {'all': 'dirt_with_roots.png'},  # ROOTED_DIRT
            
            # Cobblestone variants
            377: {'all': 'cobblestone.png'},  # COBBLESTONE_SLAB
            378: {'all': 'cobblestone.png'},  # COBBLESTONE_STAIRS
            586: {'all': 'cobblestone_mossy.png'},  # MOSSY_COBBLESTONE_SLAB
            587: {'all': 'cobblestone_mossy.png'},  # MOSSY_COBBLESTONE_STAIRS
            588: {'all': 'cobblestone_mossy.png'},  # MOSSY_COBBLESTONE_WALL
            
            # Command blocks
            34: {'all': 'chain_command_block_conditional.png'},  # CHAIN_COMMAND_BLOCK
            193: {'all': 'repeating_command_block_front.png'},  # REPEATING_COMMAND_BLOCK
            
            # Composter
            379: {'top': 'composter_top.png', 'side': 'composter_side.png', 'bottom': 'composter_bottom.png'},  # COMPOSTER
            
            # Concrete (use gray as default, would need variants)
            46: {'all': 'concrete_gray.png'},  # CONCRETE_POWDER
            
            # Copper variants
            382: {'all': 'copper_block.png'},  # COPPER_BLOCK
            390: {'all': 'copper_ore.png'},  # COPPER_ORE
            423: {'all': 'cut_copper.png'},  # CUT_COPPER
            424: {'all': 'cut_copper.png'},  # CUT_COPPER_SLAB
            425: {'all': 'cut_copper.png'},  # CUT_COPPER_STAIRS
            381: {'all': 'copper_bars.png'},  # COPPER_BARS
            383: {'all': 'copper_bulb.png'},  # COPPER_BULB
            386: {'all': 'copper_door_top.png'},  # COPPER_DOOR
            388: {'all': 'copper_grate.png'},  # COPPER_GRATE
            389: {'all': 'copper_lantern.png'},  # COPPER_LANTERN
            392: {'all': 'copper_trapdoor.png'},  # COPPER_TRAPDOOR
            
            # Exposed Copper
            493: {'all': 'exposed_copper.png'},  # EXPOSED_COPPER
            503: {'all': 'exposed_cut_copper.png'},  # EXPOSED_CUT_COPPER
            492: {'all': 'exposed_chiseled_copper.png'},  # EXPOSED_CHISELED_COPPER
            494: {'all': 'exposed_copper_bars.png'},  # EXPOSED_COPPER_BARS
            495: {'all': 'exposed_copper_bulb.png'},  # EXPOSED_COPPER_BULB
            498: {'all': 'exposed_copper_door_top.png'},  # EXPOSED_COPPER_DOOR
            500: {'all': 'exposed_copper_grate.png'},  # EXPOSED_COPPER_GRATE
            502: {'all': 'exposed_copper_trapdoor.png'},  # EXPOSED_COPPER_TRAPDOOR
            
            # Oxidized Copper
            618: {'all': 'oxidized_copper.png'},  # OXIDIZED_COPPER
            628: {'all': 'oxidized_cut_copper.png'},  # OXIDIZED_CUT_COPPER
            617: {'all': 'oxidized_chiseled_copper.png'},  # OXIDIZED_CHISELED_COPPER
            619: {'all': 'oxidized_copper_bars.png'},  # OXIDIZED_COPPER_BARS
            620: {'all': 'oxidized_copper_bulb.png'},  # OXIDIZED_COPPER_BULB
            623: {'all': 'oxidized_copper_door_top.png'},  # OXIDIZED_COPPER_DOOR
            625: {'all': 'oxidized_copper_grate.png'},  # OXIDIZED_COPPER_GRATE
            627: {'all': 'oxidized_copper_trapdoor.png'},  # OXIDIZED_COPPER_TRAPDOOR
            
            # Weathered Copper
            951: {'all': 'weathered_copper.png'},  # WEATHERED_COPPER
            961: {'all': 'weathered_cut_copper.png'},  # WEATHERED_CUT_COPPER
            950: {'all': 'weathered_chiseled_copper.png'},  # WEATHERED_CHISELED_COPPER
            952: {'all': 'weathered_copper_bars.png'},  # WEATHERED_COPPER_BARS
            953: {'all': 'weathered_copper_bulb.png'},  # WEATHERED_COPPER_BULB
            956: {'all': 'weathered_copper_door_top.png'},  # WEATHERED_COPPER_DOOR
            958: {'all': 'weathered_copper_grate.png'},  # WEATHERED_COPPER_GRATE
            960: {'all': 'weathered_copper_trapdoor.png'},  # WEATHERED_COPPER_TRAPDOOR
            
            # Coral blocks
            312: {'all': 'coral_pink.png'},  # BRAIN_CORAL
            313: {'all': 'coral_pink.png'},  # BRAIN_CORAL_BLOCK
            314: {'all': 'brain_coral_fan.png'},  # BRAIN_CORAL_FAN
            315: {'all': 'brain_coral_fan.png'},  # BRAIN_CORAL_WALL_FAN
            322: {'all': 'coral_purple.png'},  # BUBBLE_CORAL
            323: {'all': 'coral_purple.png'},  # BUBBLE_CORAL_BLOCK
            324: {'all': 'bubble_coral_fan.png'},  # BUBBLE_CORAL_FAN
            325: {'all': 'bubble_coral_fan.png'},  # BUBBLE_CORAL_WALL_FAN
            509: {'all': 'coral_red.png'},  # FIRE_CORAL
            510: {'all': 'coral_red.png'},  # FIRE_CORAL_BLOCK
            511: {'all': 'fire_coral_fan.png'},  # FIRE_CORAL_FAN
            512: {'all': 'fire_coral_fan.png'},  # FIRE_CORAL_WALL_FAN
            533: {'all': 'coral_yellow.png'},  # HORN_CORAL
            534: {'all': 'coral_yellow.png'},  # HORN_CORAL_BLOCK
            535: {'all': 'horn_coral_fan.png'},  # HORN_CORAL_FAN
            536: {'all': 'horn_coral_fan.png'},  # HORN_CORAL_WALL_FAN
            850: {'all': 'coral_blue.png'},  # TUBE_CORAL
            851: {'all': 'coral_blue.png'},  # TUBE_CORAL_BLOCK
            852: {'all': 'tube_coral_fan.png'},  # TUBE_CORAL_FAN
            853: {'all': 'tube_coral_fan.png'},  # TUBE_CORAL_WALL_FAN
            
            # Crafter
            399: {'all': 'crafter_top.png'},  # CRAFTER
            
            # Crimson (Nether wood)
            407: {'all': 'crimson_fungus.png'},  # CRIMSON_FUNGUS
            410: {'top': 'crimson_nylium_top.png', 'side': 'crimson_nylium_side.png', 'bottom': 'netherrack.png'},  # CRIMSON_NYLIUM
            411: {'all': 'crimson_planks.png'},  # CRIMSON_PLANKS
            413: {'all': 'crimson_roots.png'},  # CRIMSON_ROOTS
            416: {'all': 'crimson_planks.png'},  # CRIMSON_SLAB
            417: {'all': 'crimson_planks.png'},  # CRIMSON_STAIRS
            418: {'all': 'crimson_stem.png'},  # CRIMSON_STEM (1.17 texture)
            419: {'all': 'crimson_trapdoor.png'},  # CRIMSON_TRAPDOOR
            
            # Crying Obsidian
            422: {'all': 'crying_obsidian.png'},  # CRYING_OBSIDIAN
            
            # Dark Oak
            53: {'all': 'oak_planks.png'},  # DARK_OAK_STAIRS
            434: {'all': 'dark_oak_shelf.png'},  # DARK_OAK_SHELF
            436: {'all': 'empty.png'},  # DARK_OAK_WALL_SIGN
            437: {'all': 'log_big_oak.png'},  # DARK_OAK_WOOD
            
            # Dark Prismarine
            438: {'all': 'prismarine_dark.png'},  # DARK_PRISMARINE
            439: {'all': 'prismarine_dark.png'},  # DARK_PRISMARINE_SLAB
            440: {'all': 'prismarine_dark.png'},  # DARK_PRISMARINE_STAIRS
            
            # Daylight detector inverted
            55: {'all': 'daylight_detector_inverted_top.png'},  # DAYLIGHT_DETECTOR_INVERTED
            
            # Decorated Pot
            461: {'top': 'decorated_pot_base.png', 'side': 'decorated_pot_side.png', 'bottom': 'decorated_pot_base.png'},  # DECORATED_POT
            
            # Deepslate
            462: {'top': 'deepslate_top.png', 'side': 'deepslate.png', 'bottom': 'deepslate_top.png'},  # DEEPSLATE
            361: {'all': 'deepslate_bricks.png'},  # CHISELED_DEEPSLATE
            373: {'all': 'cobbled_deepslate.png'},  # COBBLED_DEEPSLATE
            374: {'all': 'cobbled_deepslate.png'},  # COBBLED_DEEPSLATE_SLAB
            375: {'all': 'cobbled_deepslate.png'},  # COBBLED_DEEPSLATE_STAIRS
            376: {'all': 'cobbled_deepslate.png'},  # COBBLED_DEEPSLATE_WALL
            463: {'all': 'deepslate_bricks.png'},  # DEEPSLATE_BRICKS
            464: {'all': 'deepslate_bricks.png'},  # DEEPSLATE_BRICK_SLAB
            465: {'all': 'deepslate_bricks.png'},  # DEEPSLATE_BRICK_STAIRS
            466: {'all': 'deepslate_bricks.png'},  # DEEPSLATE_BRICK_WALL
            475: {'all': 'deepslate_tiles.png'},  # DEEPSLATE_TILES
            476: {'all': 'deepslate_tiles.png'},  # DEEPSLATE_TILE_SLAB
            477: {'all': 'deepslate_tiles.png'},  # DEEPSLATE_TILE_STAIRS
            478: {'all': 'deepslate_tiles.png'},  # DEEPSLATE_TILE_WALL
            681: {'all': 'polished_deepslate.png'},  # POLISHED_DEEPSLATE
            682: {'all': 'polished_deepslate.png'},  # POLISHED_DEEPSLATE_SLAB
            683: {'all': 'polished_deepslate.png'},  # POLISHED_DEEPSLATE_STAIRS
            684: {'all': 'polished_deepslate.png'},  # POLISHED_DEEPSLATE_WALL
            394: {'all': 'cracked_deepslate_bricks.png'},  # CRACKED_DEEPSLATE_BRICKS
            395: {'all': 'cracked_deepslate_tiles.png'},  # CRACKED_DEEPSLATE_TILES
            
            # Deepslate Ores (1.17 textures - some use deepslate.png fallback)
            467: {'all': 'deepslate_coal_ore.png'},  # DEEPSLATE_COAL_ORE
            468: {'all': 'deepslate.png'},  # DEEPSLATE_COPPER_ORE (fallback - texture not in 1.17)
            469: {'all': 'deepslate_diamond_ore.png'},  # DEEPSLATE_DIAMOND_ORE
            470: {'all': 'deepslate.png'},  # DEEPSLATE_EMERALD_ORE (fallback - texture not in 1.17)
            471: {'all': 'deepslate.png'},  # DEEPSLATE_GOLD_ORE (fallback - texture not in 1.17)
            472: {'all': 'deepslate_iron_ore.png'},  # DEEPSLATE_IRON_ORE
            473: {'all': 'deepslate_lapis_ore.png'},  # DEEPSLATE_LAPIS_ORE
            474: {'all': 'deepslate_redstone_ore.png'},  # DEEPSLATE_REDSTONE_ORE
            
            # Diorite
            479: {'all': 'stone_diorite.png'},  # DIORITE
            480: {'all': 'stone_diorite.png'},  # DIORITE_SLAB
            481: {'all': 'stone_diorite.png'},  # DIORITE_STAIRS
            482: {'all': 'stone_diorite.png'},  # DIORITE_WALL
            685: {'all': 'stone_diorite_smooth.png'},  # POLISHED_DIORITE
            686: {'all': 'stone_diorite_smooth.png'},  # POLISHED_DIORITE_SLAB
            687: {'all': 'stone_diorite_smooth.png'},  # POLISHED_DIORITE_STAIRS
            
            # Dirt Path
            483: {'top': 'grass_path_top.png', 'side': 'grass_path_side.png', 'bottom': 'dirt.png'},  # DIRT_PATH
            
            # Dripstone
            488: {'all': 'dripstone_block.png'},  # DRIPSTONE_BLOCK
            666: {'all': 'pointed_dripstone_down_tip.png'},  # POINTED_DRIPSTONE
            
            # Dried Kelp
            487: {'top': 'dried_kelp_top.png', 'side': 'dried_kelp_side_a.png', 'bottom': 'dried_kelp_top.png'},  # DRIED_KELP_BLOCK
            
            # End Stone Brick variants
            489: {'all': 'end_bricks.png'},  # END_STONE_BRICK_SLAB
            490: {'all': 'end_bricks.png'},  # END_STONE_BRICK_STAIRS
            491: {'all': 'end_bricks.png'},  # END_STONE_BRICK_WALL
            73: {'all': 'end_gateway.png'},  # END_GATEWAY
            
            # Fern
            507: {'all': 'fern.png'},  # FERN
            
            # Froglight
            613: {'top': 'ochre_froglight_top.png', 'side': 'ochre_froglight_side.png', 'bottom': 'ochre_froglight_top.png'},  # OCHRE_FROGLIGHT
            654: {'top': 'pearlescent_froglight_top.png', 'side': 'pearlescent_froglight_side.png', 'bottom': 'pearlescent_froglight_top.png'},  # PEARLESCENT_FROGLIGHT
            866: {'top': 'verdant_froglight_top.png', 'side': 'verdant_froglight_side.png', 'bottom': 'verdant_froglight_top.png'},  # VERDANT_FROGLIGHT
            
            # Glow Lichen
            518: {'all': 'glow_lichen.png'},  # GLOW_LICHEN
            
            # Granite
            519: {'all': 'stone_granite.png'},  # GRANITE
            520: {'all': 'stone_granite.png'},  # GRANITE_SLAB
            521: {'all': 'stone_granite.png'},  # GRANITE_STAIRS
            522: {'all': 'stone_granite.png'},  # GRANITE_WALL
            688: {'all': 'stone_granite_smooth.png'},  # POLISHED_GRANITE
            689: {'all': 'stone_granite_smooth.png'},  # POLISHED_GRANITE_SLAB
            690: {'all': 'stone_granite_smooth.png'},  # POLISHED_GRANITE_STAIRS
            
            # Grass Block (alternate)
            523: {'top': 'grass_top.png', 'side': 'grass_side_carried.png', 'bottom': 'dirt.png'},  # GRASS_BLOCK
            
            # Grindstone
            528: {'all': 'grindstone_side.png'},  # GRINDSTONE
            
            # Hanging Roots
            529: {'all': 'hanging_roots.png'},  # HANGING_ROOTS
            
            # Heavy Core
            530: {'all': 'heavy_core.png'},  # HEAVY_CORE
            
            # Honey
            531: {'all': 'honeycomb.png'},  # HONEYCOMB_BLOCK
            532: {'top': 'honey_top.png', 'side': 'honey_side.png', 'bottom': 'honey_bottom.png'},  # HONEY_BLOCK
            
            # Jungle
            114: {'all': 'oak_planks.png'},  # JUNGLE_STAIRS
            541: {'all': 'jungle_shelf.png'},  # JUNGLE_SHELF
            543: {'all': 'empty.png'},  # JUNGLE_WALL_SIGN
            544: {'all': 'log_jungle.png'},  # JUNGLE_WOOD
            
            # Kelp
            545: {'all': 'kelp.png'},  # KELP (1.17 texture)
            546: {'all': 'kelp_plant.png'},  # KELP_PLANT
            
            # Lantern
            547: {'all': 'lantern.png'},  # LANTERN
            802: {'all': 'soul_lantern.png'},  # SOUL_LANTERN
            
            # Lectern
            551: {'top': 'lectern_top.png', 'side': 'lectern_front.png', 'bottom': 'lectern_base.png'},  # LECTERN
            
            # Light (invisible block)
            552: {'all': 'empty.png'},  # LIGHT
            
            # Lightning Rod
            553: {'all': 'lightning_rod.png'},  # LIGHTNING_ROD
            
            # Lodestone
            562: {'top': 'lodestone_top.png', 'side': 'lodestone_side.png', 'bottom': 'lodestone_top.png'},  # LODESTONE
            
            # Loom
            563: {'top': 'loom_top.png', 'side': 'loom_side.png', 'bottom': 'loom_bottom.png'},  # LOOM
            
            # Mangrove
            566 : {'all': 'empty.png'},  # MANGROVE_BUTTON
            567: {'all': 'mangrove_door_top.png'},  # MANGROVE_DOOR
            568: {'all': 'fence.png'},  # MANGROVE_FENCE
            569: {'all': 'fence.png'},  # MANGROVE_FENCE_GATE
            571: {'all': 'mangrove_leaves_opaque.png'},  # MANGROVE_LEAVES
            572: {'top': 'mangrove_log_top.png', 'side': 'mangrove_log_side.png', 'bottom': 'mangrove_log_top.png'},  # MANGROVE_LOG
            573: {'all': 'mangrove_planks.png'},  # MANGROVE_PLANKS
            574: {'all': 'mangrove_planks.png'},  # MANGROVE_PRESSURE_PLATE
            575: {'all': 'mangrove_propagule.png'},  # MANGROVE_PROPAGULE
            576: {'top': 'mangrove_roots_top.png', 'side': 'mangrove_roots_side.png', 'bottom': 'mangrove_roots_top.png'},  # MANGROVE_ROOTS
            577: {'all': 'mangrove_shelf.png'},  # MANGROVE_SHELF
            579: {'all': 'mangrove_planks.png'},  # MANGROVE_SLAB
            580: {'all': 'mangrove_planks.png'},  # MANGROVE_STAIRS
            581: {'all': 'mangrove_trapdoor.png'},  # MANGROVE_TRAPDOOR
            583: {'all': 'empty.png'},  # MANGROVE_WALL_SIGN}
            584: {'all': 'mangrove_log_side.png'},  # MANGROVE_WOOD
            597: {'top': 'muddy_mangrove_roots_top.png', 'side': 'muddy_mangrove_roots_side.png', 'bottom': 'muddy_mangrove_roots_top.png'},  # MUDDY_MANGROVE_ROOTS
            
            # Moss
            593: {'all': 'moss_block.png'},  # MOSS_BLOCK
            594: {'all': 'moss_block.png'},  # MOSS_CARPET
            
            # Mossy Stone Bricks
            589: {'all': 'stonebrick_mossy.png'},  # MOSSY_STONE_BRICKS
            590: {'all': 'stonebrick_mossy.png'},  # MOSSY_STONE_BRICK_SLAB
            591: {'all': 'stonebrick_mossy.png'},  # MOSSY_STONE_BRICK_STAIRS
            592: {'all': 'stonebrick_mossy.png'},  # MOSSY_STONE_BRICK_WALL
            
            # Mud
            596: {'all': 'mud.png'},  # MUD
            598: {'all': 'mud_bricks.png'},  # MUD_BRICKS
            599: {'all': 'mud_bricks.png'},  # MUD_BRICK_SLAB
            600: {'all': 'mud_bricks.png'},  # MUD_BRICK_STAIRS
            601: {'all': 'mud_bricks.png'},  # MUD_BRICK_WALL
            632: {'all': 'mud.png'},  # PACKED_MUD
            
            # Mushroom Stem
            602: {'all': 'mushroom_block_skin_stem.png'},  # MUSHROOM_STEM
            
            # Nether Brick variants
            145: {'all': 'nether_brick.png'},  # NETHER_BRICK_STAIRS
            396: {'all': 'cracked_nether_bricks.png'},  # CRACKED_NETHER_BRICKS
            393: {'all': 'empty.png'},
            408: {'all': 'empty.png'},
            402 : {'all': 'empty.png'},

            

            604: {'all': 'nether_brick.png'},  # NETHER_BRICK_SLAB
            605: {'all': 'nether_brick.png'},  # NETHER_BRICK_WALL
            606: {'all': 'nether_gold_ore.png'},  # NETHER_GOLD_ORE
            607: {'all': 'nether_sprouts.png'},  # NETHER_SPROUTS
            751: {'all': 'red_nether_bricks.png'},  # RED_NETHER_BRICK_SLAB
            752: {'all': 'red_nether_bricks.png'},  # RED_NETHER_BRICK_STAIRS
            753: {'all': 'red_nether_bricks.png'},  # RED_NETHER_BRICK_WALL
            
            # Oak
            149: {'all': 'oak_planks.png'},  # OAK_STAIRS
            609: {'all': 'oak_shelf.png'},  # OAK_SHELF
            612: {'all': 'log_oak.png'},  # OAK_WOOD
            
            # Observer
            150: {'top': 'observer_top.png', 'side': 'observer_side.png', 'bottom': 'observer_back.png'},  # OBSERVER
            
            # Podzol
            665: {'top': 'dirt_podzol_top.png', 'side': 'dirt_podzol_side.png', 'bottom': 'dirt.png'},  # PODZOL
            
            # Prismarine variants
            734: {'all': 'prismarine_bricks.png'},  # PRISMARINE_BRICKS
            735: {'all': 'prismarine_bricks.png'},  # PRISMARINE_BRICK_SLAB
            736: {'all': 'prismarine_bricks.png'},  # PRISMARINE_BRICK_STAIRS
            737: {'all': 'prismarine_rough.png'},  # PRISMARINE_SLAB
            738: {'all': 'prismarine_rough.png'},  # PRISMARINE_STAIRS
            739: {'all': 'prismarine_rough.png'},  # PRISMARINE_WALL
            
            # Purpur variants
            171: {'all': 'purpur_block.png'},  # PURPUR_DOUBLE_SLAB
            172: {'all': 'purpur_pillar.png'},  # PURPUR_PILLAR
            173: {'all': 'purpur_block.png'},  # PURPUR_SLAB
            174: {'all': 'purpur_block.png'},  # PURPUR_STAIRS
            
            # Quartz variants
            177: {'all': 'quartz_block_side.png'},  # QUARTZ_STAIRS
            742: {'all': 'quartz_bricks.png'},  # QUARTZ_BRICKS
            743: {'all': 'quartz_block_lines.png'},  # QUARTZ_PILLAR
            744: {'all': 'quartz_block_side.png'},  # QUARTZ_SLAB
            787: {'all': 'quartz_block_bottom.png'},  # SMOOTH_QUARTZ
            788: {'all': 'quartz_block_bottom.png'},  # SMOOTH_QUARTZ_SLAB
            789: {'all': 'quartz_block_bottom.png'},  # SMOOTH_QUARTZ_STAIRS
            
            # Raw ore blocks
            745: {'all': 'raw_copper_block.png'},  # RAW_COPPER_BLOCK
            746: {'all': 'raw_gold_block.png'},  # RAW_GOLD_BLOCK
            747: {'all': 'raw_iron_block.png'},  # RAW_IRON_BLOCK
            
            # Red Sand/Sandstone variants
            754: {'all': 'red_sand.png'},  # RED_SAND
            190: {'all': 'red_sandstone_normal.png'},  # RED_SANDSTONE_STAIRS
            365: {'all': 'red_sandstone_carved.png'},  # CHISELED_RED_SANDSTONE
            426: {'all': 'red_sandstone_smooth.png'},  # CUT_RED_SANDSTONE
            427: {'all': 'red_sandstone_smooth.png'},  # CUT_RED_SANDSTONE_SLAB
            755: {'all': 'red_sandstone_normal.png'},  # RED_SANDSTONE_SLAB
            756: {'all': 'red_sandstone_normal.png'},  # RED_SANDSTONE_WALL
            790: {'all': 'red_sandstone_top.png'},  # SMOOTH_RED_SANDSTONE
            791: {'all': 'red_sandstone_top.png'},  # SMOOTH_RED_SANDSTONE_SLAB
            792: {'all': 'red_sandstone_top.png'},  # SMOOTH_RED_SANDSTONE_STAIRS
            
            # Sandstone variants
            428: {'all': 'sandstone_smooth.png'},  # CUT_SANDSTONE
            429: {'all': 'sandstone_smooth.png'},  # CUT_SANDSTONE_SLAB
            766: {'all': 'sandstone_normal.png'},  # SANDSTONE_SLAB
            767: {'all': 'sandstone_normal.png'},  # SANDSTONE_WALL
            793: {'all': 'sandstone_top.png'},  # SMOOTH_SANDSTONE
            794: {'all': 'sandstone_top.png'},  # SMOOTH_SANDSTONE_SLAB
            795: {'all': 'sandstone_top.png'},  # SMOOTH_SANDSTONE_STAIRS
            
            # Scaffolding
            768: {'all': 'scaffolding_side.png'},  # SCAFFOLDING
            
            # Sculk
            769: {'all': 'sculk.png'},  # SCULK
            770: {'all': 'sculk_catalyst_top.png'},  # SCULK_CATALYST
            771: {'all': 'sculk_sensor_bottom.png'},  # SCULK_SENSOR
            772: {'all': 'sculk_shrieker_top.png'},  # SCULK_SHRIEKER
            773: {'all': 'sculk_vein.png'},  # SCULK_VEIN
            
            # Seagrass / Sea Pickle / Grass variants
            774: {'all': 'seagrass.png'},  # SEAGRASS
            775: {'all': 'sea_pickle.png'},  # SEA_PICKLE
            776: {'all': 'short_dry_grass.png'},  # SHORT_DRY_GRASS
            841: {'all': 'tall_dry_grass.png'},  # TALL_DRY_GRASS
            842: {'all': 'tall_seagrass_top.png'},  # TALL_SEAGRASS
            
            # Bush / Cactus flower / Potted plants
            327: {'all': 'bush.png'},  # BUSH
            328: {'all': 'cactus_flower.png'},  # CACTUS_FLOWER
            508: {'all': 'firefly_bush.png'},  # FIREFLY_BUSH
            703: {'all': 'flower_pot.png'},  # POTTED_CACTUS
            839: {'all': 'suspicious_sand_0.png'},  # SUSPICIOUS_SAND
            
            # Cave Vines
            337: {'all': 'cave_vines.png'},  # CAVE_VINES
            338: {'all': 'cave_vines_plant.png'},  # CAVE_VINES_PLANT
            
            # Chain
            339: {'all': 'chain1.png'},  # CHAIN
            
            # Shroomlight
            777: {'all': 'shroomlight.png'},  # SHROOMLIGHT
            
            # Slabs/Stairs missing
            222: {'all': 'smooth_stone_slab_side.png'},  # STONE_SLAB
            223: {'all': 'smooth_stone_slab_side.png'},  # STONE_SLAB2
            224: {'all': 'stone.png'},  # STONE_STAIRS
            219: {'all': 'stone_bricks.png'},  # STONE_BRICK_STAIRS
            249: {'all': 'oak_planks.png'},  # WOODEN_SLAB
            813: {'all': 'stone_bricks.png'},  # STONE_BRICK_SLAB
            814: {'all': 'stone_bricks.png'},  # STONE_BRICK_WALL
            
            # Cracked Stone Bricks
            398: {'all': 'stonebrick_cracked.png'},  # CRACKED_STONE_BRICKS
            
            # Smoker / Blast Furnace
            785: {'top': 'smoker_top.png', 'side': 'smoker_front_off.png', 'bottom': 'smoker_bottom.png'},  # SMOKER
            308: {'top': 'blast_furnace_top.png', 'side': 'blast_furnace_front_off.png', 'bottom': 'blast_furnace_top.png'},  # BLAST_FURNACE
            
            # Smooth Stone
            796: {'all': 'stone_slab_top.png'},  # SMOOTH_STONE
            797: {'all': 'smooth_stone_slab_side.png'},  # SMOOTH_STONE_SLAB
            
            # Snow Block
            799: {'all': 'snow.png'},  # SNOW_BLOCK
            
            # Soul blocks
            801: {'all': 'soul_fire_0.png'},  # SOUL_FIRE
            803: {'all': 'soul_soil.png'},  # SOUL_SOIL
            804: {'all': 'soul_torch.png'},  # SOUL_TORCH
            
            # Spore Blossom
            806: {'all': 'spore_blossom.png'},  # SPORE_BLOSSOM
            
            # Spruce
            210: {'all': 'oak_planks.png'},  # SPRUCE_STAIRS
            808: {'all': 'spruce_shelf.png'},  # SPRUCE_SHELF
            809: {'all': 'empty.png'},  # SPRUCE_WALL_HANGING_SIGN
            811: {'all': 'log_spruce.png'},  # SPRUCE_WOOD
            
            # Stonecutter
            812: {'top': 'stonecutter_top.png', 'side': 'stonecutter_side.png', 'bottom': 'stonecutter_bottom.png'},  # STONECUTTER
            
            # Stripped Logs (note: texture files are named without "_side" suffix)
            815: {'all': 'stripped_acacia_log.png'},  # STRIPPED_ACACIA_LOG
            816: {'all': 'stripped_acacia_log.png'},  # STRIPPED_ACACIA_WOOD
            818: {'all': 'stripped_birch_log.png'},  # STRIPPED_BIRCH_LOG
            819: {'all': 'stripped_birch_log.png'},  # STRIPPED_BIRCH_WOOD
            824: {'all': 'stripped_dark_oak_log.png'},  # STRIPPED_DARK_OAK_LOG
            825: {'all': 'stripped_dark_oak_log.png'},  # STRIPPED_DARK_OAK_WOOD
            826: {'all': 'stripped_jungle_log.png'},  # STRIPPED_JUNGLE_LOG
            827: {'all': 'stripped_jungle_log.png'},  # STRIPPED_JUNGLE_WOOD
            828: {'all': 'stripped_mangrove_log_side.png'},  # STRIPPED_MANGROVE_LOG
            829: {'all': 'stripped_mangrove_log_side.png'},  # STRIPPED_MANGROVE_WOOD
            830: {'all': 'stripped_oak_log.png'},  # STRIPPED_OAK_LOG
            831: {'all': 'stripped_oak_log.png'},  # STRIPPED_OAK_WOOD
            834: {'all': 'stripped_spruce_log.png'},  # STRIPPED_SPRUCE_LOG
            835: {'all': 'stripped_spruce_log.png'},  # STRIPPED_SPRUCE_WOOD
            836: {'all': 'stripped_warped_stem.png'},  # STRIPPED_WARPED_HYPHAE
            
            # Sweet Berry Bush
            840: {'all': 'sweet_berry_bush_stage3.png'},  # SWEET_BERRY_BUSH
            
            # Target
            843: {'top': 'target_top.png', 'side': 'target_side.png', 'bottom': 'target_top.png'},  # TARGET
            
            # Tinted Glass
            846: {'all': 'tinted_glass.png'},  # TINTED_GLASS
            
            # Torchflower
            847: {'all': 'torchflower.png'},  # TORCHFLOWER
            
            # Trapdoor (generic)
            230: {'all': 'trapdoor.png'},  # TRAPDOOR
            
            # Tuff
            854: {'all': 'tuff.png'},  # TUFF
            855: {'all': 'tuff_bricks.png'},  # TUFF_BRICKS
            856: {'all': 'tuff_bricks.png'},  # TUFF_BRICK_SLAB
            857: {'all': 'tuff_bricks.png'},  # TUFF_BRICK_STAIRS
            858: {'all': 'tuff_bricks.png'},  # TUFF_BRICK_WALL
            859: {'all': 'tuff.png'},  # TUFF_SLAB
            860: {'all': 'tuff.png'},  # TUFF_STAIRS
            861: {'all': 'tuff.png'},  # TUFF_WALL
            691: {'all': 'polished_tuff.png'},  # POLISHED_TUFF
            692: {'all': 'polished_tuff.png'},  # POLISHED_TUFF_SLAB
            693: {'all': 'polished_tuff.png'},  # POLISHED_TUFF_STAIRS
            694: {'all': 'polished_tuff.png'},  # POLISHED_TUFF_WALL
            
            # Turtle Egg
            862: {'all': 'turtle_egg_not_cracked.png'},  # TURTLE_EGG
            
            # Twisting/Weeping Vines
            863: {'all': 'twisting_vines_base.png'},  # TWISTING_VINES
            864: {'all': 'twisting_vines_bottom.png'},  # TWISTING_VINES_PLANT
            965: {'all': 'weeping_vines.png'},  # WEEPING_VINES
            966: {'all': 'weeping_vines_plant.png'},  # WEEPING_VINES_PLANT
            
            # Warped (Nether wood)
            873: {'all': 'warped_fungus.png'},  # WARPED_FUNGUS
            876: {'top': 'warped_nylium_top.png', 'side': 'warped_nylium_side.png', 'bottom': 'netherrack.png'},  # WARPED_NYLIUM
            877: {'all': 'warped_planks.png'},  # WARPED_PLANKS
            879: {'all': 'warped_roots.png'},  # WARPED_ROOTS
            880: {'all': 'warped_shelf.png'},  # WARPED_SHELF
            882: {'all': 'warped_planks.png'},  # WARPED_SLAB
            883: {'all': 'warped_planks.png'},  # WARPED_STAIRS
            884: {'all': 'warped_stem.png'},  # WARPED_STEM (1.17 texture)
            885: {'all': 'warped_trapdoor.png'},  # WARPED_TRAPDOOR
            888: {'all': 'warped_wart_block.png'},  # WARPED_WART_BLOCK
            
            # Water Cauldron
            889: {'top': 'cauldron_top.png', 'side': 'cauldron_side.png', 'bottom': 'cauldron_bottom.png'},  # WATER_CAULDRON
            
            # Wet Sponge
            967: {'all': 'sponge_wet.png'},  # WET_SPONGE
            
            # Wither Rose
            971: {'all': 'flower_wither_rose.png'},  # WITHER_ROSE
            
            # ===== ADDITIONAL MISSING BLOCKS =====
            # Dead coral blocks
            458: {'all': 'coral_blue_dead.png'},  # DEAD_TUBE_CORAL_BLOCK
            441: {'all': 'coral_pink_dead.png'},  # DEAD_BRAIN_CORAL
            442: {'all': 'coral_pink_dead.png'},  # DEAD_BRAIN_CORAL_BLOCK
            445: {'all': 'coral_purple_dead.png'},  # DEAD_BUBBLE_CORAL
            446: {'all': 'coral_purple_dead.png'},  # DEAD_BUBBLE_CORAL_BLOCK
            449: {'all': 'coral_red_dead.png'},  # DEAD_FIRE_CORAL
            450: {'all': 'coral_red_dead.png'},  # DEAD_FIRE_CORAL_BLOCK
            453: {'all': 'coral_yellow_dead.png'},  # DEAD_HORN_CORAL
            454: {'all': 'coral_yellow_dead.png'},  # DEAD_HORN_CORAL_BLOCK
            457: {'all': 'coral_blue_dead.png'},  # DEAD_TUBE_CORAL
            
            # Wall torches
            868: {'all': 'torch.png'},  # WALL_TORCH
            748: {'all': 'redstone_torch.png'},  # REDSTONE_WALL_TORCH
            
            # Carved pumpkin
            335: {'top': 'pumpkin_top.png', 'side': 'pumpkin_face_off.png', 'bottom': 'pumpkin_top.png'},  # CARVED_PUMPKIN
            
            # Petrified oak slab (same as oak planks)
            655: {'all': 'oak_planks.png'},  # PETRIFIED_OAK_SLAB
            
            # Wall signs (render as empty/sprite)
            810: {'all': 'empty.png'},  # SPRUCE_WALL_SIGN
            611: {'all': 'empty.png'},  # OAK_WALL_SIGN
            
            # Player/mob heads (render as empty/sprite)
            663: {'all': 'empty.png'},  # PLAYER_HEAD
            664: {'all': 'empty.png'},  # PLAYER_WALL_HEAD
            
            # Potted plants
            711: {'all': 'flower_pot.png'},  # POTTED_DEAD_BUSH
            724: {'all': 'flower_pot.png'},  # POTTED_RED_MUSHROOM
            
            # Wool (colored) - using white as default
            # (variants would need separate entries)
            
            # Slabs and stairs base IDs
            3000: {'all': 'oak_planks.png'},
            3001: {'all': 'oak_planks.png'},
            3010: {'all': 'oak_planks.png'},
            3011: {'all': 'oak_planks.png'},
            3012: {'all': 'oak_planks.png'},
            3013: {'all': 'oak_planks.png'},
            3020: {'all': 'oak_planks.png'},
            3021: {'all': 'oak_planks.png'},
            3022: {'all': 'oak_planks.png'},
            3023: {'all': 'oak_planks.png'},
            3030: {'all': 'oak_planks.png'},
            3031: {'all': 'oak_planks.png'},
            3032: {'all': 'oak_planks.png'},
            3033: {'all': 'oak_planks.png'},
            3034: {'all': 'oak_planks.png'},
            3035: {'all': 'oak_planks.png'},
            3036: {'all': 'oak_planks.png'},
            3037: {'all': 'oak_planks.png'},
            3040: {'all': 'oak_planks.png'},
            3041: {'all': 'oak_planks.png'},
            3042: {'all': 'oak_planks.png'},
            3043: {'all': 'oak_planks.png'},
            3044: {'all': 'oak_planks.png'},
            3045: {'all': 'oak_planks.png'},
            3046: {'all': 'oak_planks.png'},
            3047: {'all': 'oak_planks.png'},
            3050: {'all': 'oak_planks.png'},
            3051: {'all': 'oak_planks.png'},
            3052: {'all': 'oak_planks.png'},
            3053: {'all': 'oak_planks.png'},
            3054: {'all': 'oak_planks.png'},
            3055: {'all': 'oak_planks.png'},
            3056: {'all': 'oak_planks.png'},
            3057: {'all': 'oak_planks.png'},
            3060: {'all': 'oak_planks.png'},
            3061: {'all': 'oak_planks.png'},
            3062: {'all': 'oak_planks.png'},
            3063: {'all': 'oak_planks.png'},
            3064: {'all': 'oak_planks.png'},
            3065: {'all': 'oak_planks.png'},
            3066: {'all': 'oak_planks.png'},
            3067: {'all': 'oak_planks.png'},
        }

    def set_textures(self, block_texture_map=None, rebuild=True):
        """Update texture mapping and optionally rebuild the atlas."""
        if block_texture_map is not None:
            self.block_texture_map = block_texture_map
        if rebuild:
            self._load_and_build_atlas()

    def _load_and_build_atlas(self):
        tex_names = set()
        for spec in self.block_texture_map.values():
            tex_names.update(spec.values())
        # Always include a fallback texture in the atlas
        tex_names.add('missing.png')
        # Resolve texture paths by searching all texture directories
        texture_paths, missing_files = _resolve_texture_paths(
            tex_names,
            self.texture_dirs,
        )
        
        # In strict mode, raise an error if any texture files are missing
        if self.strict_mode and missing_files:
            error_lines = [
                f"Missing {len(missing_files)} texture file(s):",
                "  Searched directories:",
            ]
            for d in self.texture_dirs:
                error_lines.append(f"    - {d}")
            error_lines.append("")
            error_lines.append("Missing textures:")
            for name in sorted(missing_files):
                error_lines.append(f"  - {name}")
            raise MissingTextureError("\n".join(error_lines))
        
        imgs = _load_images_rgba(texture_paths, strict_mode=self.strict_mode)
        self.atlas_rgba, self.uv_rects = _build_texture_atlas(imgs)
        self.atlas_texture = _numpy_to_texture(self.atlas_rgba)

    # Main visualization function.
    def visualize_chunk(self, voxels, plotter=None, interactive=False, show_axis=True):
        """
        Generate visualization of a chunk of Minecraft voxels
        Args:
            voxels: numpy array/torch.Tensor of block IDs
            plotter: Optional existing PyVista plotter
            interactive: Whether to create an interactive display
            show_axis: Whether to render the X, Y, and Z axes in visualization
        """
        
        # Convert to numpy if needed
        if isinstance(voxels, torch.Tensor):
            if voxels.dim() == 4:  # One-hot encoded [C,H,W,D]
                voxels = voxels.detach().cpu()
                voxels = torch.argmax(voxels, dim=0).numpy()
            else:
                voxels = voxels.detach().cpu().numpy()

        orig_dim_x = voxels.shape[0]
        orig_dim_y = voxels.shape[1]
        orig_dim_z = voxels.shape[2]
        # Apply the same transformations as original
        voxels = voxels.transpose(2, 0, 1)
        # Rotate the voxels 90 degrees around the height axis
        voxels = np.rot90(voxels, 1, (0, 1))
        render_dim_x = voxels.shape[0]
        render_dim_y = voxels.shape[1]
        render_dim_z = voxels.shape[2]
                
        # Create grid
        grid = pv.ImageData()
        grid.dimensions = np.array(voxels.shape) + 1
        grid.cell_data["values"] = voxels.flatten(order="F")
        
        # Create plotter if not provided
        if plotter is None:
            if interactive:
                plotter = pv.Plotter(notebook=True)
            else:
                plotter = pv.Plotter(off_screen=True)
        
        # Remove existing lights
        plotter.remove_all_lights()
        
        # Add the three-point lighting setup
        plotter.add_light(pv.Light(
            position=(1, -1, 1),
            intensity=1.0,
            color='white'
        ))
        
        plotter.add_light(pv.Light(
            position=(-1, 1, 0.5),
            intensity=0.5,
            color='white'
        ))
        
        plotter.add_light(pv.Light(
            position=(-0.5, -0.5, -1),
            intensity=0.3,
            color='white'
        ))
        
        # Plot each block type
        mask = (voxels != 5) & (voxels != -1)
        unique_blocks = np.unique(voxels[mask])
        
        for block_id in unique_blocks:
            threshold = grid.threshold([block_id-0.5, block_id+0.5])
            if block_id in self.blocks_to_cols:
                color = self.blocks_to_cols[int(block_id)]
                opacity = 1.0 if isinstance(color, str) or len(color) == 3 else color[3]
            else:
                color = (1.0, 0.0, 0.0)
                opacity = 0.2
            
            plotter.add_mesh(threshold, 
                        color=color,
                        opacity=opacity,
                        show_edges=True,
                        edge_color='black',
                        line_width=.2,
                        edge_opacity=0.2,
                        lighting=True)
            
        # Add dummy cube for bounds
        outline = pv.Cube(bounds=(0, render_dim_x, 0, render_dim_y, 0, render_dim_z))
        plotter.add_mesh(outline, opacity=0.0)
        
        # Add bounds with consistent settings
        if show_axis:
            plotter.show_bounds(
                grid='back',
                location='back',
                font_size=8,
                bold=False,
                font_family='arial',
                use_2d=False,
                bounds=[0, render_dim_x, 0, render_dim_y, 0, render_dim_z],
                axes_ranges=[0, render_dim_x, 0, render_dim_y, 0, render_dim_z],
                padding=0.0,
                n_xlabels=2,
                n_ylabels=2,
                n_zlabels=2
            )
        
        # Set camera position and zoom
        plotter.camera_position = 'iso'
        plotter.camera.zoom(1)
        
        return plotter
    
    def validate_block_ids(self, voxels, block_types_json_path=None):
        """
        Validate that all block IDs in voxels have texture mappings.
        
        Args:
            voxels: numpy array of block IDs
            block_types_json_path: Optional path to block_types.json for human-readable names
            
        Returns:
            list of (block_id, count, name) tuples for missing blocks
        """
        if isinstance(voxels, torch.Tensor):
            voxels = voxels.detach().cpu().numpy()
        
        unique, counts = np.unique(voxels, return_counts=True)
        
        # Load block type names if available
        block_names = {}
        if block_types_json_path and os.path.isfile(block_types_json_path):
            import json
            with open(block_types_json_path) as f:
                block_names = json.load(f)
        
        # AIR block ID
        AIR_ID = 5
        
        missing = []
        for bid, cnt in zip(unique, counts):
            bid = int(bid)
            if bid == AIR_ID:
                continue
            if bid not in self.block_texture_map:
                name = block_names.get(str(bid), 'UNKNOWN')
                missing.append((bid, cnt, name))

        return missing
    
    def visualize_chunk_textured(self, voxels, plotter=None, interactive=False, show_axis=True):
        if plotter is None:
            if interactive:
                plotter = pv.Plotter(notebook=True)
            else:
                plotter = pv.Plotter(off_screen=True)
        if self.atlas_texture is None or self.uv_rects is None:
            raise RuntimeError("Textures not loaded. Call set_textures(...) or pass textures_dir at init.")

        if isinstance(voxels, torch.Tensor):
            if voxels.dim() == 4:
                voxels = voxels.detach().cpu()
                voxels = torch.argmax(voxels, dim=0).numpy()
            else:
                voxels = voxels.detach().cpu().numpy()

        # Validate block IDs have texture mappings; warn and skip any unknown blocks
        missing = self.validate_block_ids(voxels, block_types_json_path="assets/block_types_updated.json")
        if missing:
            warn_lines = [f"[warn] missing texture mapping for {len(missing)} block type(s); skipping those voxels:"]
            for bid, cnt, name in sorted(missing, key=lambda x: -x[1]):
                warn_lines.append(f"[warn]   ID {bid:4d} ({name:40s}): {cnt:>10,} voxels")
            print("\n".join(warn_lines))

            # Replace unknown blocks with AIR so they won't be rendered
            AIR_ID = 5
            voxels = voxels.copy()
            for bid, _, _ in missing:
                voxels[voxels == bid] = AIR_ID

        orig_dim_x = voxels.shape[0]
        orig_dim_y = voxels.shape[1]
        orig_dim_z = voxels.shape[2]
        voxels = voxels.transpose(2, 0, 1)
        voxels = np.rot90(voxels, 1, (0, 1))
        render_dim_x = voxels.shape[0]
        render_dim_y = voxels.shape[1]
        render_dim_z = voxels.shape[2]

        def _init_bounds_actor():
            # Textured off-screen renders need this bounds initialization even when
            # axis labels are hidden; otherwise screenshots can come back blank.
            actor = plotter.show_bounds(
                grid='back' if show_axis else None,
                location='back',
                font_size=8,
                bold=False,
                font_family='arial',
                use_2d=False,
                bounds=[0, render_dim_x, 0, render_dim_y, 0, render_dim_z],
                axes_ranges=[0, render_dim_x, 0, render_dim_y, 0, render_dim_z],
                padding=0.0,
                n_xlabels=2,
                n_ylabels=2,
                n_zlabels=2,
                show_xlabels=show_axis,
                show_ylabels=show_axis,
                show_zlabels=show_axis,
            )
            if not show_axis:
                # Do not disable the actor or the axes entirely; textured
                # off-screen rendering appears to depend on the live bounds
                # actor being present. Instead, hide the visible axis lines by
                # making their line properties fully transparent.
                for getter_name in (
                    "GetXAxesLinesProperty",
                    "GetYAxesLinesProperty",
                    "GetZAxesLinesProperty",
                ):
                    try:
                        prop = getattr(actor, getter_name)()
                        prop.SetOpacity(0.0)
                    except Exception:
                        pass
            return actor

        mesh = _build_textured_voxel_mesh(voxels, self.block_texture_map, self.uv_rects, self.block_render_modes, strict_mode=self.strict_mode)
        if mesh is None:
            # Render a blank scene (no faces) instead of raising; return a valid plotter
            outline = pv.Cube(bounds=(0, render_dim_x, 0, render_dim_y, 0, render_dim_z))
            plotter.add_mesh(outline, opacity=0.0)
            _init_bounds_actor()
            plotter.camera_position = 'iso'
            plotter.camera.zoom(1)
            return plotter

        if plotter is None:
            plotter = pv.Plotter(notebook=True) if interactive else pv.Plotter(off_screen=True)

        plotter.remove_all_lights()
        plotter.add_light(pv.Light(position=(1, -1, 1), intensity=1.0, color='white'))
        plotter.add_light(pv.Light(position=(-1, 1, 0.5), intensity=0.5, color='white'))
        plotter.add_light(pv.Light(position=(-0.5, -0.5, -1), intensity=0.3, color='white'))

        plotter.enable_depth_peeling()
        plotter.add_mesh(
            mesh,
            texture=self.atlas_texture,
            show_edges=False,
            lighting=True,
            culling=None,  # draw both sides so sprites are visible from any angle
        )

        outline = pv.Cube(bounds=(0, render_dim_x, 0, render_dim_y, 0, render_dim_z))
        plotter.add_mesh(outline, opacity=0.0)

        _init_bounds_actor()

        plotter.camera_position = 'iso'
        plotter.camera.zoom(1)
        return plotter

    def render_chunk_isometric_fitted(
        self,
        voxels,
        output_path,
        *,
        image_height_px=512,
        show_axis=False,
        use_textures=True,
        fit_padding=1.04,
        camera_distance_scale=2.5,
        interactive=False,
    ):
        """
        Render a chunk/world with the usual isometric view but fit the camera to
        the projected scene footprint instead of a conservative cube-like frame.

        This is intentionally separate from the default visualize/render helpers
        so existing rendering behavior elsewhere in the codebase is unchanged.
        """
        if image_height_px <= 0:
            raise ValueError("image_height_px must be > 0")
        if fit_padding <= 0:
            raise ValueError("fit_padding must be > 0")
        if camera_distance_scale <= 0:
            raise ValueError("camera_distance_scale must be > 0")

        render_fn = self.visualize_chunk_textured if use_textures else self.visualize_chunk
        plotter = render_fn(voxels, plotter=None, interactive=interactive, show_axis=show_axis)

        # Reuse the standard isometric orientation, then tighten the framing
        # based on the projected 2D footprint in camera space.
        plotter.camera_position = 'iso'
        bounds = plotter.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
        center = np.array(plotter.center, dtype=np.float64)
        extents = {
            "x": float(bounds[1] - bounds[0]),
            "y": float(bounds[3] - bounds[2]),
            "z": float(bounds[5] - bounds[4]),
        }
        max_extent = max(extents.values())

        cam = plotter.camera
        focal = np.array(cam.focal_point, dtype=np.float64)
        position = np.array(cam.position, dtype=np.float64)
        up = np.array(cam.up, dtype=np.float64)

        forward = focal - position
        forward = forward / max(1e-8, np.linalg.norm(forward))
        right = np.cross(forward, up)
        right = right / max(1e-8, np.linalg.norm(right))
        true_up = np.cross(right, forward)
        true_up = true_up / max(1e-8, np.linalg.norm(true_up))

        x0, x1, y0, y1, z0, z1 = bounds
        corners = np.array([
            [x0, y0, z0],
            [x0, y0, z1],
            [x0, y1, z0],
            [x0, y1, z1],
            [x1, y0, z0],
            [x1, y0, z1],
            [x1, y1, z0],
            [x1, y1, z1],
        ], dtype=np.float64)
        centered = corners - focal
        proj_x = centered @ right
        proj_y = centered @ true_up
        proj_width = max(1e-8, float(proj_x.max() - proj_x.min()))
        proj_height = max(1e-8, float(proj_y.max() - proj_y.min()))

        aspect = proj_width / proj_height
        image_width_px = max(1, int(np.ceil(float(image_height_px) * aspect)))

        cam.position = tuple((center - forward * (max_extent * float(camera_distance_scale))).tolist())
        cam.focal_point = tuple(center.tolist())
        cam.up = tuple(true_up.tolist())
        cam.parallel_projection = True
        cam.parallel_scale = 0.5 * proj_height * float(fit_padding)

        plotter.screenshot(
            filename=output_path,
            window_size=(int(image_width_px), int(image_height_px)),
            transparent_background=False,
        )

        try:
            plotter.close()
        except Exception:
            pass

    def render_chunk_side_views(
        self,
        voxels,
        output_dir,
        filename_prefix="chunk",
        image_px=512,
        show_axis=False,
        use_textures=True,
        fit_padding=1.02,
        camera_distance_scale=2.5,
        projection_mode="orthographic",
        perspective_view_angle=30.0,
        oblique_tilt=0.18,
        interactive=False,
    ):
        """
        Render four side views by rotating around vertical axis.

        This is a wrapper around existing chunk visualizers. It reuses normal
        rendering logic, then overrides camera settings.

        projection_mode:
            - "orthographic": fully flat side projections
            - "perspective": side views with depth cues
            - "oblique": perspective side views with slight downward/upward tilt

        Returns:
            dict mapping side labels -> saved image paths
        """
        if image_px <= 0:
            raise ValueError("image_px must be > 0")
        if fit_padding <= 0:
            raise ValueError("fit_padding must be > 0")
        if camera_distance_scale <= 0:
            raise ValueError("camera_distance_scale must be > 0")
        projection_mode = str(projection_mode).lower().strip()
        if projection_mode not in {"orthographic", "perspective", "oblique"}:
            raise ValueError(
                "projection_mode must be one of: 'orthographic', 'perspective', 'oblique'"
            )
        if perspective_view_angle <= 1 or perspective_view_angle >= 170:
            raise ValueError("perspective_view_angle must be in (1, 170)")

        os.makedirs(output_dir, exist_ok=True)
        render_fn = self.visualize_chunk_textured if use_textures else self.visualize_chunk

        # Rendering code applies axis transforms before mesh build, so the
        # effective "up" axis in render-space is Z. Rotate around this axis.
        side_views = [
            ("x_pos", (1.0, 0.0, 0.0), ("z", "y")),
            ("y_pos", (0.0, 1.0, 0.0), ("z", "x")),
            ("x_neg", (-1.0, 0.0, 0.0), ("z", "y")),
            ("y_neg", (0.0, -1.0, 0.0), ("z", "x")),
        ]

        image_paths = {}
        for side_name, look_dir, visible_axes in side_views:
            plotter = render_fn(voxels, plotter=None, interactive=interactive, show_axis=show_axis)
            plotter.enable_parallel_projection()

            bounds = plotter.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
            center = np.array(plotter.center, dtype=np.float64)
            extents = {
                "x": float(bounds[1] - bounds[0]),
                "y": float(bounds[3] - bounds[2]),
                "z": float(bounds[5] - bounds[4]),
            }
            max_extent = max(extents.values())
            dir_vec = np.array(look_dir, dtype=np.float64)
            if projection_mode == "oblique":
                dir_vec = dir_vec + np.array((0.0, 0.0, float(oblique_tilt)), dtype=np.float64)
            dir_vec = dir_vec / max(1e-8, np.linalg.norm(dir_vec))

            # Place camera far enough from scene center; with parallel projection
            # distance does not change scale, but avoids near-clipping.
            camera_position = center + dir_vec * (max_extent * float(camera_distance_scale))
            plotter.camera.position = tuple(camera_position.tolist())
            plotter.camera.focal_point = tuple(center.tolist())
            plotter.camera.up = (0.0, 0.0, 1.0)

            # Fit visible vertical/horizontal extents tightly in a square image.
            visible_height = extents[visible_axes[0]]
            visible_width = extents[visible_axes[1]]
            target_size = max(visible_height, visible_width)
            if projection_mode == "orthographic":
                plotter.camera.parallel_projection = True
                plotter.camera.parallel_scale = 0.5 * target_size * float(fit_padding)
            else:
                plotter.camera.parallel_projection = False
                plotter.camera.view_angle = float(perspective_view_angle)
                # Fit the target size in perspective view with a small depth margin.
                half_fov_rad = np.deg2rad(float(perspective_view_angle) * 0.5)
                fit_dist = (0.5 * target_size * float(fit_padding)) / max(1e-8, np.tan(half_fov_rad))
                camera_position = center + dir_vec * (fit_dist * 1.25)
                plotter.camera.position = tuple(camera_position.tolist())

            out_path = os.path.join(output_dir, f"{filename_prefix}_{side_name}.png")
            plotter.screenshot(
                filename=out_path,
                window_size=(int(image_px), int(image_px)),
                transparent_background=False,
            )
            image_paths[side_name] = out_path

            try:
                plotter.close()
            except Exception:
                pass

        return image_paths
    
    
    
    def visualize_chunk_with_biomes(self, voxels, biomes, plotter=None, interactive=False):
        """
        3D visualization of a Minecraft chunk with biome overlay using PyVista.
        
        Args:
            voxels: numpy array/torch.Tensor of block IDs
            biomes: numpy array/torch.Tensor of biome strings
            plotter: Optional existing PyVista plotter
            interactive: Whether to create an interactive display
        """
        # Convert tensors to numpy if needed
        if isinstance(voxels, torch.Tensor):
            if voxels.dim() == 4:  # One-hot encoded
                voxels = voxels.detach().cpu()
                voxels = torch.argmax(voxels, dim=0).numpy()
            else:
                voxels = voxels.detach().cpu().numpy()
        if isinstance(biomes, torch.Tensor):
            if biomes.dim() == 4:  # One-hot encoded
                biomes = biomes.detach().cpu()
                biomes = torch.argmax(biomes, dim=0).numpy()
            else:
                biomes = biomes.detach().cpu().numpy()

        dim = voxels.shape[2]  
        # Apply the same transformations to both arrays
        voxels = voxels.transpose(2, 0, 1)
        voxels = np.rot90(voxels, 1, (0, 1))
        biomes = biomes.transpose(2, 0, 1)
        biomes = np.rot90(biomes, 1, (0, 1))

        # Create plotter if not provided
        if plotter is None:
            if interactive:
                plotter = pv.Plotter(notebook=True)
            else:
                plotter = pv.Plotter(off_screen=True)

        # Remove existing lights and add three-point lighting
        plotter.remove_all_lights()
        plotter.add_light(pv.Light(position=(1, -1, 1), intensity=1.0, color='white'))
        plotter.add_light(pv.Light(position=(-1, 1, 0.5), intensity=0.5, color='white'))
        plotter.add_light(pv.Light(position=(-0.5, -0.5, -1), intensity=0.3, color='white'))

        # First plot the regular blocks
        grid = pv.ImageData()
        grid.dimensions = np.array(voxels.shape) + 1
        grid.cell_data["values"] = voxels.flatten(order="F")

        # Plot each block type
        mask = (voxels != 5) & (voxels != -1)
        unique_blocks = np.unique(voxels[mask])
        
        for block_id in unique_blocks:
            threshold = grid.threshold([block_id-0.5, block_id+0.5])
            if block_id in self.blocks_to_cols:
                color = self.blocks_to_cols[int(block_id)]
                opacity = 1.0 if isinstance(color, str) or len(color) == 3 else color[3]
            else:
                color = (1.0, 0.0, 0.0)
                opacity = 0.2
            
            plotter.add_mesh(threshold, 
                        color=color,
                        opacity=opacity,
                        show_edges=True,
                        edge_color='black',
                        line_width=.2,
                        edge_opacity=0.2,
                        lighting=True)

        # Create a colormap for biomes using distinct RGB values
        unique_biomes = np.unique(biomes)
        num_biomes = len(unique_biomes)
        
        # Generate distinct colors using HSV color space
        hsv_colors = [(i/num_biomes, 0.8, 0.8) for i in range(num_biomes)]
        rgb_colors = [colorsys.hsv_to_rgb(*hsv) for hsv in hsv_colors]
        biome_color_map = dict(zip(unique_biomes, rgb_colors))
        
        # Plot biome overlay
        biome_grid = pv.ImageData()
        biome_grid.dimensions = np.array(biomes.shape) + 1
        
        legend_entries = []
        
        for biome in unique_biomes:
            # Create mask for this biome
            biome_mask = (biomes == biome).astype(float)
            biome_grid.cell_data["values"] = biome_mask.flatten(order="F")
            
            # Get RGB color for this biome
            rgb_color = biome_color_map[biome]
            
            # Add semi-transparent overlay
            threshold = biome_grid.threshold([0.5, 1.5])
            plotter.add_mesh(threshold,
                            color=rgb_color,
                            opacity=0.2,
                            show_edges=False,
                            lighting=False)
            
            # Add to legend entries - IMPORTANT: first item must be the RGB color
            legend_entries.append([str(biome), rgb_color])

        # Add dummy cube for bounds
        outline = pv.Cube(bounds=(0, dim, 0, dim, 0, dim))
        plotter.add_mesh(outline, opacity=0.0)

        # Add bounds with consistent settings
        plotter.show_bounds(
            grid='back',
            location='back',
            font_size=8,
            bold=False,
            font_family='arial',
            use_2d=False,
            bounds=[0, dim, 0, dim, 0, dim],
            axes_ranges=[0, dim, 0, dim, 0, dim],
            padding=0.0,
            n_xlabels=2,
            n_ylabels=2,
            n_zlabels=2
        )

        # Add legend if we have entries
        if legend_entries:
            plotter.add_legend(legend_entries, bcolor=(0.9, 0.9, 0.9, 0.3))

        # Set camera position and zoom
        plotter.camera_position = 'iso'
        plotter.camera.zoom(1)

        return plotter
    
def save_chunks(sampled_chunks, epoch, results_out_dir, converter, visualizer=None, classes=None, textured=False, filename_prefix='sampled_chunks'):
    if visualizer is None:
        if textured:
            visualizer = MinecraftVisualizerPyVista(
                build_textures=True,
            )
            render_fn = visualizer.visualize_chunk_textured
        else:
            visualizer = MinecraftVisualizerPyVista()
            render_fn = visualizer.visualize_chunk

    # TODO: convert back to integer (fill in once embedding implemented)
    batch_size = min(sampled_chunks.shape[0], 32)
    ncols = 8
    nrows = (batch_size + ncols - 1) // ncols

    single_size = 200  # Size of each subplot in pixels
    
    # Create matplotlib figure with subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2.5))
    
    # Handle case where we only have one row
    if nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)
    
    # Generate chunk images and display in subplots
    for i in range(batch_size):
        row = i // ncols
        col = i % ncols
        
        # Generate chunk visualization
        plotter = render_fn(sampled_chunks[i])
        img = plotter.screenshot(window_size=(single_size, single_size), 
                               transparent_background=True, 
                               return_img=True)
        plotter.close()
        
        # Display image in subplot
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
        
        # Add class label if provided
        if classes is not None and i < len(classes):
            axes[row, col].set_title(str(classes[i]), fontsize=12, fontweight='bold')
    
    # Hide any unused subplots
    for i in range(batch_size, nrows * ncols):
        row = i // ncols
        col = i % ncols
        axes[row, col].axis('off')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(results_out_dir, f'{filename_prefix}_ep_{epoch}.png'), 
                dpi=150, bbox_inches='tight')
    plt.close(fig)

def _resolve_texture_paths(tex_names, texture_dirs):
    """
    Resolve texture file paths, checking directories in order.

    For each texture name, try each directory in texture_dirs until found.

    Args:
        tex_names: set/list of texture file names to resolve
        texture_dirs: list of directories to search (in priority order)

    Returns:
        tuple: (dict mapping texture name -> resolved path, list of missing texture names)
    """
    resolved = {}
    missing = []
    for name in tex_names:
        if name == 'empty.png':
            # Special case: empty.png is intentionally empty/transparent
            resolved[name] = None  # Will be handled specially
            continue
        
        found = False
        for tex_dir in texture_dirs:
            path = os.path.join(tex_dir, name)
            if os.path.isfile(path):
                resolved[name] = path
                found = True
                break
        
        if not found:
            missing.append(name)
            # Use first directory as placeholder path (will fail on load)
            resolved[name] = os.path.join(texture_dirs[0], name) if texture_dirs else name
    
    return resolved, missing


def _load_images_rgba(texture_paths, missing_fallback_color=(255, 0, 255, 128), strict_mode=False):
    """
    Load textures from resolved paths into RGBA numpy arrays.

    Args:
        texture_paths: dict mapping texture name -> file path (or None for empty.png)
        missing_fallback_color: RGBA color for placeholder textures (default: translucent magenta)
        strict_mode: if True, raise MissingTextureError for any missing textures

    If a texture file doesn't exist and strict_mode is False, it will be replaced
    with a translucent magenta "missing" placeholder.
    """
    imgs = {}
    size = None
    failed_textures = []
    
    # Try to load missing.png first to get the canonical size and use as fallback
    missing_img = None
    if 'missing.png' in texture_paths and texture_paths['missing.png'] is not None:
        try:
            missing_img = Image.open(texture_paths['missing.png']).convert('RGBA')
            size = missing_img.size
        except Exception:
            pass

    for name, path in texture_paths.items():
        # Handle empty.png specially - create a transparent texture
        if path is None:  # empty.png
            placeholder_size = size if size else (16, 16)
            im = Image.new('RGBA', placeholder_size, (0, 0, 0, 0))  # Fully transparent
        else:
            try:
                im = Image.open(path).convert('RGBA')
            except Exception as e:
                failed_textures.append((name, path, str(e)))
                # Create a placeholder translucent magenta texture
                if missing_img is not None:
                    im = missing_img.copy()
                else:
                    placeholder_size = size if size else (16, 16)
                    im = Image.new('RGBA', placeholder_size, missing_fallback_color)
        if size is None:
            size = im.size
        else:
            if im.size != size:
                im = im.resize(size, Image.NEAREST)
        imgs[name] = np.array(im)
    
    # In strict mode, raise an error listing all missing textures
    if strict_mode and failed_textures:
        error_lines = ["Missing texture files:"]
        for name, path, err in failed_textures:
            error_lines.append(f"  - {name}: {path} ({err})")
        raise MissingTextureError("\n".join(error_lines))
    
    return imgs

def _build_texture_atlas(images_by_name):
    names = list(images_by_name.keys())
    if not names:
        raise ValueError("No textures provided.")
    h, w, _ = next(iter(images_by_name.values())).shape
    cols = int(ceil(sqrt(len(names))))
    rows = int(ceil(len(names) / cols))
    atlas = np.zeros((rows * h, cols * w, 4), dtype=np.uint8)
    uv_rects = {}
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= len(names):
                break
            name = names[idx]
            tile = images_by_name[name]
            y0, x0 = r * h, c * w
            atlas[y0:y0+h, x0:x0+w, :] = tile
            u0 = x0 / (cols * w)
            u1 = (x0 + w) / (cols * w)
            v1 = 1.0 - (y0 / (rows * h))
            v0 = 1.0 - ((y0 + h) / (rows * h))
            uv_rects[name] = (u0, v0, u1, v1)
            idx += 1
    return atlas, uv_rects

_FACE_DIRS = {'px': (1,0,0), 'nx': (-1,0,0), 'py': (0,1,0), 'ny': (0,-1,0), 'pz': (0,0,1), 'nz': (0,0,-1)}
_FACE_VERTS = {
    'px': [(1,0,0),(1,1,0),(1,1,1),(1,0,1)],
    'nx': [(0,0,0),(0,0,1),(0,1,1),(0,1,0)],
    'py': [(0,1,0),(0,1,1),(1,1,1),(1,1,0)],
    'ny': [(0,0,0),(1,0,0),(1,0,1),(0,0,1)],
    'pz': [(0,0,1),(1,0,1),(1,1,1),(0,1,1)],
    'nz': [(0,0,0),(0,1,0),(1,1,0),(1,0,0)],
}
def _is_air_or_void(v): return (v == 5) or (v == -1)

def _build_textured_voxel_mesh(voxels, block_uv_keys, uv_rects, block_render_modes=None, strict_mode=True):
    sx, sy, sz = voxels.shape
    points, faces, tcoords = [], [], []
    block_render_modes = block_render_modes or {}
    
    # Track missing textures for error reporting
    missing_texture_refs = set()  # (block_id, texture_name)
    
    # fallback texture if a face texture is missing: try to pick an opaque block
    def pick_fallback_tex():
        # Prefer explicit missing texture if present in the atlas
        if 'missing.png' in uv_rects:
            return 'missing.png'
        preferred = [
            'planks_oak.png', 'stone.png', 'cobblestone.png', 'sand.png', 'dirt.png', 'gravel.png',
            'sandstone_side.png', 'sandstone_top.png', 'bedrock.png'
        ]
        for name in preferred:
            if name in uv_rects:
                return name
        # avoid transparent/sprite-like textures
        avoid_substrings = (
            'grass', 'tall', 'vine', 'water', 'lava', 'leaves', 'flower', 'glass', 'pane', 'lily', 'sapling', 'wheat', 'reeds', 'double_plant'
        )
        # choose the first key that doesn't look transparent
        for name in uv_rects.keys():
            lname = name.lower()
            if not any(s in lname for s in avoid_substrings):
                return name
        # last resort
        return next(iter(uv_rects.keys()))

    fallback_tex = pick_fallback_tex()

    def add_quad_local(x, y, z, verts_local, uv_rect):
        base = len(points)
        u0, v0, u1, v1 = uv_rect
        uv_map = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
        for (lx, ly, lz), (uu, vv) in zip(verts_local, uv_map):
            points.append((x + lx, y + ly, z + lz))
            tcoords.append((uu, vv))
        faces.extend([4, base + 0, base + 1, base + 2, base + 3])

    def _face_uvs(face_key, uv_rect):
        """Return per-vertex UVs for a face so that the V axis points toward +Z (up)
        and the U axis points toward +X or +Y consistently across faces.
        This keeps side textures (e.g., grass_side) upright on all sides.
        """
        u0, v0, u1, v1 = uv_rect
        # Faces whose existing vertex order already matches (u along +tangent, v along +Z)
        if face_key in ('px', 'ny', 'pz'):
            # default ordering: (u0,v0),(u1,v0),(u1,v1),(u0,v1)
            return [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
        else:
            # For 'nx', 'py', and 'nz' the default vertex order would rotate the texture.
            # Swap V on the second vertex pair to keep the top of the texture aligned with +Z.
            return [(u0, v0), (u0, v1), (u1, v1), (u1, v0)]

    def add_face(x, y, z, face_key, uv_rect):
        base = len(points)
        verts = _FACE_VERTS[face_key]
        uv_map = _face_uvs(face_key, uv_rect)
        for (dx, dy, dz), (uu, vv) in zip(verts, uv_map):
            points.append((x + dx, y + dy, z + dz))
            tcoords.append((uu, vv))
        faces.extend([4, base + 0, base + 1, base + 2, base + 3])

    def neighbor_occludes(x, y, z, face_key):
        dx, dy, dz = _FACE_DIRS[face_key]
        nxp, nyp, nzp = x + dx, y + dy, z + dz
        if nxp < 0 or nyp < 0 or nzp < 0 or nxp >= sx or nyp >= sy or nzp >= sz:
            return False
        nbid = int(voxels[nxp, nyp, nzp])
        if _is_air_or_void(nbid):
            return False
        return (block_render_modes.get(nbid, 'cube') == 'cube')

    def _partial_face_vertices(face_key, x0, x1, y0, y1, z0, z1):
        if face_key == 'px':
            return [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)]
        if face_key == 'nx':
            return [(x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)]
        if face_key == 'py':
            return [(x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0)]
        if face_key == 'ny':
            return [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)]
        if face_key == 'pz':
            return [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        if face_key == 'nz':
            return [(x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0)]
        return None

    def add_partial_face(x, y, z, face_key, x0, x1, y0, y1, z0, z1, uv_rect):
        base = len(points)
        verts = _partial_face_vertices(face_key, x0, x1, y0, y1, z0, z1)
        uv_map = _face_uvs(face_key, uv_rect)
        for (vx, vy, vz), (uu, vv) in zip(verts, uv_map):
            points.append((x + vx, y + vy, z + vz))
            tcoords.append((uu, vv))
        faces.extend([4, base + 0, base + 1, base + 2, base + 3])

    def add_box(x, y, z, xr, yr, zr, uv_rects_by_face, exclude_faces=None, block_id=None):
        exclude_faces = exclude_faces or set()
        x0, x1 = xr
        y0, y1 = yr
        z0, z1 = zr
        # for each face, decide visibility: internal faces always visible unless excluded;
        # boundary faces (at 0 or 1) can be occluded by neighbor cubes
        for fk in ('px', 'nx', 'py', 'ny', 'pz', 'nz'):
            if fk in exclude_faces:
                continue
            # determine if this face lies on a block boundary
            on_boundary = (
                (fk == 'px' and x1 == 1.0) or (fk == 'nx' and x0 == 0.0) or
                (fk == 'py' and y1 == 1.0) or (fk == 'ny' and y0 == 0.0) or
                (fk == 'pz' and z1 == 1.0) or (fk == 'nz' and z0 == 0.0)
            )
            if on_boundary and neighbor_occludes(x, y, z, fk):
                continue
            tex = uv_rects_by_face.get(fk)
            original_tex = tex
            if tex is None or tex not in uv_rects:
                if tex is not None:
                    missing_texture_refs.add((block_id, tex))
                tex = fallback_tex
            add_partial_face(x, y, z, fk, x0, x1, y0, y1, z0, z1, uv_rects[tex])

    def face_tex_map(face_keys):
        return {
            'px': face_keys.get('side', face_keys.get('all')),
            'nx': face_keys.get('side', face_keys.get('all')),
            'py': face_keys.get('side', face_keys.get('all')),
            'ny': face_keys.get('side', face_keys.get('all')),
            'pz': face_keys.get('top',  face_keys.get('all')),
            'nz': face_keys.get('bottom', face_keys.get('all')),
        }

    def mirror_z(zr):
        z0, z1 = zr
        return (1.0 - z1, 1.0 - z0)

    def internal_face_key_for_half(axis, rng):
        # rng is a tuple (min,max) within [0,1]; if the face sits at 0.5, return which face key it corresponds to
        if axis == 'x':
            return 'nx' if rng[0] == 0.5 else ('px' if rng[1] == 0.5 else None)
        if axis == 'y':
            return 'ny' if rng[0] == 0.5 else ('py' if rng[1] == 0.5 else None)
        if axis == 'z':
            return 'nz' if rng[0] == 0.5 else ('pz' if rng[1] == 0.5 else None)
        return None

    def straight_stair_boxes(facing):
        # returns list of (xr, yr, zr, exclude_faces)
        # Construct as: bottom slab split into near+far (exclude top face on near to avoid z-fight),
        # plus upper half on near side (exclude bottom face to avoid z-fight).
        if facing in ('px', 'nx'):
            near = (0.5, 1.0) if facing == 'px' else (0.0, 0.5)
            far  = (0.0, 0.5) if facing == 'px' else (0.5, 1.0)
            xr_near, xr_far = near, far
            yr_full = (0.0, 1.0)
            boxes = [
                (xr_far,  yr_full, (0.0, 0.5), set()),        # lower slab on far half
                (xr_near, yr_full, (0.0, 0.5), {'pz'}),        # lower slab on near half (no top face)
                (xr_near, yr_full, (0.5, 1.0), {'nz'}),        # upper near (no bottom face)
            ]
            return boxes
        else:  # 'py' or 'ny'
            near = (0.5, 1.0) if facing == 'py' else (0.0, 0.5)
            far  = (0.0, 0.5) if facing == 'py' else (0.5, 1.0)
            yr_near, yr_far = near, far
            xr_full = (0.0, 1.0)
            boxes = [
                (xr_full, yr_far,  (0.0, 0.5), set()),        # lower slab on far half
                (xr_full, yr_near, (0.0, 0.5), {'pz'}),        # lower slab on near half (no top face)
                (xr_full, yr_near, (0.5, 1.0), {'nz'}),        # upper near (no bottom face)
            ]
            return boxes

    def corner_mapping(facing, turn):
        # returns primary_axis, lateral_axis, primary near/far ranges, lateral near/far ranges
        if facing in ('px', 'nx'):
            primary_axis = 'x'
            lateral_axis = 'y'
            p_near = (0.5, 1.0) if facing == 'px' else (0.0, 0.5)
            p_far  = (0.0, 0.5) if facing == 'px' else (0.5, 1.0)
            if facing == 'px':
                l_near = (0.5, 1.0) if turn == 'left' else (0.0, 0.5)
            else:  # 'nx'
                l_near = (0.0, 0.5) if turn == 'left' else (0.5, 1.0)
            l_far = (0.0, 0.5) if l_near == (0.5, 1.0) else (0.5, 1.0)
        else:
            primary_axis = 'y'
            lateral_axis = 'x'
            p_near = (0.5, 1.0) if facing == 'py' else (0.0, 0.5)
            p_far  = (0.0, 0.5) if facing == 'py' else (0.5, 1.0)
            if facing == 'py':
                l_near = (0.0, 0.5) if turn == 'left' else (0.5, 1.0)
            else:  # 'ny'
                l_near = (0.5, 1.0) if turn == 'left' else (0.0, 0.5)
            l_far = (0.0, 0.5) if l_near == (0.5, 1.0) else (0.5, 1.0)
        return primary_axis, lateral_axis, p_near, p_far, l_near, l_far

    def build_stair(mode_cfg, x, y, z, face_to_tex, block_id=None):
        facing = mode_cfg.get('facing')
        half = mode_cfg.get('half', 'bottom')
        shape = mode_cfg.get('shape', 'straight')  # 'straight', 'inner', 'outer'
        tex_map = face_to_tex
        if shape == 'straight':
            boxes = straight_stair_boxes(facing)
            if half == 'top':
                # swap internal face exclusions along z when mirroring
                mirrored = []
                for (xr, yr, zr, ex) in boxes:
                    ex2 = set(('nz' if f == 'pz' else 'pz' if f == 'nz' else f) for f in ex)
                    mirrored.append((xr, yr, mirror_z(zr), ex2))
                boxes = mirrored
            for xr, yr, zr, ex in boxes:
                add_box(x, y, z, xr, yr, zr, tex_map, ex, block_id=block_id)
            return
        # corner variants
        primary_axis, lateral_axis, p_near, p_far, l_near, l_far = corner_mapping(facing, mode_cfg.get('turn', 'left'))
        # helpers to assemble ranges by axes
        def make_ranges(ax_x, ax_y, xr, yr, zr):
            return xr if ax_x == 'x' else yr, yr if ax_y == 'y' else xr, zr
        boxes = []
        if mode_cfg.get('variant', mode_cfg.get('type', 'outer')) in ('outer', 'outer_corner', 'outer-right', 'outer-left'):
            # near corner split in z + two adjacent lower strips
            # determine xr, yr from primary/lateral composition
            def to_xy(pr, lr):
                if primary_axis == 'x':
                    return pr, lr
                else:
                    return lr, pr
            # near corner box split in z
            nc_xr, nc_yr = to_xy(p_near, l_near)
            boxes.append((nc_xr, nc_yr, (0.0, 0.5), set()))
            boxes.append((nc_xr, nc_yr, (0.5, 1.0), set()))
            # adjacent lower strips
            b_xr, b_yr = to_xy(p_far, l_near)
            c_xr, c_yr = to_xy(p_near, l_far)
            boxes.append((b_xr, b_yr, (0.0, 0.5), set()))
            boxes.append((c_xr, c_yr, (0.0, 0.5), set()))
            # exclude internal faces: between near-corner lower and b along primary half plane
            if primary_axis == 'x':
                boxes[0][3].add('nx' if p_near[0] == 0.5 else 'px')
                boxes[2][3].add('px' if p_near[0] == 0.5 else 'nx')
            else:
                boxes[0][3].add('ny' if p_near[0] == 0.5 else 'py')
                boxes[2][3].add('py' if p_near[0] == 0.5 else 'ny')
            # between near-corner lower and c along lateral half plane
            if lateral_axis == 'y':
                boxes[0][3].add('ny' if l_near[0] == 0.5 else 'py')
                boxes[3][3].add('py' if l_near[0] == 0.5 else 'ny')
            else:
                boxes[0][3].add('nx' if l_near[0] == 0.5 else 'px')
                boxes[3][3].add('px' if l_near[0] == 0.5 else 'nx')
            # between near-corner lower and upper at z=0.5
            boxes[0][3].add('pz')
            boxes[1][3].add('nz')
        else:
            # inner corner
            def to_xy(pr, lr):
                if primary_axis == 'x':
                    return pr, lr
                else:
                    return lr, pr
            # upper: three tiles (corner + two arms without overlap)
            a_xr, a_yr = to_xy(p_near, l_far)
            b_xr, b_yr = to_xy(p_far, l_near)
            c_xr, c_yr = to_xy(p_near, l_near)
            boxes.append((a_xr, a_yr, (0.5, 1.0), set()))
            boxes.append((b_xr, b_yr, (0.5, 1.0), set()))
            boxes.append((c_xr, c_yr, (0.5, 1.0), set()))
            # lower: two arms
            boxes.append((a_xr, a_yr, (0.0, 0.5), set()))
            boxes.append((b_xr, b_yr, (0.0, 0.5), set()))
            # exclusions along z between upper/lower arms
            boxes[3][3].add('pz'); boxes[0][3].add('nz')
            boxes[4][3].add('pz'); boxes[1][3].add('nz')
            # internal planes between upper corner and upper arms
            if primary_axis == 'x':
                boxes[2][3].add('nx' if p_near[0] == 0.5 else 'px')
                boxes[0][3].add('px' if p_near[0] == 0.5 else 'nx')
            else:
                boxes[2][3].add('ny' if p_near[0] == 0.5 else 'py')
                boxes[0][3].add('py' if p_near[0] == 0.5 else 'ny')
            if lateral_axis == 'y':
                boxes[2][3].add('ny' if l_near[0] == 0.5 else 'py')
                boxes[1][3].add('py' if l_near[0] == 0.5 else 'ny')
            else:
                boxes[2][3].add('nx' if l_near[0] == 0.5 else 'px')
                boxes[1][3].add('px' if l_near[0] == 0.5 else 'nx')
        if half == 'top':
            boxes = [(xr, yr, mirror_z(zr), ex) for (xr, yr, zr, ex) in boxes]
        for xr, yr, zr, ex in boxes:
            add_box(x, y, z, xr, yr, zr, tex_map, ex, block_id=block_id)

    for x in range(sx):
        for y in range(sy):
            for z in range(sz):
                bid = int(voxels[x, y, z])
                if _is_air_or_void(bid):
                    continue

                face_keys = block_uv_keys.get(bid) or {'all': str(bid)}
                mode = block_render_modes.get(bid, 'cube')

                # choose per-face texture name
                face_to_tex = face_tex_map(face_keys)

                if mode == 'sprite_cross':
                    # two vertical quads that cross through the center
                    tex = face_keys.get('all') or face_keys.get('side') or face_keys.get('top') or face_keys.get('bottom')
                    original_tex = tex
                    if tex is None or tex not in uv_rects:
                        if tex is not None:
                            missing_texture_refs.add((bid, tex))
                        tex = fallback_tex
                    uv = uv_rects[tex]
                    # plane at y=0.5 (x-z plane)
                    add_quad_local(x, y, z, [(0.0, 0.5, 0.0),(1.0, 0.5, 0.0),(1.0, 0.5, 1.0),(0.0, 0.5, 1.0)], uv)
                    # plane at x=0.5 (y-z plane)
                    add_quad_local(x, y, z, [(0.5, 0.0, 0.0),(0.5, 1.0, 0.0),(0.5, 1.0, 1.0),(0.5, 0.0, 1.0)], uv)
                    continue

                if mode == 'sides_only':
                    # render only vertical sides, regardless of neighbors; skip top/bottom
                    for fk in ('px', 'nx', 'py', 'ny'):
                        tex = face_to_tex[fk]
                        original_tex = tex
                        if tex is None or tex not in uv_rects:
                            if tex is not None:
                                missing_texture_refs.add((bid, tex))
                            tex = fallback_tex
                        add_face(x, y, z, fk, uv_rects[tex])
                    continue

                # advanced shapes via dict configs
                if isinstance(mode, dict):
                    m = mode.get('mode', 'cube')
                    if m == 'slab':
                        half = mode.get('half', 'bottom')
                        zr = (0.0, 0.5) if half == 'bottom' else (0.5, 1.0)
                        add_box(x, y, z, (0.0, 1.0), (0.0, 1.0), zr, face_to_tex, block_id=bid)
                        continue
                    if m == 'stair':
                        build_stair(mode, x, y, z, face_to_tex, block_id=bid)
                        continue

                # default 'cube' with neighbor-based culling
                for fk, (dx, dy, dz) in _FACE_DIRS.items():
                    nxp, nyp, nzp = x + dx, y + dy, z + dz
                    if nxp < 0 or nyp < 0 or nzp < 0 or nxp >= sx or nyp >= sy or nzp >= sz:
                        neighbor_blocks = False
                    else:
                        nbid = int(voxels[nxp, nyp, nzp])
                        # air/void never blocks
                        if _is_air_or_void(nbid):
                            neighbor_blocks = False
                        else:
                            # only 'cube' neighbors occlude; sprite/transparent-style blocks shouldn’t
                            neighbor_blocks = (block_render_modes.get(nbid, 'cube') == 'cube')

                    if neighbor_blocks:
                        continue
                    tex = face_to_tex[fk]
                    original_tex = tex
                    if tex is None or tex not in uv_rects:
                        if tex is not None:
                            missing_texture_refs.add((bid, tex))
                        tex = fallback_tex
                    add_face(x, y, z, fk, uv_rects[tex])

    # In strict mode, raise an error if any textures were missing
    if strict_mode and missing_texture_refs:
        # Load block names for better error messages
        block_names = {}
        try:
            import json
            if os.path.isfile("assets/block_types_updated.json"):
                with open("assets/block_types_updated.json") as f:
                    block_names = json.load(f)
        except Exception:
            pass
        
        error_lines = [
            f"Missing {len(missing_texture_refs)} texture reference(s) in atlas:",
            "",
        ]
        for bid, tex_name in sorted(missing_texture_refs, key=lambda x: x[0]):
            name = block_names.get(str(bid), 'UNKNOWN')
            error_lines.append(f"  Block ID {bid:4d} ({name:40s}): texture '{tex_name}'")
        error_lines.append("")
        error_lines.append("Either add the missing texture files to block_textures/ or update the")
        error_lines.append("block_texture_map in _default_block_texture_map() to use existing textures.")
        raise MissingTextureError("\n".join(error_lines))

    if not points:
        return None

    mesh = pv.PolyData(np.asarray(points, dtype=np.float32), np.asarray(faces, dtype=np.int64))
    mesh.active_t_coords = np.asarray(tcoords, dtype=np.float32)
    return mesh

def _numpy_to_texture(img_rgba):
    return pv.numpy_to_texture(img_rgba)

def visualize_sampling_gif(
    steps,
    out_path,
    *,
    converter,
    textures_dir=None,
    image_px=256,
    fps=8,
    show_axis=False,
    zoom=1.0,
    sample_index=0,
    include_initial_empty=True,
    hold_final_frames=0,
    frame_stride=1,
    max_frames=None,
    optimize_gif=False,
):
    """
    Render a GIF showing blocks spawning in as the discrete diffusion unmasking proceeds.

    Args:
        steps: Sampling intermediates containing mask tokens. Accepts:
            - torch.Tensor of shape [T, H, W, D]
            - torch.Tensor of shape [B, T, H, W, D] (uses sample_index)
            - list of length T with tensors/arrays shaped [H, W, D]
        out_path: Destination .gif path
        converter: BlockBiomeConverter used to map indices -> original block IDs
        textures_dir: If provided, render textured; else fallback to color rendering
        image_px: Frame resolution in pixels
        fps: Frames per second for GIF
        show_axis: Whether to show axes in renders
        zoom: Camera zoom factor
        sample_index: For batched steps, which sample to visualize
        include_initial_empty: If True, prepend an empty frame
        hold_final_frames: Repeat the final frame this many extra times
        frame_stride: Only render every Nth step (>= 1)
        max_frames: If set, cap total rendered frames by subsampling evenly
        optimize_gif: Pass optimize flag to Pillow (slower but smaller)
    """
    # Normalize steps to tensor [T, H, W, D]
    if isinstance(steps, list):
        steps_t = torch.stack([torch.as_tensor(s) for s in steps], dim=0)
    else:
        steps_t = torch.as_tensor(steps)
    if steps_t.dim() == 5:
        # Assume [B, T, H, W, D] as returned by MD4Discrete3D with return_intermediates
        steps_t = steps_t[int(sample_index)]
    if steps_t.dim() != 4:
        raise ValueError(f"Expected steps of shape [T,H,W,D] or [B,T,H,W,D], got {tuple(steps_t.shape)}")

    steps_t = steps_t.detach().to('cpu').long()
    T, H, W, D = steps_t.shape

    # Infer mask token as the maximum value seen across steps (MD4 uses K for mask)
    mask_token_id = int(steps_t.max().item())

    # Determine index for air so hidden voxels render as empty
    try:
        air_index = int(converter.get_air_block_index())
    except Exception:
        air_index = 0

    # Build a fast LUT for indices -> original block IDs (vectorized)
    index_keys = list(getattr(converter, 'index_to_block', {}).keys())
    if len(index_keys) == 0:
        # Fallback: identity LUT up to max observed (safe because masked entries are replaced by air_index)
        K = int(max(mask_token_id, int(steps_t.max().item())) + 1)
        lut = torch.arange(K, dtype=torch.long)
    else:
        K = int(max(index_keys)) + 1
        lut = torch.full((K,), fill_value=0, dtype=torch.long)
        for idx, bid in converter.index_to_block.items():
            lut[int(idx)] = int(bid)

    def indices_to_block_ids_fast(indices_grid: torch.Tensor) -> torch.Tensor:
        return lut[indices_grid]

    # Setup visualizer
    if textures_dir:
        visualizer = MinecraftVisualizerPyVista(
            textures_dir=textures_dir,
            build_textures=True,
        )
        render_fn = visualizer.visualize_chunk_textured
    else:
        visualizer = MinecraftVisualizerPyVista()
        render_fn = visualizer.visualize_chunk

    frames = []
    accum_indices = torch.full((H, W, D), fill_value=air_index, dtype=torch.long)
    revealed = torch.zeros((H, W, D), dtype=torch.bool)

    # Determine which time indices to render (downsample if requested)
    frame_indices = list(range(T))
    s = max(1, int(frame_stride))
    if s > 1:
        frame_indices = frame_indices[::s]
    if max_frames not in (None, 0) and len(frame_indices) > int(max_frames):
        sel = np.linspace(0, len(frame_indices) - 1, num=int(max_frames), dtype=int).tolist()
        frame_indices = [frame_indices[i] for i in sel]
    # Always include last step
    if len(frame_indices) == 0 or frame_indices[-1] != (T - 1):
        frame_indices = sorted(set(frame_indices + [T - 1]))
    frame_set = set(frame_indices)

    plotter = None
    def render_current(indices_grid: torch.Tensor):
        nonlocal plotter
        block_ids = indices_to_block_ids_fast(indices_grid)
        # Clear previous actors if reusing plotter
        if plotter is not None:
            try:
                plotter.clear()
            except Exception:
                pass
        plotter = render_fn(block_ids, plotter=plotter, interactive=False, show_axis=bool(show_axis))
        if zoom != 1.0 and hasattr(plotter, 'camera') and hasattr(plotter.camera, 'zoom'):
            plotter.camera.zoom(float(zoom))
        img = plotter.screenshot(
            window_size=(int(image_px), int(image_px)),
            transparent_background=False,
            return_img=True,
        )
        frames.append(Image.fromarray(img))

    # Optional initial empty frame
    if include_initial_empty:
        render_current(accum_indices)

    # Accumulate newly unmasked blocks each step; render only selected frames
    for t in range(T):
        curr = steps_t[t]
        new_positions = (~revealed) & (curr != mask_token_id)
        if new_positions.any():
            accum_indices[new_positions] = curr[new_positions]
            revealed[new_positions] = True
        if t in frame_set:
            render_current(accum_indices)

    # Optionally hold on the final frame for a beat
    if int(hold_final_frames) > 0 and len(frames) > 0:
        for _ in range(int(hold_final_frames)):
            frames.append(frames[-1].copy())

    # Ensure output directory exists
    out_dir = os.path.dirname(str(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    duration_ms = max(1, int(1000 / max(1, int(fps))))
    save_kwargs = dict(save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
    if optimize_gif:
        save_kwargs['optimize'] = True
    frames[0].save(str(out_path), **save_kwargs)
    # Close plotter if open
    try:
        if plotter is not None:
            plotter.close()
    except Exception:
        pass
    return out_path