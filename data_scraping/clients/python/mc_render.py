import os
from pdb import set_trace as TT

import grpc

from utils import square_spiral


# HACK: auto-generated proto files do relative imports. Will this hack cause a conflict with src/main ??
# if "PATH" not in os.environ: 
#     os.environ["PATH"] = "clients/python"
# elif "clients/python" not in os.environ["PATH"]:
#     os.environ["PATH"] += ":clients/python"
# print(os.environ["PATH"])
# os.chdir("clients/python")

# HACK: Replace relative import
with open("clients/python/src/main/proto/minecraft_pb2_grpc.py", "r") as f:
    contents = f.read()
contents = contents.replace("from src.main.proto", "from clients.python.src.main.proto")
# Write
with open("clients/python/src/main/proto/minecraft_pb2_grpc.py", "w") as f:
    f.write(contents)

import clients
import clients.python.src.main.proto.minecraft_pb2_grpc
from clients.python.src.main.proto.minecraft_pb2 import *

def mc_render(preds, labels, start_idx, cfg, name="train"):
    # input("Make sure you are running the server from `server_test`, or you may ruin the world from which we collect " 
    # "data (in which case, delete the `server/world` folder and re-launch the data-gen server). Press enter to continue.")
    channel = grpc.insecure_channel('localhost:5001')
    client = clients.python.src.main.proto.minecraft_pb2_grpc.MinecraftServiceStub(channel)

    client.initDataGen(Point(x=0, y=0, z=0))  # dummy point variable

    if start_idx == 0:
        clear(client)
        # loc = client.setLoc(Point(x=-5, y=0, z=0))  # dummy point variable
        # y0 = loc.y

    y0 = 4
    xi, yi, zi = preds[0].shape[1:]
    xj = 2 * xi + 9
    zj = zi + 8

    if name == "train":
        border_block = PURPUR_BLOCK
    else:
        border_block = REDSTONE_BLOCK

    border_idxs = [((0, i), (xi + 1, i), (xi * 2 + 2, i)) for i in range(zi + 2)] + [((i, 0), (i, zi + 1)) for i in range(1, xi * 2 + 2)]
    # Unzip into shallow list
    border_idxs = [item for sublist in border_idxs for item in sublist]
    border_blocks = [((xi, y0, zi), border_block, NORTH) for xi, zi in border_idxs]

    for i in range(len(preds)):
        j = start_idx + i
        x, z = square_spiral(j)
        x = x * xj
        z = z * zj
        pred, label = preds[i], labels[i]
        pred = pred.argmax(0)
        label = label.argmax(0)
        # client.setLoc(Point(x=x, y=y0, z=z))
        print(f"Building at ({x}, {y0}, {z}), j={j}")
        render_block_arr(client, (x, y0, z), pred)
        render_block_arr(client, (x + xi + 1, y0, z), label)
        render_blocks(client, (x - 1, y0, z - 1 ), border_blocks)

def render_blocks(client, orig, blocks):
    x0, y0, z0 = orig
    block_lst = []
    for (x, y, z), block_type, orientation in blocks:
        x += x0
        z += z0
        block_lst.append(Block(position=Point(x=x, y=y,  z=z), type=block_type, orientation=orientation))
    client.spawnBlocks(Blocks(blocks=block_lst))

def render_block_arr(client, orig, blocks):
    # Only operate on a handful of blocks or less at once
    x0, y0, z0 = orig
    block_lst = []
    for k in range(len(blocks)):
        for j in range(len(blocks[k])):
            for i in range(len(blocks[k][j])):
                block = int((blocks[k][j][i]))
                block_lst.append(Block(position=Point(x=x0+i, y=y0+j,  z=z0+k),
                                    type=block, orientation=NORTH))
    # client.spawnBlocks(Blocks(blocks=block_lst))
    n_blocks = 10000
    for i in range(0, len(block_lst), n_blocks):
        client.spawnBlocks(Blocks(blocks=block_lst[i:i+n_blocks]))
        # print("Spawned {} blocks".format(i + n_blocks))


def clear(client, w=1000):
    # clear(20,20)
    client.fillCube(FillCubeRequest(
        cube=Cube(
            min=Point(x=-w//2, y=3, z=-w//2),
            max=Point(x=w//2, y=3, z=w//2)
        ),
        type=GRASS
    ))
    client.fillCube(FillCubeRequest(
        cube=Cube(
            min=Point(x=-w//2, y=4, z=-w//2),
            max=Point(x=w//2, y=200, z=w//2)
        ),
        type=AIR
    ))
