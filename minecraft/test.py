import grpc

import minecraft_pb2_grpc
from minecraft_pb2 import *

channel = grpc.insecure_channel('localhost:5001')
client = minecraft_pb2_grpc.MinecraftServiceStub(channel)

client.spawnBlocks(Blocks(blocks=[  # Spawn a flying machine
    Block(position=Point(x=-18, y=75, z=-75), type=AIR, orientation=NORTH),
    Block(position=Point(x=-18, y=74, z=-75), type=FARMLAND, orientation=NORTH),
    Block(position=Point(x=-18, y=75, z=-75), type=WHEAT, orientation=NORTH),
    Block(position=Point(x=-19, y=75, z=-75), type=WHEAT, orientation=NORTH),
    Block(position=Point(x=-17, y=74, z=-75), type=WATER, orientation=NORTH),
    Block(position=Point(x=-15, y=78, z=-75), type=FIRE, orientation=NORTH),
    Block(position=Point(x=-15, y=77, z=-75), type=LOG, orientation=NORTH),
]))


blocks = client.readCube(Cube(
    min=Point(x=-19, y=75, z=-76),
    max=Point(x=-17, y=75, z=-74)
))

print(blocks)
