import grpc

import minecraft_pb2_grpc
from minecraft_pb2 import *

channel = grpc.insecure_channel('localhost:5001')
client = minecraft_pb2_grpc.MinecraftServiceStub(channel)

client.fillCube(FillCubeRequest(  # Clear a 20x10x20 working area
    cube=Cube(
        min=Point(x=-10, y=4, z=-10),
        max=Point(x=10, y=14, z=10)
    ),
    type=AIR
))

client.spawnBlocks(Blocks(blocks=
    [Block(position=Point(x=1, y=x, z=1), type=STONE, orientation=NORTH) for x in range(4, 10, 2)] +
    [Block(position=Point(x=1, y=x, z=1), type=FIRE, orientation=NORTH) for x in range(5, 11, 2)] +
    [Block(position=Point(x=x, y=6, z=1), type=STONE, orientation=NORTH) for x in range(2, 11, 2)] +
    [Block(position=Point(x=x, y=7, z=1), type=STONE, orientation=NORTH) for x in range(2, 11, 2)]
))


blocks = client.readCube(Cube(
    min=Point(x=-2, y=-2, z=-2),
    max=Point(x=4, y=2, z=2)
))

print(blocks.blocks)
