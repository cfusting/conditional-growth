import grpc
import grow.utils.minecraft_pb2_grpc as mc
from grow.utils.minecraft_pb2 import *
import numpy as np


class MinecraftAPI:

    max_indices = (29999999, 255, 29999999)
    min_indices = (-30000000, 0, -30000000)

    def __init__(
        self,
        max_steps,
        address="localhost:5001",
        x_offset=0,
        z_offset=0,
    ):
        channel = grpc.insecure_channel(address)
        self.client = mc.MinecraftServiceStub(channel)
        self.x_offset = x_offset
        self.z_offset = z_offset
        self.y_offset = 0
        self.y_offset = self.find_the_floor(max_steps)
        print(f"Offsets: ({self.x_offset}, {self.y_offset}, {self.z_offset})")

    def blocks_to_tensor(self, blocks, x_length, y_length, z_length):
        X = np.zeros((x_length, y_length, z_length))
        for block in blocks.blocks:
            x = block.position.x - self.x_offset
            y = block.position.y - self.y_offset
            z = block.position.z - self.z_offset
            X[x, y, z] = block.type
        return X

    def read_tensor(self, x_min, x_max, y_min, y_max, z_min, z_max):
        blocks = self.client.readCube(
            Cube(
                min=Point(
                    x=x_min + self.x_offset,
                    y=y_min + self.y_offset,
                    z=z_min + self.z_offset,
                ),
                # API is inclusive on both ends [min, max].
                # We follow numpy [min, max) and thus subtract 1.
                max=Point(
                    x=x_max + self.x_offset - 1,
                    y=y_max + self.y_offset - 1,
                    z=z_max + self.z_offset - 1,
                ),
            )
        )
        return self.blocks_to_tensor(
            blocks, x_max - x_min, y_max - y_min, z_max - z_min
        )

    def tensor_to_blocks(self, X, skip=AIR):
        blocks = []
        it = np.nditer(X, flags=["multi_index"])
        for v in it:
            x, y, z = it.multi_index
            if (
                v != skip
                and (
                    MinecraftAPI.min_indices[0]
                    <= x + self.x_offset
                    <= MinecraftAPI.max_indices[0]
                )
                and (
                    MinecraftAPI.min_indices[1]
                    <= y + self.y_offset
                    <= MinecraftAPI.max_indices[1]
                )
                and (
                    MinecraftAPI.min_indices[2]
                    <= z + self.z_offset
                    <= MinecraftAPI.max_indices[2]
                )
            ):
                blocks.append(
                    Block(
                        position=Point(
                            x=x + self.x_offset,
                            y=y + self.y_offset,
                            z=z + self.z_offset,
                        ),
                        type=int(v),
                    )
                )

        return Blocks(blocks=blocks)

    def write_tensor(self, X, skip=AIR):
        blocks = self.tensor_to_blocks(X, skip)
        self.client.spawnBlocks(blocks)

    def find_the_floor(self, max_steps):
        x = self.x_offset + max_steps + 1
        y_min = 0
        y_max = 200
        z = self.z_offset + max_steps + 1
        X = self.read_tensor(x, x, y_min, y_max, z, z)
        X = X.flatten()
        y = y_min
        while y < y_max and y != AIR:
            y += 1
        if y == y_max:
            raise ValueError("No floor found. Exiting.")
        return y - 1
