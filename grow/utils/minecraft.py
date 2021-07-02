import grpc
import grow.utils.minecraft_pb2_grpc as mc
from grow.utils.minecraft_pb2 import *
import numpy as np


class MinecraftAPI:

    min_indices = (-30000000, 0, -30000000)
    max_indices = (29999999, 255, 29999999)

    def __init__(
        self,
        max_steps,
        max_length,
        address="localhost:5001",
        x_offset=0,
        z_offset=0,
    ):
        channel = grpc.insecure_channel(address)
        self.client = mc.MinecraftServiceStub(channel)
        self.x_offset = x_offset
        self.z_offset = z_offset
        self.y_offset = 0
        self.max_steps = max_steps
        self.max_length = max_length
        self.find_the_floor()
        self.y_offset = self.Z[self.max_steps + 1, self.max_steps + 1]
        print(f"Offsets: ({self.x_offset}, {self.y_offset}, {self.z_offset})")

    def to_global_coordinates(self, x, y, z):
        """Inputs in local coordinates."""
        return x + self.x_offset, y + self.y_offset, z + self.z_offset

    def to_local_coordinates(self, x, y, z):
        """Inputs in global coordinates."""
        return x - self.x_offset, y - self.y_offset, z - self.z_offset

    def to_hyper_local_coordinates(
        self, x, y, z, x_local_offset, y_local_offset, z_local_offset
    ):
        """Inputs in local coordinates."""
        return x - x_local_offset, y - y_local_offset, z - z_local_offset

    def blocks_to_tensor(self, blocks, x_min, x_max, y_min, y_max, z_min, z_max):
        X = np.zeros((x_max - x_min, y_max - y_min, z_max - z_min))

        for block in blocks.blocks:
            x, y, z = self.to_local_coordinates(
                block.position.x, block.position.y, block.position.z
            )
            x, y, z = self.to_hyper_local_coordinates(
                block.position.x,
                block.position.y,
                block.position.z,
                x_min,
                y_min,
                z_min,
            )
            X[x, y, z] = block.type
        return X

    def read_tensor(self, x_min, x_max, y_min, y_max, z_min, z_max):
        x_min, y_min, z_min = self.to_global_coordinates(x_min, y_min, z_min)
        x_max, y_max, z_max = self.to_global_coordinates(x_max, y_max, z_max)
        blocks = self.client.readCube(
            Cube(
                min=Point(
                    x=x_min,
                    y=y_min,
                    z=z_min,
                ),
                # API is inclusive on both ends [min, max].
                # We follow numpy [min, max) and thus subtract 1.
                max=Point(
                    x=x_max - 1,
                    y=y_max - 1,
                    z=z_max - 1,
                ),
            )
        )
        return self.blocks_to_tensor(blocks, x_min, x_max, y_min, y_max, z_min, z_max)

    def tensor_to_blocks(self, X, skip=AIR, only=None):
        blocks = []
        it = np.nditer(X, flags=["multi_index"])
        for v in it:
            x, y, z = it.multi_index
            x, y, z = self.to_global_coordinates(x, y, z)
            if (
                (
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

                if (only is None and v != skip) or (only is not None and v in only):
                    blocks.append(
                        Block(
                            position=Point(
                                x=x,
                                y=y,
                                z=z,
                            ),
                            type=int(v),
                        )
                    )

        return Blocks(blocks=blocks)

    def write_tensor(self, X, skip=AIR, only=None):
        blocks = self.tensor_to_blocks(X, skip, only)
        self.client.spawnBlocks(blocks)

    def find_the_floor(self):
        X = self.read_tensor(
            0,
            self.max_length,
            MinecraftAPI.min_indices[1],
            MinecraftAPI.max_indices[1],
            0,
            self.max_length,
        )

        self.Z = np.zeros((self.max_length, self.max_length), dtype=np.int)
        for y in reversed(range(X.shape[1])):
            M = X[:, y, :].reshape((self.max_length, self.max_length)) != AIR
            self.Z[np.bitwise_and(self.Z == 0, M)] = y
