from grow.utils.minecraft import MinecraftAPI
from grow.utils.minecraft_pb2 import AIR, BEDROCK, GRASS, DIRT, STONE, SAND
from grow.entities.voxel import Voxel
import numpy as np
import time


# All tests to be run on superflat.

def test_find_floor():
    mc = MinecraftAPI(4)
    X = mc.read_tensor(0, 1, 0, 1, 0, 1)
    print(X)
    assert mc.x_offset == 0
    assert mc.y_offset == 4
    assert mc.z_offset == 0
    assert np.array_equal(X, [[[5]]])


def test_write_tensor():
    mc = MinecraftAPI(0)
    X = np.array([[[STONE]]])
    mc.write_tensor(X)
    X = mc.read_tensor(0, 1, 0, 1, 0, 1)
    assert np.array_equal(X, [[[STONE]]])
    # Reset what we did.
    X = np.array([[[AIR]]])
    mc.write_tensor(X, skip=None)


def test_sand_falling():
    mc = MinecraftAPI(0)
    X = np.full((3, 3, 3), AIR)
    X[0, 2, 0] = SAND
    mc.write_tensor(X)
    time.sleep(.1)
    X = mc.read_tensor(0, 3, 0, 3, 0, 3)
    X = np.full((3, 3, 3), AIR)
    mc.write_tensor(X, skip=None)

