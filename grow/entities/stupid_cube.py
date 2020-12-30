import numpy as np


class StupidCube:

    def __init__(self, length=3, width=3, height=3):
        self.length = length
        self.width = width
        self.height = height
        self.reset()

    def reset(self):
        self.creature = np.zeros((self.length, self.width, self.height))
        self.current_step = 0
        
    def step(self, voxels_to_add):
        self.current_step += 1
        for v in voxels_to_add:
            self.creature[v] = 1
