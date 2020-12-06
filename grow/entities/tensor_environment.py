import numpy as np


class TensorEnvironment():
    """Randomly distribute voxel tensors.

    This class takes a list of tensors, randomly distributes
    them about a space, and generates a new tensor for simulation.

    """

    def __init__(self, tensors, num_spacing_voxels=20):
        self.num_spacing_voxels = num_spacing_voxels

    def distribute_tensors(self):

        def get_midpoint(X):
            x = int(np.floor(X.shape[0]))
            y = int(np.floor(X.shape[1]))
            z = int(np.floor(X.shape[2]))
            return (x, y, z)



