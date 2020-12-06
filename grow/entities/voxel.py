class Voxel:
    """A voxel of material type, build level and connections.

    IE a graph representation of a voxel.

    """

    def __init__(self, material):
        self.material = material
        self.negative_x = None
        self.positive_x = None
        self.positive_z = None
        self.negative_z = None
        self.positive_y = None
        self.negative_y = None
        self.level = None
        self.searched = False
        self.processed = False

    def __str__(self):
        return f"Voxel with material: {self.material} and level: {self.level}"


