class Voxel:
    """A voxel of material type, build level and connections.

    IE a graph representation of a voxel.

    """

    def __init__(self, material, x=None, y=None, z=None):
        self.material = material
        self.negative_x = None
        self.positive_x = None
        self.positive_z = None
        self.negative_z = None
        self.positive_y = None
        self.negative_y = None
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return f"Voxel at ({self.x}, {self.y}, {self.z}) with material: {self.material}"

    def __key(self):
        return (
            self.material,
            self.x,
            self.y,
            self.z,
        )

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, Voxel):
            return self.__key() == other.__key()
        return NotImplemented
