from grow.entities.voxel import Voxel
import numpy as np
from collections import deque
import itertools


class GrowthFunction:
    """Generate patterns given patterns.

    Use the local context to decide what pattern to generate next.
    IE the the configuration of voxels added depend on the proportion
    of the voxel types.

    """

    def __init__(
        self,
        materials=(0, 1, 2),
        max_voxels=6,
        search_radius=3,
        axiom_material=1,
        num_timestep_features=1,
        max_steps=10,
        empty_material=0,
        initial_state=None,
        max_length=None,
    ):
        directions = (
            "negative_x",
            "positive_x",
            "negative_y",
            "positive_y",
            "negative_z",
            "positive_z",
        )
        self.materials = materials
        self.empty_material = empty_material
        self.initial_state = initial_state
        self.directions = directions
        self.max_voxels = max_voxels
        self.search_radius = search_radius
        self.axiom_material = axiom_material
        self.max_steps = max_steps
        self.max_length = max_length
        # n-dim creatures next.
        self.num_coordinates = 3

        self.num_features = len(materials) * len(directions) * num_timestep_features

        self.axiom_coordinate = self.max_steps + 1

        self.initialize_configurations()
        self.reset()

    def initialize_axiom(self):
        self.body = deque([])
        self.get_new_voxel(
            self.axiom_material,
            self.axiom_coordinate,
            self.axiom_coordinate,
            self.axiom_coordinate,
            self.initial_state,
            clobber=True,
        )

    def reset(self):
        self.historic_representation = [
            0 for _ in range(self.num_features + self.num_coordinates)
        ]
        self.steps = 0
        self.X = np.full(
            (self.max_length, self.max_length, self.max_length), self.empty_material
        )
        self.initialize_axiom()

    def building(self):
        """Returns True if there is more to build."""

        return len(self.body) > 0

    def get_new_voxel(self, material, x, y, z, E, clobber=False):
        v = Voxel(material, x, y, z)
        if E[x, y, z] == self.empty_material or clobber:
            self.X[x, y, z] = material
            self.body.appendleft(v)

    def attach_voxels(self, configuration, current_voxel, E):
        """Attach a configuration of voxels to the current voxel.

        Attach a configuration of voxels (IE a
        combination of voxels of a given material and placements)
        to to the current voxel.

        """

        if configuration is None:
            return []

        for c in configuration:
            material = c[0]
            direction = c[1]

            if direction == "negative_x":
                self.get_new_voxel(
                    material, current_voxel.x - 1, current_voxel.y, current_voxel.z, E
                )
            if direction == "positive_x":
                self.get_new_voxel(
                    material, current_voxel.x + 1, current_voxel.y, current_voxel.z, E
                )
            if direction == "negative_y":
                self.get_new_voxel(
                    material, current_voxel.x, current_voxel.y - 1, current_voxel.z, E
                )
            if direction == "positive_y":
                self.get_new_voxel(
                    material, current_voxel.x, current_voxel.y + 1, current_voxel.z, E
                )
            if direction == "negative_z":
                self.get_new_voxel(
                    material, current_voxel.x, current_voxel.y, current_voxel.z - 1, E
                )
            if direction == "positive_z":
                self.get_new_voxel(
                    material, current_voxel.x, current_voxel.y, current_voxel.z + 1, E
                )

    def step(self, configuration_index, environment_tensor=None):
        """Add one configuration. Do not use with ``expand()``."""

        assert self.building(), "Step called without anything left to build."

        voxel = self.body.pop()
        configuration = self.configuration_map[configuration_index]
        self.attach_voxels(configuration, voxel, environment_tensor)
        self.steps += 1

    def get_local_voxel_representation(self):
        """Get a representation of voxels nearby the next voxel in queue."""

        if self.building():
            local_representation = self.get_function_input(self.body[-1])
        else:
            # Maximum number of directions for each zero.
            # Covers the edge case in which no observations are available.
            local_representation = [
                0
                for _ in range(
                    len(self.materials) * len(self.directions) + self.num_coordinates
                )
            ]

        self.historic_representation = (
            local_representation + self.historic_representation
        )

        return self.historic_representation[: self.num_features + self.num_coordinates]

    def get_next_building_voxel(self):
        return self.body[-1]

    def get_function_input(self, voxel):
        proportions = []  # Ordered by -x, +x, -y, ...

        # x axis positive.
        v = min(voxel.x + self.search_radius + 1, self.max_length)
        material_totals = []
        for m in self.materials:
            material_totals.append(np.sum(self.X[voxel.x : v, :, :] == m))
        for i in range(len(self.materials)):
            proportions.append(material_totals[i] / np.sum(material_totals))

        # x axis negative.
        v = max(voxel.x - self.search_radius, 0)
        u = min(voxel.x + 1, self.max_length)
        material_totals = []
        for m in self.materials:
            material_totals.append(np.sum(self.X[v:u, :, :] == m))
        for i in range(len(self.materials)):
            proportions.append(material_totals[i] / np.sum(material_totals))

        # y axis positive.
        v = min(voxel.y + self.search_radius + 1, self.max_length)
        material_totals = []
        for m in self.materials:
            material_totals.append(np.sum(self.X[:, voxel.y : v, :] == m))
        for i in range(len(self.materials)):
            proportions.append(material_totals[i] / np.sum(material_totals))

        # x axis negative.
        v = max(voxel.y - self.search_radius, 0)
        u = min(voxel.y + 1, self.max_length)
        material_totals = []
        for m in self.materials:
            material_totals.append(np.sum(self.X[:, v:u, :] == m))
        for i in range(len(self.materials)):
            proportions.append(material_totals[i] / np.sum(material_totals))

        # z axis positive.
        v = min(voxel.z + self.search_radius + 1, self.max_length)
        material_totals = []
        for m in self.materials:
            material_totals.append(np.sum(self.X[:, :, voxel.z : v] == m))
        for i in range(len(self.materials)):
            proportions.append(material_totals[i] / np.sum(material_totals))

        # z axis negative.
        v = max(voxel.z - self.search_radius, 0)
        u = min(voxel.z + 1, self.max_length)
        material_totals = []
        for m in self.materials:
            material_totals.append(np.sum(self.X[:, :, v:u] == m))
        for i in range(len(self.materials)):
            proportions.append(material_totals[i] / np.sum(material_totals))

        return proportions + [
            voxel.x / self.max_length,
            voxel.y / self.max_length,
            voxel.z / self.max_length,
        ]

    def initialize_configurations(self):
        """Map every possible configuration.

        Create a map from i = 0 to n of every possible way in which
        voxels with materials could be placed on three-dimensional
        surfaces.

        """

        self.building_materials = [
            x for x in self.materials if x != self.empty_material
        ]

        # Get all possible voxels.
        possible_voxels = []
        for m in self.building_materials:
            for d in self.directions:
                possible_voxels.append((m, d))

        self.configuration_map = {}
        self.configuration_map[0] = None
        i = 1
        for num_voxels in range(1, self.max_voxels + 1):
            for subset in itertools.combinations(possible_voxels, num_voxels):
                self.configuration_map[i] = subset
                i += 1

        print(f"Found {len(self.configuration_map)} possible voxel configurations.")

    def __len__(self):
        return np.sum(self.X != 0)

    def atrophy_disconnected_voxels(self):
        raise NotImplementedError
