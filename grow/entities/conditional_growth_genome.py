from grow.entities.voxel import Voxel
from time import time
import numpy as np
from collections import deque
import itertools


class ConditionalGrowthGenome:
    """Generate patterns given patterns.

    Use the local context to decide what pattern to generate next.
    IE the the configuration of voxels added depend on the proportion
    of the voxel types.

    """

    def __init__(
        self,
        materials=(0, 1, 2),
        directions=(
            "negative_x",
            "positive_x",
            "negative_y",
            "positive_y",
            "negative_z",
            "positive_z",
        ),
        max_voxels=5,
        search_radius=1,
        axiom_material=1,
    ):
        self.materials = materials
        self.directions = directions
        self.max_voxels = max_voxels
        self.search_radius = search_radius
        self.axiom_material = axiom_material

        self.initialize_configurations()
        print(f"Found {len(self.configuration_map)} possible voxel configurations.")
        self.reset()

    def reset(self):
        self.num_voxels = 1
        self.next_voxel_id = 0
        self.axiom = self.get_new_voxel(self.axiom_material)
        self.axiom.level = 0
        self.body = deque([self.axiom])
        self.id = str(time())
        self.max_level = 0
        self.steps = 0

    def building(self):
        """Returns True if there is more to build."""

        return len(self.body) > 0

    def get_new_voxel(self, material):
        v = Voxel(material, self.next_voxel_id)
        self.next_voxel_id += 1
        self.num_voxels += 1
        return v

    def attach_voxels(self, configuration, current_voxel):
        """Attach a configuration of voxels to the current voxel.

        Attach a configuration of voxels (IE a
        combination of voxels of a given material and placements)
        to to the current voxel.

        """

        if configuration is None:
            return []

        voxels = []
        for c in configuration:
            material = c[0]
            direction = c[1]
            voxel = self.get_new_voxel(material)

            def increment_level(voxel, current_voxel):
                voxel.level = current_voxel.level + 1
                if voxel.level > self.max_level:
                    self.max_level = voxel.level

            if direction == "negative_x" and not current_voxel.negative_x:
                current_voxel.negative_x = voxel
                voxel.positive_x = current_voxel
                increment_level(voxel, current_voxel)
                voxels.append(voxel)
            if direction == "positive_x" and not current_voxel.positive_x:
                current_voxel.positive_x = voxel
                voxel.negative_x = current_voxel
                increment_level(voxel, current_voxel)
                voxels.append(voxel)
            if direction == "negative_y" and not current_voxel.negative_y:
                current_voxel.negative_y = voxel
                voxel.positive_y = current_voxel
                increment_level(voxel, current_voxel)
                voxels.append(voxel)
            if direction == "positive_y" and not current_voxel.positive_y:
                current_voxel.positive_y = voxel
                voxel.negative_y = current_voxel
                increment_level(voxel, current_voxel)
                voxels.append(voxel)
            if direction == "negative_z" and not current_voxel.negative_z:
                current_voxel.negative_z = voxel
                voxel.positive_z = current_voxel
                increment_level(voxel, current_voxel)
                voxels.append(voxel)
            if direction == "positive_z" and not current_voxel.positive_z:
                current_voxel.positive_z = voxel
                voxel.negative_z = current_voxel
                increment_level(voxel, current_voxel)
                voxels.append(voxel)

        return voxels

    def step(self, configuration_index):
        """Add one configuration. Do not use with ``expand()``."""
        voxel = self.body.pop()
        configuration = self.configuration_map[configuration_index]
        voxels = self.attach_voxels(configuration, voxel)
        self.body.extendleft(voxels)
        self.steps += 1

    def get_local_voxel_representation(self):
        """Get a representation of voxels nearby the next voxel in queue."""
        if self.building():
            return self.get_function_input(self.body[-1])
        else:
            # Maximum number of directions for each zero.
            # Covers the edge case in which no observations are available.
            return [0 for _ in range(len(self.materials) * 6)]

    def get_function_input(self, voxel):
        """Get the material proportions of nearby voxels"""

        initial_level = voxel.level
        total_voxels = 0

        material_proportions = {
            "negative_x": {},
            "positive_x": {},
            "negative_y": {},
            "positive_y": {},
            "negative_z": {},
            "positive_z": {},
        }

        for k in material_proportions:
            for m in self.materials:
                material_proportions[k][m] = 0

        # Initialize the start directions from the voxel.
        if voxel.negative_x:
            voxel.negative_x.start_direction = "negative_x"
        if voxel.positive_x:
            voxel.positive_x.start_direction = "positive_x"
        if voxel.negative_y:
            voxel.negative_y.start_direction = "negative_y"
        if voxel.positive_y:
            voxel.positive_y.start_direction = "positive_y"
        if voxel.negative_z:
            voxel.negative_z.start_direction = "negative_z"
        if voxel.positive_z:
            voxel.positive_z.start_direction = "positive_z"

        searched_voxel_ids = set()
        search_voxels = deque([voxel])
        while len(search_voxels) > 0:
            voxel = search_voxels.pop()
            searched_voxel_ids.add(voxel.id)

            if np.abs(initial_level - voxel.level) > self.search_radius:
                break

            if hasattr(voxel, "start_direction"):
                material_proportions[voxel.start_direction][
                    voxel.material
                ] += 1

            if voxel.negative_x and voxel.negative_x.id not in searched_voxel_ids:
                if not hasattr(voxel.negative_x, "start_direction"):
                    voxel.negative_x.start_direction = voxel.start_direction
                search_voxels.appendleft(voxel.negative_x)
            if voxel.positive_x and voxel.positive_x.id not in searched_voxel_ids:
                if not hasattr(voxel.positive_x, "start_direction"):
                    voxel.positive_x.start_direction = voxel.start_direction
                search_voxels.appendleft(voxel.positive_x)
            if voxel.negative_y and voxel.negative_y.id not in searched_voxel_ids:
                if not hasattr(voxel.negative_y, "start_direction"):
                    voxel.negative_y.start_direction = voxel.start_direction
                search_voxels.appendleft(voxel.negative_y)
            if voxel.positive_y and voxel.positive_y.id not in searched_voxel_ids:
                if not hasattr(voxel.positive_y, "start_direction"):
                    voxel.positive_y.start_direction = voxel.start_direction
                search_voxels.appendleft(voxel.positive_y)
            if voxel.negative_z and voxel.negative_z.id not in searched_voxel_ids:
                if not hasattr(voxel.negative_z, "start_direction"):
                    voxel.negative_z.start_direction = voxel.start_direction
                search_voxels.appendleft(voxel.negative_z)
            if voxel.positive_z and voxel.positive_z.id not in searched_voxel_ids:
                if not hasattr(voxel.positive_z, "start_direction"):
                    voxel.positive_z.start_direction = voxel.start_direction
                search_voxels.appendleft(voxel.positive_z)

            total_voxels += 1

        for k in material_proportions:
            total_voxels = 0
            for m in material_proportions[k]:
                total_voxels += material_proportions[k][m]
            for m in material_proportions[k]:
                if total_voxels != 0:
                    material_proportions[k][m] /= total_voxels

        representation = []
        for k in material_proportions:
            representation.extend(list(material_proportions[k].values()))

        return representation

    def initialize_configurations(self):
        """Map every possible configuration.

        Create a map from i = 0 to n of every possible way in which
        voxels with materials could be placed on three-dimensional
        surfaces.

        """
        # Get all possible voxels.
        possible_voxels = []
        for m in self.materials[1:]:
            for d in self.directions:
                possible_voxels.append((m, d))

        self.configuration_map = {}
        self.configuration_map[0] = None
        i = 1
        for num_voxels in range(1, self.max_voxels + 1):
            for subset in itertools.combinations(possible_voxels, num_voxels):
                self.configuration_map[i] = subset
                i += 1

    def to_tensor(self):
        """Convert the graph representation of the body to a tensor.

        Fill a three-dimensional tensor with the material types of
        each voxel IE:
            X[i][j][k] = m

        """

        extent = (2 * self.max_level) + 1
        X = np.zeros((extent, extent, extent))
        middle = int(np.floor(extent / 2))
        x, y, z = middle, middle, middle

        searched_voxel_ids = set()
        to_process = deque([(x, y, z, self.axiom)])
        while len(to_process) > 0:
            x, y, z, voxel = to_process.pop()
            searched_voxel_ids.add(voxel.id)
            X[x, y, z] = voxel.material

            if voxel.negative_x and voxel.negative_x.id not in searched_voxel_ids:
                x -= 1
                to_process.appendleft((x, y, z, voxel.negative_x))
            if voxel.positive_x and voxel.positive_x.id not in searched_voxel_ids:
                x += 1
                to_process.appendleft((x, y, z, voxel.positive_x))
            if voxel.negative_y and voxel.negative_y.id not in searched_voxel_ids:
                y -= 1
                to_process.appendleft((x, y, z, voxel.negative_y))
            if voxel.positive_y and voxel.positive_y.id not in searched_voxel_ids:
                y += 1
                to_process.appendleft((x, y, z, voxel.positive_y))
            if voxel.negative_z and voxel.negative_z.id not in searched_voxel_ids:
                z -= 1
                to_process.appendleft((x, y, z, voxel.negative_z))
            if voxel.positive_z and voxel.positive_z.id not in searched_voxel_ids:
                z += 1
                to_process.appendleft((x, y, z, voxel.positive_z))
        return X
