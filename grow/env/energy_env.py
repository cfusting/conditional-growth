import gym
from collections import deque
import numpy as np
from gym.spaces import Box, Discrete
from grow.utils.fitness import (
    max_z,
    table,
    max_volume,
    max_surface_area,
    twist,
    convex_hull_volume,
)
from grow.utils.plotting import plot_voxels
from grow.entities.growth_function import GrowthFunction


"""A 3D grid environment in which creatures iteratively grow."""


class EnergyGrowthEnvironment(gym.Env):

    metadata = {"render.modes": ["rgb_array"]}

    (
        empty,
        storage,
        consumption,
        photosynthetic,
        sense,
        motor,
    ) = (0, 1, 2, 3, 4, 5)
    materials_array = [empty, storage, consumption, photosynthetic, sense, motor]

    def __init__(self, config):
        self.genome = GrowthFunction(
            materials=(
                EnergyGrowthEnvironment.empty,
                EnergyGrowthEnvironment.storage,
                EnergyGrowthEnvironment.consumption,
                EnergyGrowthEnvironment.photosynthetic,
                EnergyGrowthEnvironment.sense,
                EnergyGrowthEnvironment.motor,
            ),
            max_voxels=config["max_voxels"],
            search_radius=config["search_radius"],
            axiom_material=config["axiom_material"],
            num_timestep_features=config["num_timestep_features"],
            max_steps=config["max_steps"],
        )

        self.max_steps = config["max_steps"]
        self.action_space = Discrete(len(self.genome.configuration_map))
        self.observation_space = Box(
            np.array([0 for x in range(self.genome.num_features)] + [-1, -1, -1]),
            np.array([1 for x in range(self.genome.num_features)] + [1, 1, 1]),
        )
        self.reward_range = (0, float("inf"))
        self.reward_type = config["reward_type"]
        self.reward_interval = config["reward_interval"]

        self.voxel_energy = {
            # Material type: (energy in, energy out) per timestep.
            EnergyGrowthEnvironment.empty: (
                0,
                0,
            ),  # Empty space (for indexing compatibility)
            EnergyGrowthEnvironment.storage: (0, 1),  # Storage
            EnergyGrowthEnvironment.consumption: (10, 4),  # Consumption
            EnergyGrowthEnvironment.photosynthetic: (3, 2),  # Photosynthetic
            EnergyGrowthEnvironment.sense: (0, 2),  # Sense
            EnergyGrowthEnvironment.motor: (0, 4),  # Motor
        }
        self.creature_energy = [10]
        self.num_materials = len(self.genome.materials)
        self.num_faces = 6

    def move_creature(self, X, direction, magnitude):
        if direction == "negative_x":

    def add_creature_to_simulation(self):
        pass

    def normalize_creature_to_floor(self, X):
        """Set the minimum z coordinate to zero for collision detection.

        """
        min_z = np.argwhere(X > 0).min(axis=2)
        return X - min_z

    def calculate_energy_for_creature(self, axiom):
        """Gets the energy in / out of a creature.

        Storage voxels increase the total energy a creature can store.
        Consumptions voxels rapidly take energy in when one or more of their faces in contact with another creature.
        Photosynthetic voxels take in energy when their +z face is unobstructed.
        Sense voxels cause a creature to move (mostly) deterministically toward the closest creature they sense.
        Motor voxels provide movement power if they are in contact with the -z floor.

        """

        def get_material_concentrations_for_faces(start):
            i = 0
            x = np.zeros(self.num_faces)
            material = start.material
            nodes = deque([start])
            while len(nodes) > 0:
                i += 1
                node = nodes.pop()
                visited.add(node)

                # Search for nodes with the same material and indicate the directions
                # on which open faces exist.
                if node.negative_x:
                    if node.negative_x.material == material:
                        nodes.appendleft(node.negative_x)
                else:
                    x[0] = 1
                if node.positive_x:
                    if node.positive_x.material == material:
                        nodes.appendleft(node.positive_x)
                else:
                    x[1] = 1
                if node.negative_y:
                    if node.negative_y.material == material:
                        nodes.appendleft(node.negative_y)
                else:
                    x[2] = 1
                if node.positive_y:
                    if node.positive_y.material == material:
                        nodes.appendleft(node.positive_y)
                else:
                    x[3] = 1
                if node.negative_z:
                    if node.negative_z.material == material:
                        nodes.appendleft(node.negative_z)
                else:
                    x[4] = 1
                if node.positive_z:
                    if node.positive_z.material == material:
                        nodes.appendleft(node.positive_z)
                else:
                    x[5] = 1

                # Edge case when the material does not depend on exposed faces.
                if material == 0:
                    return np.exp(i) * np.ones(6), i

            return np.exp(i) * x, i

        # Number of materials x number of faces.
        material_concentrations = np.zeros((self.num_materials, self.num_faces))
        materials_counts = np.zeros(self.num_materials)
        # Breadth first traversal to gather material types and exposed faces.
        visited = set()
        remaining = deque([axiom])
        while len(remaining) > 0:
            current = remaining.pop()

            if current not in visited and current.material != 0:
                face_concentrations, i = get_material_concentrations_for_faces(current)
                material_concentrations[current.material, :] += face_concentrations
                materials_counts[current.material] += i

            if current.negative_x:
                remaining.appendleft(current.negative_x)
            if current.positive_x:
                remaining.appendleft(current.positive_x)
            if current.negative_y:
                remaining.appendleft(current.negative_y)
            if current.positive_y:
                remaining.appendleft(current.positive_y)
            if current.negative_z:
                remaining.appendleft(current.negative_z)
            if current.positive_z:
                remaining.appendleft(current.positive_z)

        return material_concentrations, materials_counts

    def apply_simulation_rules_to_creature(self):
        # Material types x faces.
        energy_in_by_face, materials_counts = self.calculate_energy_for_creature()
        energy_out = 0
        energy_in = 0

        # Energy out is just the product if oer voxel use and voxel count by type.
        for material_type in EnergyGrowthEnvironment.materials_array:
            energy_out += (
                materials_counts[material_type]
                * self.voxel_energy[material_type][1]
            )

        # Energy in depends on concentration of voxels.
        # Storage does not depend on face, just pick index 0.

        # Consumption.
        faces_in_contact = self.get_faces_in_contact()
        for face in faces_in_contact:
        energy_in += energy_in_by_face[EnergyGrowthEnvironment.consumption, face] * self.voxel_energy[EnergyGrowthEnvironment.consumption][0]

        # Photosynthetic applies only to positive z.
        energy_in += energy_in_by_face[EnergyGrowthEnvironment.photosynthetic, 5]

    def reset(self):
        self.genome.reset()
        self.previous_reward = 0
        return self.get_representation()

    def get_representation(self):
        x = np.array(self.genome.get_local_voxel_representation())
        return x

    def step(self, action):
        self.genome.step(action)
        if 0 < self.genome.steps and self.genome.steps % self.reward_interval == 0:

            reward = self.get_reward(self.genome.positions, self.genome.X)
            self.previous_reward = reward

        done = (not self.genome.building()) or (self.genome.steps == self.max_steps)
        return self.get_representation(), self.previous_reward, done, {}

    def get_reward(self, final_positions, X):
        if self.reward_type == "max_z":
            reward = max_z(final_positions)
        elif self.reward_type == "table":
            reward = table(final_positions)
        elif self.reward_type == "max_volume":
            reward = max_volume(X)
        elif self.reward_type == "max_surface_area":
            reward = max_surface_area(X)
        elif self.reward_type == "tree":
            reward = twist(self.genome.axiom)
        elif self.reward_type == "convex_hull_volume":
            reward = convex_hull_volume(self.genome.axiom)
        else:
            raise Exception("Unknown reward type: {self.reward}")
        return reward

    def render(self, mode="rgb_array"):
        if mode == "rgb_array":
            # Most unfortunetly this calls vtk which has a memory leak.
            # Best to only call during a short evaluation.
            img = plot_voxels(self.genome.positions, self.genome.values)
            return img
