import gym
import sys
import numpy as np
from gym.spaces import Box, Discrete
from grow.utils.fitness import get_height_from_floor
from grow.entities.growth_function import GrowthFunction
from grow.utils.minecraft import MinecraftAPI
from grow.utils.observations import get_voxel_material_proportions


"""Minecraft Environment."""


class MinecraftEnvironment(gym.Env):
    def __init__(self, config):
        self.empty_material = config["empty_material"]
        self.observing_materials = config["observing_materials"]

        self.growth_function = GrowthFunction(
            materials=config["materials"],
            max_voxels=config["max_voxels"],
            search_radius=config["search_radius"],
            axiom_material=config["axiom_material"],
            num_timestep_features=config["num_timestep_features"],
            max_steps=config["max_steps"],
            empty_material=self.empty_material,
        )

        self.search_radius = config["search_radius"]
        self.max_steps = config["max_steps"]
        self.action_space = Discrete(len(self.growth_function.configuration_map))
        self.num_proportion_features = len(self.observing_materials) * 6
        self.observation_space = Box(
            np.array([0 for x in range(self.num_proportion_features + 3)]),
            np.array([1 for x in range(self.num_proportion_features + 3)]),
        )
        self.last_features = (
            np.array([0 for x in range(self.num_proportion_features + 3)]),
        )
        self.reward_range = (0, float("inf"))
        self.reward_type = config["reward_type"]
        self.reward_interval = config["reward_interval"]

        # The grow location is based on worker and vector indices.
        sigma = self.growth_function.max_length
        x_offset = int(
            np.floor(
                np.random.normal(
                    config.worker_index * self.growth_function.max_length * 3, sigma
                )
            )
        )
        y_offset = int(
            np.floor(
                np.random.normal(
                    config.vector_index * self.growth_function.max_length * 3, sigma
                )
            )
        )
        self.mc = MinecraftAPI(
            self.max_steps,
            self.growth_function.max_length,
            x_offset=x_offset,
            z_offset=y_offset,
        )

        ###
        # Ensure the axiom coordinate is one block above the floor.
        #
        # Having found the y_offset during self.mc initialization,
        # we must now account for the axiom coordinate being in the
        # center of the growth function's tensor so that it starts
        # one block above the floor.
        self.mc.y_offset -= self.growth_function.max_steps + 1
        ###

        # Save the landscape before growth.
        self.initial_state = self.get_grow_area_tensor()

    def reset(self):
        self.growth_function.reset()
        self.mc.write_tensor(self.growth_function.X, skip=self.empty_material)
        self.previous_reward = 0
        self.previous_height = 1
        return self.get_representation()

    def get_grow_area_tensor(self):
        """Get the tensor representation of the entire grow area.

        The grow area is defined as the maximum extent the growth
        function could reach in any direction given the number of
        growth steps.

        """

        X = self.mc.read_tensor(
            0,
            self.growth_function.max_length,
            0,
            self.growth_function.max_length,
            0,
            self.growth_function.max_length,
        )
        return X

    def update_growth_function_tensor(self):
        """Update the state of the creature.

        Remove any blocks that have dissapeared from the creature.
        This occurs when sand falls, wood burns, etc.

        """

        X = self.get_grow_area_tensor()
        M = self.growth_function.X != X
        self.growth_function.X[M] = self.empty_material

    def get_representation(self):
        current_voxel = self.growth_function.get_next_building_voxel()
        x, y, z = current_voxel.x, current_voxel.y, current_voxel.z
        X = self.mc.read_tensor(
            x - self.search_radius,
            x + self.search_radius,
            y - self.search_radius,
            y + self.search_radius,
            z - self.search_radius,
            z + self.search_radius,
        )
        material_proportions = get_voxel_material_proportions(
            X, x, y, z, self.observing_materials
        )
        relative_locations = [
            x / self.growth_function.max_length,
            y / self.growth_function.max_length,
            z / self.growth_function.max_length,
        ]
        features = np.array(material_proportions + relative_locations)
        return features

    def step(self, action):
        self.update_growth_function_tensor()
        self.growth_function.step(action)
        self.mc.write_tensor(self.growth_function.X, skip=self.empty_material)

        if (
            0 < self.growth_function.steps
            and self.growth_function.steps % self.reward_interval == 0
        ):
            reward = self.get_reward(self.growth_function.X)
            self.previous_reward = reward

        done = (not self.growth_function.building()) or (
            self.growth_function.steps == self.max_steps
        )

        if done:
            self.mc.write_tensor(self.initial_state, skip=None)
            features = self.last_features
        else:
            features = self.get_representation()
            self.last_features = features

        return features, self.previous_reward, done, {}

    def get_reward(self, X):
        if self.reward_type == "max_y":
            height = get_height_from_floor(
                X, self.growth_function.building_materials, self.max_steps + 1
            )
            reward = height - self.previous_height
            self.previous_height = height
        else:
            raise Exception("Unknown reward type: {self.reward}")
        return reward
