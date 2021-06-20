import gym
import numpy as np
from gym.spaces import Box, Discrete
from grow.utils.fitness import get_max_connected_y_for_tensor
from grow.entities.growth_function import GrowthFunction
from grow.utils.minecraft import MinecraftAPI
from grow.utils.minecraft_pb2 import AIR


"""Minecraft Environment."""


class MinecraftEnvironment(gym.Env):
    def __init__(self, config):
        self.growth_function = GrowthFunction(
            materials=config["materials"],
            max_voxels=config["max_voxels"],
            search_radius=config["search_radius"],
            axiom_material=config["axiom_material"],
            num_timestep_features=config["num_timestep_features"],
            max_steps=config["max_steps"],
            empty_material=AIR,
        )

        self.max_steps = config["max_steps"]
        self.action_space = Discrete(len(self.growth_function.configuration_map))
        self.observation_space = Box(
            np.array(
                [0 for x in range(self.growth_function.num_features)] + [-1, -1, -1]
            ),
            np.array([1 for x in range(self.growth_function.num_features)] + [1, 1, 1]),
        )
        self.reward_range = (-float("inf"), float("inf"))
        self.reward_type = config["reward_type"]
        self.reward_interval = config["reward_interval"]
        self.mc = MinecraftAPI(
            self.max_steps,
            x_offset=0,
            z_offset=0
        )
        self.initial_state = self.get_grow_area_tensor()

    def reset(self):
        self.mc.write_tensor(self.initial_state)
        self.growth_function.reset()
        self.previous_reward = 0
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
        M = self.growth_function.X == X
        self.growth_function.X = self.growth_function.X[~M]

    def get_representation(self):
        x = np.array(self.growth_function.get_local_voxel_representation())
        return x

    def step(self, action):
        self.update_growth_function_tensor()
        self.growth_function.step(action)
        self.mc.write_tensor(self.growth_function.X)
        if (
            0 < self.growth_function.steps
            and self.growth_function.steps % self.reward_interval == 0
        ):
            reward = self.get_reward(self.growth_function.X)
            self.previous_reward = reward

        done = (not self.growth_function.building()) or (
            self.growth_function.steps == self.max_steps
        )
        return self.get_representation(), self.previous_reward, done, {}

    def get_reward(self, X):
        if self.reward_type == "max_y":
            reward = get_max_connected_y_for_tensor(X)
        else:
            raise Exception("Unknown reward type: {self.reward}")
        return reward
