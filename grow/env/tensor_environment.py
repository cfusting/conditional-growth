import gym
import numpy as np
from gym.spaces import Box, Discrete
from grow.utils.fitness import (
    max_z,
    table,
    max_volume,
    max_surface_area,
    twist,
    get_convex_hull_volume,
    max_hull_volume_min_density,
)
from grow.utils.plotting import plot_voxels
from grow.entities.growth_function import GrowthFunction


"""A 3D grid environment in which creatures iteratively grow."""


class TensorGrowthEnvironment(gym.Env):

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, config):
        self.genome = GrowthFunction(
            materials=config["materials"],
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
            reward = get_convex_hull_volume(final_positions)
        elif self.reward_type == "max_hull_volume_min_density":
            reward = max_hull_volume_min_density(final_positions)
        else:
            raise Exception("Unknown reward type: {self.reward}")
        return reward

    def render(self, mode="rgb_array"):
        if mode == "rgb_array":
            # Most unfortunetly this calls vtk which has a memory leak.
            # Best to only call during a short evaluation.
            img = plot_voxels(self.genome.positions, self.genome.values)
            return img
