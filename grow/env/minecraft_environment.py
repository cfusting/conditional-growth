import gym
import itertools
import numpy as np
from gym.spaces import Box, Discrete
from grow.utils.fitness import get_height_from_floor, inverse_distance_from_block_type
from grow.entities.growth_function import GrowthFunction
from grow.utils.minecraft import MinecraftAPI
from grow.utils.observations import get_voxel_material_proportions


"""Minecraft Environment.

This environment communicates with Minecraft via a read blocks / 
write blocks API. It iteratively grows a creasture based on the
world around it where each time step is a (potential) growth step.

If multiple workers and vectors are available, this environment will
use the corresponding indices to psudo-randomly place the creature in
the Minecraft environment. 

Notes:
    Minecraft indices block locations as (axis1, height, axis2)
    which throughout the API and this class is called (x, y, z).

    References in this class are in local coordinates with respect
    to the creature. The MinecraftAPI handles local to global
    coordinate transforms using the x, y, and z offsets set during
    initialization.

"""


class MinecraftEnvironment(gym.Env):
    def __init__(self, config):
        self.empty_material = config["empty_material"]
        self.reward_block_type = config["reward_block_type"]
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
        feature_zeros = np.array([0 for x in range(self.num_proportion_features + 3)])
        self.observation_space = Box(
            feature_zeros,
            np.array([1 for x in range(self.num_proportion_features + 3)]),
        )
        self.last_features = (feature_zeros,)
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
        self.set_growth_area_tensor()
        self.initial_state = self.growth_area_tensor

        self.initialize_rewards()

    def initialize_rewards(self):
        self.previous_reward = 0
        
        if self.reward_type == "y_max":
            self.previous_height = 1
        elif self.reward_type == "distance_from_blocks":
            self.previous_inverse_distance = 0

            # Uniformly at random select points to place blocks.
            # Excluding Axiom.
            possible_points = itertools.combinations(
                [x for x in range(self.growth_function.max_length)], 2
            )
            possible_points.remove(
                (
                    self.growth_function.axiom_coordinate,
                    self.growth_function.axiom_coordinate,
                )
            )
            points = np.random.choice(possible_points, size=np.random.randint(1, 3))
            for p in points:
                X = np.full_like(self.growth_function.X, self.empty_material)
                X[p[0], p[1]] = self.reward_block_type
                MinecraftAPI.write_tensor(X)
        else:
            raise Exception("Unknown reward type: {self.reward}")

    def clear_creature(self):
        X = np.full_like(self.growth_function.X, np.nan)
        for material in self.growth_function.building_materials:
            M = self.growth_function.X == material
            X = np.bitwise_or(X, M)

        X[X == 1] = self.empty_material
        self.mc.write_tensor(X, skip=None, only=[self.empty_material])

    def reset(self):
        self.clear_creature()
        self.growth_function.reset()
        self.initialize_rewards()
        return self.get_representation()

    def set_grow_area_tensor(self):
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
        self.growth_area_tensor = X

    def update_growth_function_tensor(self):
        """Update the state of the creature.

        Remove any blocks that have dissapeared from the creature.
        This occurs when sand falls, wood burns, etc.

        """

        self.set_grow_area_tensor()
        M = self.growth_function.X != self.growth_area_tensor
        self.growth_function.X[M] = self.empty_material
        # self.growth_function.atrophy_disconnected_voxels()

    def get_representation(self):
        """Get the local representation of blocks.

        Return the proportions of blocks around the block on which
        growth is about to occur. A proportion is returned for each
        of the sixes faces of the block.

        """
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
        # Ensure the growth functiona and Minecraft are in parity.
        # Take a growth step and update Minecraft.
        self.update_growth_function_tensor()
        self.growth_function.step(action)
        self.mc.write_tensor(self.growth_function.X, skip=self.empty_material)

        # Determine the reward and end state.
        if (
            0 < self.growth_function.steps
            and self.growth_function.steps % self.reward_interval == 0
        ):
            reward = self.get_reward()
            self.previous_reward = reward

        done = (not self.growth_function.building()) or (
            self.growth_function.steps == self.max_steps
        )

        # Get the representation around the next block on
        # which to build and return the features, reward, etc.
        if done:
            self.mc.write_tensor(self.initial_state, skip=None)
            features = self.last_features
        else:
            features = self.get_representation()
            self.last_features = features

        return features, self.previous_reward, done, {}

    def get_reward(self):
        if self.reward_type == "max_y":
            height = get_height_from_floor(
                self.growth_function.X,
                self.growth_function.building_materials,
                self.max_steps + 1,
            )
            reward = height - self.previous_height
            self.previous_height = height
        elif self.reward_type == "distance_from_blocks":
            inverse_distance = inverse_distance_from_block_type(
                self.growth_function.X,
                self.growth_area_tensor,
                self.reward_block_type,
                self.empty_material,
            )
            reward = self.inverse_distance - self.previous_inverse_distance
            self.previous_inverse_distance = inverse_distance
        else:
            raise Exception("Unknown reward type: {self.reward}")
        return reward
