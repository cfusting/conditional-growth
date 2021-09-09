import gym
import numpy as np
from gym.spaces import Box, Discrete
from grow.utils.fitness import get_height_from_floor, distance_from_block_type
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
        self.feature_type = config["feature_type"]

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
        if self.feature_type == "raw":
            self.observation_space = Box(
                0,
                1.0,
                (
                    2 * self.search_radius + 2,
                    2 * self.search_radius + 2,
                    2 * self.search_radius + 2,
                    len(self.observing_materials),
                ),
                np.uint8,
            )
        else:
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
        z_offset = int(
            np.floor(
                np.random.normal(
                    config.vector_index * self.growth_function.max_length * 3, sigma
                )
            )
        )

        if config["training"]:
            self.mc = MinecraftAPI(
                self.max_steps,
                self.growth_function.max_length,
                x_offset=x_offset,
                z_offset=z_offset,
            )
            self.mc.establish_connection()

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
            self.set_grow_area_tensor()
            self.initial_state = self.growth_area_tensor

    def initialize_rewards(self):
        self.previous_reward = 0

        if self.reward_type == "y_max":
            self.previous_height = 1
        elif self.reward_type == "distance_from_blocks":
            self.previous_distance = 0

            X = self.initial_state
            # Put something in the axiom spot to prevent
            # placing the reward where the first block
            # will be placed.
            X[
                self.growth_function.axiom_coordinate,
                self.growth_function.axiom_coordinate,
                self.growth_function.axiom_coordinate,
            ] = self.growth_function.building_materials[0]

            # Find empty space to place the reward block.
            possible_points = list(np.argwhere(X == self.empty_material))
            i = np.random.choice(
                len(possible_points),
                replace=False,
            )
            x = possible_points[i][0]
            y = possible_points[i][1]
            z = possible_points[i][2]

            X = np.full_like(self.growth_function.X, self.empty_material)
            X[x, y, z] = self.reward_block_type
            self.reward_block_coordinate = (x, y, z)
            self.mc.write_tensor(X)
        else:
            raise Exception("Unknown reward type: {self.reward}")

    def clear_construction(self):
        X = np.full_like(self.growth_function.X, 0)
        for material in self.growth_function.building_materials:
            M = self.growth_function.X == material
            X = np.bitwise_or(X, M)

        X[X == 1] = self.empty_material

        if self.reward_type == "distance_from_blocks":
            X[
                self.reward_block_coordinate[0],
                self.reward_block_coordinate[1],
                self.reward_block_coordinate[2],
            ] = self.empty_material
        self.mc.write_tensor(X, skip=None, only=[self.empty_material])

    def reset(self):
        self.growth_function.reset()
        self.initialize_rewards()
        return self.get_representation()

    def set_grow_area_tensor(self):
        """Get the tensor representation of the entire grow area.

        The grow area is defined as the maximum extent the growth
        function could reach in any direction given the number of
        growth steps.

        """

        X, _ = self.mc.read_tensor(
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
        # Should probably do this at some points.
        # self.growth_function.atrophy_disconnected_voxels()

    def get_representation(self):
        """Get the local representation of blocks.

        Return the proportions of blocks around the block on which
        growth is about to occur. A proportion is returned for each
        of the sixes faces of the block.

        """
        current_voxel = self.growth_function.get_next_building_voxel()
        x, y, z = current_voxel.x, current_voxel.y, current_voxel.z
        X, Z = self.mc.read_tensor(
            x - self.search_radius - 1,
            x + self.search_radius + 1,
            y - self.search_radius - 1,
            y + self.search_radius + 1,
            z - self.search_radius - 1,
            z + self.search_radius + 1,
            self.observing_materials,
        )
        # Use convolution to find features.
        if self.feature_type == "raw":
            return Z

        # Otherwise we'll hand craft some features.
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

        self.clear_construction()

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

        done = (
            (not self.growth_function.building())
            or (self.growth_function.steps == self.max_steps)
            or (reward == -1)
        )

        # Get the representation around the next block on
        # which to build and return the features, reward, etc.
        if done:
            self.clear_construction()
            features = self.last_features
        else:
            features = self.get_representation()
            self.last_features = features

        if reward == -1:
            self.previous_reward = 1
        return features, self.previous_reward, done, {}

    def get_reward(self):
        if self.reward_type == "max_y":
            height = get_height_from_floor(
                self.growth_function.X,
                self.growth_function.building_materials,
                self.max_steps + 1,
            )
            if height > self.previous_height:
                reward = 1
            else:
                reward = 0
            self.previous_height = height
        elif self.reward_type == "distance_from_blocks":
            distance = distance_from_block_type(
                self.growth_function.X,
                self.growth_area_tensor,
                self.reward_block_type,
                self.empty_material,
            )
            if distance == -1:
                return -1
            if distance < self.previous_distance:
                reward = 1
            else:
                reward = 0
            self.previous_distance = distance
        else:
            raise Exception("Unknown reward type: {self.reward}")
        return reward
