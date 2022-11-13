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
        self.search_radius = config["search_radius"]
        self.max_steps = config["max_steps"]

        # Extra category for all blocks not explicitly encoded
        p = len(self.observing_materials) + 1
        if self.feature_type == "raw":
            q = 2 * self.search_radius + 2
            self.observation_space = Box(
                0,
                1.0,
                (q, q, q, p),
                np.uint8,
            )
            self.last_features = np.zeros((q, q, q, p), dtype=np.uint8)
        else:
            num_proportion_features = p * 6
            feature_zeros = np.array([0 for x in range(num_proportion_features + 3)])
            self.observation_space = Box(
                feature_zeros,
                np.array([1 for x in range(num_proportion_features + 3)]),
            )
            self.last_features = feature_zeros

        self.reward_range = (0, float("inf"))
        self.reward_type = config["reward_type"]
        self.reward_interval = config["reward_interval"]

        # If it builds on one axis in one direction.
        self.max_length = 2 * (self.max_steps) + 2

        if config["axiom_position"] is not None:
            x_offset, y_offset, z_offset = config["axiom_position"]
            x_offset -= self.max_steps
            z_offset -= self.max_steps
            should_find_the_floor = False
        else:
            # The grow location is based on worker and vector indices.
            sigma = self.max_length
            x_offset = int(
                np.floor(
                    np.random.normal(
                        config.worker_index * self.max_length * 3, sigma
                    )
                )
            )
            z_offset = int(
                np.floor(
                    np.random.normal(
                        config.vector_index * self.max_length * 3, sigma
                    )
                )
            )
            y_offset = 0
            should_find_the_floor = True

        self.mc = MinecraftAPI(
            self.max_steps,
            self.max_length,
            x_offset=x_offset,
            z_offset=z_offset,
            y_offset=y_offset,
            should_find_the_floor=should_find_the_floor,
        )

        ###
        # Ensure the axiom coordinate is one block above the floor.
        #
        # Having found the y_offset during self.mc initialization,
        # we must now account for the axiom coordinate being in the
        # center of the growth function's tensor so that it starts
        # one block above the floor.
        self.mc.y_offset -= self.max_steps + 1
        ###

        # Save the landscape before growth.
        E = self.get_grow_area_tensor()
        self.initial_state = np.copy(E)

        self.growth_function = GrowthFunction(
            materials=config["materials"],
            max_voxels=config["max_voxels"],
            search_radius=config["search_radius"],
            axiom_material=config["axiom_material"],
            num_timestep_features=config["num_timestep_features"],
            max_steps=config["max_steps"],
            empty_material=self.empty_material,
            initial_state=self.initial_state,
            max_length=self.max_length,
        )

        self.action_space = Discrete(len(self.growth_function.configuration_map))

    def initialize_rewards(self):
        self.previous_reward = 0

        if self.reward_type == "y_max":
            self.previous_height = 1
        elif self.reward_type == "distance_from_blocks":
            self.previous_distance = 0

            X = np.copy(self.initial_state)
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

            X[x, y, z] = self.reward_block_type
            self.reward_block_coordinate = (x, y, z)
            self.mc.write_tensor(X)
        else:
            raise Exception("Unknown reward type: {self.reward}")

    def clear_construction(self):
        self.mc.write_tensor(self.initial_state, skip=None)

    def reset(self):
        self.clear_construction()
        self.growth_function.reset()
        self.initialize_rewards()
        return self.get_representation()

    def get_grow_area_tensor(self):
        """Get the tensor representation of the entire grow area.

        The grow area is defined as the maximum extent the growth
        function could reach in any direction given the number of
        growth steps.

        """

        E, _ = self.mc.read_tensor(
            0,
            self.max_length,
            0,
            self.max_length,
            0,
            self.max_length,
        )
        return E

    def update_growth_function_tensor(self, E):
        """Update the state of the creature.

        Remove any blocks that have dissapeared from the creature.
        This occurs when sand falls, wood burns, etc.

        """

        M = self.growth_function.X != E
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
            x / self.max_length,
            y / self.max_length,
            z / self.max_length,
        ]
        features = np.array(material_proportions + relative_locations)
        return features

    def step(self, action):
        # Ensure the growth functiona and Minecraft are in parity.
        # Take a growth step and update Minecraft.
        E = self.get_grow_area_tensor()
        self.update_growth_function_tensor(E)
        self.growth_function.step(action, E)
        self.mc.write_tensor(self.growth_function.X, skip=self.empty_material)
        E = self.get_grow_area_tensor()

        # Determine the reward and end state.
        if (
            0 < self.growth_function.steps
            and self.growth_function.steps % self.reward_interval == 0
        ):
            reward = self.get_reward(E)
            self.previous_reward = reward

        done = (
            (not self.growth_function.building())
            or (self.growth_function.steps == self.max_steps)
            or (reward == -1)
        )

        # Get the representation around the next block on
        # which to build and return the features, reward, etc.
        if done:
            features = self.last_features
        else:
            features = self.get_representation()
            self.last_features = features

        if reward == -1:
            self.previous_reward = 1
        return features, self.previous_reward, done, {}

    def get_reward(self, E):
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
                E,
                self.reward_block_type,
                self.empty_material,
            )
            if distance < 3:
                print(f"distance: {distance}")
            if distance == -1:
                return -1
            if distance < self.previous_distance:
                reward = 1
            elif distance == 0:
                reward = 1
            else:
                reward = 0
            self.previous_distance = distance
        else:
            raise Exception("Unknown reward type: {self.reward}")
        return reward
