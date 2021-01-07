import gym
from grow.utils.output import get_voxel_positions
import os
import numpy as np
from gym.spaces import Box, Discrete
from grow.utils.tensor_to_cdata import tensor_to_cdata, add_cdata_to_xml
from grow.utils.fitness import max_z, table
from grow.entities.conditional_growth_genome import ConditionalGrowthGenome
import subprocess


class VoxcraftGrowthEnvironment(gym.Env):
    def __init__(self, config):
        self.genome = ConditionalGrowthGenome(
            materials=config["materials"],
            max_voxels=config["max_voxels"],
            search_radius=config["search_radius"],
            axiom_material=config["axiom_material"],
            num_timesteps=config["num_timesteps"],
        )

        self.action_space = Discrete(len(self.genome.configuration_map))
        self.num_features = (
            len(self.genome.materials)
            * len(self.genome.directions)
            * config["num_timesteps"]
        )
        self.observation_space = Box(low=0, high=1, shape=(self.num_features,))
        self.reward_range = (0, float("inf"))

        self.path_to_sim_build = config["path_to_sim_build"]
        self.path_to_base_vxa = config["path_to_base_vxa"]
        self.ranked_simulation_file_path = config["ranked_simulation_file_path"]
        self.record_history = config["record_history"]
        subprocess.run(f"mkdir -p {self.ranked_simulation_file_path}".split())

        self.reward = config["reward"]
        self.max_steps = config["max_steps"]
        self.voxel_size = config["voxel_size"]
        self.simulation_interval = config["simulation_interval"]
        self.surrogate_simulation = config["surrogate_simulation"]

    def get_representation(self):
        x = np.array(self.genome.get_local_voxel_representation())
        x = x[: self.num_features]
        return x

    def step(self, action):
        if self.genome.steps != 0 and self.genome.steps % self.simulation_interval == 0:
            if self.surrogate_simulation:
                reward = self.get_surrogate_reward_for_action(action)
            else:
                reward = self.get_sim_reward_for_action(action)
            self.previous_reward = reward

        else:
            self.genome.step(action)

        done = not self.genome.building() or (self.genome.steps == self.max_steps)

        return self.get_representation(), self.previous_reward, done, {}

    def reset(self):
        self.genome.reset()
        self.previous_reward = 0
        return self.get_representation()

    def get_sim_reward_for_action(self, action):
        self.genome.step(action)

        (
            simulation_folder,
            data_dir_path,
            simulation_file_path,
            out_file_path,
        ) = self.prep_simulation_folders()
        self.generate_sim_data(action, data_dir_path)
        run_command = f"./voxcraft-sim -i {data_dir_path} -o {out_file_path}"
        initial_positions, final_positions = self.get_sim_final_positions(
            run_command, simulation_file_path, out_file_path
        )
        reward = self.get_reward(initial_positions, final_positions)
        self.update_file_fitness(
            simulation_file_path, simulation_folder, reward, data_dir_path
        )

        return reward

    def get_surrogate_reward_for_action(self, action):
        self.genome.step(action)

        initial_positions, final_positions = self.genome.to_tensor_and_tuples()
        reward = self.get_reward(initial_positions, final_positions)
        return reward

    def get_reward(self, initial_positions, final_positions):
        if self.reward == "max_z":
            reward = max_z(initial_positions, final_positions)
        elif self.reward == "table":
            reward = table(initial_positions, final_positions)
        else:
            raise Exception("Unknown reward type: {self.reward}")
        return reward

    def update_file_fitness(
        self, simulation_file_path, simulation_folder, fitness, data_dir_path
    ):
        updated_data_dir_path = f"{self.ranked_simulation_file_path}/{fitness:.20f}_{self.genome.steps}_{simulation_folder}"

        if os.path.isdir(updated_data_dir_path):
            raise ("WARNING: Output directory exists. This not happen.")
        subprocess.run(f"mv {data_dir_path} {updated_data_dir_path}".split())

    def get_sim_final_positions(self, run_command, simulation_file_path, out_file_path):
        with open(simulation_file_path, "w") as f:
            subprocess.run(
                run_command.split(),
                cwd=self.path_to_sim_build,
                stdout=f,
            )
        subprocess.run(f"cp {simulation_file_path} /tmp/latest.history".split())
        initial_positions, final_positions = get_voxel_positions(out_file_path)
        initial_positions = self.normalize_positions(initial_positions)
        final_positions = self.normalize_positions(final_positions)
        return initial_positions, final_positions

    def prep_simulation_folders(self):
        simulation_folder = f"{self.genome.id}_{self.genome.steps}"
        data_dir_path = f"/tmp/{simulation_folder}"
        simulation_file_path = f"{data_dir_path}/simulation.history"
        out_file_path = f"{data_dir_path}/output.xml"

        # Create the necessary directory and copy the base sim config.
        subprocess.run(f"mkdir -p {data_dir_path}".split())
        subprocess.run(f"cp {self.path_to_base_vxa} {data_dir_path}".split())
        return simulation_folder, data_dir_path, simulation_file_path, out_file_path

    def normalize_positions(self, positions):
        normalized_positions = []
        for p in positions:
            normalized_positions.append(
                (
                    p[0] / self.voxel_size,
                    p[1] / self.voxel_size,
                    p[2] / self.voxel_size,
                )
            )
        return normalized_positions

    def generate_sim_data(self, configuration_index, data_dir_path):
        X, _ = self.genome.to_tensor_and_tuples()
        C = tensor_to_cdata(X)
        robot_path = data_dir_path + "/robot.vxd"
        add_cdata_to_xml(C, X.shape[0], X.shape[1], X.shape[2], robot_path, self.record_history)
