import gym
from grow.utils.output import get_voxel_positions
import os
import numpy as np
from gym.spaces import Box, Discrete
from grow.utils.tensor_to_cdata import tensor_to_cdata, add_cdata_to_xml
from grow.utils.fitness import max_z, table
from grow.entities.conditional_growth_genome import ConditionalGrowthGenome
import subprocess
from time import time


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
        self.ranked_simulation_file_path = "/tmp/ranked_simulations"
        subprocess.run(f"mkdir -p {self.ranked_simulation_file_path}".split())

        self.reward = config["reward"]
        self.max_steps = config["max_steps"]
        self.voxel_size = config["voxel_size"]

    def get_representation(self):
        x = np.array(self.genome.get_local_voxel_representation())
        x = x[: self.num_features]
        return x

    def step(self, action):
        fitness = self.get_fitness_for_action(action)

        done = not self.genome.building() or (self.genome.steps == self.max_steps)

        return self.get_representation(), fitness, done, {}

    def reset(self):
        self.genome.reset()
        return self.get_representation()

    def get_fitness_for_action(self, action):
        simulation_folder = f"{self.genome.id}_{self.genome.steps}"
        data_dir_path = f"/tmp/{simulation_folder}"
        simulation_file_path = f"{data_dir_path}/simulation.history"
        out_file_path = f"{data_dir_path}/output.xml"

        # Create the necessary directory and copy the base sim config.
        subprocess.run(f"mkdir -p {data_dir_path}".split())
        subprocess.run(f"cp {self.path_to_base_vxa} {data_dir_path}".split())

        # Stage the simulation file (robot.vxd).
        self.generate_sim_data(action, data_dir_path)

        # Run the simulation.
        run_command = f"./voxcraft-sim -i {data_dir_path} -o {out_file_path}"
        # print(f"Running: {run_command}")
        with open(simulation_file_path, "w") as f:
            t1 = time()
            subprocess.run(
                run_command.split(),
                cwd=self.path_to_sim_build,
                stdout=f,
            )
        # print(f"Simulation complete with time: {time() - t1}")
        subprocess.run(f"cp {simulation_file_path} /tmp/latest.history".split())

        # The fitness is written to out_file_path.
        if self.reward == "max_z":
            _, final_positions = get_voxel_positions(out_file_path)
            fitness = max_z(final_positions)
        elif self.reward == "table":
            _, final_positions = get_voxel_positions(out_file_path)
            voxels = []
            for p in final_positions:
                voxels.append((p[0] / self.voxel_size, p[1] / self.voxel_size, p[2] / self.voxel_size))
            fitness = table(voxels)
        else:
            raise Exception("Unknown reward type: {self.reward}")

        updated_data_dir_path = f"{self.ranked_simulation_file_path}/{fitness:.20f}_{self.genome.steps}_{simulation_folder}"
        if os.path.isdir(updated_data_dir_path):
            raise ("WARNING: Output directory exists. This not happen.")
        subprocess.run(f"mv {data_dir_path} {updated_data_dir_path}".split())
        return fitness

    def generate_sim_data(self, configuration_index, data_dir_path):
        self.genome.step(configuration_index)

        # Prepare creature for simulation.
        X = self.genome.to_tensor()
        C = tensor_to_cdata(X)
        robot_path = data_dir_path + "/robot.vxd"
        add_cdata_to_xml(C, X.shape[0], X.shape[1], X.shape[2], robot_path)
