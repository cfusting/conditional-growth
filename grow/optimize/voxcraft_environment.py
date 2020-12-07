import gym
from gym.spaces import Box, Discrete
from grow.utils.tensor_to_cdata import tensor_to_cdata, add_cdata_to_xml, get_fitness
from grow.entities.conditional_growth_genome import ConditionalGrowthGenome
import subprocess
from time import time


class VoxcraftGrowthEnvironment(gym.Env):
    def __init__(self, config):
        self.genome = ConditionalGrowthGenome(
            materials=config["materials"],
            directions=config["directions"],
            growth_iterations=config["growth_iterations"],
            max_voxels=config["max_voxels"],
            search_radius=config["search_radius"],
            axiom_material=config["axiom_material"],
        )

        self.action_space = Discrete(len(self.genome.configuration_map))
        self.observation_space = Box(
            low=0, high=1, shape=(len(self.genome.materials), )
        )
        self.path_to_sim_build = config["path_to_sim_build"]
        self.path_to_base_vxa = config["path_to_base_vxa"]
        self.reward_range = (0, float("inf"))

    def step(self, action):
        fitness = float(self.get_fitness_for_action(action))
        next_observation = self.genome.get_local_voxel_representation()
        return next_observation, fitness, not self.genome.building(), {}

    def reset(self):
        self.genome.reset()
        return self.genome.get_local_voxel_representation()

    def get_fitness_for_action(self, action):
        data_dir_path = f"/tmp/{time()}_voxcraft_data"
        out_file_path = f"{data_dir_path}/voxcraft_output.xml"
        subprocess.run(f"mkdir {data_dir_path}".split())

        # Create the necessary directory and copy the base sim config.
        subprocess.run(f"cp {self.path_to_base_vxa} {data_dir_path}".split())

        # Stage the simulation data.
        self.generate_sim_data(action, data_dir_path)

        # Run the simulation.
        run_command = f"./voxcraft-sim -l -f -i {data_dir_path} -o {out_file_path}"
        subprocess.run(
            run_command.split(),
            cwd=self.path_to_sim_build,
        )

        # The results are written to a file.
        return get_fitness(out_file_path)

    def generate_sim_data(self, configuration_index, data_dir_path):
        self.genome.step(configuration_index)

        # Prepare creature for simulation.
        X = self.genome.to_tensor()
        C = tensor_to_cdata(X)
        robot_path = data_dir_path + "/robot.vxd"
        add_cdata_to_xml(C, X.shape[0], X.shape[1], X.shape[2], robot_path)
