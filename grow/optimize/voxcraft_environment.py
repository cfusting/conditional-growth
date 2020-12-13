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
        self.observation_space = Box(low=0, high=1, shape=(len(self.genome.materials),))
        self.path_to_sim_build = config["path_to_sim_build"]
        self.path_to_base_vxa = config["path_to_base_vxa"]
        self.reward_range = (0, float("inf"))

        self.ranked_simulation_file_path = "/tmp/ranked_simulations"
        subprocess.run(f"mkdir -p {self.ranked_simulation_file_path}".split())

    def step(self, action):
        fitness = float(self.get_fitness_for_action(action))
        next_observation = self.genome.get_local_voxel_representation()
        return next_observation, fitness, not self.genome.building(), {}

    def reset(self):
        self.genome.reset()
        return self.genome.get_local_voxel_representation()

    def get_fitness_for_action(self, action):
        simulation_folder = "voxcraft_data"
        data_dir_path = f"{self.ranked_simulation_file_path}/{simulation_folder}"
        simulation_file_path = f"{data_dir_path}/simulation.history"
        out_file_path = f"{data_dir_path}/output.xml"

        # Create the necessary directory and copy the base sim config.
        subprocess.run(f"mkdir -p {data_dir_path}".split())
        subprocess.run(f"cp {self.path_to_base_vxa} {data_dir_path}".split())

        # Stage the simulation file (robot.vxd).
        self.generate_sim_data(action, data_dir_path)

        # Run the simulation.
        run_command = f"./voxcraft-sim -l -f -i {data_dir_path} -o {out_file_path}"
        with open(simulation_file_path, "w") as f:
            t1 = time()
            subprocess.run(
                run_command.split(),
                cwd=self.path_to_sim_build,
                stdout=f,
            )
        print(f"Simulation complete with time: {time() - t1}")
        subprocess.run(f"cp {simulation_file_path} /tmp/latest.history".split())

        # The fitness is written to out_file_path.
        fitness = get_fitness(out_file_path) / self.genome.num_voxels
        iteration = self.genome.num_steps
        updated_data_dir_path = f"{self.ranked_simulation_file_path}/{fitness:.20f}_{iteration}_{simulation_folder}"
        subprocess.run(
            f"mv {data_dir_path} {updated_data_dir_path}".split()
        )
        return fitness

    def generate_sim_data(self, configuration_index, data_dir_path):
        self.genome.step(configuration_index)

        # Prepare creature for simulation.
        X = self.genome.to_tensor()
        C = tensor_to_cdata(X)
        robot_path = data_dir_path + "/robot.vxd"
        add_cdata_to_xml(C, X.shape[0], X.shape[1], X.shape[2], robot_path)
