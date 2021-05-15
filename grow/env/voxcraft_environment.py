import gym
from uuid import uuid4
from grow.utils.output import get_voxel_positions
import numpy as np
from gym.spaces import Box, Discrete
from grow.utils.tensor_to_cdata import tensor_to_cdata, add_cdata_to_xml
from grow.utils.fitness import max_z, table, distance_traveled, has_fallen
from grow.entities.growth_function import GrowthFunction
import subprocess


class VoxcraftGrowthEnvironment(gym.Env):

    metadata = {"render.modes": ["ansi"]}

    def __init__(self, config):
        print("Init")
        self.genome = GrowthFunction(
            materials=config["materials"],
            max_voxels=config["max_voxels"],
            search_radius=config["search_radius"],
            axiom_material=config["axiom_material"],
            num_timestep_features=config["num_timestep_features"],
        )
        self.max_steps = config["max_steps"]
        self.action_space = Discrete(len(self.genome.configuration_map))
        self.observation_space = Box(low=0, high=1, shape=(self.genome.num_features,))
        self.reward_range = (0, float("inf"))
        self.path_to_sim_build = config["path_to_sim_build"]
        self.base_template_path = config["base_template_path"]
        self.reward_type = config["reward_type"]
        self.voxel_size = config["voxel_size"]
        self.reward_interval = config["reward_interval"]
        self.fallen_threshold = config["fallen_threshold"]
        self.robot = "\n"

    def get_representation(self):
        x = np.array(self.genome.get_local_voxel_representation())
        return x

    def step(self, action):
        print(f"Step: {self.genome.step}")
        self.genome.step(action)
        if self.genome.steps != 0 and self.genome.steps % self.reward_interval == 0:
            reward = self.get_reward_for_action(action)
            self.previous_reward = reward

        done = not self.genome.building() or (self.genome.steps == self.max_steps)
        return self.get_representation(), self.previous_reward, done, {}

    def reset(self):
        self.genome.reset()
        self.previous_reward = 0
        return self.get_representation()

    def get_reward_for_action(self, action):
        folder = uuid4()
        sim_path = f"/tmp/{folder}"
        out_path = f"{sim_path}/output.xml"
        base_path = f"{sim_path}/base.vxa"
        subprocess.run(f"mkdir -p {sim_path}".split())
        subprocess.run(f"cp {self.base_template_path} {base_path}".split())

        self.generate_sim_data(action, sim_path)
        run_command = f"./voxcraft-sim -f -i {sim_path} -o {out_path}"
        initial_positions, final_positions = self.get_sim_positions(
            run_command, out_path
        )
        reward = self.get_reward(
            initial_positions, final_positions, out_path
        )
        subprocess.run(f"rm -fr {sim_path}".split())
        print(f"Reward: {reward}")
        return reward

    def generate_sim_data(self, configuration_index, data_dir_path):
        X, _, _, = self.genome.to_tensor_and_tuples()
        C = tensor_to_cdata(X)
        robot_path = data_dir_path + "/robot.vxd"
        self.robot = add_cdata_to_xml(
            C, X.shape[0], X.shape[1], X.shape[2], robot_path, record_history=False)

    def get_sim_positions(self, run_command, out_file_path):
        subprocess.run(
            run_command.split(),
            cwd=self.path_to_sim_build,
        )
        initial_positions, final_positions = get_voxel_positions(out_file_path, voxel_size=self.voxel_size)
        return initial_positions, final_positions

    def get_reward(
        self, initial_positions, final_positions, out_file_path
    ):
        if self.reward_type == "max_z":
            if has_fallen(initial_positions, final_positions, self.fallen_threshold):
                reward = 0
            else:
                reward = max_z(final_positions)
        elif self.reward_type == "table":
            if has_fallen(initial_positions, final_positions, self.fallen_threshold):
                reward = 0
            else:
                reward = table(final_positions)
        elif self.reward_type == "distance_traveled":
            reward = distance_traveled(initial_positions, final_positions)
        else:
            raise Exception("Unknown reward type: {self.reward_type}")
        print(reward)
        return reward

    def render(self, mode="ansi"):
        if mode == "ansi":
            return self.robot + "\n"
