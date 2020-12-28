from ray.rllib.env.external_env import ExternalEnv
from grow.entities.conditional_growth_genome import ConditionalGrowthGenome
import subprocess
from gym.spaces import Box, Discrete


class VoxcraftExternalGrowthEnvironment(ExternalEnv):

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

        self.reward = config["reward"]
        self.level_step = config["level_step"]
        self.level = 0

    def run(self):
        level = self.level
        self.genome.reset()
        episode_id = self.start_episode()

        while level < level + level_step: 
            configuration_index = self.get_action(observation)
            level = self.genome.step(configuration_index)

        self.log_returns(reward)
        self.end_episode(episode_id)
        
