from ray.rllib.env.external_env import ExternalEnv
import subproces
from time import time


class VoxcraftGrowthEnvironment(ExternalEnv):

    def run(self, path_to_sim_build, 
            growth_iterations=7):
        episode_id = self.start_episode()
        for i in growth_iterations:
            action = self.get_action(episode_id, observation) 
            fitness = get_fitness_for_action(action)
            log_returns(episode_id, fitness)

    def get_fitness_for_action(self, action):
        data_dir = f"{time()_voxcraft_data}"
        out_file = f"{time()_voxcraft_output}"
        generate_sim_data(data_dir)
        run_command = f"./voxcraft -l -i {data_dir} -o {out_file}"
        subproces.run(
            run_command,
            cwd=path_to_sim_build,
        )
        return extract_fitness(out_file)

    def generate_sim_data(self, data_dir):
        
