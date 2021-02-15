import ray
import numpy as np
from ray.rllib.agents.ppo import PPOTrainer
from grow.env.tensor_environment import TensorGrowthEnvironment
from ray.rllib.agents.callbacks import DefaultCallbacks


class Callbacks(DefaultCallbacks):

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        episode.custom_metrics["final_reward"] = episode.prev_reward_for()


ray.init()
max_steps = 20
reward_interval = 3

for seed in range(5):
    config = {
        "env": TensorGrowthEnvironment,
        "env_config": {
            "materials": (0, 1),
            "num_timestep_features": 1,
            "max_steps": max_steps,
            "reward_interval": reward_interval,
            "max_voxels": 6,
            "search_radius": 3,
            "axiom_material": 1,
            "reward_type": "max_surface_area",
            "voxel_size": 0.01,
        },
        "vf_clip_param": 10**5,
        "seed": seed,
        "num_workers": 8,
        "num_gpus": 0,
        "num_gpus_per_worker": 0,
        "num_envs_per_worker": 1,
        "framework": "torch",
        "callbacks": Callbacks,
    }

    ray.tune.run(
        PPOTrainer,
        name=f"max_surface_area_{reward_interval}",
        config=config,
        checkpoint_freq=10,
        keep_checkpoints_num=0,
        stop={"training_iteration": 10**2},
    )
