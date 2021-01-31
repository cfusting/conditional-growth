import ray
import numpy as np
from ray.rllib.agents.ppo import PPOTrainer
from grow.optimize.voxcraft_environment import VoxcraftGrowthEnvironment


"""
IMPORTANT: You MUST configure data/base.vxa to match the relevant
configurations in this file.
"""


ray.init()

config = {
    "env": VoxcraftGrowthEnvironment,
    "env_config": {
        "materials": (0, 1),
        "num_timesteps": 1,
        "max_steps": 200,
        "simulation_interval": 66,
        "max_voxels": 6,
        "search_radius": 3,
        "axiom_material": 1,
        "path_to_sim_build": "/home/badger/voxcraft-sim/build",
        "reward": "shape",
        "voxel_size": 0.01,
        "record_history": True,
    },
    "vf_clip_param": 10**5,
    "seed": np.random.randint(10**5),
    "num_workers": 1,
    "num_gpus": 1,
    "num_gpus_per_worker": 0,
    "num_envs_per_worker": 1,
    "framework": "torch",
    "log_level": "WARN",
    "monitor": True, 
    "evaluation_num_workers": 7,
}

ray.tune.run(
    PPOTrainer,
    name="eval-sphere",
    config=config,
    checkpoint_freq=0,
    keep_checkpoints_num=0,
    verbose=1,
    restore="/root/ray_results/sphere/PPO_VoxcraftGrowthEnvironment_3db32_00000_0_2021-01-24_21-58-07/checkpoint_800/checkpoint-800"
)