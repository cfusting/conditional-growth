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
        "num_timestep_feature": 1,
        "max_steps": 10,
        "simulation_interval": 1,
        "max_voxels": 6,
        "search_radius": 3,
        "axiom_material": 1,
        "path_to_sim_build": "/root/voxcraft-sim/build",
        "base_template_path": "/root/conditional-growth/experiments/grow/data/base.vxa",
        "reward_type": "distance_traveled",
        "voxel_size": 0.01,
        "fallen_threshold": 0.25,
    },
    "vf_clip_param": 10**5,
    "seed": np.random.randint(10**5),
    "num_workers": 3,
    "num_gpus": .25,
    "num_gpus_per_worker": 0.75,
    "num_envs_per_worker": 1,
    "framework": "torch",
}

ray.tune.run(
    PPOTrainer,
    name="",
    config=config,
    checkpoint_freq=0,
    keep_checkpoints_num=0,
)