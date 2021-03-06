import ray
import numpy as np
from ray.rllib.agents.ppo import PPOTrainer
from grow.env.voxcraft_environment import VoxcraftGrowthEnvironment


"""
IMPORTANT: You MUST configure data/base.vxa to match the relevant
configurations in this file.
"""


ray.init()

config = {
    "env": VoxcraftGrowthEnvironment,
    "env_config": {
        "materials": (0, 1, 2),
        "num_timestep_features": 1,
        "max_steps": 100,
        "reward_interval": 9,
        "max_voxels": 6,
        "search_radius": 3,
        "axiom_material": 1,
        "path_to_sim_build": "/cluster/home/skrieg01/voxcraft-sim/build",
        "base_template_path": "/cluster/home/skrieg01/conditional-growth/experiments/distance_traveled/data/base.vxa",
        "reward_type": "distance_traveled",
        "voxel_size": 0.01,
        "fallen_threshold": 0.25,
    },
    "vf_clip_param": 10**5,
    "seed": np.random.randint(10**5),
    "num_workers": 1,
    "num_gpus": 1,
    "num_gpus_per_worker": 1,
    "num_envs_per_worker": 1,
    "framework": "torch",
    #"monitor": True, 
    # "evaluation_num_workers": 7,
}

ray.tune.run(
    PPOTrainer,
    name="catnap",
    config=config,
    checkpoint_freq=0,
    keep_checkpoints_num=0,
    # restore=,
)
