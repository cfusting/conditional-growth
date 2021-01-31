import ray
import numpy as np
from ray.rllib.agents.ppo import PPOTrainer
from grow.optimize.voxcraft_environment import TensorGrowthEnvironment


ray.init()

config = {
    "env": TensorGrowthEnvironment,
    "env_config": {
        "materials": (0, 1),
        "num_timestep_features": 1,
        "max_steps": 200,
        "reward_interval": 66,
        "max_voxels": 6,
        "search_radius": 3,
        "axiom_material": 1,
        "reward_type": "max_volume",
        "voxel_size": 0.01,
    },
    "vf_clip_param": 10**5,
    "seed": np.random.randint(10**5),
    "num_workers": 8,
    "num_gpus": 1,
    "num_gpus_per_worker": 0,
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
