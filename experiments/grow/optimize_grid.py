import ray
import numpy as np
from ray.rllib.agents.ppo import PPOTrainer
from grow.env.tensor_environment import TensorGrowthEnvironment


ray.init()

config = {
    "env": TensorGrowthEnvironment,
    "env_config": {
        "materials": (0, 1),
        "num_timestep_features": 1,
        "max_steps": 50,
        "reward_interval": 49,
        "max_voxels": 5,
        "search_radius": 3,
        "axiom_material": 1,
        "reward_type": "tree",
        "voxel_size": 0.01,
    },
    "vf_clip_param": 10**5,
    "seed": np.random.randint(10**5),
    "num_workers": 1,
    "num_gpus": 0,
    "num_gpus_per_worker": 0,
    "num_envs_per_worker": 1,
    "framework": "torch",
    # "monitor": True, 
    # "evaluation_num_workers": 7,
}

ray.tune.run(
    PPOTrainer,
    name="tree",
    config=config,
    checkpoint_freq=0,
    keep_checkpoints_num=0,
    # restore=,
)
