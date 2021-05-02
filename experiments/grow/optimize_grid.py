import ray
import numpy as np
from ray.rllib.agents.ppo import PPOTrainer
from grow.env.tensor_environment import TensorGrowthEnvironment


ray.init()

config = {
    "env": TensorGrowthEnvironment,
    "env_config": {
        "materials": (0, 1),
        "num_timestep_features": 3,
        "max_steps": 500,
        "reward_interval": 499,
        "max_voxels": 6,
        "search_radius": 9,
        "axiom_material": 1,
        "reward_type": "tree",
    },
    "vf_clip_param": 10**5,
    "seed": np.random.randint(10**5),
    "num_workers": 1,
    "num_gpus": 0,
    "num_gpus_per_worker": 0,
    "num_envs_per_worker": 1,
    "framework": "torch",
    # "record_env": True, 
    # "evaluation_num_workers": 7,
}

ray.tune.run(
    PPOTrainer,
    name="big-tree",
    config=config,
    checkpoint_freq=10,
    keep_checkpoints_num=None,
    # restore="/home/ray/ray_results/fig-tree/PPO_TensorGrowthEnvironment_5c0d8_00000_0_2021-05-02_11-15-22/checkpoint_000200/checkpoint-200"
    )
