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
        "max_steps": 500,
        "reward_interval": 499,
        "max_voxels": 6,
        "search_radius": 3,
        "axiom_material": 1,
        "reward_type": "tree",
    },

    # Hypers
    # See https://openreview.net/pdf?id=nIAxjsniDzg
    # WHAT MATTERS FOR ON-POLICY DEEP ACTOR-CRITIC METHODS? A LARGE-SCALE STUDY
    "model": {
       "fcnet_hiddens": [256, 256],
       "fcnet_activation": "tanh",
    },
    "gamma": 0.99,  #  One of the most important. Tune this!
    "lr": 0.0003,  #  Tune this as well.
    "clip_param": 0.25,  #  Tune: 0.1 to 0.5
    "vf_clip_param": 10**10,  #  No clipping.
    "lambda": 0.9, #  Optimal 0.9

    # The remaining choices were not addressed in the paper.
    "entropy_coeff": 0.00,  # Tune?

    # Settings
    "seed": np.random.randint(2**32),
    "num_workers": 1,
    "num_gpus": 0,
    "num_gpus_per_worker": 0,
    "num_envs_per_worker": 1,
    "framework": "torch",
    "record_env": True, 
    # "evaluation_num_workers": 7,
}


ray.tune.run(
    PPOTrainer,
    # name="fat-cat",
    name="visualize",
    config=config,
    checkpoint_freq=10,
    keep_checkpoints_num=None,
    restore="/home/ray/ray_results/fat-cat/PPO_TensorGrowthEnvironment_cc9b7_00000_0_2021-05-08_17-30-47/checkpoint_000240/checkpoint-240",
    )
