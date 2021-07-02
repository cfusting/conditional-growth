import ray
import numpy as np
from ray.rllib.agents.ppo import PPOTrainer
from grow.env.minecraft_environment import MinecraftEnvironment
from grow.utils.minecraft_pb2 import AIR, SEA_LANTERN, GLOWSTONE

ray.init()

config = {
    "env": MinecraftEnvironment,
    "env_config": {
        "materials": (AIR, SEA_LANTERN),
        "num_timestep_features": 1,
        "max_steps": 10,
        "reward_interval": 1,
        "max_voxels": 6,
        "search_radius": 10,
        "axiom_material": SEA_LANTERN,
        "reward_type": "distance_from_blocks",
        "empty_material": AIR,
        "observing_materials": (AIR, SEA_LANTERN, GLOWSTONE),
        "reward_block_type": GLOWSTONE,
    },
    # Hypers

    # See https://openreview.net/pdf?id=nIAxjsniDzg
    # WHAT MATTERS FOR ON-POLICY DEEP ACTOR-CRITIC METHODS? A LARGE-SCALE STUDY
    "model": {
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "tanh",
    },
    "gamma": 0.99,  # One of the most important. Tune this!
    "lr": 0.0003,  # Tune this as well.
    "clip_param": 0.25,  # Tune: 0.1 to 0.5
    # "vf_clip_param": 10 ** 10,  # No clipping.
    "lambda": 0.9,  # Optimal 0.9
    # The remaining choices were not addressed in the paper.
    "entropy_coeff": 0.00,  # Tune?
    # Size of batches collected from each worker.
    "rollout_fragment_length": 10,
    # Number of timesteps collected for each SGD round. This defines the size
    # of each SGD epoch.
    "train_batch_size": 100,
    # Total SGD batch size across all devices for SGD. This defines the
    # minibatch size within each epoch.
    "sgd_minibatch_size": 10,
    # Settings
    "seed": np.random.randint(2 ** 32),
    "num_workers": 3,
    "num_gpus": 0,
    "num_gpus_per_worker": 0,
    "num_envs_per_worker": 2,
    "framework": "torch",
}

ray.tune.run(
    PPOTrainer,
    name="ddog",
    config=config,
    checkpoint_freq=10,
    keep_checkpoints_num=None,
)
