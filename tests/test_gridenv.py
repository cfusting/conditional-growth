import ray
import numpy as np
from ray.rllib.agents.ppo import PPOTrainer
from grow.env.tensor_environment import TensorGrowthEnvironment


def test_optimize_grid():
    ray.init()

    config = {
        "env": TensorGrowthEnvironment,
        "env_config": {
            "materials": (0, 1),
            "num_timestep_features": 1,
            "max_steps": 10,
            "reward_interval": 1,
            "max_voxels": 6,
            "search_radius": 3,
            "axiom_material": 1,
            "reward_type": "max_volume",
            "voxel_size": 0.01,
        },
        "vf_clip_param": 10**5,
        "seed": np.random.randint(10**5),
        "num_workers": 4,
        "num_gpus": 0,
        "num_gpus_per_worker": 0,
        "num_envs_per_worker": 1,
        "framework": "torch",
    }

    ray.tune.run(
        PPOTrainer,
        name="grid-test",
        config=config,
        stop={"timesteps_total": 1},
        local_dir="/tmp"
    )
