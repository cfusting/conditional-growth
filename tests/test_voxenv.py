import ray
import numpy as np
from ray.rllib.agents.ppo import PPOTrainer
from grow.env.voxcraft_environment import VoxcraftGrowthEnvironment


def test_optimize():

    ray.init()

    config = {
        "env": VoxcraftGrowthEnvironment,
        "env_config": {
            "materials": (0, 1),
            "num_timestep_features": 1,
            "max_steps": 3,
            "reward_interval": 1,
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
        "num_workers": 1,
        "num_gpus": .25,
        "num_gpus_per_worker": 0.75,
        "num_envs_per_worker": 1,
        "framework": "torch",
        "monitor": True,
    }

    ray.tune.run(
        PPOTrainer,
        name="vox-test",
        config=config,
        stop={"timesteps_total": 1},
        local_dir="/tmp"
    )

test_optimize()
