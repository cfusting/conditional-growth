import ray
import numpy as np
from ray.rllib.agents.ppo import PPOTrainer
from grow.env.tensor_environment import TensorGrowthEnvironment


ray.init()
seed = np.random.randint(10**5)
max_steps = 20
reward_interval = 19

config = {
    "env": TensorGrowthEnvironment,
    "env_config": {
        "materials": (0, 1),
        "num_timestep_features": 1,
        "max_steps": max_steps,
        "reward_interval": reward_interval,
        "max_voxels": 6,
        "search_radius": 3,
        "axiom_material": 1,
        "reward_type": "max_surface_area",
        "voxel_size": 0.01,
    },
    "vf_clip_param": 10**5,
    "seed": seed,
    "num_workers": 4,
    "num_gpus": 0,
    "num_gpus_per_worker": 0,
    "num_envs_per_worker": 1,
    "framework": "torch",
}


for i in range(3):
    ray.tune.run(
        PPOTrainer,
        name=f"wombat_{i}_{max_steps}_{reward_interval}",
        config=config,
        checkpoint_freq=10,
        keep_checkpoints_num=0,
        stop={"timesteps_total": 10**4},
    )
