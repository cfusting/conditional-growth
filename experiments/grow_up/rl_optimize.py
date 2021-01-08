import ray
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
        "num_timesteps": 3,
        "max_steps": 10,
        "simulation_interval": 3,
        "max_voxels": 6,
        "search_radius": 3,
        "axiom_material": 1,
        "path_to_sim_build": "/root/voxcraft-sim/build",
        "path_to_base_vxa": "/root/conditional-growth/experiments/grow_up/data/base.vxa",
        "reward": "table",
        "voxel_size": 0.01,
        "surrogate_simulation": False,
    },
    "seed": 32432,
    "num_workers": 1,
    "num_gpus": .25,
    "num_gpus_per_worker": .25,
    "num_envs_per_worker": 3,
    "framework": "torch",
    # "create_env_on_driver": True,
    "output": "logdir",
    "log_level": "WARN",
}

ray.tune.run(
    PPOTrainer,
    name="stupid_table",
    config=config,
    checkpoint_freq=1,
    keep_checkpoints_num=0,
    verbose=1,
    # resume="LOCAL",
)
