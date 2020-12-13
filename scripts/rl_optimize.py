import ray
from ray.rllib.agents.sac import SACTrainer
from grow.optimize.voxcraft_environment import VoxcraftGrowthEnvironment


ray.init()

config = {
    "env": VoxcraftGrowthEnvironment,
    "env_config": {
        "materials": (0, 1, 2),
        "directions": (
            "negative_x",
            "positive_x",
            "negative_y",
            "positive_y",
            "negative_z",
            "positive_z",
        ),
        "growth_iterations": 3,
        "max_voxels": 2,
        "search_radius": 3,
        "axiom_material": 1,
        "path_to_sim_build": "/root/voxcraft-sim/build",
        "path_to_base_vxa": "/root/conditional-growth/data/base.vxa",
    },
    "num_gpus_per_worker": 1,
    "num_gpus": 1,
    "framework": "torch",
    "num_workers": 0,
    "create_env_on_driver": True,
    "output": "/root/simulation-history",
    # "input": "/root/simulation-history",
    "timesteps_per_iteration": 1,
    "log_level": "INFO",
    # Hypers
    "learning_starts": 1500,
    "metrics_smoothing_episodes": 1,
    "gamma": 0.99,
    "optimization": {
         "actor_learning_rate": 3e-4,
         "critic_learning_rate": 3e-4,
         "entropy_learning_rate": 3e-4,
    },
    "tau": 1.0,
}

ray.tune.run(
    SACTrainer, 
    config=config,
    checkpoint_freq=1,
    keep_checkpoints_num=10,
)
