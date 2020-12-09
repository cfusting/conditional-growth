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
        "growth_iterations": 5,
        "max_voxels": 3,
        "search_radius": 3,
        "axiom_material": 1,
        "path_to_sim_build": "/root/voxcraft-sim/build",
        "path_to_base_vxa": "/root/conditional-growth/data/base.vxa",
    },
    "num_gpus_per_worker": 1,
    "num_gpus": 1,
    "framework": "torch",
    "timesteps_per_iteration": 10,
    "learning_starts": 10,
    "output": "/root/simulation-history",
    "num_workers": 0,
    "create_env_on_driver": True,
    "rollout_fragment_length": 10,
    "train_batch_size": 10,
    "metrics_smoothing_episodes": 10,
}

ray.tune.run(
    SACTrainer, 
    config=config,
    checkpoint_freq=1,
    restore="/root/ray_results/SAC_2020-12-08_17-53-54/SAC_VoxcraftGrowthEnvironment_567e2_00000_0_2020-12-08_17-53-54/checkpoint_299/checkpoint-299",
)
