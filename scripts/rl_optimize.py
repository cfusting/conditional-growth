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
        "max_voxels": 3,
        "search_radius": 2,
        "axiom_material": 1,
        "path_to_sim_build": "/root/voxcraft-sim/build",
        "path_to_base_vxa": "/root/conditional-growth/data/base.vxa",
    },
    "num_gpus_per_worker": 1,
    "num_gpus": 1,
    "framework": "torch",
    "timesteps_per_iteration": 30,
}

ray.tune.run(
    SACTrainer, 
    config=config,
)
