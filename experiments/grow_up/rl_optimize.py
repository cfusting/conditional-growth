import ray
from ray.rllib.agents.sac import SACTrainer
from grow.optimize.voxcraft_environment import VoxcraftGrowthEnvironment


ray.init()

config = {
    "env": VoxcraftGrowthEnvironment,
    "env_config": {
        "materials": (0, 1),
        "directions": (
            "negative_x",
            "positive_x",
            "negative_y",
            "positive_y",
            "negative_z",
            "positive_z",
        ),
        "max_steps": 20,
        "max_voxels": 5,
        "search_radius": 4,
        "axiom_material": 1,
        "path_to_sim_build": "/root/voxcraft-sim/build",
        "path_to_base_vxa": "/root/conditional-growth/experiments/grow_up/data/base.vxa",
        "reward": "max_z",
    },
    "num_gpus_per_worker": 1,
    "num_gpus": 1,
    "framework": "torch",
    "num_workers": 0,
    "create_env_on_driver": True,
    "output": "logdir",
    "tau": 1.0,
}

ray.tune.run(
    SACTrainer, 
    name="sac_max_z",
    config=config,
    checkpoint_freq=1,
    keep_checkpoints_num=10,
    # resume="LOCAL",
)
