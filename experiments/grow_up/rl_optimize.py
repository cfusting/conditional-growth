import ray
from ray.rllib.agents.sac import SACTrainer
from grow.optimize.voxcraft_environment import VoxcraftGrowthEnvironment


ray.init()

config = {
    "env": VoxcraftGrowthEnvironment,
    "env_config": {
        "materials": (0, 1),
        "num_timesteps": 4,
        "max_steps": 12,
        "max_voxels": 6,
        "search_radius": 6,
        "axiom_material": 1,
        "path_to_sim_build": "/root/voxcraft-sim/build",
        "path_to_base_vxa": "/root/conditional-growth/experiments/grow_up/data/base.vxa",
        "reward": "table",
        "voxel_size": 0.01,
        "simulation_interval": 4,
    },
    "num_gpus_per_worker": 1,
    "num_gpus": 1,
    "framework": "torch",
    "num_workers": 0,
    "create_env_on_driver": True,
    "output": "logdir",
    "log_level": "DEBUG",
    "tau": 1.0,
    "learning_starts": 100,
    # "batch_mode": "complete_episodes",
    # "buffer_size": 100,
    # "prioritized_replay": False,
    # "rollout_fragment_length": 1,
    # "timesteps_per_iteration": 5,
    "train_batch_size": 32,
    "target_network_update_freq": 32,
    # "evaluation_interval": None,
    # "evaluation_num_episodes": 10,
    # "metrics_smoothing_episodes": 1,
    # "n_step": 1,
    # "target_entropy": "auto",
    # "clip_rewards": False,
    # "clip_actions": False,
    "optimization": {
        "actor_learning_rate": 0.0001,
        "critic_learning_rate": 0.0001,
        "entropy_learning_rate": 0.0001,
    },
}

ray.tune.run(
    SACTrainer,
    name="table",
    config=config,
    checkpoint_freq=100,
    keep_checkpoints_num=0,
    verbose=3,
    # resume="LOCAL",
)
