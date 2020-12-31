import ray
from ray.rllib.agents.ppo import PPOTrainer
from grow.optimize.voxcraft_environment import VoxcraftGrowthEnvironment


ray.init()

config = {
    "env": VoxcraftGrowthEnvironment,
    "env_config": {
        "materials": (0, 1),
        "num_timesteps": 6,
        "max_steps": 24,
        "max_voxels": 6,
        "search_radius": 6,
        "axiom_material": 1,
        "path_to_sim_build": "/root/voxcraft-sim/build",
        "path_to_base_vxa": "/root/conditional-growth/experiments/grow_up/data/base.vxa",
        "reward": "table",
        "voxel_size": 0.01,
        "simulation_interval": 6,
        "surrogate_simulation": True,
    },
    "num_gpus_per_worker": 1,
    "num_gpus": 1,
    "framework": "torch",
    "num_workers": 0,
    "create_env_on_driver": True,
    "output": "logdir",
    "log_level": "WARN",
    #"tau": 1.0,
    #"learning_starts": 10000,
    # "batch_mode": "complete_episodes",
    # "buffer_size": 100,
    # "prioritized_replay": False,
    # "rollout_fragment_length": 1,
    # "timesteps_per_iteration": 5,
    #"train_batch_size": 256,
    #"target_network_update_freq": 256,
    # "evaluation_interval": None,
    # "evaluation_num_episodes": 10,
    # "metrics_smoothing_episodes": 1,
    # "n_step": 1,
    # "target_entropy": "auto",
    # "clip_rewards": False,
    # "clip_actions": False,
    # "optimization": {
    #     "actor_learning_rate": 0.0003,
    #     "critic_learning_rate": 0.0003,
    #     "entropy_learning_rate": 0.0003,
    # },
}

ray.tune.run(
    PPOTrainer,
    name="table",
    config=config,
    checkpoint_freq=100,
    keep_checkpoints_num=0,
    verbose=3,
    # resume="LOCAL",
)
