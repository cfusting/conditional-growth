import ray
from ray.rllib.agents.sac import SACTrainer
from grow.optimize.voxcraft_environment import VoxcraftGrowthEnvironment


ray.init()

config = {
    "env": VoxcraftGrowthEnvironment,
    "env_config": {
        "materials": (0, 1),
        "num_timesteps": 6, 
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
    "learning_starts": 100,
    # "rollout_fragment_length": 1,
    "batch_mode": "complete_episodes",
    "prioritized_replay": False,
    "train_batch_size": 32, 
    "target_network_update_freq": 100,
    "evaluation_interval": 4,
    "metrics_smoothing_episodes": 4,
    "n_step": 1,
    "target_entropy": "auto",
    "timesteps_per_iteration": 4,
    "clip_rewards": False,
    "clip_actions": False,
}

ray.tune.run(
    SACTrainer, 
    name="max_z",
    config=config,
    checkpoint_freq=1,
    keep_checkpoints_num=10,
    # resume="LOCAL",
)
