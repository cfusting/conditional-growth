import ray
import torch
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from grow.utils.minecraft_pb2 import AIR, SEA_LANTERN, GLOWSTONE
import os
import numpy as np
from ray.rllib.offline.json_reader import JsonReader
from ray.rllib.agents.ppo import PPOTrainer
from grow.env.minecraft_environment import MinecraftEnvironment
from ray.rllib.models import ModelCatalog
from grow.utils.nn import ThreeDimensionalConvolution
from grow.utils.vim import VariationInformationMaximization


ModelCatalog.register_custom_model("3dconv", ThreeDimensionalConvolution)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ray.init(num_gpus=0)

config = {
    "env": MinecraftEnvironment,
    "env_config": {
        "materials": (AIR, SEA_LANTERN),
        "num_timestep_features": 1,
        "max_steps": 10,
        "reward_interval": 1,
        "max_voxels": 6,
        # Make sure to calibrate this with the convolutions.
        "search_radius": 10,
        "axiom_material": SEA_LANTERN,
        "reward_type": "distance_from_blocks",
        "empty_material": AIR,
        "observing_materials": (AIR, SEA_LANTERN, GLOWSTONE),
        "reward_block_type": GLOWSTONE,
        "feature_type": "raw",
    },
    # Hypers
    # See https://openreview.net/pdf?id=nIAxjsniDzg
    # WHAT MATTERS FOR ON-POLICY DEEP ACTOR-CRITIC METHODS? A LARGE-SCALE STUDY
    "model": {
        "custom_model": "3dconv",
        # "fcnet_hiddens": [256, 256],
        # "fcnet_activation": "tanh",
    },
    "gamma": 0.99,  # One of the most important. Tune this!
    "lr": 0.0003,  # Tune this as well.
    "clip_param": 0.25,  # Tune: 0.1 to 0.5
    # "vf_clip_param": 10 ** 10,  # No clipping.
    "lambda": 0.9,  # Optimal 0.9
    # The remaining choices were not addressed in the paper.
    "entropy_coeff": 0.00,  # Tune?
    # Size of batches collected from each worker.
    # "rollout_fragment_length": 10,
    # Number of timesteps collected for each SGD round. This defines the size
    # of each SGD epoch.
    # "train_batch_size": 100,
    # Total SGD batch size across all devices for SGD. This defines the
    # minibatch size within each epoch.
    # "sgd_minibatch_size": 10,
    # Settings
    "seed": np.random.randint(2 ** 32),
    "num_workers": 1,
    "num_gpus": 0,
    "num_gpus_per_worker": 0,
    "num_envs_per_worker": 1,
    "framework": "torch",
}


agent = PPOTrainer(env=MinecraftEnvironment, config=config)
agent.restore(
    "/home/ray/ray_results/escape/PPO_MinecraftEnvironment_898b6_00000_0_2021-08-29_09-16-59/checkpoint_000040/checkpoint-40"
)
model = agent.get_policy(DEFAULT_POLICY_ID).model

k = 5
vim = VariationInformationMaximization(24, 64, k, 0)
# torch.autograd.set_detect_anomaly(True)
reader = JsonReader("/home/ray/ray_results/escape_output")
action_decoder_losses = 0
source_losses = 0
empowerments = 0
j = 1
temperature = 1
for _ in range(10**3):
    batch = reader.next()
    for episode in batch.split_by_episode():
        obs, actions, dones = episode.columns(["obs", "actions", "dones"])
        if True not in dones or len(dones) < 10:
            # print(f"Episode length: {len(dones)}")
            # print("Incomplete episode, skipping.")
            continue
        X = torch.from_numpy(obs[[0, k], ...]).permute(0, 4, 1, 2, 3)
        Z = model._hidden_layers(X).squeeze()
        z_start = Z[0, ...].unsqueeze(0)
        z_end = Z[1, ...].unsqueeze(0)
        actions = torch.from_numpy(actions[:k])

        action_decoder_loss, source_action_loss, empowerment = vim.step(
            z_start, z_end, actions, temperature
        )
        action_decoder_losses += action_decoder_loss
        source_losses += source_action_loss
        empowerments += empowerment
        j += 1

        if j % 10**2 == 0:
            print(f"Action Decoder Loss: {(action_decoder_losses / j):.3f}")
            print(f"Source Loss: {(source_losses / j):.3f}")
            print(f"Empowerment: {(empowerments / j):.3f}")

        if j % 10**4 == 0:
            temperature = 1
