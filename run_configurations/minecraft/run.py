import ray
import os
import numpy as np
from ray.rllib.agents.ppo import PPOTrainer
from grow.env.minecraft_environment import MinecraftEnvironment
from grow.utils.minecraft_pb2 import AIR, SEA_LANTERN, GLOWSTONE
from torch.nn import Module, Sequential, ReLU, Linear, Flatten, Conv3d
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog


class ThreeDimensionalConvolution(TorchModelV2, Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        Module.__init__(self)
        n = obs_space.shape[3]
        self._hidden_layers = Sequential(
            Conv3d(n, 2 * n, 4, 2),
            ReLU(),
            Conv3d(2 * n, 4 * n, 4, 2),
            ReLU(),
            Conv3d(4 * n, 8 * n, 4, 1),
            ReLU(),
            Flatten(),
            Linear(8 * n, num_outputs),
        )

        self.vf = Sequential(
            Conv3d(n, 2 * n, 4, 2),
            ReLU(),
            Conv3d(2 * n, 4 * n, 4, 2),
            ReLU(),
            Conv3d(4 * n, 8 * n, 4, 1),
            ReLU(),
            Flatten(),
            Linear(8 * n, 1),
        )

        self.x = None

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].float()
        self.x = x.permute(0, 4, 1, 2, 3)
        return self._hidden_layers(self.x), state

    def value_function(self):
        assert self.x is not None, "You must call forward() before value_function()."
        return self.vf(self.x).squeeze(1)


ModelCatalog.register_custom_model("3dconv", ThreeDimensionalConvolution)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ray.init(num_gpus=1)

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
    "num_workers": 15,
    "num_gpus": 1,
    "num_gpus_per_worker": 0,
    "num_envs_per_worker": 1,
    "framework": "torch",
}

ray.tune.run(
    PPOTrainer,
    name="bigbadger",
    config=config,
    checkpoint_freq=1,
    keep_checkpoints_num=None,
    # restore=""
)
