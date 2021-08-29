from torch.nn import Module, Sequential, ReLU, Linear, Flatten, Conv3d
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


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
        )

        self._output_layers = Sequential(
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
        latent_representation = self._hidden_layers(self.x)
        return self._output_layers(latent_representation), state

    def value_function(self):
        assert self.x is not None, "You must call forward() before value_function()."
        return self.vf(self.x).squeeze(1)
