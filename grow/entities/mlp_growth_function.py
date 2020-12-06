from mlp import MLP
import torch
import numpy as np


class MLPGrowthFunction:
    def __init__(self, input_size, output_size, sample=False):
        self.growth_function = MLP(input_size, output_size)
        self.sample = sample

    def predict(self, X):
        """Predict the configuration index based on nearby voxels.

        Attributes:
            X: List of material proportions.

        Returns:
            index of configuration as an int.

        """

        X = torch.tensor([X])
        probabilities = self.growth_function(X).squeeze()

        if self.sample:
            index = np.random.choice(
                [i for i in range(probabilities.shape[0])],
                size=1,
                replace=False,
                p=probabilities.detach().numpy(),
            )[0]
            print(index)
        else:
            max_value = 0
            index = 0
            for i in range(probabilities.shape[0]):
                if probabilities[i] > max_value:
                    max_value = probabilities[i]
                    index = i

        return int(index)


