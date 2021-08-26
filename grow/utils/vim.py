import torch
from torch import nn
from torch.optim import Adam
import numpy as np


class VariationInformationMaximization:
    def __init__(
        self,
        state_dimensions,
        num_actions,
        num_action_steps,
        null_action,
        num_neurons=256,
        beta=1,
    ):
        self.state_dimensions = state_dimensions
        self.num_action_steps = num_action_steps
        self.null_action = null_action
        self.num_actions = num_actions
        self.beta = beta

        self.action_decoder = TwoLayer(
            self.num_actions + 2 * state_dimensions, num_actions, num_neurons
        )
        self.source_action_state = TwoLayer(
            self.num_actions + state_dimensions, num_actions, num_neurons
        )
        self.source_state = TwoLayer(self.state_dimensions, 1, num_neurons)

        # Loss to be defined in run.
        self.action_decoder_optimizer = Adam(self.action_decoder.parameters())

        self.source_loss = nn.MSELoss()
        self.source_optimizer = Adam(
            list(self.action_decoder.parameters()) + list(self.scalar)
        )

    def get_action_decoder_probability(self, start_state, end_state, actions=None):
        # One hot encoding of actions concatenated by start and end states.
        X = np.zeros(
            (
                start_state.shape[0],
                self.num_actions + 2 * self.state_dimensions,
            )
        )
        X[:, self.num_actions : self.null_actions + self.state_dimensions] = start_state
        X[:, self.null_actions + self.state_dimensions :] = end_state

        selected_action = self.null_action
        probabilities = np.ones((X.shape[0], self.num_actions))
        for i in self.num_action_steps:
            if actions is None:
                X[:, selected_action] = np.ones(X.shape[0])
            else:
                X[:, actions[i]] = np.ones(X.shape[0])
            probabilities *= self.action_decoder(X)
            selected_action = np.argmax(probabilities, axis=1)

        return np.max(probabilities, axis=1)

    def get_source_action_state_probabilities(self, start_state, actions):
        X = np.zeros(
            (
                start_state.shape[0],
                self.num_actions + self.state_dimensions,
            )
        )
        X[:, self.num_actions : self.null_actions + self.state_dimensions] = start_state

        probabilities = np.ones((X.shape[0], self.num_actions))
        for i in self.num_action_steps:
            X[:, actions[i]] = np.ones(X.shape[0])
            probabilities *= self.source_action_state(X)

        return probabilities

    def sample_source_distribution(self, start_state):
        probabilities = self.get_source_action_state_probabilities(start_state)
        sample_actions = list(
            np.random.choice(
                self.num_actions,
                size=self.num_action_steps,
                replace=True,
                p=probabilities,
            )
        )
        sample_actions.insert(0, self.null_action)
        return sample_actions

    def optimize(self, start_state, end_state):
        action_decoder_probability = self.get_action_decoder_probability(
            start_state, end_state
        )

        # Maximize the log likelihood of the decoder..
        self.action_decoder_optimizer.zero_grads()
        action_decoder_loss = -torch.sum(torch.log(action_decoder_probability))
        action_decoder_loss.backwards()
        self.action_decoder_optimizer.step()

        # We use samples from the source distribution to optimize.
        action_sequence_sample = self.sample_source_distribution(start_state)

        # Minimize the loss of the approximation to the source distribution.
        self.source_optimizer.zero_grads()
        action_decoder_probability = self.get_action_decoder_probability(
            start_state, end_state, actions=action_sequence_sample
        )
        source_state_probabilities = (
            self.get_source_action_state_probabilities(
                start_state, end_state, actions=action_sequence_sample
            )
        )
        scalar = self.source_state(start_state)
        source_loss = self.explortation_loss(
            self.beta * torch.log(action_decoder_probability)
            - (torch.log(source_state_probabilities) + scalar)
        )
        source_loss.backwards()
        self.source_optimizer.step()

    def get_empowerment(self, start_state):
        return self.beta**-1 * self.scalar(start_state)


class TwoLayer(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_neurons=256):
        super(TwoLayer, self).__init()
        self.two_layer = nn.Sequential(
            nn.Linear(num_inputs, num_neurons),
            nn.ReLU(),
            nn.Linear(num_neurons, num_neurons),
            nn.ReLU(),
            nn.Linear(num_neurons, num_outputs),
            nn.Softmax(num_outputs),
        )

    def forward(self, x):
        return self.two_layer(x)


class ScalarTwoLayer(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_neurons=256):
        super(TwoLayer, self).__init()
        self.two_layer = nn.Sequential(
            nn.Linear(num_inputs, num_neurons),
            nn.ReLU(),
            nn.Linear(num_neurons, num_neurons),
            nn.ReLU(),
            nn.Linear(num_neurons, num_outputs),
        )

    def forward(self, x):
        return self.two_layer(x)
