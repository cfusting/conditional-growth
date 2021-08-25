from torch import nn
import numpy as np


class VariationInformationMaximization:
    def __init__(
        self,
        state_dimensions,
        num_actions,
        num_action_steps,
        null_action,
        num_neurons=256,
    ):
        self.state_dimensions = state_dimensions
        self.num_action_steps = num_action_steps
        self.null_action = null_action
        self.num_actions = num_actions
        self.action_decoder = TwoLayer(
            self.num_actions + 2 * state_dimensions, num_actions, num_neurons
        )
        self.exploration_action_state = TwoLayer(
            self.num_actions + state_dimensions, num_actions, num_neurons
        )
        self.exploration_state = TwoLayer(self.state_dimensions, 1, num_neurons)

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

    def get_exploration_action_state_probabilities(self, start_state, actions):
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
            probabilities *= self.exploration_action_state(X)

        return probabilities

    def sample_exploration_distribution(self, start_state):
        probabilities = self.get_exploration_action_state_probabilities(start_state)
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
