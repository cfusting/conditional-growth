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
        self.source_state = ScalarTwoLayer(self.state_dimensions, 1, num_neurons)

        # Loss to be defined in run.
        self.action_decoder_optimizer = Adam(self.action_decoder.parameters())

        self.source_loss = nn.MSELoss()
        self.source_optimizer = Adam(self.source_state.parameters())

    def get_feature_tensor(self, actions, start_state, end_state=None):
        if end_state:
            num_states = 2
        else:
            num_states = 1
        X = torch.zeros(
            (
                start_state.shape[0],
                self.num_actions + num_states * self.state_dimensions,
            )
        )
        X[:, self.num_actions : self.num_actions + self.state_dimensions] = start_state
        if end_state:
            X[:, self.num_actions + self.state_dimensions :] = end_state
        X[:, actions] = torch.ones(X.shape[0])
        return X

    def get_action_decoder_probability(self, start_state, end_state, actions=None):
        selected_actions = torch.full(start_state.shape[0], self.null_action)
        probabilities = torch.ones((start_state.shape[0], self.num_actions))
        for i in range(self.num_action_steps):
            if actions is None:
                X = self.get_feature_tensor(selected_actions, start_state, end_state)
            else:
                X = self.get_feature_tensor(actions[..., i], start_state, end_state)
            y = self.action_decoder(X)
            probabilities = probabilities * y
            selected_actions = torch.argmax(probabilities, dim=1)
        y, _ = torch.max(probabilities, dim=1)
        return y

    def get_source_action_state_probabilities_per_action(
        self, start_state, actions=None
    ):
        selected_actions = torch.full(start_state.shape[0], self.null_action)
        probabilities = torch.ones(
            (start_state.shape[0], self.num_actions, self.num_action_steps)
        )
        for i in range(self.num_action_steps):
            if actions is None:
                X = self.get_feature_tensor(selected_actions, start_state)
            else:
                X = self.get_feature_tensor(actions[..., i], start_state)
            probabilities[..., i] = self.source_action_state(X)

        return probabilities

    def get_source_action_state_probability(self, start_state, actions=None):
        probabilities = self.get_source_action_state_probabilities_per_action(
            start_state, actions
        )
        torch.prod(probabilities, dim=2)
        y, _ = torch.max(probabilities, dim=1)
        return y

    def sample_source_distribution(self, start_state):
        # (obs, start_state) -> (obs, actions, action_steps)
        probabilities = self.get_source_action_state_probabilities_per_action(
            start_state,
        )
        actions = np.zeros(probabilities.shape)
        for i in range(probabilities.shape[0]):
            for j in range(probabilities.shape[2]):
                sample_action = np.random.choice(
                    self.num_actions,
                    size=1,
                    p=probabilities[i, :, j]
                )
                actions[i, :, j] = sample_action
        return actions

    def optimize(self, start_state, end_state):
        action_decoder_probability = self.get_action_decoder_probability(
            start_state, end_state
        )

        # Maximize the log likelihood of the decoder..
        self.action_decoder_optimizer.zero_grad()
        action_decoder_loss = -torch.sum(torch.log(action_decoder_probability))
        action_decoder_loss.backward()
        self.action_decoder_optimizer.step()
        print(f"Action Decoder Loss: {action_decoder_loss}")

        # We use samples from the source distribution to optimize.
        action_sequence_sample = self.sample_source_distribution(start_state)

        # Minimize the loss of the approximation to the source distribution.
        self.source_optimizer.zero_grad()
        action_decoder_probability = self.get_action_decoder_probability(
            start_state, end_state, actions=action_sequence_sample
        )
        source_action_state_probability = self.get_source_action_state_probability(
            start_state, end_state, actions=action_sequence_sample
        )
        scalar = self.source_state(start_state)
        source_loss = self.explortation_loss(
            self.beta * torch.log(action_decoder_probability)
            - (torch.log(source_action_state_probability) + scalar)
        )
        source_loss.backward()
        self.source_optimizer.step()
        print(f"Source Loss: {source_loss}")

    def get_empowerment(self, start_state):
        return self.beta ** -1 * self.scalar(start_state)


class TwoLayer(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_neurons=256):
        super(TwoLayer, self).__init__()
        self.two_layer = nn.Sequential(
            nn.Linear(num_inputs, num_neurons),
            nn.ReLU(),
            nn.Linear(num_neurons, num_neurons),
            nn.ReLU(),
            nn.Linear(num_neurons, num_outputs),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.two_layer(x)


class ScalarTwoLayer(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_neurons=256):
        super(ScalarTwoLayer, self).__init__()
        self.two_layer = nn.Sequential(
            nn.Linear(num_inputs, num_neurons),
            nn.ReLU(),
            nn.Linear(num_neurons, num_neurons),
            nn.ReLU(),
            nn.Linear(num_neurons, num_outputs),
        )

    def forward(self, x):
        return self.two_layer(x)
