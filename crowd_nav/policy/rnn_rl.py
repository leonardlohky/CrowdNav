import torch
import torch.nn as nn
import numpy as np
import logging
from crowd_nav.policy.cadrl import mlp
from crowd_nav.policy.multi_human_rl import MultiHumanRL


class ValueNetwork1(nn.Module):
    def __init__(self, input_dim, self_state_dim, mlp_dims, rnn_hidden_dim):
        super().__init__()
        self.self_state_dim = self_state_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.mlp = mlp(self_state_dim + rnn_hidden_dim, mlp_dims)
        self.rnn = nn.RNN(input_dim, rnn_hidden_dim, batch_first=True)

    def forward(self, state_input):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation
        """
        if isinstance(state_input, tuple):
            state, lengths = state_input
        else:
            state = state_input
            lengths = torch.IntTensor([state.size()[1]])
        self_state = state[:, 0, :self.self_state_dim]
        packed_sequence = torch.nn.utils.rnn.pack_padded_sequence(state, lengths, batch_first=True)
        _, hn = self.rnn(packed_sequence)
        hn = hn.squeeze(0)
        joint_state = torch.cat([self_state, hn], dim=1)
        value = self.mlp(joint_state)
        return value


class ValueNetwork2(nn.Module):
    def __init__(self, input_dim, self_state_dim, mlp1_dims, mlp_dims, rnn_hidden_dim):
        super().__init__()
        self.self_state_dim = self_state_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.mlp1 = mlp(input_dim, mlp1_dims)
        self.rnn = nn.RNN(mlp1_dims[-1], rnn_hidden_dim, batch_first=True)
        self.mlp = mlp(self_state_dim + rnn_hidden_dim, mlp_dims)

    def forward(self, state_input):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation
        :return:
        """
        if isinstance(state_input, tuple):
            state, lengths = state_input
        else:
            state = state_input
            lengths = torch.IntTensor([state.size()[1]])

        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]

        state = torch.reshape(state, (-1, size[2]))
        mlp1_output = self.mlp1(state)
        mlp1_output = torch.reshape(mlp1_output, (size[0], size[1], -1))
        packed_mlp1_output = torch.nn.utils.rnn.pack_padded_sequence(mlp1_output, lengths, batch_first=True)

        output, hn = self.rnn(packed_mlp1_output)
        hn = hn.squeeze(0)
        joint_state = torch.cat([self_state, hn], dim=1)
        value = self.mlp(joint_state)
        return value


class RnnRL(MultiHumanRL):
    def __init__(self):
        super().__init__()
        self.name = 'RNN-RL'
        self.with_interaction_module = None

    def configure(self, config):
        self.set_common_parameters(config)
        self.with_om = config.rnn_rl.with_om
        self.multiagent_training = config.rnn_rl.multiagent_training

        mlp_dims = config.rnn_rl.mlp2_dims
        global_state_dim = config.rnn_rl.global_state_dim
        with_interaction_module = config.rnn_rl.with_interaction_module
        if with_interaction_module:
            mlp1_dims = config.rnn_rl.mlp1_dims
            self.model = ValueNetwork2(self.input_dim(), self.self_state_dim, mlp1_dims, mlp_dims, global_state_dim)
        else:
            self.model = ValueNetwork1(self.input_dim(), self.self_state_dim, mlp_dims, global_state_dim)
        logging.info('Policy: {}RNN-RL {} pairwise interaction module'.format(
            'OM-' if self.with_om else '', 'w/' if with_interaction_module else 'w/o'))

    def predict(self, state):
        """
        Input state is the joint state of robot concatenated with the observable state of other agents
        To predict the best action, agent samples actions and propagates one step to see how good the next state is
        thus the reward function is needed
        """

        def dist(human):
            # sort human order by decreasing distance to the robot
            return np.linalg.norm(np.array(human.position) - np.array(state.robot_state.position))

        state.human_states = sorted(state.human_states, key=dist, reverse=True)
        return super().predict(state)