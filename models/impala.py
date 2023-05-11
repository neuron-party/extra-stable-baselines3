import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

    def forward(self, x):
        inputs = x
        x = torch.relu(x)
        x = self.conv0(x)
        x = torch.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super(ConvSequence, self).__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3, padding=1)
        self.max_pool2d = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool2d(x)
        x = self.res_block0(x)
        x = self.res_block1(x)
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return self._out_channels, (h + 1) // 2, (w + 1) // 2

class SB_Impala(nn.Module): # procgen and atari
    '''
    PPO Requirements for stable baselines 3 compatability:
        forward() needs to return action, value, log_prob, takes in a dictionary
        need an evaluate_actions() method that takes states and actions and returns the log prob, value, and entropy
        need a predict_values() method that takes states and outputs the value estimate
    '''
    def __init__(self, obs_space, num_outputs, lr):
        super().__init__()

        h, w, c = obs_space.shape
        shape = (c, h, w)

        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
            
        self.conv_seqs = nn.ModuleList(conv_seqs)
        self.hidden_fc = nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256)
        self.logits_fc = nn.Linear(in_features=256, out_features=num_outputs)
        self.value_fc = nn.Linear(in_features=256, out_features=1)
        # self.aux_value_fc = nn.Linear(in_features=256, out_features=1)
        
        # Initialize weights of logits_fc
        nn.init.orthogonal_(self.logits_fc.weight, gain=0.01)
        nn.init.zeros_(self.logits_fc.bias)
        
        # stable baselines keeps the optimizer inside the nn Module class for some reason
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
    def preprocess(self, obs):
        '''
        I think stable baselines reshapes the environment arrays into the correct order for us already
        '''
        obs = obs / 255.0
        # obs = obs.permute(0, 3, 1, 2)
        return obs

    def forward(self, obs):
        obs = obs['rgb']
        assert obs.ndim == 4
        x = self.preprocess(obs)
        
        for conv_seq in self.conv_seqs:
            x = conv_seq(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(x)
        x = self.hidden_fc(x)
        x = torch.relu(x)
        logits = self.logits_fc(x)
        dist = torch.distributions.Categorical(logits=logits)
        value = self.value_fc(x)
        
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, value, log_prob
        # return dist, value
        
    def evaluate_actions(self, obs, actions):
        obs = obs['rgb']
        assert obs.ndim == 4
        x = self.preprocess(obs)
        
        for conv_seq in self.conv_seqs:
            x = conv_seq(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(x)
        x = self.hidden_fc(x)
        x = torch.relu(x)
        logits = self.logits_fc(x)
        dist = torch.distributions.Categorical(logits=logits)
        
        value = self.value_fc(x)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return value, log_probs, entropy
    
    def predict_values(self, obs):
        obs = obs['rgb']
        assert obs.ndim == 4
        x = self.preprocess(obs)
        
        for conv_seq in self.conv_seqs:
            x = conv_seq(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(x)
        x = self.hidden_fc(x)
        x = torch.relu(x)
        value = self.value_fc(x)
        return value
    
    def generate_dataset(self, obs):
        assert obs.ndim == 4
        x = self.preprocess(obs)
        
        for conv_seq in self.conv_seqs:
            x = conv_seq(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(x)
        x = self.hidden_fc(x)
        x = torch.relu(x)
        logits = self.logits_fc(x)
        dist = torch.distributions.Categorical(logits=logits)
        value = self.value_fc(x)
        
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, value, logits
        # return dist, value
        
    def supervised_pred(self, obs):
        assert obs.ndim == 4
        x = self.preprocess(obs)
        
        for conv_seq in self.conv_seqs:
            x = conv_seq(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(x)
        x = self.hidden_fc(x)
        x = torch.relu(x)
        logits = self.logits_fc(x)
        value = self.value_fc(x)
        return value, logits
    
    
class SB_Impala_PPG(nn.Module): # procgen and atari
    '''
    PPO Requirements for stable baselines 3 compatability:
        forward() needs to return action, value, log_prob, takes in a dictionary
        need an evaluate_actions() method that takes states and actions and returns the log prob, value, and entropy
        need a predict_values() method that takes states and outputs the value estimate
    '''
    def __init__(self, obs_space, num_outputs, lr):
        super().__init__()

        h, w, c = obs_space.shape
        shape = (c, h, w)

        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
            
        self.conv_seqs = nn.ModuleList(conv_seqs)
        self.hidden_fc = nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256)
        self.logits_fc = nn.Linear(in_features=256, out_features=num_outputs)
        self.value_fc = nn.Linear(in_features=256, out_features=1)
        self.aux_value_fc = nn.Linear(in_features=256, out_features=1)
        
        # Initialize weights of logits_fc
        nn.init.orthogonal_(self.logits_fc.weight, gain=0.01)
        nn.init.zeros_(self.logits_fc.bias)
        
        # stable baselines keeps the optimizer inside the nn Module class for some reason
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
    def preprocess(self, obs):
        '''
        I think stable baselines reshapes the environment arrays into the correct order for us already
        '''
        obs = obs / 255.0
        # obs = obs.permute(0, 3, 1, 2)
        return obs

    def forward(self, obs):
        obs = obs['rgb']
        assert obs.ndim == 4
        x = self.preprocess(obs)
        
        for conv_seq in self.conv_seqs:
            x = conv_seq(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(x)
        x = self.hidden_fc(x)
        x = torch.relu(x)
        logits = self.logits_fc(x)
        dist = torch.distributions.Categorical(logits=logits)
        value = self.value_fc(x)
        
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, value, log_prob, logits
        # return dist, value
        
    def evaluate_actions(self, obs, actions):
        obs = obs['rgb']
        assert obs.ndim == 4
        x = self.preprocess(obs)
        
        for conv_seq in self.conv_seqs:
            x = conv_seq(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(x)
        x = self.hidden_fc(x)
        x = torch.relu(x)
        logits = self.logits_fc(x)
        dist = torch.distributions.Categorical(logits=logits)
        
        value = self.value_fc(x)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return value, log_probs, entropy
    
    def predict_values(self, obs):
        obs = obs['rgb']
        assert obs.ndim == 4
        x = self.preprocess(obs)
        
        for conv_seq in self.conv_seqs:
            x = conv_seq(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(x)
        x = self.hidden_fc(x)
        x = torch.relu(x)
        value = self.value_fc(x)
        return value
    
    def auxiliary_phase(self, obs):
        obs = obs['rgb']
        assert obs.ndim == 4
        x = self.preprocess(obs)
        
        for conv_seq in self.conv_seqs:
            x = conv_seq(x)
        
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(x)
        x = self.hidden_fc(x)
        x = torch.relu(x)
        
        logits = self.logits_fc(x)
        values = self.value_fc(x)
        aux_values = self.aux_value_fc(x)
        return values, aux_values, logits