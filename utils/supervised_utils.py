import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SupervisedFinetuning:
    def __init__(
        self, 
        observations, 
        values, 
        policy_logits,
        batch_size,
        minibatch_size,
        device
    ):
        assert len(observations) == len(values) == len(policy_logits)
        observations, values, policy_logits = np.array(observations), np.array(values), np.array(policy_logits)
        observations, values, policy_logits = torch.tensor(observations), torch.tensor(values), torch.tensor(policy_logits)
        observations = observations.flatten(start_dim=0, end_dim=1)
        values = values.flatten(start_dim=0, end_dim=1)
        policy_logits = policy_logits.flatten(start_dim=0, end_dim=1)
        
        self.observations = observations
        self.values = values
        self.policy_logits = policy_logits
        self.minibatch_size = minibatch_size
        self.batch_size = batch_size
        self.device = device
        
    def get(self):
        start_idx = 0
        indices = np.random.permutation(len(self.observations))
        while start_idx < self.batch_size:
            yield self._get_samples(indices[start_idx : start_idx + self.minibatch_size])
            start_idx += self.minibatch_size
            
    def _get_samples(self, indices):
        observations = self.observations[indices].to(self.device)
        values = self.values[indices].to(self.device)
        policy_logits = self.policy_logits[indices].to(self.device)
        return observations, values, policy_logits