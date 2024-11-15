import torch
import torch.nn as nn
from torch.distributions import Categorical

class actor_critic_network(nn.Module):
    def __init__(self, num_observations, num_actions):
        super(actor_critic_network, self).__init__()
        
        self.num_actions = num_actions
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(num_observations, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        
        self.action_layer = nn.Linear(128, self.num_actions)
        self.value_layer = nn.Linear(128, 1)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        actions = self.action_layer(features)
        action_probs = nn.functional.softmax(actions, dim=1)
        v = self.value_layer(features)
        return torch.cat([action_probs, v], dim=1)
        
    def get_state_value(self, x):
        f = self.feature_extractor(x)
        v = self.value_layer(f)
        return v
    
    def evaluate(self, state, action):
        output = self.forward(state)
        
        action_probs = torch.index_select(output, 1, torch.tensor(range(self.num_actions)))
        dist = Categorical(action_probs)
        
        logprob_action = dist.log_prob(action)
        
        state_val = torch.index_select(output, 1, torch.tensor([self.output_dim]))
        dist_entropy = dist.entropy()
        
        return logprob_action, state_val, dist_entropy
    
    def get_action_probs(self, x):
        features = self.feature_extractor(x)
        action_vals = self.action_layer(features)
        action_probs = nn.functional.softmax(action_vals, dim=1) # this could be wrong right? 
        return action_probs
    
    def act(self, x):
        # action_probs = self.get_action_probs(x)
        # dist = Categorical(action_probs)
        # chosen_action = dist.sample()
        # logprob_action = dist.log_prob(chosen_action)
        
        
        output = self.forward(x)
        
        action_probs = torch.index_select(output, 1, torch.tensor(range(self.num_actions)))
        dist = Categorical(action_probs)
        chosen_action = dist.sample()
        
        logprob_action = dist.log_prob(chosen_action)
        
        state_val = torch.index_select(output, 1, torch.tensor([self.output_dim]))
        
        return chosen_action.detach(), logprob_action.detach(), state_val.detach()
        
        