'''
PPO training algorithm
based on: https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py
'''

import random
import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from compiler_gym_wrapper import env_wrapper
from actor_critic_network import actor_critic_network
from eval import Evaluation
from tqdm import tqdm

class RolloutBuffer:
    def __init__(self):
        self.actions=[]
        self.states=[]
        self.logprobs=[]
        self.rewards=[]
        # self.state_values = []
        self.dones=[]
        
    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        # del self.state_values[:]
        del self.dones[:]
        
    def add_data(self, state, action, logprob, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        # self.state_values.append(state_value)
        self.dones.append(done)

class PPO:
    def __init__(self, env, benchmarks, name="default", EPOCHS=80, eps_clip=0.2, loss_mse_fac=0.5, 
                 loss_entr_fac=0.01, learning_rate=5e-4, trajectories_until_update=20, gamma=0.99):
        self.EPOCHS = EPOCHS
        self.env = env
        self.name = name
        self.eps_clip = eps_clip
        self.loss_mse_fac = loss_mse_fac
        self.loss_entr_fac = loss_entr_fac
        self.learning_rate = learning_rate
        self.trajectories_until_update = trajectories_until_update
        self.gamma = gamma
        self.benchmarks = benchmarks
        
        self.actor_critic_net = actor_critic_network(self.env.observation_space.shape[0], self.env.action_space.n)
        self.actor_critic_net_old = actor_critic_network(self.env.observation_space.shape[0], self.env.action_space.n)
        self.buffer = RolloutBuffer()
        self.optimizer = optim.Adam(self.actor_critic_net.parameters(), lr=self.learning_rate)
        self.mse_loss = nn.MSELoss()
        
    def train(self, log_progress=False, progress_log_rate = 60*30, checkpoint_name=None):
        if checkpoint_name is not None:
            self.actor_critic_net.load_state_dict(torch.load("models/{0}.model".format(checkpoint_name)))
            print(f"Loaded checkpoint on {checkpoint_name}")
        
        print("Training Has Started...")
        start_time = time.time()
        last_checkpoint = time.time()
        
        rewards = []
        
        while True:
            print(f"Collecting {self.trajectories_until_update} episodes of data...")
            self.collect_trajectories(self.trajectories_until_update)
            print("Updating the Policy...")
            self.update()
            
            if log_progress and (time.time() - last_checkpoint > progress_log_rate):
                torch.save(self.actor_critic_net.state_dict(), f"models/{self.name}.model")
                geo_maxima, geo_averages, _ = Evaluation.evaluate(self.benchmarks, model_name=self.name, print_progress=False,
                                                                max_trials_per_benchmark=10, max_time_per_benchmark=10)

                rewards.append(geo_averages)
                print(F"Geo of averages: {geo_averages}")
                plt.clf()
                plt.plot(rewards)
                plt.savefig(f"models/{self.name}.png")
                
                last_checkpoint = time.time()
            
            self.env.switch_benchmark()
        
    def collect_trajectories(self, num_traj):
        for _ in range(num_traj):
            obs = self.env.reset()
            done = False
            while not done:
                obs = torch.tensor(obs).float() # on CPU
                action, log_prob = self.actor_critic_net.act(obs)
                new_obs, reward, done, info = self.env.step(action.item())
                self.buffer.add_data(obs, action, log_prob, reward, done)
                obs = new_obs
    
    # TODO: below is the other update that I was working on. IDK if it was the real cause of the issue.
    # def update(self):
    #     # Calculated the rollout discounted rewards:
    #     rewards = []
    #     discounted_reward = 0
    #     for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.dones)):
    #         if is_terminal:
    #             discounted_reward
    #         discounted_reward = reward + (self.gamma*discounted_reward)
    #         rewards.insert(0, discounted_reward) # add to front
    
    #     # Normalize
    #     rewards = torch.tensor(rewards, dtype=torch.float32)
    #     rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

    #     # Convert buffer lists to tensors
    #     old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach()
    #     old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach()
    #     old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach()
    #     old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach()
        
    #     # calculate advantages
    #     advantages = rewards.detach() - old_state_values.detach()
        
    #     # Optimize for K epochs
    #     for _ in tqdm(range(self.EPOCHS)):
    #         batch_size = len(old_states)
    #         sampled_indices = torch.tensor(random.sample(range(len(old_states)), batch_size))
    #         sampled_states = torch.index_select(old_states, 0, sampled_indices)
    #         sampled_actions = torch.index_select(old_actions, 0, sampled_indices)
    #         sampled_logprobs = torch.index_select(old_logprobs, 0, sampled_indices)
    #         sampled_advantages = torch.index_select(advantages, 0, sampled_indices)
    #         sampled_state_values = torch.index_select(old_state_values, 0, sampled_indices) # TODO: do we need this? 
    #         sampled_rewards = torch.index_select(rewards, 0, sampled_indices)
            
    #         logprobs, state_values, dist_entropies = self.actor_critic_net.evaluate(sampled_states, sampled_actions)
    #         state_values = torch.squeeze(state_values)
            
    #         ratios = torch.exp(logprobs - sampled_logprobs.detach())
            
    #         surr1 = ratios * sampled_advantages
    #         surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * sampled_advantages
            
    #         loss = -torch.min(surr1, surr2) + self.loss_mse_fac * self.mse_loss(state_values, sampled_rewards) \
    #                - self.loss_entr_fac*dist_entropies

    #         self.optimizer.zero_grad()
    #         loss.mean().backward()
    #         self.optimizer.step()
    #     self.buffer.clear()
    
    def update(self):
        # Calc Advantages
        xpctd_returns = []
        current_xpctd_return = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.dones)):
            if is_terminal:
                current_xpctd_return = 0
            current_xpctd_return = reward + current_xpctd_return
            xpctd_returns.insert(0, current_xpctd_return)
        xpctd_returns = torch.tensor(xpctd_returns)
        xpctd_returns = (xpctd_returns - xpctd_returns.mean()) / (xpctd_returns.std() + 1e-7)

        rollouts_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach()
        rollouts_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach()
        rollouts_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach()

        # Perform update
        for _ in range(self.EPOCHS):
            batch_size = len(rollouts_states)

            sampled_indices = torch.tensor(random.sample(range(len(rollouts_states)), batch_size))
            sampled_states = torch.index_select(rollouts_states, 0, sampled_indices)
            sampled_actions = torch.index_select(rollouts_actions, 0, sampled_indices)
            sampled_logprobs = torch.index_select(rollouts_logprobs, 0, sampled_indices)
            sampled_xpctd_returns = torch.index_select(xpctd_returns, 0, sampled_indices)

            logprobs, state_values, dist_entropies = self.actor_critic_net.evaluate(sampled_states, sampled_actions)

            state_values = torch.squeeze(state_values)

            prob_ratios = torch.exp(logprobs - sampled_logprobs)

            advantages = (sampled_xpctd_returns - state_values).detach()

            surr1 = prob_ratios * advantages
            surr2 = torch.clamp(prob_ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + self.loss_mse_fac * self.mse_loss(state_values, sampled_xpctd_returns) \
                   - self.loss_entr_fac * dist_entropies

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.buffer.clear()