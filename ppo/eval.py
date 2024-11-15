
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

class Evaluation:
    def geom_mean(self, input_list: List):
        output_list = np.array(input_list)
        return output_list.prod()**(1/len(output_list))
    
    def evaluate(self, benchmarks, model_name, print_progress=True,
                 max_trials_per_benchmark=10, max_time_per_benchmark=10 * 1):
        if print_progress:
            print("Evaluating {0}:".format(model_name))
            
        episode_len = 200
        performances=[]
        
        for benchmark in benchmarks:
            env = env_wrapper([benchmark], max_episode_steps=episode_len, steps_in_observation=True)
            
            model = actor_critic_network(env.observation_space.shape[0], env.action_space.n)
            model.load_state_dict(torch.load(f"models/{model_name}.model"))
            
            max_reward = 0
            best_action_sequence = []
            total_reward = 0
            trials = 0
            start_time = time.time()
            while trials < max_trials_per_benchmark and (time.time() - start_time) < max_time_per_benchmark:
                trials += 1
                obs = env.reset()
                done = False
                action_sequence = []
                cum_rewards = []
                running_reward = 0
                while not done:
                    action = model.act(torch.tensor(obs).float())[0].item()
                    action_sequence.append(action)
                    obs, reward, done, info = env.step(action)
                    running_reward += reward
                    cum_rewards.append(running_reward)
                
                # finished
                if max(cum_rewards) > max_reward:
                    max_reward = max(cum_rewards)
                    best_action_sequence = action_sequence
                total_reward += max(cum_rewards)
                
                performance = [max_reward, total_reward / trials, trials]
                performances.append(performance)
                
                if print_progress:
                    print(f"Environment: {benchmark}. Found max of {max_reward} \
                          and average of {total_reward / trials} in {trials} trials.")
                
                env.close()
                
            performances = np.array(performances)
            if print_progress:
                print("Geometric mean of maxima: {0}".format(Evaluation.geom_mean(performances[:, 0])))
                print("Geometric mean of averages: {0}".format(Evaluation.geom_mean(performances[:, 1])))
            
            return Evaluation.geom_mean(performances[:, 0]), Evaluation.geom_mean(performances[:, 1])

                    