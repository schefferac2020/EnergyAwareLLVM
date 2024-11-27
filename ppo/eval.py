
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
    def geom_mean(input_list: List):
        output_list = np.array(input_list)
        return output_list.prod()**(1/len(output_list))
    
    def evaluate(benchmarks, model_name, print_progress=True,
                 max_trials_per_benchmark=10, max_time_per_benchmark=10 * 1, additional_steps_for_max=0):
        if print_progress:
            print("Evaluating {0}:".format(model_name))
            
        episode_len = 200
        performances=[]
        
        for benchmark in benchmarks:
            env = env_wrapper([benchmark], max_episode_steps=episode_len, steps_in_observation=True)
            long_env = env_wrapper([benchmark], max_episode_steps=episode_len + additional_steps_for_max,
                                    steps_in_observation=True)
            
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
               
            obs = long_env.reset()
            done = False
            cum_of_max = []
            for action in best_action_sequence:
                _, reward, done, _ = long_env.step(action)
                cum_of_max.append(reward + (cum_of_max[-1] if len(cum_of_max) > 0 else 0))
            while not done:
                action = model.act(torch.tensor(obs).float())[0].item()
                obs, reward, done, _ = long_env.step(action)
                cum_of_max.append(reward + (cum_of_max[-1] if len(cum_of_max) > 0 else 0))
            
            if max(cum_of_max) > max_reward:
                print("Improvement! {0} -> {1}".format(max(cum_of_max), max_reward))
            
            performance = [max(cum_of_max), total_reward / trials, trials]
            performances.append(performance)
            
            if print_progress:
                print("Environment: {0}. Found max of {1} and average of {2} in {3} trials.".format(benchmark,
                                                                                                    performance[0],
                                                                                                    performance[1],
                                                                                                    performance[2]))
            
            env.close()
            long_env.close()
                
        performances = np.array(performances)
        if print_progress:
            print("Geometric mean of maxima: {0}".format(Evaluation.geom_mean(performances[:, 0])))
            print("Geometric mean of averages: {0}".format(Evaluation.geom_mean(performances[:, 1])))
            
        return Evaluation.geom_mean(performances[:, 0]), Evaluation.geom_mean(performances[:, 1])

                    