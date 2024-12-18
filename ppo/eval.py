
import random
import time
import subprocess
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from compiler_gym_wrapper import env_wrapper
from actor_critic_network import actor_critic_network
from energy_estimate import estimate_program_energy

class Evaluation:
    def geom_mean(input_list: List):
        output_list = np.array(input_list)
        return output_list.prod()**(1/len(output_list))
    
    def arith_mean(input_list: List):
        output_list = np.array(input_list)
        return output_list.sum() / len(output_list)
    
    def evaluate_baseline(benchmarks, print_progress=True, opt_mode='-Oz', reward_type="SIZE"):
        assert(reward_type=="BOTH" or reward_type=="NRG" or reward_type=="SIZE")
        
        if print_progress:
            print("Evaluating {0}:".format(opt_mode))

        episode_len = 300
        performances=[]

        for benchmark in benchmarks:
            # Not stepping with the env
            env = env_wrapper([benchmark], max_episode_steps=episode_len, steps_in_observation=True)
            obs = env.reset()
            if opt_mode == '-Oz':
                bitmode = "IrInstructionCountOz"
            elif opt_mode == '-O3':
                bitmode = "IrInstructionCountO3"
            else:
                raise NotImplementedError('Invalid Optimization level')

            initial_energy = env.initial_energy
            initial_bitcode = env.env.observation["IrInstructionCount"]

            bitcode_reward = (initial_bitcode - env.env.observation[bitmode]) / initial_bitcode

            input_bitcode_file = env.env.observation["BitcodeFile"]
            optimized_bitcode_file = os.path.join(os.path.dirname(input_bitcode_file), "opt.bc")
            output_asm_file = os.path.join(os.path.dirname(input_bitcode_file), "asm.s")

            # Apply opt_mode to the file
            subprocess.run([
                'opt',
                opt_mode,
                input_bitcode_file,
                '-o',
                optimized_bitcode_file
            ], check=True, capture_output=True)
            
            # Compile bitcode to ASM using clang
            '''clang -S --target=arm-none-eabi -march=armv7e-m -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -nostdlib ir_code.bc -o file.s'''
            subprocess.run([
                'clang',
                '-S',
                '--target=arm-none-eabi',
                '-march=armv7e-m',
                '-mcpu=cortex-m4',
                '-mfpu=fpv4-sp-d16',
                '-mfloat-abi=hard',
                '-mthumb',
                '-nostdlib',
                optimized_bitcode_file,
                '-o',
                output_asm_file
            ], check=True, capture_output=True)
            
            with open(output_asm_file, "r") as file:
                # Read the entire file content as a string
                asm_code = file.read()
            
            total_energy = estimate_program_energy(asm_code).total_energy
            nrg_reward = (initial_energy - total_energy)/ initial_energy

            # TODO: change rewards if needed
            if reward_type == "SIZE":
                reward = bitcode_reward
            elif reward_type == "NRG":
                reward = nrg_reward
            elif reward_type == "BOTH":
                reward = 0.5*bitcode_reward + 0.5*nrg_reward
            else:
                raise AssertionError
            
            performances.append(reward)

            env.close()
        
        # Switching to arithmetic mean because of negatives
        return Evaluation.arith_mean(performances), performances

    def evaluate(benchmarks, model_name, is_random_policy=False, print_progress=True,
                 max_trials_per_benchmark=10, max_time_per_benchmark=10 * 1, additional_steps_for_max=0, reward_type="SIZE"):
        if print_progress:
            print("Evaluating {0}:".format(model_name))
            
        episode_len = 200
        performances=[]
        
        for benchmark in benchmarks:
            env = env_wrapper([benchmark], max_episode_steps=episode_len, steps_in_observation=True, reward_type=reward_type)
            long_env = env_wrapper([benchmark], max_episode_steps=episode_len + additional_steps_for_max,
                                    steps_in_observation=True, reward_type=reward_type)
            
            if not is_random_policy:
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
                    if is_random_policy:
                        action = env.action_space.sample()
                    else:
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
                if is_random_policy:
                    action = env.action_space.sample()
                else:
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
            print("Arithmetric mean of maxima: {0}".format(Evaluation.arith_mean(performances[:, 0])))
            print("Arithmetric mean of averages: {0}".format(Evaluation.arith_mean(performances[:, 1])))
        
        # Switching to arithmetic mean because of negatives
        return Evaluation.arith_mean(performances[:, 0]), Evaluation.arith_mean(performances[:, 1]), performances

                    