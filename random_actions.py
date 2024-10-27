'''
An example script that loads an LLVM environment and takes random actions for 100 timesteps
'''

import gym
import compiler_gym

env = gym.make("llvm-autophase-ic-v0")

env.reset(benchmark="benchmark://npb-v0/50")

episode_reward = 0

for i in range(1, 101):

    observation, reward, done, info = env.step(env.action_space.sample())

    if done:
        break

    episode_reward += reward

    print(f"Step {i}, quality={episode_reward:.3%}")