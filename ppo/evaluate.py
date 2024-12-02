import gym
import compiler_gym
from compiler_gym_wrapper import env_wrapper
import torch
from ppo_algo import PPO
from eval import Evaluation


# TODO: plot and visual representation
# TODO: change model name
model_name = "defaultDec01bitcode2" # idek what this is
eval_benchmarks = ["cbench-v1/susan",
                   "cbench-v1/lame",
                   "cbench-v1/stringsearch",
                   "cbench-v1/blowfish",
                   "cbench-v1/sha",
                   "cbench-v1/gsm"]

env = env_wrapper(eval_benchmarks, max_episode_steps=200, steps_in_observation=True)

geo_maxima, geo_averages = Evaluation.evaluate(eval_benchmarks, model_name=model_name, print_progress=True,
                                                                max_trials_per_benchmark=10, max_time_per_benchmark=10)
