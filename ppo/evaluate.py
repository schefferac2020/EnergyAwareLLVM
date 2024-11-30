import gym
import compiler_gym
from compiler_gym_wrapper import env_wrapper
import torch
from ppo_algo import PPO
from eval import Evaluation


# TODO: plot and visual representation
# TODO: change model name
model_name = "defaultNov25" # idek what this is
eval_benchmarks = ["cbench-v1/susan",
                   "cbench-v1/lame",
                   "cbench-v1/stringsearch",
                   "cbench-v1/blowfish",
                   "cbench-v1/sha",
                   "cbench-v1/gsm"]

env = env_wrapper(eval_benchmarks, max_episode_steps=200, steps_in_observation=True)

# TODO: Remember to change the reward in compiler_gym_wrapper and evaluate_baseline.

result, performances = Evaluation.evaluate_baseline(eval_benchmarks, opt_mode="-O3")
result, performances = Evaluation.evaluate_baseline(eval_benchmarks, opt_mode="-Oz")

print(f"O3 results: {result}, {performances}")
print(f"Oz results: {result}, {performances}")

# geo_maxima, geo_averages, _ = Evaluation.evaluate(eval_benchmarks, model_name=model_name, print_progress=True,
                                                                # max_trials_per_benchmark=10, max_time_per_benchmark=10)
