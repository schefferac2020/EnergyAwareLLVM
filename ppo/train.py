import gym
import compiler_gym
from compiler_gym_wrapper import env_wrapper

from ppo_algo import PPO


benchmarks_all=["adpcm",
            "blowfish",
            "crc32",
            "ghostscript",
            "ispell",
            "jpeg-d",
            "patricia",
            "rijndael",
            "stringsearch2",
            "susan",
            "tiff2rgba",
            "tiffmedian",
            "bitcount",
            "bzip2",
            "dijkstra",
            "gsm",
            "jpeg-c",
            "lame",
            "qsort",
            "sha",
            "stringsearch",
            "tiff2bw",
            "tiffdither"]
benchmarks_small = ["qsort", "bitcount"]
benchmarks = benchmarks_all

print("Training with these benchmarks:", " ".join(benchmarks), "\n--------------")

env = env_wrapper(benchmarks, max_episode_steps=200, steps_in_observation=True)

log_rate_in_seconds = 5 # 5 seconds

ppo_training = PPO(env, benchmarks, name="model_test_Nov27")
ppo_training.train(log_progress=True, progress_log_rate=log_rate_in_seconds)


env.close()
