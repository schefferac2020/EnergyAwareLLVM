import gym
import compiler_gym
from compiler_gym_wrapper import env_wrapper

from ppo_algo import PPO


benchmarks_all=["cbench-v1/adpcm",
            "cbench-v1/blowfish",
            "bench-v1c/crc32",
            "cbench-v1/ghostscript",
            "cbench-v1/ispell",
            "cbench-v1/jpeg-d",
            "cbench-v1/patricia",
            "cbench-v1/rijndael",
            "cbench-v1/stringsearch2",
            "cbench-v1/susan",
            "cbench-v1/tiff2rgba",
            "cbench-v1/tiffmedian",
            "cbench-v1/bitcount",
            "cbench-v1/bzip2",
            "cbench-v1/dijkstra",
            "cbench-v1/gsm",
            "cbench-v1/jpeg-c",
            "cbench-v1/lame",
            "cbench-v1/qsort",
            "cbench-v1/sha",
            "cbench-v1/stringsearch",
            "cbench-v1/tiff2bw",
            "cbench-v1/tiffdither"]
benchmarks_small = ["adpcm.decode"]

benchmarks_train = ["cbench-v1/adpcm",
            "cbench-v1/crc32",
            "cbench-v1/ghostscript",
            "cbench-v1/ispell",
            "cbench-v1/jpeg-d",
            "cbench-v1/patricia",
            "cbench-v1/rijndael",
            "cbench-v1/stringsearch2",
            "cbench-v1/tiff2rgba",
            "cbench-v1/tiffmedian",
            "cbench-v1/bitcount",
            "cbench-v1/bzip2",
            "cbench-v1/dijkstra",
            "cbench-v1/jpeg-c",
            "cbench-v1/qsort",
            "cbench-v1/tiff2bw",
            "cbench-v1/tiffdither"]

benchmarks = benchmarks_train

print("Training with these benchmarks:", " ".join(benchmarks), "\n--------------")

env = env_wrapper(benchmarks, max_episode_steps=200, steps_in_observation=True)

log_rate_in_seconds = 600 # 5 seconds

#TODO: Change name
ppo_training = PPO(env, benchmarks, name="energy_reward_model_Nov30")
ppo_training.train(log_progress=True, progress_log_rate=log_rate_in_seconds, checkpoint_name="energy_reward_model_Nov30")


env.close()

# started training at 10:25 am
# still training at 12:36 am... 14 hours, wtf
