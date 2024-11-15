import gym
import compiler_gym
from compiler_gym_wrapper import env_wrapper


benchmarks=["adpcm",
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
print("Training with these benchmarks:", " ".join(benchmarks), "\n--------------")

env = env_wrapper(benchmarks, max_episode_steps=200, steps_in_observation=True)

env.close()
