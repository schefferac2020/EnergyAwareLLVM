import gym
import numpy as np
import random
from compiler_gym.envs.llvm import make_benchmark
import subprocess
import os
from energy_estimate import estimate_program_energy


class env_wrapper(gym.Env):
    """
    benchmarks: Names of programs in cbench-v1 which are to be cycled through in training
    max_episode_steps: If specified: Number of maximum steps an episode can last up to. Otherwise no episode limit
    patience: If specified:  Number of consecutive steps with reward 0 for episode to terminate.
    allowed_actions: If specified: Subset of action space given as integers.
    """

    def __init__(self, benchmarks, max_episode_steps=None, steps_in_observation=False, patience=None,
                 allowed_actions=None):
        self.env = gym.make("llvm-autophase-ic-v0", benchmark="cbench-v1/{0}".format(benchmarks[0]))
        # self.env = gym.make("llvm-v0", benchmark="cbench-v1/{0}".format(benchmarks[0]))
        self.benchmarks = benchmarks

        # patience
        self.patience = patience
        self.fifo = []

        # Observation space
        self.limited_time = max_episode_steps is not None
        if self.limited_time:
            self.max_steps = max_episode_steps
            self.elapsed_steps = 0
            self.steps_in_observation = steps_in_observation
            if steps_in_observation:
                self.observation_space = gym.spaces.Box(0, 9223372036854775807,
                                                        np.shape(list(range(1 + self.env.observation_space.shape[0]))),
                                                        dtype=np.int64)
            else:
                self.observation_space = self.env.observation_space

        else:
            self.observation_space = self.env.observation_space

        # Action space
        self.limited_action_set = allowed_actions is not None
        self.action_mapping = dict()
        if self.limited_action_set:
            for idx, action in enumerate(allowed_actions):
                self.action_mapping[idx] = action
            self.action_space = gym.spaces.Discrete(
                self.env.action_space.n if allowed_actions is None else len(allowed_actions))
        else:
            self.action_space = self.env.action_space

    def close(self):
        self.env.close()
        super().close()

    def switch_benchmark(self):
        idx = random.randint(0, -1 + len(self.benchmarks))
        print("Switched to {0}".format(self.benchmarks[idx]))
        self.env.close()
        self.env = gym.make("llvm-autophase-ic-v0", benchmark="cbench-v1/{0}".format(self.benchmarks[idx]))

    def step(self, action):
        # If necessary, map action
        if self.limited_action_set:
            action = self.action_mapping[action]

        observation, reward, done, info = self.env.step(action)
        compute_energy = False
        if compute_energy:
        
            ir_string = self.env.observation["Ir"]
            input_bitcode_file = self.env.observation["BitcodeFile"]
            output_asm_file = os.path.join(os.path.dirname(input_bitcode_file), "asm.s")
            
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
                input_bitcode_file,
                '-o',
                output_asm_file
            ], check=True, capture_output=True)
            
            with open(output_asm_file, "r") as file:
                # Read the entire file content as a string
                asm_code = file.read()
            
                # print("asm contents:", asm_code)
                print("Estimated energy:", estimate_program_energy(asm_code))
                print("Estimated energy:", estimate_program_energy(asm_code).total_energy)


        if self.patience is not None:
            self.fifo.append(reward)
            while len(self.fifo) > self.patience:
                del self.fifo[0]
            all_zero = True
            for x in self.fifo:
                if x != 0:
                    all_zero = False
                    break
            if all_zero and len(self.fifo) >= self.patience:
                done = True

        if self.limited_time:
            self.elapsed_steps += 1
            if self.elapsed_steps >= self.max_steps:
                done = True

        if self.steps_in_observation:
            return np.concatenate((observation, np.array([self.max_steps - self.elapsed_steps]))), reward, done, info
        else:
            return observation, reward, done, info

    def reset(self):
        self.fifo = []
        if self.limited_time:
            self.elapsed_steps = 0

        if self.steps_in_observation:
            return np.concatenate((self.env.reset(), np.array([self.max_steps])))
        else:
            return self.env.reset()
