# EnergyAwareLLVM

## Installation 
**Note: This was tested on Ubuntu 20.04 and my Intel Mac**

Create and activate conda environment:
```bash
conda create -y -n cse583 python=3.8
conda activate cse583
```

Fix some packages:
```bash
pip install setuptools==65.5.0 pip==21  # gym 0.21 installation is broken with more recent versions
pip install wheel==0.38.0
```

Install `CompilerGym`:
```bash
pip install -U compiler_gym
```

Install Other Requirements
```bash
pip install -r requirements.txt
```

## Reinforcement Learning
run the training code here:

```bash
cd ppo/
python3 train.py
```

At the moment, the code uses the [Autophase](https://compilergym.com/llvm/index.html#autophase) observation space. 


## Static Energy Estimation
Our implementation was heavily based on [Neville Grech's Code](https://github.com/nevillegrech/llvm-energy/tree/master) adapted to use llvmlite.
```bash
cd llvm-energy
python3 inferCR_converted.py ir_tests/test1.ll
``` 

## Usage
Run the testing script:
```bash
python3 random_actions.py
```
