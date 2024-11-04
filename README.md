# EnergyAwareLLVM

## Installation 
**Note: This was tested on Ubuntu 20.04**

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
