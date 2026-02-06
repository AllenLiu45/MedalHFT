# MedalHFT
This is the implementation of our paper "Multi-Scale Frequency Domain Learning for High-Frequency Trading via Hierarchical Reinforcement Learning".

## Framework
Let me show you the structure of our framework:
```python
./download_data  # This folder contains a pipeline for data acquisition, data processing, and factor synthesis.
./env # This folder contains the environments for agents.
./logs    # This folder records the logs.
./model # This folder contains the models we trained.
./MyData # This folder contains the dataset after data processing.
./preprocess # This folder contains the code for MAD module.
./result # This folder contains the results of the experiment.
./RL # This folder contains the hierarchical reinforcement learning algorithm.
./scripts # This folder contains the scripts for running the experiment.
./tools # This folder contains the tools for the experiment.
```
Then we can start to use our framework:
## step 0
pip install -r requirements.txt

## step 1
Run down_data/Raw_data_process.py to process the raw data that we obtained from Binance. We have already provided the pre-processed sample file (feather format) [Download dataset here](https://drive.google.com/file/d/1PDA7EI7HwQa-HmCy8dfIp4ytKjT0nzzF/view?usp=sharing). It is in compressed format, please extract it and replace the MyData folder. Then you can use the dataset directly.

## Step 2
Run scripts/decomposition.sh for data decomposition and labeling. 

## Step 3
Run scripts/low_level.sh for training expert agents in stage 1. 

## Step 4
Run scripts/high_level.sh for training a manager agent in stage 2. 
