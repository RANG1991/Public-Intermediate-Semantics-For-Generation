#!/bin/bash

# Change number of tasks, amount of memory and time limit according to your needs

#SBATCH -n 2
#SBATCH --time=100:0:0
#SBATCH --mem=100G
#SBATCH --gres=gpu:a30:1

# Uncomment and enter path of code
cd /sci/labs/sagieb/ranga/InterSem/

# virtual_env location
# virtual_env=/sci/labs/sagieb/ranga/python_env_intermediate_representations/bin/activate

python_exec=/sci/labs/sagieb/ranga/anaconda3/envs/InterSemEnv/bin/python

export LD_LIBRARY_PATH=/usr/local/nvidia/cuda/default/lib64/
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128

$python_exec -m accelerate.commands.launch ./src/align_your_latents_controls_gen/align_latents_train_2_controls.py --yaml_config_file_name ./src/align_your_latents_controls_gen/config_files/run_on_regular_dataset_2_controls.yml