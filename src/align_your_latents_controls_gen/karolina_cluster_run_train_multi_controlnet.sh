#!/usr/bin/bash
#SBATCH --job-name=train-multi-controlnet
#SBATCH --gpus=3
#SBATCH --account=dd-24-37
#SBATCH --partition=qgpu
#SBATCH --time=1-00:00:00
#SBATCH --output=/mnt/proj3/dd-24-37/ran/train_multi_controlnet.out
#SBATCH --error=/mnt/proj3/dd-24-37/ran/train_multi_controlnet.err
#SBATCH --export=ALL  # This line ensures all environment variables are passed to the job
# Load Anaconda and activate the environment
cd /mnt/proj3/dd-24-37/ran/InterSem

python_exec=/mnt/proj3/dd-24-37/ran/anaconda3/envs/InterSemEnv/bin/python

export LD_LIBRARY_PATH=/usr/local/nvidia/cuda/default/lib64/

$python_exec -m accelerate.commands.launch ./src/align_your_latents_controls_gen/align_latents_train_multi_controlnet.py --yaml_config_file_name ./src/align_your_latents_controls_gen/config_files/run_on_regular_dataset_multi_controlnet.yml
