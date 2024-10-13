#!/usr/bin/bash
#SBATCH --job-name=sample-control-gen
#SBATCH --gpus=3
#SBATCH --account=dd-24-37
#SBATCH --partition=qgpu
#SBATCH --time=1-00:00:00
#SBATCH --output=/mnt/proj3/dd-24-37/ran/sample_control_gen.out
#SBATCH --error=/mnt/proj3/dd-24-37/ran/sample_control_gen.err
#SBATCH --export=ALL  # This line ensures all environment variables are passed to the job

# Load Anaconda and activate the environment
cd /mnt/proj3/dd-24-37/ran/InterSem

python_exec=/mnt/proj3/dd-24-37/ran/anaconda3/envs/InterSemEnv/bin/python

export LD_LIBRARY_PATH=/usr/local/nvidia/cuda/default/lib64/
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128

$python_exec ./src/align_your_latents_controls_gen/align_latents_sample_control_gen.py