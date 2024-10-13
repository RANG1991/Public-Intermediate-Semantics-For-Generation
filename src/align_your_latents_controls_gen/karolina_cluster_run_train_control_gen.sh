#!/bin/sh
#SBATCH --export=ALL  # This line ensures all environment variables are passed to the job
#SBATCH --gpus=1
#SBATCH --account=dd-24-37
#SBATCH --partition=qgpu
#SBATCH --time=2-00:00:00
#SBATCH --output /mnt/proj3/dd-24-37/ran/%x-train-control-gen.out
#SBATCH --error /mnt/proj3/dd-24-37/ran/%x-train-control-gen.err

echo "$SLURM_JOB_NAME-train-control-gen"

# Load Anaconda and activate the environment
cd /mnt/proj3/dd-24-37/ran/InterSem

python_exec=/mnt/proj3/dd-24-37/ran/anaconda3/envs/InterSemEnv/bin/python

export LD_LIBRARY_PATH=/usr/local/nvidia/cuda/default/lib64/
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128

$python_exec -m accelerate.commands.launch --main_process_port=29502 ./src/align_your_latents_controls_gen/align_latents_train_control_gen.py --yaml_config_file_name ./src/align_your_latents_controls_gen/config_files/run_on_regular_dataset_control_gen.yml --control_type $SLURM_JOB_NAME