#!/bin/bash
#SBATCH --job-name=lstm           # create a short name for your job
#SBATCH --partition=pli
#SBATCH --account=oasis
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=48       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=480G
#SBATCH --gres=gpu:4             # number of gpus per node
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=logs/long_rotate-%J.log
#SBATCH --error=logs/long_rotate-%J.log

module purge
module load anaconda3/2024.6
conda activate oasis
module load cudatoolkit/11.8

# python /home/tc0786/Project/train-oasis/train_oasis/inference/sample.py
torchrun --nnodes=1 --nproc_per_node=4 /home/tc0786/Project/train-oasis/train_oasis/main.py --config-name=lstm
# python /home/tc0786/Project/train-oasis/train_oasis/inference/batch_inference.py
# python /home/tc0786/Project/train-oasis/train_oasis/inference/lstm_chunk.py