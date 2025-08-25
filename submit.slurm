#!/bin/bash
#SBATCH --job-name=lstm           # create a short name for your job
#SBATCH --partition=pli
#SBATCH --account=oasis
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=96       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=960G
#SBATCH --gres=gpu:8             # number of gpus per node
#SBATCH --time=48:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=logs/lstm-%J.log
#SBATCH --error=logs/lstm-%J.log

module purge
module load anaconda3/2024.6
conda activate oasis
module load cudatoolkit/11.8

# python /home/tc0786/Project/train-oasis/train_oasis/inference/sample.py
torchrun --nnodes=1 --nproc_per_node=8 /home/tc0786/Project/train-oasis/train_oasis/main.py --config-name=lstm
# python /home/tc0786/Project/train-oasis/train_oasis/inference/batch_inference.py