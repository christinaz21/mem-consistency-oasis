#!/bin/bash
#SBATCH --job-name=maze           # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=24       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=480G
#SBATCH --constrain=gpu80
#SBATCH --gres=gpu:4
#SBATCH --time=00:60:00          # total run time limit (HH:MM:SS)
#SBATCH --output=logs/test-%J.log
#SBATCH --error=logs/test-%J.log

module purge
module load anaconda3/2024.6
conda activate oasis
module load cudatoolkit/11.8

# python /home/tc0786/Project/train-oasis/train_oasis/inference/evaluate.py
torchrun --nnodes=1 --nproc_per_node=4 /home/tc0786/Project/train-oasis/train_oasis/main.py --config-name=latent_rag
# python train_oasis/inference/lstm_chunk.py
# python /home/tc0786/Project/train-oasis/train_oasis/inference/batch_inference.py