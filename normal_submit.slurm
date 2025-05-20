#!/bin/bash
#SBATCH --job-name=rag_s           # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=36       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=480G
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=outputs/logs/rag_s-%J.log
#SBATCH --error=outputs/logs/rag_s-%J.log

module purge
module load anaconda3/2024.6
conda activate oasis

# python /home/tc0786/Project/train-oasis/train_oasis/inference/sample.py
torchrun --nnodes=1 --nproc_per_node=4 /home/tc0786/Project/train-oasis/train_oasis/main.py --config-name=latent_rag_add