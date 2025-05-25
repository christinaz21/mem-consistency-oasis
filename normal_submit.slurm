#!/bin/bash
#SBATCH --job-name=test_df           # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=8       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=80G
#SBATCH --constrain=gpu80
#SBATCH --gres=gpu:2
#SBATCH --time=00:20:00          # total run time limit (HH:MM:SS)
#SBATCH --output=outputs/logs/test_df-%J.log
#SBATCH --error=outputs/logs/test_df-%J.log

module purge
module load anaconda3/2024.6
conda activate oasis

# python /home/tc0786/Project/train-oasis/train_oasis/inference/evaluate.py
torchrun --nnodes=1 --nproc_per_node=2 /home/tc0786/Project/train-oasis/train_oasis/main.py --config-name=world_coordinate