#!/bin/bash
#SBATCH --job-name=pred           # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=8       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=32G
#SBATCH --time=02:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=logs/test-%J.log
#SBATCH --error=logs/test-%J.log

module purge
module load anaconda3/2024.6
conda activate oasis

# python /home/tc0786/Project/train-oasis/train_oasis/inference/evaluate.py
# torchrun --nnodes=1 --nproc_per_node=4 /home/tc0786/Project/train-oasis/train_oasis/main.py --config-name=latent_frame_pack
# python train_oasis/inference/batch_inference.py
python train_oasis/only_for_test.py