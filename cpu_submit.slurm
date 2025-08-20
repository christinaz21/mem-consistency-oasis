#!/bin/bash
#SBATCH --job-name=pred           # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=40G
#SBATCH --time=12:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=outputs/logs/predictor-%J.log
#SBATCH --error=outputs/logs/predictor-%J.log

module purge
module load anaconda3/2024.6
conda activate oasis

# python /home/tc0786/Project/train-oasis/train_oasis/inference/evaluate.py
# torchrun --nnodes=1 --nproc_per_node=4 /home/tc0786/Project/train-oasis/train_oasis/main.py --config-name=latent_frame_pack
# python train_oasis/inference/batch_inference.py
python /home/tc0786/Project/train-oasis/train_oasis/trainer/predictor.py