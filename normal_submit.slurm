#!/bin/bash
#SBATCH --job-name=eval           # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=6       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=120G
#SBATCH --constrain=gpu80
#SBATCH --gres=gpu:1
#SBATCH --time=00:60:00          # total run time limit (HH:MM:SS)
#SBATCH --output=outputs/logs/eval-%J.log
#SBATCH --error=outputs/logs/eval-%J.log

module purge
module load anaconda3/2024.6
conda activate oasis

python /home/tc0786/Project/train-oasis/train_oasis/inference/evaluate.py
# torchrun --nnodes=1 --nproc_per_node=4 /home/tc0786/Project/train-oasis/train_oasis/main.py --config-name=latent_frame_pack
# python train_oasis/inference/batch_inference.py
# python /home/tc0786/Project/train-oasis/train_oasis/inference/batch_inference.py