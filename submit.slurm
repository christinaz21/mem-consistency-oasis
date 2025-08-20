#!/bin/bash
#SBATCH --job-name=pp           # create a short name for your job
#SBATCH --partition=pli
#SBATCH --account=oasis
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=8       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=120G
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:60:00          # total run time limit (HH:MM:SS)
#SBATCH --output=outputs/logs/rag_pred_pose-%J.log
#SBATCH --error=outputs/logs/rag_pred_pose-%J.log

module purge
module load anaconda3/2024.6
conda activate oasis

# python /home/tc0786/Project/train-oasis/train_oasis/inference/sample.py
# torchrun --nnodes=1 --nproc_per_node=4 /home/tc0786/Project/train-oasis/train_oasis/main.py --config-name=latent_frame_pack
python /home/tc0786/Project/train-oasis/train_oasis/inference/batch_inference.py