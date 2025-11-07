#!/bin/bash
#SBATCH --job-name=df300           # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=24       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=480G
#SBATCH --constraint=gpu80
#SBATCH --gres=gpu:4
#SBATCH --time=15:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=logs/rnn_chunk_mamba_pos_nocomb-%J.log
#SBATCH --error=logs/rnn_chunk_mamba_pos_nocomb-%J.log

module purge
module load anaconda3/2024.6
conda activate oasis
module load cudatoolkit/11.8

torchrun --nnodes=1 --nproc_per_node=4 /home/tc0786/Project/train-oasis/train_oasis/main.py --config-name=mc_rnn_ttt
# python train_oasis/inference/lstm_chunk.py
# python /home/tc0786/Project/train-oasis/train_oasis/inference/batch_inference.py
# python /home/tc0786/Project/train-oasis/train_oasis/inference/evaluate.py
# python /home/tc0786/Project/train-oasis/train_oasis/inference/rnn_inference_temp.py