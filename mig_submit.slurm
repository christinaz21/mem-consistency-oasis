#!/bin/bash
#SBATCH --job-name=test           # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=logs/test-%J.log
#SBATCH --error=logs/test-%J.log

module purge
module load anaconda3/2024.6
conda activate oasis
module load cudatoolkit/11.8

# torchrun --nnodes=1 --nproc_per_node=1 /home/tc0786/Project/train-oasis/train_oasis/main.py --config-name=df_maze
# python train_oasis/inference/lstm_chunk.py
# python /home/tc0786/Project/train-oasis/train_oasis/inference/batch_inference.py
# python /home/tc0786/Project/train-oasis/train_oasis/inference/evaluate.py
# python /home/tc0786/Project/train-oasis/train_oasis/inference/rnn_inference.py
python /home/tc0786/Project/train-oasis/train_oasis/inference/rnn_inference_temp.py \
    --model_name Mamba_comb \
    --ckpt_path outputs/rnn/mamba_maze_pos_comb/checkpoints/epoch=2-step=2445.ckpt \
    --combine_actions