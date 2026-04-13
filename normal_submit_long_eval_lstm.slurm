#!/bin/bash
# SBATCH --exclude=neu306
#SBATCH --job-name=minecraft           # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=24      # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=59:00          # total run time limit (HH:MM:SS)
#SBATCH --output=/n/fs/videogen/train-oasis/output/logs/test-%J.log
#SBATCH --error=/n/fs/videogen/train-oasis/output/logs/test-%J.log
#SBATCH --mail-type=all
#SBATCH --mail-user=christinaz@princeton.edu

module purge
source ~/.bashrc
eval "$(conda shell.bash hook)"  # this is needed to load python packages correctly
conda activate oasis
module load cudatoolkit/12.4

# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=0
# export CUDA_LAUNCH_BLOCKING=1  # only for debugging slowdowns

# srun --ntasks-per-node=8 --gpus-per-task=1 \
#   python /n/fs/videogen/train-oasis/train_oasis/main.py --config-name=mc_rnn_ttt

# # srun --ntasks=1 --gres=gpu:8 \
# #   torchrun --standalone --nproc_per_node=8 \
# #   /n/fs/videogen/train-oasis/train_oasis/main.py --config-name=mc_rnn_ttt

# python /home/tc0786/Project/train-oasis/train_oasis/inference/evaluate.py
CUDA_LAUNCH_BLOCKING=1 
export TRITON_CACHE_DIR=/n/fs/videogen/triton_cache

python train_oasis/inference/rnn_inference2.py \
    --save_dir /n/fs/videogen/train-oasis/outputs/rnn/eval_results_droid/lstm_shorter \
    --ckpt_path /n/fs/videogen/train-oasis/outputs/rnn/frame_lstm/checkpoints/ckpt-epoch=0-step=5000-v1.ckpt \
    --metadata_file_path /n/fs/videogen/data/minecraft_data/data/eval_data/paths_shorter.json \
    --inference_type framewise \
    --vae_name oasis \
    --vae_ckpt /n/fs/videogen/train-oasis/models/oasis500m/vit-l-20.safetensors \
    --batch_size 1 \
    --dataset minecraft \
    --rnn_type LSTM \
    --total_frames 100 \
    --prompt_frames 60 \
    --onemem \
    --combine_actions