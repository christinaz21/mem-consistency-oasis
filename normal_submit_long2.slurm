#!/bin/bash
#SBATCH --exclude=neu306
#SBATCH --job-name=minecraft           # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=8       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=250G
#SBATCH --gres=gpu:8
#SBATCH --time=18:59:00          # total run time limit (HH:MM:SS)
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
torchrun --nnodes=1 --nproc_per_node=8 /n/fs/videogen/train-oasis/train_oasis/main.py --config-name=mc_rnn_ttt2
# python train_oasis/inference/lstm_chunk.py
# python /home/tc0786/Project/train-oasis/train_oasis/inference/batch_inference.py