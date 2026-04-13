#!/bin/bash
#SBATCH --exclude=neu306
#SBATCH --job-name=minecraft           # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=48       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=480G
#SBATCH --gres=gpu:8
#SBATCH --time=23:59:00          # total run time limit (HH:MM:SS)
#SBATCH --output=/n/fs/videogen/train-oasis/output/logs/test-%J.log
#SBATCH --error=/n/fs/videogen/train-oasis/output/logs/test-%J.log
#SBATCH --mail-type=all
#SBATCH --mail-user=christinaz@princeton.edu

module purge
source ~/.bashrc
eval "$(conda shell.bash hook)"  # this is needed to load python packages correctly
conda activate oasis
module load cudatoolkit/12.4

# python /home/tc0786/Project/train-oasis/train_oasis/inference/evaluate.py
torchrun --nnodes=1 --nproc_per_node=8 /n/fs/videogen/train-oasis/train_oasis/main.py --config-name=mc_rnn_lstm
# python train_oasis/inference/lstm_chunk.py
# python /home/tc0786/Project/train-oasis/train_oasis/inference/batch_inference.py