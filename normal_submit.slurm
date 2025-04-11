#!/bin/bash
#SBATCH --job-name=apt           # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=48       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=768G
#SBATCH --gres=gpu:4             # number of gpus per node
#SBATCH --time=00:40:00          # total run time limit (HH:MM:SS)
#SBATCH --output=outputs/logs/apt-test-%J.log
#SBATCH --error=outputs/logs/apt-test-%J.log

module purge
module load anaconda
conda activate oasis
echo "gcc version: $(gcc --version)"

torchrun --nnodes=1 --nproc_per_node=4 /home/tc0786/Project/train-oasis/train_oasis/main.py --config-name=apt_test