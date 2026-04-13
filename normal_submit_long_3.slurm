#!/bin/bash
#SBATCH --job-name=minecraft
#SBATCH --nodes=3                   # 3 nodes
#SBATCH --ntasks-per-node=1         # one launcher task per node (torchrun will spawn 8 procs on each)
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --time=59:00
#SBATCH --output=/n/fs/videogen/train-oasis/output/logs/test-%J.log
#SBATCH --error=/n/fs/videogen/train-oasis/output/logs/test-%J.log
#SBATCH --mail-type=all
#SBATCH --mail-user=christinaz@princeton.edu

# --- Environment ---
module purge
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate oasis
module load cudatoolkit/12.4

# Good defaults
export OMP_NUM_THREADS=6            # 48 CPU / 8 GPU ≈ 6 threads per proc
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1

# --- Rendezvous (master) info from SLURM ---
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export NNODES=$SLURM_NNODES                 # should be 3
export NPROC_PER_NODE=8                     # 8 GPUs per node

# (Optional) sanity:
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT NNODES=$NNODES NPROC_PER_NODE=$NPROC_PER_NODE"

# --- Launch ---
# Run torchrun ONCE (not once per node). srun here just pins to the first node.
srun --nodes=1 --ntasks=1 bash -lc "
  source ~/.bashrc
  eval \"\$(conda shell.bash hook)\"
  conda activate oasis
  which torchrun || true
  torchrun \
    --nnodes=\$NNODES \
    --nproc_per_node=\$NPROC_PER_NODE \
    --rdzv_backend=c10d \
    --rdzv_endpoint=\$MASTER_ADDR:\$MASTER_PORT \
    /n/fs/videogen/train-oasis/train_oasis/main.py --config-name=mc_rnn_ttt
"
