#!/bin/bash
#SBATCH --exclude=neu306
#SBATCH --job-name=grpo-clipped
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
# Single GPU (default). For multi-GPU DDP on one node, use e.g.:
#   #SBATCH --gres=gpu:4
#   #SBATCH --cpus-per-task=32
# and keep `experiment.training.devices: -1` in config/grpo.yaml (or override
# `experiment.training.devices=4` below).
#SBATCH --gres=gpu:2
#SBATCH --time=59:00
#SBATCH --output=/n/fs/videogen/train-oasis/output/logs/grpo-clipped-%J.log
#SBATCH --error=/n/fs/videogen/train-oasis/output/logs/grpo-clipped-%J.log
#SBATCH --mail-type=all
#SBATCH --mail-user=christinaz@princeton.edu

module purge
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate oasis
module load cudatoolkit/12.4

export TRITON_CACHE_DIR=/n/fs/videogen/triton_cache

echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"

# torchrun rendezvous port. Don't rely on job-id modulo (collisions happen).
# Ask the OS for a free port instead of risking EADDRINUSE.
export MASTER_PORT=$(python - <<'PY'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("", 0))  # let OS pick a free port
port = s.getsockname()[1]
s.close()
print(port)
PY
)
echo "MASTER_PORT=${MASTER_PORT}"

# GRPO with PPO-style clipped ratios + KL to ref (DiffusionForcingGRPOClipped).
# Do not set model.gradient_checkpointing=false: full activations for T=100 OOM a single L40.
# Uses ~3 DiT copies (policy, policy_old, ref) — same memory caveats as baseline GRPO + extra old policy.
python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port="${MASTER_PORT}" \
  /n/fs/videogen/train-oasis/train_oasis/main.py --config-name=grpo \
  algorithm=df_grpo_clipped \
  hydra.run.dir='outputs/grpo_clipped/${now:%Y-%m-%d}/${now:%H-%M-%S}' \
  name=grpo-clipped-droid
