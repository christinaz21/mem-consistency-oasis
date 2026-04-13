#!/bin/bash
# Same as normal_submit_grpo.slurm but requests 4 GPUs for Lightning DDP.
# Effective batch ≈ batch_size * num_gpus (see config/grpo.yaml batch_size).
# Increase --time if steps are slow (GRPO is heavy per step).

#SBATCH --exclude=neu306
#SBATCH --job-name=grpo-droid-4gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --gres=gpu:4
#SBATCH --time=59:00
#SBATCH --output=/n/fs/videogen/train-oasis/output/logs/grpo-%J.log
#SBATCH --error=/n/fs/videogen/train-oasis/output/logs/grpo-%J.log
#SBATCH --mail-type=all
#SBATCH --mail-user=christinaz@princeton.edu

module purge
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate oasis
module load cudatoolkit/12.4

export TRITON_CACHE_DIR=/n/fs/videogen/triton_cache

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

echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"

# Use torchrun so WORLD_SIZE matches allocated GPUs (plain `python` often stays 1 process).
NGPUS=$(python - <<'PY'
import os
v = os.environ.get("CUDA_VISIBLE_DEVICES", "")
print(len(v.split(",")) if v.strip() else 4)
PY
)
echo "torchrun --nproc_per_node=${NGPUS} --master_port=${MASTER_PORT}"
torchrun --standalone --nproc_per_node="${NGPUS}" --master_port="${MASTER_PORT}" \
  /n/fs/videogen/train-oasis/train_oasis/main.py --config-name=grpo
