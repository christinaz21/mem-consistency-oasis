#!/bin/bash
#SBATCH --exclude=neu306
#SBATCH --job-name=grpo-clipped-sd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --time=47:59:00
#SBATCH --output=/n/fs/videogen/train-oasis/output/logs/grpo-clipped-sd-%J.log
#SBATCH --error=/n/fs/videogen/train-oasis/output/logs/grpo-clipped-sd-%J.log
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

# Clipped GRPO with spatial-distance reward (VGGT + Chamfer vs batch GT;
# DiffusionForcingGRPOClippedSpatialDistance, config/algorithm/df_grpo_clipped_sd.yaml).
# Validation is wired into trainer.fit(..., val_dataloaders=...).
# val_every_n_epoch / val_every_n_step override config/grpo.yaml defaults so short runs still validate.
python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port="${MASTER_PORT}" \
  /n/fs/videogen/train-oasis/train_oasis/main.py --config-name=grpo \
  algorithm=df_grpo_clipped_sd \
  hydra.run.dir='outputs/grpo_clipped_sd/${now:%Y-%m-%d}/${now:%H-%M-%S}' \
  name=grpo-clipped-sd \
  experiment.validation.val_every_n_epoch=1 \
  experiment.validation.val_every_n_step=100 \
  experiment.validation.limit_batch=1

