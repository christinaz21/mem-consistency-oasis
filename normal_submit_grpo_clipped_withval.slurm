#!/bin/bash
#SBATCH --exclude=neu306
#SBATCH --job-name=grpo-clipped-val
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --time=59:00
#SBATCH --output=/n/fs/videogen/train-oasis/output/logs/grpo-clipped-val-%J.log
#SBATCH --error=/n/fs/videogen/train-oasis/output/logs/grpo-clipped-val-%J.log
#SBATCH --mail-type=all
#SBATCH --mail-user=christinaz@princeton.edu

module purge
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate oasis
module load cudatoolkit/12.4

export TRITON_CACHE_DIR=/n/fs/videogen/triton_cache

# Hydra cwd: relative paths (hydra.run.dir, checkpoints) resolve from here.
cd /n/fs/videogen/train-oasis

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

# GRPO with PPO-style clipped ratios + KL to ref (DiffusionForcingGRPOClipped),
# with validation enabled frequently so eval videos are logged.
#
# Notes:
# - Validation is already wired into `trainer.fit(..., val_dataloaders=...)`.
# - In config/grpo.yaml, `val_every_n_epoch` defaults to 50, so with max_epochs=3
#   you otherwise never hit validation.
# - `experiment.validation.val_every_n_step` = Lightning `val_check_interval`: validation
#   (and GRPO val MP4s under val_videos/) runs every N *training batches*, not every epoch.
#   Example: 100 means no videos until step 100 (~hours at ~2 min/step). Use 10 for quicker checks.
# - Resume training from a Lightning checkpoint (`main.py` passes this to `trainer.fit(ckpt_path=...)`).
#   Hydra cannot parse unquoted paths whose basename contains '=' (e.g. ckpt-epoch=0-step=1000.ckpt).
#   Pass `load='...'` as one override so the value is a quoted Hydra string.
# GRPO_RESUME_CKPT='/u/cz5047/videogen/train-oasis/outputs/grpo_clipped_val/2026-04-10/15-14-41/checkpoints/ckpt-epoch=0-step=1000.ckpt'
python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port="${MASTER_PORT}" \
  /n/fs/videogen/train-oasis/train_oasis/main.py --config-name=grpo \
  algorithm=df_grpo_clipped \
  hydra.run.dir='outputs/grpo_clipped_val/${now:%Y-%m-%d}/${now:%H-%M-%S}' \
  name=grpo-clipped-droid-val \
  experiment.validation.val_every_n_epoch=null \
  experiment.validation.val_every_n_step=100 \
  experiment.validation.limit_batch=1 \
  algorithm.grpo.val_video_max_frames=100
# Resume (optional): define GRPO_RESUME_CKPT above, then add:
#   "load='${GRPO_RESUME_CKPT}'" \
