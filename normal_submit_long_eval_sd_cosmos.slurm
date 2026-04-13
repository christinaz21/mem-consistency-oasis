#!/bin/bash
#SBATCH --exclude=neu306,neu322
#SBATCH --job-name=cosmos_sd           # spatial-distance eval on Cosmos eval split
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=15:59:00
#SBATCH --output=/n/fs/videogen/train-oasis/output/logs/test-%J.log
#SBATCH --error=/n/fs/videogen/train-oasis/output/logs/test-%J.log
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
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"

nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"

# Submit from train-oasis (same as normal_submit_long_eval_sd.slurm).
python -m train_oasis.inference.fast_inference_spatial_distance \
  --metadata-path /u/cz5047/videogen/data/Cosmos-Drive-Dreams/synthetic/vehicle_video_action/eval_paths_sunny.json \
  --save-dir /n/fs/videogen/train-oasis/outputs/droid/eval_results_sd/sd_cosmos_120_p16/\
  --oasis-ckpt /u/cz5047/videogen/train-oasis-2/outputs/2025-09-23/23-46-55/checkpoints/epoch=2-step=21000.ckpt \
  --model-name cosmos \
  --action-dim 6 \
  --vae-ckpt /n/fs/videogen/train-oasis/models/oasis500m/vit-l-20.safetensors \
  --sd-device cuda:0 \
  --sd-max-frames 30 \
  --num_particles 16 \
  --num-frames 120 \
