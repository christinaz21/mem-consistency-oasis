#!/bin/bash
#SBATCH --exclude=neu306,neu322
#SBATCH --job-name=sd_interval      # interval BoN with spatial-distance (VGGT) scoring
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=5:59:00
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

# Submit from train-oasis. BoN frequency per video = --chunk-frames (re-score every N new frames).
# python -m train_oasis.inference.fast_inference_spatial_distance_interval \
#   --metadata-path /n/fs/videogen/data/minecraft_data/data/eval_data/paths.json \
#   --save-dir /n/fs/videogen/train-oasis/outputs/droid/eval_results_sd/sd_bon_200_p16/\
#   --oasis-ckpt /u/cz5047/videogen/train-oasis/models/df_20_full_1epoch.bin \
#   --model-name cosmos \
#   --action-dim 6 \
#   --vae-ckpt /n/fs/videogen/train-oasis/models/oasis500m/vit-l-20.safetensors \
#   --num-frames 120 \
#   --num_particles 4 \
#   --chunk-frames 15 \
#   --sd-device cuda:0 \
#   --sd-max-frames 30

# # Submit from train-oasis. BoN frequency per video = --chunk-frames (re-score every N new frames).
# python -m train_oasis.inference.fast_inference_spatial_distance_interval \
#   --metadata-path /u/cz5047/videogen/data/Cosmos-Drive-Dreams/synthetic/vehicle_video_action/eval_paths_sunny.json \
#   --save-dir /n/fs/videogen/train-oasis/outputs/droid/eval_results_sd/sd_cosmos_interval_c15_p4/\
#   --oasis-ckpt /u/cz5047/videogen/train-oasis-2/outputs/2025-09-23/23-46-55/checkpoints/epoch=2-step=21000.ckpt \
#   --model-name cosmos \
#   --action-dim 6 \
#   --vae-ckpt /n/fs/videogen/train-oasis/models/oasis500m/vit-l-20.safetensors \
#   --num-frames 120 \
#   --num_particles 4 \
#   --chunk-frames 15 \
#   --sd-device cuda:0 \
#   --sd-max-frames 30

# Minecraft-style metadata (same layout as normal_submit_long_eval_droid_cosmos.slurm), e.g. chunk-frames 25 and p=4:
python -m train_oasis.inference.fast_inference_spatial_distance_interval \
    --metadata-path /n/fs/videogen/data/minecraft_data/data/eval_data/paths.json \
    --save-dir /n/fs/videogen/train-oasis/outputs/droid/eval_results_sd/sd_mc_interval_c200_p4/\
    --oasis-ckpt /u/cz5047/videogen/train-oasis/models/df_20_full_1epoch.bin \
    --vae-ckpt /n/fs/videogen/train-oasis/models/oasis500m/vit-l-20.safetensors \
    --num-frames 200 \
    --num_particles 4 \
    --chunk-frames 200 \
    --sd-device cuda:0 \
    --sd-max-frames 30 \
