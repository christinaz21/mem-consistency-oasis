#!/bin/bash
#SBATCH --exclude=neu306,neu322
#SBATCH --job-name=minecraft           # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=24      # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=7:59:00          # total run time limit (HH:MM:SS)
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


python train_oasis/inference/fast_inference_droid_interval.py \
    --oasis-ckpt /u/cz5047/videogen/train-oasis/models/df_20_full_1epoch.bin \
    --save-dir /n/fs/videogen/train-oasis/outputs/droid/eval_results_interval/droid_bon_25_p4/\
    --vae-ckpt /n/fs/videogen/train-oasis/models/oasis500m/vit-l-20.safetensors \
    --num-frames 200 \
    --metadata-path /n/fs/videogen/data/minecraft_data/data/eval_data/paths.json \
    --droid-weights /n/fs/videogen/data/models/droid_models/droid.pth \
    --num_particles 4 \
    --chunk-frames 25 \
    
# python train_oasis/inference/fast_inference_droid.py \
#     --oasis-ckpt /u/cz5047/videogen/train-oasis-2/outputs/2025-09-23/23-46-55/checkpoints/epoch=2-step=21000.ckpt \
#     --save-dir /n/fs/videogen/train-oasis/outputs/droid/eval_results_mc/droid_200_p16/\
#     --vae-ckpt /n/fs/videogen/train-oasis/models/oasis500m/vit-l-20.safetensors \
#     --num-frames 120 \
#     --metadata-path /u/cz5047/videogen/data/Cosmos-Drive-Dreams/synthetic/vehicle_video_action/eval_paths_sunny.json \
#     --droid-weights /n/fs/videogen/data/models/droid_models/droid.pth \
#     --num_particles 16 \
#     --action-dim 6 \
#     --model-name cosmos \
