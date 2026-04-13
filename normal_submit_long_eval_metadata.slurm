#!/bin/bash
#SBATCH --exclude=neu306,neu322
#SBATCH --job-name=minecraft           # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=24      # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=59:00          # total run time limit (HH:MM:SS)
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

# srun --ntasks-per-node=8 --gpus-per-task=1 \
#   python /n/fs/videogen/train-oasis/train_oasis/main.py --config-name=mc_rnn_ttt

# # srun --ntasks=1 --gres=gpu:8 \
# #   torchrun --standalone --nproc_per_node=8 \
# #   /n/fs/videogen/train-oasis/train_oasis/main.py --config-name=mc_rnn_ttt

# python /home/tc0786/Project/train-oasis/train_oasis/inference/evaluate.py
# CUDA_LAUNCH_BLOCKING=1 
export TRITON_CACHE_DIR=/n/fs/videogen/triton_cache
# export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS}

echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"

nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"

# python train_oasis/inference/fast_inference_droid_frame.py \
#     --oasis-ckpt /u/cz5047/videogen/train-oasis/models/df_20_full_1epoch.bin \
#     --output-path /n/fs/videogen/train-oasis/outputs/droid/eval_results/droid_baseline/hex_000000_chunk1.mp4 \
#     --vae-ckpt /n/fs/videogen/train-oasis/models/oasis500m/vit-l-20.safetensors \
#     --num-frames 100 \
#     --actions-path /n/fs/videogen/data/minecraft_data/data/eval_data/memory/hex/000000.npz \
#     --prompt-path /n/fs/videogen/data/minecraft_data/data/eval_data/memory/hex/000000.mp4 \
#     --droid-weights /n/fs/videogen/data/models/droid_models/droid.pth \
#     --chunk-frames 1 \
# python train_oasis/inference/batch_inference_droid_frame.py \
#     --oasis-ckpt /u/cz5047/videogen/train-oasis/models/df_20_full_1epoch.bin \
#     --save-dir /n/fs/videogen/train-oasis/outputs/droid/eval_results_try2/droid_bon_chunk10_200/\
#     --vae-ckpt /n/fs/videogen/train-oasis/models/oasis500m/vit-l-20.safetensors \
#     --num-frames 200 \
#     --metadata-path /n/fs/videogen/data/minecraft_data/data/eval_data/paths.json \
#     --droid-weights /n/fs/videogen/data/models/droid_models/droid.pth \
#     --chunk-frames 10 \
#     --batch-size 1 \
#     --score-workers 1 \
    # --batch_size 5 \
    # --dataset minecraft \
    # --total_frames 100 \
    # --prompt_frames 60 \
    # --num-particles 4 \
    # --base-seed 0 \
    # --metadata_file_path /n/fs/videogen/data/minecraft_data/data/eval_data/paths.json \
    # --ckpt-path /n/fs/videogen/train-oasis/outputs/rnn/frame_droid/checkpoints/ckpt-epoch=0-step=5000-v1.ckpt \
    # --inference-type framewise \
    # --vae-name oasis \

# python train_oasis/inference/fast_inference_droid.py \
#     --oasis-ckpt /u/cz5047/videogen/train-oasis/models/df_20_full_1epoch.bin \
#     --save-dir /n/fs/videogen/train-oasis/outputs/droid/eval_results_try3/droid_bon_200_p16/\
#     --vae-ckpt /n/fs/videogen/train-oasis/models/oasis500m/vit-l-20.safetensors \
#     --num-frames 200 \
#     --metadata-path /n/fs/videogen/data/minecraft_data/data/eval_data/paths.json \
#     --droid-weights /n/fs/videogen/data/models/droid_models/droid.pth \
#     --num_particles 16 \

python train_oasis/inference/fast_inference_metadata.py \
    --oasis-ckpt /u/cz5047/videogen/train-oasis/outputs/grpo_clipped_sd/2026-04-07/17-34-57/checkpoints/ckpt-epoch=0-step=1000.ckpt \
    --save-dir /n/fs/videogen/train-oasis/outputs/eval_grpo/sd/2026-04-07/17-34-57/1000/\
    --vae-ckpt /n/fs/videogen/train-oasis/models/oasis500m/vit-l-20.safetensors \
    --num-frames 100 \
    --metadata-path /n/fs/videogen/data/minecraft_data/data/eval_data/paths.json \
# python train_oasis/inference/fast_inference_metadata.py \
#     --oasis-ckpt /u/cz5047/videogen/train-oasis/models/df_20_full_1epoch.bin \
#     --save-dir /n/fs/videogen/train-oasis/outputs/droid/eval_results_grpo/baseline_100/\
#     --vae-ckpt /n/fs/videogen/train-oasis/models/oasis500m/vit-l-20.safetensors \
#     --num-frames 100 \
#     --metadata-path /n/fs/videogen/data/minecraft_data/data/eval_data/paths.json \

    # --chunk-frames 25 \
