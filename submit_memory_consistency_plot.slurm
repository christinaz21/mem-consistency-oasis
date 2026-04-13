#!/bin/bash
#SBATCH --exclude=neu306,neu322
#SBATCH --job-name=mem-consistency
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --output=/n/fs/videogen/train-oasis/output/logs/memory-consistency-%J.log
#SBATCH --error=/n/fs/videogen/train-oasis/output/logs/memory-consistency-%J.log
#SBATCH --mail-type=all
#SBATCH --mail-user=christinaz@princeton.edu

# set -euo pipefail

module purge
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate oasis
module load cudatoolkit/12.4

export TRITON_CACHE_DIR=/n/fs/videogen/triton_cache

echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"

# -----------------------------
# Edit these paths as needed
# -----------------------------
BASELINE_DIR="/u/cz5047/videogen/train-oasis/outputs/droid/eval_results_try3/droid_bon_200_p1/memory"
SD_BON_DIR="/u/cz5047/videogen/train-oasis/outputs/droid/eval_results_sd/sd_bon_200_p16/memory"
SD_GRPO_DIR="/u/cz5047/videogen/train-oasis/outputs/eval_grpo/sd/2026-04-08/13-00-49/1000/memory"
DROID_BON_DIR="/u/cz5047/videogen/train-oasis/outputs/droid/eval_results_try3/droid_bon_200_p16/memory"
DROID_GRPO_DIR="/u/cz5047/videogen/train-oasis/outputs/eval_grpo/droid/2026-04-06/02-52-33/500/memory"

REFERENCE_ROOT="/u/cz5047/videogen/data/minecraft_data/data/eval_data/memory"
OUTPUT_DIR="/u/cz5047/videogen/train-oasis/outputs/memory_consistency/$(date +%F)/$(date +%H-%M-%S)"

# Frame sweep config
START_FRAME=10
END_FRAME=100
STRIDE=10
BATCH_SIZE=8

# Add --invert-score if your DROID output is reward-like (higher is better)
INVERT_FLAG="--invert-score"

python /u/cz5047/videogen/train-oasis/train_oasis/inference/plot_memory_consistency_vs_time.py \
  --method "baseline=${BASELINE_DIR}" \
  --method "droid_bon=${DROID_BON_DIR}" \
  --method "droid_grpo=${DROID_GRPO_DIR}" \
  --method "sd_bon=${SD_BON_DIR}" \
  --method "sd_grpo=${SD_GRPO_DIR}" \
  --reference-root "${REFERENCE_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --compute-droid \
  --compute-spatial-distance \
  --plot-metric memory_error \
  --device cuda \
  --batch-size "${BATCH_SIZE}" \
  --start-frame "${START_FRAME}" \
  --end-frame "${END_FRAME}" \
  --stride "${STRIDE}" \
  ${INVERT_FLAG}

echo "Done. Outputs written to: ${OUTPUT_DIR}"

