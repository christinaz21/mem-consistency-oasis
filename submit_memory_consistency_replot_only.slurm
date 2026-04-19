#!/bin/bash
#SBATCH --exclude=neu306,neu322
#SBATCH --job-name=mem-replot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=59:00
#SBATCH --output=/n/fs/videogen/train-oasis/output/logs/memory-replot-%J.log
#SBATCH --error=/n/fs/videogen/train-oasis/output/logs/memory-replot-%J.log
#SBATCH --mail-type=all
#SBATCH --mail-user=christinaz@princeton.edu


module purge
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate oasis

echo "HOST=$(hostname)"

# Hardcoded to your existing evaluated run folder
OUTPUT_DIR="/u/cz5047/videogen/train-oasis/outputs/memory_consistency/2026-04-13/05-18-25"

# Method folders must match names used in raw/<method>/frames_XXXX/summary.json
BASELINE_DIR="/u/cz5047/videogen/train-oasis/outputs/droid/eval_results_try3/droid_bon_200_p1/memory"
SD_BON_DIR="/u/cz5047/videogen/train-oasis/outputs/droid/eval_results_sd/sd_bon_200_p16/memory"
SD_GRPO_DIR="/u/cz5047/videogen/train-oasis/outputs/eval_grpo/sd/2026-04-08/13-00-49/1000/memory"
DROID_BON_DIR="/u/cz5047/videogen/train-oasis/outputs/droid/eval_results_try3/droid_bon_200_p16/memory"
DROID_GRPO_DIR="/u/cz5047/videogen/train-oasis/outputs/eval_grpo/droid/2026-04-06/02-52-33/500/memory"
REFERENCE_ROOT="/u/cz5047/videogen/data/minecraft_data/data/eval_data/memory"

python /u/cz5047/videogen/train-oasis/train_oasis/inference/plot_memory_consistency_vs_time.py \
  --method "baseline=${BASELINE_DIR}" \
  --method "droid_bon=${DROID_BON_DIR}" \
  --method "droid_grpo=${DROID_GRPO_DIR}" \
  --reference-root "${REFERENCE_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --plot-metrics memory_error,ssim,psnr,lpips \
  --start-frame 10 \
  --end-frame 100 \
  --stride 10 \
  --reuse-existing \
  --invert-score

echo "Replot complete. Outputs in: ${OUTPUT_DIR}"
