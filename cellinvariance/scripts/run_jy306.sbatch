#!/bin/bash
#SBATCH --job-name=ci_jy306
#SBATCH --account=torch_pr_62_tandon_priority
#SBATCH --partition=h100_tandon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/sd5963/slurm_logs/ci_jy306_%j.out
#SBATCH --error=/scratch/sd5963/slurm_logs/ci_jy306_%j.err

set -euo pipefail
cd /scratch/sd5963/cell_registration_2026/cellinvariance

EPOCHS="${EPOCHS:-50}"
AUGS="${AUGS:-100}"
TAG="${TAG:-jy306_smoke}"

echo "node: $(hostname)"
echo "gpu:  $(nvidia-smi --query-gpu=name,memory.free --format=csv,noheader || echo none)"
echo "config: epochs=${EPOCHS}  augs=${AUGS}  tag=${TAG}"

singularity exec --nv \
    --bind /scratch/sd5963:/scratch/sd5963 \
    /scratch/sd5963/containers/w2v2_cuda_cu128.sif \
    env CELLINVARIANCE_SUBJECT=jy306 \
        python3 scripts/reproduce_jy306.py \
            --epochs "${EPOCHS}" \
            --augs   "${AUGS}" \
            --tag    "${TAG}"
