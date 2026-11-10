#!/usr/bin/env bash
# G4: PIE-metric scoring of the PIE seed retrains (HepG2 fold) vs official checkpoint and G3 baselines. CPU.
#SBATCH --job-name=g4_pie_score
#SBATCH --partition=cpu
#SBATCH --exclude=g[1-4]
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/g4_pie_seed_score_%j.log
set -euo pipefail
source experiments/feedback/g3_pie_env.sh
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
echo "host=$(hostname) job=${SLURM_JOB_ID:-} start=$(date -Is)"
PRE=$(ls -d "$PIE_DATA_ROOT"/hf/datasets/arcinstitute/*/20c9faef76fc96fdc809871340bd93499413dfe7/preprocessed | head -1)
SPL=$(ls -d "$PIE_DATA_ROOT"/hf/datasets/arcinstitute/*/396ab9563175ee887750c9eed7ccaea6f5fdbf50/replogle_wdataset | head -1)
"$PIE_PY" -u src/feedback/g4_pie_seed_score.py --pre "$PRE" --splits "$SPL" --fold hepg2 "$@"
echo "end=$(date -Is)"
