#!/usr/bin/env bash
# G3 / G1-b: simple baselines on PIE's split, scored with pie.metrics (CPU).
#SBATCH --job-name=g3_pie_base
#SBATCH --partition=cpu
#SBATCH --exclude=g[1-4]
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/g3_pie_baselines_%j.log
set -euo pipefail
source experiments/feedback/g3_pie_env.sh
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
FOLD=${1:-hepg2}; shift || true
DEV=${1:-cuda}; shift || true   # which reproduced eval to read (cuda or cpu)
echo "host=$(hostname) job=${SLURM_JOB_ID:-} fold=$FOLD start=$(date -Is)"
PRE=$(ls -d "$PIE_DATA_ROOT"/hf/datasets/arcinstitute/*/20c9faef76fc96fdc809871340bd93499413dfe7/preprocessed | head -1)
SPL=$(ls -d "$PIE_DATA_ROOT"/hf/datasets/arcinstitute/*/396ab9563175ee887750c9eed7ccaea6f5fdbf50/replogle_wdataset | head -1)
REPRO="$PIE_RUNS_ROOT/$DEV/replogle_wdataset/$FOLD/eval"
"$PIE_PY" -u src/feedback/g3_pie_baselines.py --pre "$PRE" --splits "$SPL" --fold "$FOLD" \
  --pub "$G3_ROOT/models/replogle_wdataset/$FOLD/eval" --repro "$REPRO" \
  --out reports/feedback/G3 --assets "$G3_ROOT/baselines" "$@"
echo "end=$(date -Is)"
