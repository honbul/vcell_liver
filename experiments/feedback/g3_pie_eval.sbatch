#!/usr/bin/env bash
# G3 / G1-a: official checkpoint evaluation (pie eval, best_auprc) for the replogle_wdataset folds.
# Usage: sbatch experiments/feedback/g3_pie_eval.sbatch [cuda|cpu] fold [fold ...]
# CPU path: sbatch --partition=cpu --gres=none --exclude=g[1-4] --cpus-per-task=32 experiments/feedback/g3_pie_eval.sbatch cpu hepg2
# Outputs: assets/g3_pie/runs/<device>/replogle_wdataset/<fold>/eval/<setting>/
#SBATCH --job-name=g3_pie_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/g3_pie_eval_%j.log
set -euo pipefail
source experiments/feedback/g3_pie_env.sh
# assets come from g3_pie_fetch (PIE repo locks serialise any overlap); no auth needed
DEVICE=${1:-cuda}; shift || true
FOLDS=${*:-hepg2}
echo "host=$(hostname) job=${SLURM_JOB_ID:-} device=$DEVICE folds=$FOLDS cuda_visible=${CUDA_VISIBLE_DEVICES:-unset} start=$(date -Is)"
if [ "$DEVICE" = cuda ]; then nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; else export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}; fi
SPLITS=hf://datasets/arcinstitute/PIE_splits@396ab9563175ee887750c9eed7ccaea6f5fdbf50
for fold in $FOLDS; do
  for setting in unseen_ctx_pert unseen_ctx unseen_pert; do
    t0=$(date +%s)
    "$PIE_BIN" eval experiment_name=replogle_wdataset/$fold ckpt=best_auprc \
      run_dir="$PIE_RUNS_ROOT/$DEVICE/replogle_wdataset/$fold" \
      split_path=$SPLITS/replogle_wdataset/$setting/$fold/test.json row_set=$setting \
      save_predictions=true overwrite=true device=$DEVICE
    echo "[g3] fold=$fold setting=$setting seconds=$(( $(date +%s) - t0 ))"
    [ "$DEVICE" = cuda ] && nvidia-smi --query-gpu=memory.used --format=csv,noheader || true
  done
done
echo "end=$(date -Is)"
