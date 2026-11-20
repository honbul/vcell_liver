#!/usr/bin/env bash
# G3: download pinned PIE checkpoints (4 folds) + the data/sources/splits they reference (~50 GB).
#SBATCH --job-name=g3_pie_fetch
#SBATCH --partition=cpu
#SBATCH --exclude=g[1-4]
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=logs/g3_pie_fetch_%j.log
set -euo pipefail
source experiments/feedback/g3_pie_env.sh
echo "host=$(hostname) job=${SLURM_JOB_ID:-} start=$(date -Is)"
"$PIE_PY" -u src/feedback/g3_pie_fetch.py --models "$G3_ROOT/models/replogle_wdataset" \
  --out reports/feedback/G3/provenance_fetch.json "$@"
# run dirs: symlink the published checkpoint + config so `pie eval` never writes into models/
for dev in cuda cpu; do for f in hepg2 jurkat k562 rpe1; do
  mkdir -p "$PIE_RUNS_ROOT/$dev/replogle_wdataset/$f"
  for x in best_auprc.ckpt config.yaml data_stats.json; do
    ln -sfn "$G3_ROOT/models/replogle_wdataset/$f/$x" "$PIE_RUNS_ROOT/$dev/replogle_wdataset/$f/$x"
  done
done; done
du -sh "$PIE_DATA_ROOT" "$G3_ROOT/models" || true
echo "end=$(date -Is)"
