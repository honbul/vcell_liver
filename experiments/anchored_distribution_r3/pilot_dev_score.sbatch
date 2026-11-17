#!/usr/bin/env bash
# Pilot, development stage: score the CellFlow dev output with the unchanged evaluation code and
# write one dev table with short method names (baselines + DDPM grid + CellFlow).
#SBATCH --job-name=r3_pdevs
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/r3_%x_%j.log
set -euo pipefail
source experiments/anchored_distribution_r3/env.sh
echo "host=$(hostname) job=${SLURM_JOB_ID:-} commit=$(git rev-parse HEAD) start=$(date -Is)"
PY="$R3_PY -u -m src.anchored_distribution_r3"
NAMES=configs/anchored_distribution_r3/pilot_dev_names.json
CF='cells:assets/gen/cellflow/dev/cf009-dev-x2000-it60000-bs1024-s0-b9fe4a5c__ss{seed}.npz'
$PY.evaluate --setting dev --tag pilot_dev_cf --methods "$CF"
$R3_PY - <<PYEOF
import pandas as pd, os
p = os.environ["R3_ASSETS"] + "/runs/eval/dev/"
a = pd.concat([pd.read_csv(p + t + ".tsv.gz", sep="\t") for t in ["pilot_dev", "pilot_dev_cf"]])
a.to_csv(p + "pilot_dev_all.tsv.gz", sep="\t", index=False, float_format="%.6g")
PYEOF
$PY.pilot_relabel --setting dev --tag pilot_dev_all --out-tag pilot_dev_named --map $NAMES
$PY.aggregate --setting dev --tags c1_base,pilot_dev_named --out reports/anchored_distribution_r3/tables/pilot_dev \
  --refs shifted:anchored2,gauss_lr50:anchored2,control:native
echo "end=$(date -Is)"
