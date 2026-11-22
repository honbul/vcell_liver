#!/usr/bin/env bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=36G
#SBATCH --time=04:00:00
set -euo pipefail
study=$1
code=$2
destination=$3
cd "$code"
r19="$study/vcell_liver_r19_plan"
python="$study/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
t2="$study/vcell_liver_goal_tracks/v1/t2"
export PYTHONPATH="$code:$t2/dependencies:$t2-public/deps"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
"$python" -m tune_t4_v2.pan \
  --hepg2 "$r19/t4-run.fS2rpsIm" --phh "$r19/t4-phh.p1AqqE6o" \
  --cmap "$study/vcell_liver_transigen_adaptation_b1/assets/raw/cmap2020" \
  --r18 "$r19/preflight.kwjS2aLo/r18-input-lock" \
  --donors "$r19/audit.YBfVmbuo/r18-registered/r18_B_donor_exceptions.tsv" \
  --landmarks "$study/vcell_liver_r15_lincs/v1/cache/GSE92742/landmarks.tsv" --out "$destination"
