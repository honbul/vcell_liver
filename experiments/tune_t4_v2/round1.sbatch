#!/usr/bin/env bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=02:00:00
set -euo pipefail
study=$1
code=$2
destination=$3
cd "$code"
r19="$study/vcell_liver_r19_plan"
python="$study/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
t2="$study/vcell_liver_goal_tracks/v1/t2"
export PYTHONPATH="$code:$t2/dependencies:$t2-public/deps"
export OPENBLAS_NUM_THREADS=3 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTHONDONTWRITEBYTECODE=1
cells=(HEPG2 PHH)
roles=(A B C)
task=${SLURM_ARRAY_TASK_ID:?}
cell=${cells[task/9]}
group="f$(((task%9)/3))-${roles[task%3]}"
"$python" -m tune_t4_v2.round1 --reports "$code/reports/r19_plan" \
  --hepg2 "$r19/t4-run.fS2rpsIm" --phh "$r19/t4-phh.p1AqqE6o" --cell "$cell" \
  --group "$group" --out "$destination/$cell/$group"
