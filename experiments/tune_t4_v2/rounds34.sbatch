#!/usr/bin/env bash
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=01:30:00
set -euo pipefail
study=$1
code=$2
runs=$3
out=$4
cd "$code"
python="$study/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
t2="$study/vcell_liver_goal_tracks/v1/t2"
export PYTHONPATH="$code:$t2/dependencies:$t2-public/deps"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
cells=(HEPG2 PHH)
roles=(A B C)
task=${SLURM_ARRAY_TASK_ID:?}
cell=${cells[task/9]}
group="f$(((task%9)/3))-${roles[task%3]}"
"$python" -m tune_t4_v2.rounds34 --round1 "$runs/round1-01/$cell/$group" \
  --round2 "$runs/round2-01/$cell/$group" --pretrain "$runs/pretrain-01" --out "$out/$cell/$group"
