#!/usr/bin/env bash
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --time=02:00:00
set -euo pipefail
study=$1
code=$2
destination=$3
round1=$4
cd "$code"
python="$study/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
t2="$study/vcell_liver_goal_tracks/v1/t2"
runtime="$study/vcell_liver/vcell_r12_runtime_cu126"
export PYTHONPATH="$code:$runtime:$t2/dependencies:$t2-public/deps"
export OPENBLAS_NUM_THREADS=3 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTHONDONTWRITEBYTECODE=1
cells=(HEPG2 PHH)
roles=(A B C)
task=${SLURM_ARRAY_TASK_ID:?}
cell=${cells[task/9]}
group="f$(((task%9)/3))-${roles[task%3]}"
"$python" -m tune_t4_v2.round2 --round1 "$round1/$cell/$group" --out "$destination/$cell/$group"
