#!/usr/bin/env bash
#SBATCH --cpus-per-task=6
#SBATCH --mem=28G
#SBATCH --time=01:45:00
set -euo pipefail
study=$1
code=$2
pairs=$3
out=$4
cd "$code"
runtime="$study/vcell_liver/vcell_r12_runtime_cu126"
t2="$study/vcell_liver_goal_tracks/v1/t2"
export PYTHONPATH="$code:$runtime:$t2/dependencies:$t2-public/deps"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python="$study/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
"$python" -m tune_t4_v2.pretrain --pairs "$pairs" --out "$out"
