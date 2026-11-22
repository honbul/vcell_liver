#!/usr/bin/env bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=00:50:00
set -euo pipefail
study=$1
code=$2
output=$3
cd "$code"
t2="$study/vcell_liver_goal_tracks/v1/t2"
export PYTHONPATH="$code:$code/src:$t2/dependencies:$t2-public/deps"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=3 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
python="$study/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
"$python" -m v0plus_v2.cmap --cmap /mnt/S1/sdata/pubdata/LINCS/LINCS2020 \
  --bundle "$study/vc_build_v1/function/cross-v1" \
  --landmarks "$study/vcell_liver_r15_lincs/v1/cache/GSE92742/landmarks.tsv" --out "$output"
