#!/usr/bin/env bash
#SBATCH --cpus-per-task=3
#SBATCH --mem=10G
#SBATCH --time=01:00:00
set -euo pipefail
study=$1
code=$2
input=$3
cd "$code"
export PYTHONPATH="$code" PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
python="$study/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
"$python" -m liver_dili_v2.assays --tox21 "$input/tox21-training.zip" --out "$input/assays-01"
