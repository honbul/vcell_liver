#!/usr/bin/env bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:15:00
set -euo pipefail
study=$1
code=$2
input=$3
cd "$code"
export PYTHONPATH="$code:$study/liver_dili_v1/deps" PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2
python="$study/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
"$python" -m liver_dili_v2.epa --epa "$input/toxcast-v43.zip" --dsstox "$input/dsstox-2018.zip" \
  --prior "$input/assays-01" --out "$input/assays-04"
