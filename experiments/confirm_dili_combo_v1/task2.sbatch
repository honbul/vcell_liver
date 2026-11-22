#!/usr/bin/env bash
#SBATCH --cpus-per-task=5
#SBATCH --mem=19G
#SBATCH --time=01:30:00
set -euo pipefail
root=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
cd "$1"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
"$root/vcell_liver_r11_chemical_probe/assets/study-env/bin/python" -m confirm_combo.evaluate --out "$2"
