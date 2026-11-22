#!/usr/bin/env bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=13G
#SBATCH --time=01:15:00
set -euo pipefail
root=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd "$1"
"$root/vcell_liver_r11_chemical_probe/assets/study-env/bin/python" -m confirm_dili.exposure \
    --archive "$2/pubchem18.zip" --cache "$2/parquet-01" --out "$2/audit-01"
