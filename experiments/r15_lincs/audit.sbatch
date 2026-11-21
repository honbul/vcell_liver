#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
set -euo pipefail
study="$1"
destination="$2"
base=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
export PYTHONPATH="$PWD:$base/vcell_liver_goal_tracks/v1/t2-public/deps"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
exec "$python" -m src.r15_lincs.position_audit --root /mnt/S1/sdata/pubdata/LINCS \
  --study "$study" --out "$destination"
