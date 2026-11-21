#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail

cd "$1"
base=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OMP_NUM_THREADS=$OPENBLAS_NUM_THREADS MKL_NUM_THREADS=$OPENBLAS_NUM_THREADS

exec "$python" -m src.tahoe_position.sciplex_replicate \
  --cache "$base/vcell_liver_r10_dose_composition/assets/cache/r10_v1" \
  --output "$2" --protocol "$3" --protocol-sha256 "$4"
