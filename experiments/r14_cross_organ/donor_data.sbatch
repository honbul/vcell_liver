#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:30:00
set -euo pipefail
base="$(realpath "$PWD/../..")"
r13="$base/vcell_liver_r13_optimization"
export PYTHONPATH="$PWD:$r13:$r13/assets/external-gse306429/python310-h5py-3.15.1"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
if [[ "$1" == emeraldbay ]]; then
  exec "$python" -m src.r14_cross_organ.emeraldbay_data --source /mnt/S1/sdata/pubdata/Virtual_Cell/EmeraldBay --workers 4 --protocol "$PWD/reports/r14_cross_organ/EXECUTION_PROTOCOL.md" --out "$PWD/data/$1"
fi
exec "$python" -m src.r14_cross_organ.donor_data "$1" --base "$base" \
  --protocol "$PWD/reports/r14_cross_organ/EXECUTION_PROTOCOL.md" --output "$PWD/data/$1"
