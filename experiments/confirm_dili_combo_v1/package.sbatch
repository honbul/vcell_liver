#!/usr/bin/env bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=00:20:00
set -euo pipefail
root=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
cd "$1"
run=$2
export PYTHONPATH="$root/liver_dili_v1/deps:$root/vcell_liver/vcell_r12_runtime_cu126"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
"$root/vcell_liver_r11_chemical_probe/assets/study-env/bin/python" -m confirm_dili.package \
    --history "$run/historical" --encoder "$root/nextgen_5track/checkpoints/clamp" \
    --replay "$run/cli" --out "$run/package"
