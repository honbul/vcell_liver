#!/usr/bin/env bash
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=02:30:00
set -euo pipefail
root=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
vc="$root/vc_build_v1"
work="$root/nextgen_h123/h2_multiview"
python="$root/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
code=$1
shift
hdf="$vc/v0plus-runs/lincs-pathway-e382c8a-hdf/dependencies/wheel"
export PYTHONPATH="$code/src:$hdf:$vc/v0plus-model-deps:$vc/function/prepare-deps:$root/vcell_liver/vcell_r12_runtime_cu126"
export OPENBLAS_NUM_THREADS=5 OMP_NUM_THREADS=5 MKL_NUM_THREADS=5
export PYTHONDONTWRITEBYTECODE=1
exec "$python" -m "nextgen_h2.$1" "${@:2}"
