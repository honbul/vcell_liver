#!/usr/bin/env bash
#SBATCH --cpus-per-task=3
#SBATCH --mem=12G
#SBATCH --time=00:20:00
set -euo pipefail
code=$1
root=$2
run=$3
storage=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
python="$storage/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
export PYTHONPATH="$code/src:$root/v0plus-model-deps:$root/function/prepare-deps"
export OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 OMP_NUM_THREADS=3
exec "$python" -m vc_build.oslo_public_curves --source "$root/3d-public/oslo-crlm-hr94h42xdc-v3" \
    --licob "$root/v0plus-runs/licob-cold-5bf53fc/inputs" --out "$run/registered"
