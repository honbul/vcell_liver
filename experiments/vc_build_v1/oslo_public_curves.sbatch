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
case "${4:-register}" in
  register)
    args=(--out "$run/registered" register --source "$root/3d-public/oslo-crlm-hr94h42xdc-v3"
          --licob "$root/v0plus-runs/licob-cold-5bf53fc/inputs")
    ;;
  bridge)
    args=(--out "$run/dose_bridge.json" bridge --registered "$run/registered"
          --dose "$root/v0plus-runs/licob-dose-c471062/inputs")
    ;;
  *) exit 2 ;;
esac
exec "$python" -m vc_build.oslo_public_curves "${args[@]}"
