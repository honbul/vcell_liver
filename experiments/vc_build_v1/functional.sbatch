#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:45:00
set -euo pipefail

repository=$1
data=$2
run=$3
shift 3
storage=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
python="$storage/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
export PYTHONPATH="$repository/src"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
cd "$repository"
exec /usr/bin/time -v "$python" -m vc_build.train \
  --config configs/vc_build_v1/contract.yaml --data-dir "$data" --run-dir "$run" "$@"
