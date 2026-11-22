#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:20:00
set -euo pipefail

workspace="$1"
stage="$2"
base="$(dirname "$(dirname "$workspace")")"
code="$workspace/code"
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
deps="$base/vcell_liver_goal_tracks/v1"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=0
export PYTHONPATH="$code:$deps/t2/dependencies:$deps/t2-public/deps"
export CUDA_VISIBLE_DEVICES=''
cd "$code"
exec "$python" -m src.vc_build.state_heparg "$stage" \
    --source "$base/vcell_liver_r14_cross_organ/execution-v1/data/heparg_284321" \
    --run "$workspace/heparg-state-v1" --protocol "$code/state/HEPARG_PROTOCOL.md"
