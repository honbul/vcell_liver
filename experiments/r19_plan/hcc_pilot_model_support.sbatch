#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=00:20:00
set -euo pipefail

run=$1
phase=$2
base=$(dirname "$(dirname "$run")")
r="$run/repo/reports"
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
deps="$base/vcell_liver_goal_tracks/v1"
export PYTHONPATH="$run/repo:$base/vcell_liver/vcell_r12_runtime_cu126:$base/vcell_liver/vcell_r12_runtime_transformers:$deps/t2/dependencies:$deps/t2-public/deps"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

exec /usr/bin/time -v "$python" -m src.r19_plan.hcc_pilot_model_support "$phase" \
    --base "$base" --reports "$r" --protocol "$r/r19_plan/HCC_PILOT_MODEL_SUPPORT_PROTOCOL.md" --out "$run/$phase"
