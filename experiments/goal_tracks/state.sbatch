#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
set -euo pipefail
v1=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_goal_tracks/v1
base=$(realpath "$v1/../..")
r12="$base/vcell_liver_r12_multiaxis"
r11="$base/vcell_liver_r11_chemical_probe/assets"
public="$v1/t2-public"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$PWD:$v1/t1-code:$base/vcell_liver/vcell_r12_runtime_cu126:$base/vcell_liver/vcell_r12_runtime_transformers:$public/state-deps:$public/deps"
args=(--upstream "$public/upstream-state")
if [[ "${1:-}" == smoke ]]; then
  args+=(--smoke)
else
  args+=(--assets "$r12/assets" --config "$r12/configs/r12_multiaxis/config.yaml"
         --protocol reports/goal_tracks/PROTOCOL.md --staged "$v1/t1-cache"
         --out "$public/state-v2" --extra-cells "$v1/t2/context-v2/cells-v2" --task "$SLURM_ARRAY_TASK_ID")
fi
exec "$r11/study-env/bin/python" -m src.goal_tracks.t2_state "${args[@]}"
