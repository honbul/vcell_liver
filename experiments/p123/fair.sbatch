#!/usr/bin/env bash
set -euo pipefail
study="$1"
lane="$2"
base="$(dirname "$(dirname "$study")")"
t2="$base/vcell_liver_goal_tracks/v1"
r12="$base/vcell_liver_r12_multiaxis"
r11="$base/vcell_liver_r11_chemical_probe/assets"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONHASHSEED=0
export PYTHONPATH="$PWD:$base/vcell_liver/vcell_r12_runtime_cu126:$base/vcell_liver/vcell_r12_runtime_transformers:$t2/t2/dependencies:$t2/t2-public/state-deps:$t2/t2-public/deps"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec "$r11/study-env/bin/python" -m src.p123.p1_fair \
  --lane "$lane" --task "$SLURM_ARRAY_TASK_ID" --assets "$r12/assets" \
  --config "$r12/configs/r12_multiaxis/config.yaml" \
  --protocol reports/p123/PROTOCOL.md --old-protocol reports/goal_tracks/PROTOCOL.md \
  --old-reports reports/goal_tracks/data --positions "$study/p1-position" \
  --t2-root "$t2" --out "$study/p1-fair"
