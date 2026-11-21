#!/usr/bin/env bash
set -euo pipefail

root="$1"
action="$2"
shift 2
lab="$(dirname "$(dirname "$root")")"
case_root="$root/t2"
python="$lab/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONHASHSEED=0
export PYTHONPATH="$lab/vcell_liver/vcell_r12_runtime_cu126:$case_root/dependencies:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
printf 'job=%s task=%s host=%s start=%s\n' "$SLURM_JOB_ID" "${SLURM_ARRAY_TASK_ID:-none}" "$(hostname)" "$(date -Iseconds)"
stage="$root/t1-cache"
protocol=reports/goal_tracks/PROTOCOL.md

if [ "$action" = predict ]; then
    model="$1"
    "$python" -m src.goal_tracks.t2_predict --model "$model" --task "$SLURM_ARRAY_TASK_ID" \
        --assets "$lab/vcell_liver_r12_multiaxis/assets" --config configs/r12_multiaxis/config.yaml \
        --stage "$stage" --protocol "$protocol" --extra-context "$case_root/context-v2" \
        --out "$case_root/predictions/$model"
else
    "$python" -m src.goal_tracks.t2_score "$action" --stage "$stage" --protocol "$protocol" "$@"
fi
