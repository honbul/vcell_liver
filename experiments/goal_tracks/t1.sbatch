#!/usr/bin/env bash
set -euo pipefail

root="$1"
trial="$2"
dose="$3"
action="$4"
lab="$(dirname "$(dirname "$root")")"
cache="$root/t1-cache"
out="$root/t1/$trial/d$dose"
exports=4
export OPENBLAS_NUM_THREADS="$exports" OMP_NUM_THREADS="$exports" MKL_NUM_THREADS="$exports" PYTHONHASHSEED=0
export PYTHONPATH="$lab/vcell_liver/vcell_r12_runtime_cu126:${PYTHONPATH:-}"
export R13_DEVICE=cuda CUBLAS_WORKSPACE_CONFIG=:4096:8
python="$lab/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
protocol="reports/goal_tracks/PROTOCOL.md"
printf 'job=%s task=%s host=%s started=%s\n' "$SLURM_JOB_ID" "${SLURM_ARRAY_TASK_ID:-none}" "$(hostname)" "$(date -Iseconds)"
if [ "$action" = predict ]; then
    task="$SLURM_ARRAY_TASK_ID"
    minimum=20
    if [ "$task" -ge 6 ]; then
        minimum=200
    fi
    fold=$((task % 6 / 2))
    role=A
    if [ "$((task % 2))" -eq 1 ]; then
        role=B
    fi
    "$python" -m src.goal_tracks.t1_predict --config "$lab/vcell_liver_r13_optimization/configs/r12_multiaxis/config.yaml" \
        --assets "$lab/vcell_liver_r12_multiaxis/assets" --stage "$cache" --dose "$dose" --minimum-cells "$minimum" \
        --history "$lab/vcell_liver_r13_optimization/runs/r13-round1/compound" --protocol "$protocol" \
        --fold "$fold" --role "$role" --out "$out/predictions"
else
    "$python" -m src.goal_tracks.t1_score "$action" --stage "$cache" --protocol "$protocol" --predictions "$out/predictions" \
        --freeze "$out/freeze.json" --dose "$dose" --out "$out/scored"
fi
