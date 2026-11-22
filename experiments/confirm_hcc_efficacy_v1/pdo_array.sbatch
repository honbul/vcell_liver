#!/usr/bin/env bash
set -euo pipefail
root=$1
stage=$2
studies=(LICOB Wong Yang)
sizes=(2 4 8)
id=${SLURM_ARRAY_TASK_ID:?}
study=${studies[id/3]}
k=${sizes[id%3]}
out="$root/task3/$study/k$k"
extra=()
if [[ "$stage" == score ]]; then
    extra=(--forecast "$out/predict")
fi
exec bash "$root/code/jobs/run.sbatch" "$root/code" -m confirm_hcc.fewshot "$stage" \
    --study "$study" --k "$k" --prior "$root/task3/source.parquet" \
    --contract "$root/code/jobs/contract.json" --registration "$root/task3/registration" \
    --out "$out/$stage" "${extra[@]}"
