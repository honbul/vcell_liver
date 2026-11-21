#!/usr/bin/env bash
# Submit a score only when the upstream model forecasts and the matching T1 freeze succeed.
set -euo pipefail
root="$1"
work="$2"
state_root="$3"
model_dependencies="$4"
shift 4
out="$root/t2"
models=(XPert_morgan chemCPA_P0 TranSiGen_Tahoe ChemBERTa_R11 FiLM_R12 Concat_R12 scGPT_C7 UCE_C8)
arguments=()
for name in "${models[@]}"; do
    arguments+=(--model "$name=$out/predictions/$name")
done
arguments+=(--model "PRnet-well-adaptation=$root/t2-public/prnet")
arguments+=(--model "State-Morgan-well-adaptation=$state_root")
for slice in "$@"; do
    dose="${slice%:*}"
    anchor="${slice##*:}"
    dose_dir="$out/d$dose"
    mkdir -p "$dose_dir"
    freeze=$(sbatch --parsable -J "goal_t2_freeze_d$dose" -p cpu -c 4 --mem 18G -t 03:00:00 \
        -d "afterok:$anchor:$model_dependencies" --kill-on-invalid-dep=yes -D "$work" \
        -o "$out/logs/freeze-d$dose-%j.log" "$work/t2.sbatch" "$root" freeze --dose "$dose" \
        --t1 "$root/t1/first/d$dose" --freeze "$dose_dir/freeze.json" "${arguments[@]}")
    score=$(sbatch --parsable -J "goal_t2_score_d$dose" -p cpu -c 6 --mem 36G -t 04:00:00 \
        -d "afterok:$freeze" --kill-on-invalid-dep=yes -D "$work" -o "$out/logs/score-d$dose-%j.log" \
        "$work/t2.sbatch" "$root" score --dose "$dose" --freeze "$dose_dir/freeze.json" --out "$dose_dir/scored")
    printf '%s %s %s\n' "$dose" "$freeze" "$score" | tee -a "$out/score_jobs.tsv"
done
