#!/usr/bin/env bash
set -euo pipefail
# Run on G1 after staging a code snapshot. The Round 1 allocation may already exist.
code=$1
r1=$2
log=logs
wrapper="$code/experiments/nextgen_h3_pcgrad/run.sbatch"
fit="$code/experiments/nextgen_h3_pcgrad/fit-array.sbatch"
cpu=(-p cpu -c 4 --mem=14G -t 00:30:00)
gpu=(-p gpu -c 4 --mem=14G -t 00:35:00 --gres=gpu:a40:1 --array=0-8%2)
submit() {
    local stage=$1
    local predecessor=$2
    shift 2
    local id
    id=$(sbatch --parsable -J "ng-h3-$stage" --dependency="afterok:$predecessor" --kill-on-invalid-dep=yes "$@")
    printf '%s\t%s\n' "$stage" "$id" >> "$log/submissions.tsv"
    printf '%s\n' "$id"
}
parent=$(submit r1-select "$r1" "${cpu[@]}" -o "$log/r1-select-%j.log" "$wrapper" "$code" -m h3_pcgrad.select round1)
for stage in round2-block round2-opt round2-head; do
    fit_id=$(submit "$stage" "$parent" "${gpu[@]}" -o "$log/$stage-%A_%a.log" "$fit" "$code" "$stage")
    parent=$(submit "$stage-select" "$fit_id" "${cpu[@]}" -o "$log/$stage-select-%j.log" "$wrapper" "$code" -m h3_pcgrad.select "$stage")
done
submit outer-fit "$parent" "${gpu[@]}" -o "$log/outer-fit-%A_%a.log" "$fit" "$code" outer
