#!/usr/bin/env bash
set -euo pipefail
root="$1"
p1="$2"
p2="$3"
runner="$root/p1-code/experiments/p123/p1.sbatch"
for dose in 5 0.5 0.05; do
    target="$root/p2/d$dose"
    mkdir -p "$target"
    common=(--track P2 --extra-root "$root/p2/predictions" --coverage "$root/lincs/tahoe_coverage.tsv"
            --freeze "$target/freeze.json")
    freeze=$(sbatch --parsable --dependency="afterok:$p1:$p2" --mem=20G \
        --chdir "$root/p1-code" -J "p2-freeze-$dose" -o "$root/logs/p2-freeze-%j.out" \
        "$runner" "$root" freeze "$dose" "${common[@]}")
    score=$(sbatch --parsable --dependency="afterok:$freeze" --mem=22G \
        --chdir "$root/p1-code" -J "p2-score-$dose" -o "$root/logs/p2-score-%j.out" \
        "$runner" "$root" score "$dose" "${common[@]}" --out "$target/scored")
    printf "%s\t%s\t%s\n" "$dose" "$freeze" "$score"
done
