#!/usr/bin/env bash
# Submit every registered F1 ablation run once (run on G1 from the worktree root).
# Records <run-id>\t<job-id> in logs/f1_ablation_submissions.tsv; refuses to resubmit a run ID.
set -euo pipefail
cd "$(dirname "$0")/../.."
journal=logs/f1_ablation_submissions.tsv
touch "$journal"
gres="${F1_GRES:-gpu:a100:1}"
# Optional extra sbatch flags, e.g. F1_SBATCH_EXTRA="--dependency=afterok:<smoke job>"
read -r -a extra <<< "${F1_SBATCH_EXTRA:-}"
ids=$(.venvs/scgpt/bin/python -c 'import json;print(" ".join(r["run_id"] for r in json.load(open("experiments/feedback/f1_ablation_runs.json"))["runs"]))')
for id in $ids; do
  if grep -q "^$id	" "$journal"; then echo "$id already submitted: $(grep "^$id	" "$journal" | cut -f2)"; continue; fi
  if [ -e "assets/runs/A/$id" ]; then echo "$id already has a run directory; resume manually"; continue; fi
  job=$(sbatch --parsable "${extra[@]}" --gres="$gres" --job-name="f1a_${id#A-Norman-v1-scgpt-}" experiments/feedback/f1_ablation_train.sbatch --run-id "$id")
  printf '%s\t%s\t%s\t%s\t%s\n' "$id" "$job" "$gres" "$(date -Is)" "${F1_SBATCH_EXTRA:-}" >> "$journal"
  echo "$id $job"
done
