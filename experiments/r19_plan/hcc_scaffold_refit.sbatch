#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=18G
#SBATCH --time=00:30:00
set -euo pipefail

run=$1
phase=$2
base=$(dirname "$(dirname "$run")")
repo="$run/repo"
goal="$base/vcell_liver_goal_tracks/v1"
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
export PYTHONPATH="$repo:$goal/t2/dependencies:$goal/t2-public/deps"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
args=(--run "$run" --base "$base" --archive "$repo/reports"
      --protocol "$repo/reports/r19_plan/HCC_SCAFFOLD_REFIT_PROTOCOL.md")
case "$phase" in
  predict)
    roles=(A B)
    args+=(--fold "$((SLURM_ARRAY_TASK_ID / 2))" --role "${roles[SLURM_ARRAY_TASK_ID % 2]}")
    ;;
  score)
    args+=(--expected-sha256 "$(cat "$run/freeze.sha256")")
    ;;
esac
/usr/bin/time -v "$python" -u -m src.r19_plan.hcc_scaffold_refit "$phase" "${args[@]}"
if [[ "$phase" == freeze ]]; then
  sha256sum "$run/freeze.json" | cut -d' ' -f1 > "$run/freeze.sha256"
fi
