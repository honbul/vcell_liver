#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:20:00
set -euo pipefail

run=$1
phase=$2
audit=$3
plan_root=$(dirname "$run")
base=$(dirname "$plan_root")
repo="$run/repo"
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
tahoe="$base/vcell_liver_goal_tracks/v1"
export PYTHONPATH="$repo:$tahoe/t2/dependencies:$tahoe/t2-public/deps"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
parameters=(--protocol "$repo/reports/p4/PROTOCOL.md" --archive "$repo/reports" --prior "$tahoe"
    --p123 "$base/vcell_liver_p123/v1" --out "$base/vcell_liver_p4/v1"
    --audit "$audit" --matrix "$repo/reports/r19_plan/data/preflight/r18/index/GSE92742/receipt.json"
    --design "$repo/reports/r19_plan/SNU423_DONOR_PROTOCOL.md" --run "$run")
case "$phase" in
  predict)
    roles=(A B)
    task=$SLURM_ARRAY_TASK_ID
    "$python" -m src.r19_plan.snu423_donor predict "${parameters[@]}" --fold "$((task/2))" --role "${roles[task%2]}"
    ;;
  freeze)
    "$python" -m src.r19_plan.snu423_donor freeze "${parameters[@]}"
    sha256sum "$run/freeze.json" | cut -d' ' -f1 > "$run/freeze.sha256"
    ;;
  score)
    "$python" -m src.r19_plan.snu423_donor score "${parameters[@]}" --expected-sha256 "$(cat "$run/freeze.sha256")"
    ;;
  *) exit 2 ;;
esac
