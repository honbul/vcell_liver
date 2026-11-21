#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:25:00
set -euo pipefail

run=$1
phase=$2
base=$(dirname "$(dirname "$run")")
repo="$run/repo"
goal_tracks="$base/vcell_liver_goal_tracks/v1"
assets="$base/vcell_liver_r11_chemical_probe/assets"
export PYTHONPATH="$repo:$goal_tracks/t2/dependencies:$goal_tracks/t2-public/deps"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
options=(--protocol "$repo/reports/p4/PROTOCOL.md" --archive "$repo/reports" --prior "$goal_tracks"
         --p123 "$base/vcell_liver_p123/v1" --out "$base/vcell_liver_p4/v1"
         --design "$repo/reports/r19_plan/HCC_SCAFFOLD_PROTOCOL.md" --features "$assets/features/features.npz")
case "$phase" in
  audit)
    "$assets/study-env/bin/python" -m src.r19_plan.hcc_scaffold audit "${options[@]}" --product "$run/structure"
    sha256sum "$run/structure/receipt.json" | cut -d' ' -f1 > "$run/structure.sha256"
    ;;
  score)
    "$assets/study-env/bin/python" -m src.r19_plan.hcc_scaffold score "${options[@]}" --product "$run/score" \
      --structure "$run/structure" --expected-sha256 "$(cat "$run/structure.sha256")"
    ;;
  *) exit 2 ;;
esac
