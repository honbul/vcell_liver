#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
set -euo pipefail

run=$1
phase=$2
root=$(dirname "$run")
base=$(dirname "$root")
repo="$run/repo"
paper="$repo/reports/r19_plan"
public="$base/vcell_liver_goal_tracks/v1"
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
export PYTHONPATH="$repo:$public/t2/dependencies:$public/t2-public/deps"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
fixtures="$run/synthetic"
common=(--audit "$root/hcc-off-pilot.GzfiFSmB/audit" --forecast "$root/hcc-off-pilot-freeze.IhWnfPBE"
  --forecast-pin "$paper/data/hcc_off_position_pilot/forecast/freeze.json")
if [[ "$phase" == lock ]]; then
  "$python" "$repo/experiments/r19_plan/hcc_off_position_score_fixture.py" "${common[@]}" --out "$fixtures"
fi
args=("${common[@]}" --wells "$fixtures/wells.tsv" --cultures "$fixtures/cultures.tsv" --qc-record "$fixtures/qc.md"
  --assay-receipt "$fixtures/assay.json" --protocol "$paper/HCC_OFF_POSITION_PILOT_SCORE_PROTOCOL.md" --synthetic)
case "$phase" in
  lock) args+=(--output "$run/intake") ;;
  score) args+=(--output "$run/score" --measurements "$fixtures/measurements.npz" --intake "$run/intake"
    --intake-sha256 "$(cat "$run/intake.sha256")") ;;
  *) exit 2 ;;
esac
/usr/bin/time -v "$python" -u -m src.r19_plan.hcc_off_position_pilot_score "$phase" "${args[@]}"
if [[ "$phase" == lock ]]; then
  sha256sum "$run/intake/receipt.json" | cut -d' ' -f1 > "$run/intake.sha256"
fi
