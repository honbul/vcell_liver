#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
set -euo pipefail

run=$1
root=$(dirname "$run")
base=$(dirname "$root")
repo="$run/repo"
paper="$repo/reports/r19_plan"
fixture="$root/hcc-off-score-smoke.zWVNzblu"
public="$base/vcell_liver_goal_tracks/v1"
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
export PYTHONPATH="$repo:$public/t2/dependencies:$public/t2-public/deps"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
exec /usr/bin/time -v "$python" -u -m src.r19_plan.hcc_off_position_pilot_repeat \
  --audit "$root/hcc-off-pilot.GzfiFSmB/audit" --forecast "$root/hcc-off-pilot-freeze.IhWnfPBE" \
  --forecast-pin "$paper/data/hcc_off_position_pilot/forecast/freeze.json" \
  --native "$root/hcc-off-pilot-native.KbzmD1lw" --native-pin "$paper/data/hcc_off_position_pilot/native/forecast/freeze.json" \
  --native-protocol "$paper/HCC_OFF_POSITION_PILOT_NATIVE_PROTOCOL.md" --protocol "$paper/HCC_OFF_POSITION_PILOT_SCORE_PROTOCOL.md" \
  --wells "$fixture/synthetic/wells.tsv" --cultures "$fixture/synthetic/cultures.tsv" --qc-record "$fixture/synthetic/qc.md" \
  --assay-receipt "$fixture/synthetic/assay.json" --measurements "$fixture/synthetic/measurements.npz" \
  --intake "$fixture/intake" --intake-sha256 "$(cat "$fixture/intake.sha256")" --repeat-protocol "$paper/HCC_OFF_POSITION_PILOT_REPEAT_PROTOCOL.md" --synthetic --output "$run/score"
