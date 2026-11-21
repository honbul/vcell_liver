#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:25:00
set -euo pipefail

run=$1
phase=$2
root=$(dirname "$run")
base=$(dirname "$root")
repo="$run/repo"
report="$repo/reports/r19_plan"
public="$base/vcell_liver_goal_tracks/v1"
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
export PYTHONPATH="$repo:$public/t2/dependencies:$public/t2-public/deps"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
protocol="$report/JHH7_POSITION_ATTRIBUTION_PROTOCOL.md"
if [[ "$phase" == audit ]]; then
  exec /usr/bin/time -v "$python" -u -m src.r19_plan.huh7_cmap_support --base "$base" --reports "$repo/reports" \
    --r18-registered "$root/audit.YBfVmbuo/r18-registered" --r18-lock "$root/preflight.kwjS2aLo/r18-input-lock" \
    --r18-access "$root/access-plan.h0O8ex4z/allowlists" --r18-index "$root/matrix-index.91pT0e4H/index" \
    --protocol "$protocol" --out "$run/huh7-support"
fi
args=(--run "$run" --previous "$root/jhh7-low-dose.ucgrUrtp" --protocol "$protocol"
  --access-pin "$report/data/jhh7_low_dose/access/receipt.json" --previous-pin "$report/data/jhh7_low_dose/freeze.json"
  --previous-protocol "$report/JHH7_LOW_DOSE_PROTOCOL.md")
roles=(A B C)
case "$phase" in
  predict) args+=(--role "${roles[${SLURM_ARRAY_TASK_ID:?}]}" ) ;;
  score) args+=(--digest "$(cat "$run/freeze.sha256")") ;;
  freeze) ;;
  *) exit 2 ;;
esac
/usr/bin/time -v "$python" -u -m src.r19_plan.jhh7_position_attribution "$phase" "${args[@]}"
if [[ "$phase" == freeze ]]; then
  sha256sum "$run/freeze.json" | cut -d' ' -f1 > "$run/freeze.sha256"
fi
