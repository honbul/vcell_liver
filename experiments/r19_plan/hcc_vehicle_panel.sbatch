#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
set -euo pipefail

run=$1
phase=$2
root=$(dirname "$run")
base=$(dirname "$root")
repo="$run/repo"
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
t2="$base/vcell_liver_goal_tracks/v1/t2"
export PYTHONPATH="$repo:$t2/dependencies:$t2-public/deps"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
args=(--reports "$repo/reports" --previous "$root/hcc-scaffold-refit.lWd6KCZt"
  --protocol "$repo/reports/r19_plan/HCC_VEHICLE_PANEL_PROTOCOL.md" --lock "$run/lock")
case "$phase" in
  lock)
    args+=(--metadata /mnt/S1/sdata/pubdata/Tahoe/Tahoe-100M/metadata/sample_metadata.parquet
      --controls "$base/vcell_liver_r12_multiaxis/assets/doses/source")
    ;;
  diagnose) args+=(--digest "$(cat "$run/lock.sha256")" --out "$run/diagnostic") ;;
  *) exit 2 ;;
esac
/usr/bin/time -v "$python" -m src.r19_plan.hcc_vehicle_panel "$phase" "${args[@]}"
if [[ "$phase" == lock ]]; then
  sha256sum "$run/lock/receipt.json" | cut -d' ' -f1 > "$run/lock.sha256"
fi
