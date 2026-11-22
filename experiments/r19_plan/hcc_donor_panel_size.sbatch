#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
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
args=(--run "$run" --reports "$report" --protocol "$report/HCC_DONOR_PANEL_SIZE_PROTOCOL.md"
  --jhh5 "$root/jhh5-transfer.r0tiL2nJ" --attribution "$root/jhh5-donor-check.qZGK4C5u"
  --jhh7 "$root/jhh7-transfer.HRBqXsAJ")
case "$phase" in
  predict)
    roles=(A B C)
    args+=(--role "${roles[${SLURM_ARRAY_TASK_ID:?}]}")
    ;;
  score) args+=(--digest "$(cat "$run/freeze.sha256")") ;;
  freeze) ;;
  *) exit 2 ;;
esac
/usr/bin/time -v "$python" -u -m src.r19_plan.hcc_donor_panel_size "$phase" "${args[@]}"
if [[ "$phase" == freeze ]]; then
  sha256sum "$run/freeze.json" | cut -d' ' -f1 > "$run/freeze.sha256"
fi
