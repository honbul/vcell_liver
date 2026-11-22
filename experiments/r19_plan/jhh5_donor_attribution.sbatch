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
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
public="$base/vcell_liver_goal_tracks/v1"
export PYTHONPATH="$repo:$public/t2/dependencies:$public/t2-public/deps"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
args=(--run "$run" --previous "$root/jhh5-transfer.r0tiL2nJ"
  --previous-protocol "$report/HCC_LINCS_24H_TRANSFER_PROTOCOL.md" --protocol "$report/JHH5_DONOR_ATTRIBUTION_PROTOCOL.md"
  --previous-pin "$report/data/hcc_lincs24/transfer/freeze.json" --access-pin "$report/data/hcc_lincs24/access/receipt.json")
case "$phase" in
  predict)
    roles=(A B C)
    args+=(--role "${roles[${SLURM_ARRAY_TASK_ID:?}]}")
    ;;
  score) args+=(--digest "$(cat "$run/freeze.sha256")") ;;
  freeze) ;;
  *) exit 2 ;;
esac
/usr/bin/time -v "$python" -u -m src.r19_plan.jhh5_donor_attribution "$phase" "${args[@]}"
if [[ "$phase" == freeze ]]; then
  sha256sum "$run/freeze.json" | cut -d' ' -f1 > "$run/freeze.sha256"
fi
