#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
set -euo pipefail

run=$1
cell=$2
phase=$3
root=$(dirname "$run")
base=$(dirname "$root")
repo="$run/repo"
reports="$repo/reports/r19_plan"
public="$base/vcell_liver_goal_tracks/v1"
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
export PYTHONPATH="$repo:$public/t2/dependencies:$public/t2-public/deps"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
case "$cell" in
  JHH5) previous=jhh5-off-position.0ldmuIkJ ;;
  JHH7) previous=jhh7-off-position.GMpuNJHy ;;
  *) exit 2 ;;
esac
prefix="${cell,,}"
out="$run/$cell"
orig="$reports/data/${prefix}_off_position"
roles=(A B C)
extra=()
case "$phase" in
  predict) extra+=(--role "${roles[${SLURM_ARRAY_TASK_ID:?}]}") ;;
  freeze) ;;
  score) extra+=(--digest "$(cat "$out/freeze.sha256")") ;;
  *) exit 2 ;;
esac
/usr/bin/time -v "$python" -u -m src.r19_plan.jhh_donor_variance "$phase" \
  --cell "$cell" --run "$out" --previous "$root/$previous" --previous-pin "$orig/freeze.json" \
  --previous-protocol "$reports/${cell}_OFF_POSITION_PROTOCOL.md" --access-pin "$orig/access/receipt.json" \
  --native-protocol "$reports/${cell}_OFF_POSITION_NATIVE_PROTOCOL.md" --previous-score-pin "$orig/score/receipt.json" \
  --protocol "$reports/JHH_DONOR_VARIANCE_PROTOCOL.md" "${extra[@]}"
if [[ "$phase" == freeze ]]; then
  sha256sum "$out/freeze.json" | cut -d' ' -f1 > "$out/freeze.sha256"
fi
