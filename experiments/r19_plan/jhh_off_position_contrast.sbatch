#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:20:00
set -euo pipefail

run=$1
root=$(dirname "$run")
base=$(dirname "$root")
repo="$run/repo"
reports="$repo/reports/r19_plan"
public="$base/vcell_liver_goal_tracks/v1"
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
export PYTHONPATH="$repo:$public/t2/dependencies:$public/t2-public/deps"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
case "${SLURM_ARRAY_TASK_ID:?}" in
  0) cell=JHH5; prefix=jhh5; native=jhh5-off-native.xTAKElO6; previous=jhh5-off-position.0ldmuIkJ ;;
  1) cell=JHH7; prefix=jhh7; native=jhh7-off-native.HAX62Mi9; previous=jhh7-off-position.GMpuNJHy ;;
  *) exit 2 ;;
esac
orig="$reports/data/${prefix}_off_position"
extended="${orig}_native"
exec /usr/bin/time -v "$python" -u -m src.r19_plan.jhh_off_position_contrast \
  --cell "$cell" --run "$root/$native" --previous "$root/$previous" \
  --protocol "$reports/JHH_OFF_POSITION_CONTRAST_PROTOCOL.md" \
  --previous-protocol "$reports/${cell}_OFF_POSITION_PROTOCOL.md" --previous-pin "$orig/freeze.json" \
  --native-protocol "$reports/${cell}_OFF_POSITION_NATIVE_PROTOCOL.md" --native-pin "$extended/freeze.json" \
  --score-pin "$extended/score/receipt.json" --access-pin "$orig/access/receipt.json" --out "$run/$cell"
