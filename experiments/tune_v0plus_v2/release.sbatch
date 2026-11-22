#!/usr/bin/env bash
#SBATCH --cpus-per-task=6
#SBATCH --mem=27G
#SBATCH --time=02:00:00
set -euo pipefail
study=$1
workspace=$2
runs=$3
action=$4
out=$5
cd "$workspace"
export PYTHONPATH="$workspace:$workspace/src:$study/vc_build_v1/v0plus-model-deps:$study/liver_dili_v1/deps"
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
python="$study/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
root="$study/vc_build_v1"
base=(--bundle "$root/function/cross-v1" --package "$out/package")
case "$action" in
  package) "$python" -m v0plus_v2.release package "${base[@]}" --runs "$runs" --out "$out/package" ;;
  predict) "$python" -m v0plus_v2.release predict "${base[@]}" --out "$out/limore.parquet" \
    --baselines "$root/v0plus-runs/final-8e5d3f4/final.parquet" --fm "$study/nextgen_5track/fm_store_v1" ;;
  score) "$python" -m v0plus_v2.release score "${base[@]}" --forecast "$out/limore.parquet" --out "$out/limore_score.json" ;;
  *) exit 2 ;;
esac
