#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:25:00
set -euo pipefail
run=$1
stage=$2
repo="$run/repo"
base=$(dirname "$(dirname "$run")")
deps="$base/vcell_liver_goal_tracks/v1"
r14="$base/vcell_liver_r14_cross_organ"
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
export PYTHONPATH="$repo:$deps/t2/dependencies:$deps/t2-public/deps"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
options=(--work "$run" --t3 "$deps/t3" --cache "$r14/execution-v1" --index "$r14/metadata/v4"
         --matrix "$repo/reports/r19_plan/data/preflight/r18/index/GSE92742/receipt.json"
         --protocol "$repo/reports/r19_plan/HEPARG_DONOR_PROTOCOL.md")
if [[ "$stage" != source ]]; then
  dataset=$3
  options+=(--dataset "$dataset")
  case "$stage" in
    predict) options+=(--fold "$SLURM_ARRAY_TASK_ID");;
    score) options+=(--expected-sha256 "$(cat "$run/$dataset/freeze.sha256")");;
  esac
fi
/usr/bin/time -v "$python" -u -m src.r19_plan.heparg_donor "$stage" "${options[@]}"
if [[ "$stage" == lock ]]; then
  sha256sum "$run/$dataset/freeze.json" | cut -d' ' -f1 > "$run/$dataset/freeze.sha256"
fi
