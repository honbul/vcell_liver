#!/usr/bin/env bash
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
#SBATCH --time=02:00:00
set -euo pipefail

repository=$1
root=$2
run=$3
stage=$4
storage=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
study_python="$storage/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
runtime="$storage/vcell_liver/vcell_r12_runtime_cu126"
export PYTHONPATH="$repository/src:$root/v0plus-model-deps:$root/function/prepare-deps:$runtime"
export OPENBLAS_NUM_THREADS=5 MKL_NUM_THREADS=5 OMP_NUM_THREADS=5
pdo="$root/v0plus-runs/improve-pdo-347009e/development"
teacher="$run/teacher.parquet"
common=(
  --priors "$root/v0plus-public-priors"
  --hallmark /mnt/S1/sdata/pubdata/Virtual_Cell/MSigDB/2026.1.Hs/h.all.v2026.1.Hs.symbols.gmt
  --targets "$root/external-code-v3/drug_targets.json"
)
case "$stage" in
  teacher)
    args=(--original "$root/function/cross-v1" --core "$root/v0plus-runs/source-core-92aa482/core"
          --extension "$root/v0plus-runs/source-panel-c351b0b/extension"
          --auxiliary "$root/v0plus-runs/depmap-25q3-bb8a47b" --out "$teacher" "${common[@]}")
    ;;
  adapt)
    studies=("IMPROVE dataset 1" "IMPROVE dataset 2" "IMPROVE dataset 3" "Pilot 1")
    study=${studies[$SLURM_ARRAY_TASK_ID]}
    args=(--study "$study" --teacher "$teacher" --out "$run/forecasts/${study// /_}.parquet" "${common[@]}")
    ;;
  score)
    args=(--teacher "$teacher" --forecasts "$run/forecasts" --out "$run/score.json")
    ;;
  *) exit 2 ;;
esac
exec "$study_python" -m vc_build.pdo_transfer_cv "$stage" --pdo "$pdo" "${args[@]}"
