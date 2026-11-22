#!/usr/bin/env bash
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
#SBATCH --time=01:30:00
set -euo pipefail
repository=$1
root=$2
run=$3
stage=$4
storage=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
python="$storage/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
runtime="$storage/vcell_liver/vcell_r12_runtime_cu126"
export PYTHONPATH="$repository/src:$root/v0plus-model-deps:$root/function/prepare-deps:$runtime"
export OPENBLAS_NUM_THREADS=5 MKL_NUM_THREADS=5 OMP_NUM_THREADS=5
archive="$root/v0plus-runs/final-8e5d3f4"
args=(--original "$root/function/cross-v1" --core "$root/v0plus-runs/source-core-92aa482/core")
case "$stage" in
  cpu|gpu)
    if [[ "$stage" == cpu ]]; then
      components=(linear lightgbm cellhit)
      extra=()
    else
      components=(catboost velodrome_inductive velodrome_transductive codeae_inductive codeae_transductive
                  v0plus_inductive v0plus_transductive structured structured_old)
      extra=(--gpu)
    fi
    method=${components[$SLURM_ARRAY_TASK_ID]}
    stage=predict
    args+=(--component "$method" --settings "$archive/settings.json"
           --priors "$root/v0plus-public-priors"
           --hallmark /mnt/S1/sdata/pubdata/Virtual_Cell/MSigDB/2026.1.Hs/h.all.v2026.1.Hs.symbols.gmt
           --targets "$root/external-code-v3/drug_targets.json"
           --auxiliary "$root/v0plus-runs/depmap-25q3-bb8a47b" --upstream "$root/v0plus-upstream"
           --out "$run/forecasts/$method.parquet" "${extra[@]}")
    ;;
  package)
    export CUDA_VISIBLE_DEVICES=""
    args+=(--package "$root/v0plus-runs/corrected-v0plus-bc00031/V0plus"
           --out "$run/forecasts/release70.parquet")
    ;;
  merge)
    args+=(--historical "$archive/final.parquet" --parts "$run/forecasts" --out "$run/final.parquet")
    ;;
  score)
    args+=(--forecast "$run/final.parquet" --out "$run/score.json")
    ;;
  *) exit 2 ;;
esac
exec "$python" -m vc_build.limore_corrected "$stage" "${args[@]}"
