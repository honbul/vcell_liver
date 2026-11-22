#!/usr/bin/env bash
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=01:30:00
set -euo pipefail
repository=$1
root=$2
run=$3
action=$4
storage=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
python="$storage/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
export PYTHONPATH="$repository/src:$root/v0plus-model-deps:$root/function/prepare-deps:/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver/vcell_r12_runtime_cu126"
export OPENBLAS_NUM_THREADS=5 MKL_NUM_THREADS=5 OMP_NUM_THREADS=5
runs="$root/v0plus-runs"
dataset="$root/licob-v0plus"
support="$dataset/author-c65cb2f"
supplement="$dataset/NIHMS1969891-supplement-Supplementary_Tables_S1-S13.xlsx"
case "$action" in
  register)
    args=(--original "$root/function/cross-v1" --core "$runs/source-core-92aa482/core"
          --extension "$runs/source-panel-c351b0b/extension" --gdsc "$runs/improve-gdsc-stereo-347009e/extension"
          --author "$runs/licob-author-2f1328b/cohort" --wide "$runs/licob-wide-rna-2833a20/cohort"
          --catalogue "$support/20261030-L1700-Bioactive-Compound-Library-I-96-well.xlsx"
          --cas "$support/cas" --stereo "$dataset/stereo-cas-v1" --supplement "$supplement" --out "$run/inputs")
    ;;
  predict)
    fold=${SLURM_ARRAY_TASK_ID:?}
    args=(--inputs "$run/inputs" --fold "$fold" --priors "$root/v0plus-public-priors"
          --targets "$root/external-code-v3/drug_targets.json" --auxiliary "$runs/depmap-25q3-bb8a47b"
          --hallmark /mnt/S1/sdata/pubdata/Virtual_Cell/MSigDB/2026.1.Hs/h.all.v2026.1.Hs.symbols.gmt
          --out "$run/parts/fold-$fold.parquet")
    ;;
  source|external)
    args=(--inputs "$run/inputs" --parts "$run/parts" --out "$run/$action.json")
    if [[ "$action" == external ]]; then args+=(--supplement "$supplement"); fi
    ;;
  *) exit 2 ;;
esac
exec "$python" -m vc_build.licob_cold "$action" "${args[@]}"
