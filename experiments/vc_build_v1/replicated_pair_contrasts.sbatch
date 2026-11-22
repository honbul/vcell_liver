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
python="$storage/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
runtime="$storage/vcell_liver/vcell_r12_runtime_cu126"
export PYTHONPATH="$repository/src:$root/v0plus-model-deps:$root/function/prepare-deps:$runtime"
export OPENBLAS_NUM_THREADS=5 MKL_NUM_THREADS=5 OMP_NUM_THREADS=5
args=(--original "$root/function/cross-v1" --core "$root/v0plus-runs/source-core-92aa482/core")
licob="$root/v0plus-runs/licob-wide-rna-2833a20/cohort"
case "$stage" in
  cv|external-predict)
    if [[ "$stage" == cv ]]; then
      assays=(pooled CTRPv2 GDSC2 PRISM20Q2)
      mode=${assays[$((SLURM_ARRAY_TASK_ID / 3))]}
      fold=$((SLURM_ARRAY_TASK_ID % 3))
      outfile="$mode-$fold.parquet"
    else
      studies=(LIMORE LICOB)
      mode=${studies[$SLURM_ARRAY_TASK_ID]}
      fold=0
      outfile="$mode.parquet"
    fi
    stage=predict
    args+=(--mode "$mode" --fold "$fold" --archive "$root/v0plus-runs/response-components-bf4fe80/archive" --licob "$licob"
          --cv-archive "$root/v0plus-runs/expanded-panel-81119c8"
          --confirm "$storage/confirm_20261123/worker1_eff/task1/limore.parquet" --priors "$root/v0plus-public-priors"
          --hallmark /mnt/S1/sdata/pubdata/Virtual_Cell/MSigDB/2026.1.Hs/h.all.v2026.1.Hs.symbols.gmt
          --targets "$root/external-code-v3/drug_targets.json" --auxiliary "$root/v0plus-runs/depmap-25q3-bb8a47b"
          --out "$run/forecasts/$outfile")
    ;;
  source)
    args+=(--forecasts "$run/forecasts" --out "$run/source_score.json")
    ;;
  external)
    args+=(--forecasts "$run/forecasts" --licob "$licob" --limore-labels "$root/function/cross-v1/test_labels/outcomes.parquet"
          --licob-labels "$root/licob-v0plus/NIHMS1969891-supplement-Supplementary_Tables_S1-S13.xlsx"
          --out "$run/external_score.json")
    ;;
  *) exit 2 ;;
esac
exec "$python" -m vc_build.replicated_pair_contrasts "$stage" "${args[@]}"
