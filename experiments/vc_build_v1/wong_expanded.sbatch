#!/usr/bin/env bash
#SBATCH --cpus-per-task=5
#SBATCH --mem=28G
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
runs="$root/v0plus-runs"
module=vc_build.wong_expanded
case "$stage" in
  register)
    args=(--original "$root/function/cross-v1" --core "$runs/source-core-92aa482/core"
          --extension "$runs/source-panel-c351b0b/extension" --gdsc "$runs/improve-gdsc-stereo-347009e/extension"
          --wong "$runs/wong-84d021d/cohort" --support "$repository/reports/vc_build_v1/v0plus/hcc_panel_support.json"
          --out "$run/inputs")
    ;;
  cpu|gpu)
    if [[ "$stage" == gpu ]]; then
      arms=(kernel codeae_inductive velodrome_inductive v0plus_inductive catboost)
      resource=(--gpu)
    else
      arms=(linear lightgbm cellhit-0 cellhit-1 cellhit-2)
      resource=()
    fi
    arm=${arms[$SLURM_ARRAY_TASK_ID]}
    stage=predict
    args=(--inputs "$run/inputs" --component "$arm" "${resource[@]}" --settings "$runs/final-8e5d3f4/settings.json"
          --priors "$root/v0plus-public-priors" --upstream "$root/v0plus-upstream"
          --hallmark /mnt/S1/sdata/pubdata/Virtual_Cell/MSigDB/2026.1.Hs/h.all.v2026.1.Hs.symbols.gmt
          --targets "$root/external-code-v3/drug_targets.json" --auxiliary "$runs/depmap-25q3-bb8a47b"
          --out "$run/parts/$arm.parquet")
    ;;
  tissue-register|joint-score)
    module=vc_build.wong_tissue
    pdo=$5
    args=(--workbook "$root/licob-v0plus/Wong_HCC_PDO_Supplementary_v1.xlsx")
    if [[ "$stage" == tissue-register ]]; then
      stage=register
      args+=(--pdo "$pdo/inputs" --out "$run/inputs")
    else
      stage=score
      args+=(--pdo "$pdo" --tissue "$run" --out "$run/score.json")
    fi
    ;;
  merge)
    args=(--inputs "$run/inputs" --parts "$run/parts" --out "$run/final.parquet")
    ;;
  score)
    args=(--inputs "$run/inputs" --forecast "$run/final.parquet"
          --workbook "$root/licob-v0plus/Wong_HCC_PDO_Supplementary_v1.xlsx" --out "$run/score.json")
    ;;
  *) exit 2 ;;
esac
exec "$python" -m "$module" "$stage" "${args[@]}"
