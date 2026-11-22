#!/usr/bin/env bash
#SBATCH --cpus-per-task=5
#SBATCH --mem=25G
#SBATCH --time=01:10:00
set -euo pipefail
code=$1
root=$2
run=$3
action=$4
storage=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
python="$storage/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
export PYTHONPATH="$code/src:$root/v0plus-model-deps:$root/function/prepare-deps:$storage/vcell_liver/vcell_r12_runtime_cu126"
export OPENBLAS_NUM_THREADS=5 OMP_NUM_THREADS=5 MKL_NUM_THREADS=5
runs="$root/v0plus-runs"
case "$action" in
  register)
    args=(--original "$root/function/cross-v1" --core "$runs/source-core-92aa482/core"
          --limore "$root/function/limore-v1" --growth "$run/growth.csv" --out "$run/inputs")
    ;;
  cpu|gpu)
    if [[ "$action" == gpu ]]; then
      methods=(structured velodrome_inductive codeae_inductive v0plus_inductive)
      gpu=(--gpu)
    else
      methods=(linear cellhit)
      gpu=()
    fi
    component=${methods[${SLURM_ARRAY_TASK_ID:?}]}
    action=predict
    args=(--registration "$run/inputs" --core "$runs/source-core-92aa482/core"
          --component "$component" --settings "$runs/final-8e5d3f4/settings.json"
          --priors "$root/v0plus-public-priors" --targets "$root/external-code-v3/drug_targets.json"
          --hallmark /mnt/S1/sdata/pubdata/Virtual_Cell/MSigDB/2026.1.Hs/h.all.v2026.1.Hs.symbols.gmt
          --auxiliary "$runs/depmap-25q3-bb8a47b" --upstream "$root/v0plus-upstream"
          --out "$run/parts/$component.parquet" "${gpu[@]}")
    ;;
  freeze)
    args=(--registration "$run/inputs" --parts "$run/parts" --out "$run/final.parquet")
    ;;
  score)
    args=(--registration "$run/inputs" --forecast "$run/final.parquet" --article "$run/article.pdf"
          --values "$run/table2.tsv" --limore "$root/function/limore-v1" --out "$run/score.json")
    ;;
  *) exit 2 ;;
esac
exec "$python" -m vc_build.tki_gr50 "$action" "${args[@]}"
