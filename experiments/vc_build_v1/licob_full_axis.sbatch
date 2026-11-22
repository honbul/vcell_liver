#!/usr/bin/env bash
#SBATCH --cpus-per-task=5
#SBATCH --mem=25G
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
supplement="$root/licob-v0plus/NIHMS1969891-supplement-Supplementary_Tables_S1-S13.xlsx"
core="$root/v0plus-runs/source-core-92aa482/core"
prefix="$root/v0plus-runs/licob-node-recovery-23dcddc"
case "$stage" in
  register)
    args=(--core "$core" --rna "$root/hcc-drugdis-fc66d2991e499a03b92778347a8a251d0bd9eb7c/LICOB.parquet"
          --prefix "$prefix/cohort" --licob "$root/v0plus-runs/licob-author-2f1328b/cohort"
          --original "$root/function/cross-v1" --extension "$root/v0plus-runs/source-panel-c351b0b/extension"
          --out "$run/cohort")
    ;;
  cpu|gpu)
    if [[ "$stage" == gpu ]]; then
      arms=(structured velodrome_inductive codeae_inductive v0plus_inductive expanded)
      resource=(--gpu)
    else
      arms=(linear cellhit-0 cellhit-1 cellhit-2)
      resource=()
    fi
    arm=${arms[$SLURM_ARRAY_TASK_ID]}
    stage=predict
    args=(--input "$run/cohort" --core "$core" --component "$arm" "${resource[@]}"
          --settings "$root/v0plus-runs/final-8e5d3f4/settings.json"
          --genomics "$root/v0plus-runs/measured-drivers-4dd2872/genomics"
          --priors "$root/v0plus-public-priors" --upstream "$root/v0plus-upstream"
          --hallmark /mnt/S1/sdata/pubdata/Virtual_Cell/MSigDB/2026.1.Hs/h.all.v2026.1.Hs.symbols.gmt
          --targets "$root/external-code-v3/drug_targets.json" --auxiliary "$root/v0plus-runs/depmap-25q3-bb8a47b"
          --out "$run/parts/$arm.parquet")
    ;;
  merge)
    args=(--input "$run/cohort" --parts "$run/parts" --historical "$prefix/final.parquet" --out "$run/final.parquet")
    ;;
  score)
    args=(--input "$run/cohort" --forecast "$run/final.parquet" --supplement "$supplement" --out "$run/score.json")
    ;;
  *) exit 2 ;;
esac
exec "$python" -m vc_build.licob_full_axis "$stage" "${args[@]}"
