#!/usr/bin/env bash
#SBATCH --cpus-per-task=5
#SBATCH --mem=25G
#SBATCH --time=03:00:00
set -euo pipefail

repository=$1
root=$2
run=$3
mode=$4
shift 4
storage=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
study_python="$storage/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
runtime="$storage/vcell_liver/vcell_r12_runtime_cu126"
export PYTHONPATH="$repository/src:$root/v0plus-model-deps:$root/function/prepare-deps:$runtime"
export OPENBLAS_NUM_THREADS=5 MKL_NUM_THREADS=5 OMP_NUM_THREADS=5
bundle="$root/function/cross-v1"
source="$root/licob-v0plus"
author="$source/author-c65cb2f"
supplement="$source/NIHMS1969891-supplement-Supplementary_Tables_S1-S13.xlsx"

case "$mode" in
  register)
    args=(--rna "$author/rna.tsv" --author "$author/data_ilicob_org" --supplement "$supplement"
          --catalog "$author/20261030-L1700-Bioactive-Compound-Library-I-96-well.xlsx"
          --cas "$author/cas" --node-archive "$source/LICOB_omics_matrix_data.xlsx.part"
          --bundle "$bundle" --out "$run/cohort")
    ;;
  cpu|gpu)
    if [[ "$mode" == cpu ]]; then
      arms=(linear lightgbm catboost cellhit)
      device=()
    else
      arms=(velodrome_inductive codeae_inductive v0plus_inductive)
      device=(--gpu)
    fi
    arm=${arms[$SLURM_ARRAY_TASK_ID]}
    mode=predict
    args=(--cohort "$run/cohort" --bundle "$bundle"
          --settings "$root/v0plus-runs/final-8e5d3f4/settings.json"
          --priors "$root/v0plus-public-priors"
          --hallmark /mnt/S1/sdata/pubdata/Virtual_Cell/MSigDB/2026.1.Hs/h.all.v2026.1.Hs.symbols.gmt
          --targets "$root/external-code-v3/drug_targets.json" --upstream "$root/v0plus-upstream"
          --auxiliary "$root/v0plus-runs/depmap-25q3-bb8a47b"
          --arm "$arm" --out "$run/predictions/$arm.parquet" "${device[@]}")
    ;;
  merge)
    args=(--cohort "$run/cohort" --parts "$run/predictions" --out "$run/forecast.parquet")
    ;;
  score)
    args=(--cohort "$run/cohort" --forecast "$run/forecast.parquet"
          --supplement "$supplement" --patient-audit "$run/patient-audit.json" --out "$run/score.json")
    ;;
  *) exit 2 ;;
esac
exec "$study_python" -m vc_build.v0plus_licob "$mode" "${args[@]}"
