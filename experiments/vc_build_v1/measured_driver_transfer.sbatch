#!/usr/bin/env bash
#SBATCH --cpus-per-task=5
#SBATCH --mem=25G
#SBATCH --time=02:00:00
set -euo pipefail
repository=$1
root=$2
run=$3
stage=$4
study=${5:-}
storage=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
python="$storage/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
runtime="$storage/vcell_liver/vcell_r12_runtime_cu126"
export PYTHONPATH="$repository/src:$root/v0plus-model-deps:$root/function/prepare-deps:$runtime"
export OPENBLAS_NUM_THREADS=5 MKL_NUM_THREADS=5 OMP_NUM_THREADS=5
sources="$root/function/sources"
licob="$root/v0plus-runs/licob-author-2f1328b"
args=(--core "$root/v0plus-runs/source-core-92aa482/core" --limore "$root/function/cross-v1" --licob "$licob/cohort")
case "$stage" in
  prepare)
    args+=(--limore-s2 "$sources/LIMORE-CancerCell-2019-TableS2.xlsx"
           --limore-s3 "$sources/LIMORE-CancerCell-2019-TableS3.xlsx"
           --licob-node "$root/licob-v0plus/LICOB_omics_matrix_data.xlsx.part"
           --depmap-mutations /mnt/S1/sdata/pubdata/DepMap/25Q3/OmicsSomaticMutations.csv
           --depmap-copy "$sources/DepMap-Public-24Q4-OmicsAbsoluteCNGene.csv" --out "$run/genomics")
    ;;
  gpu|cpu)
    if [[ "$stage" == gpu ]]; then
      studies=(LIMORE LICOB LICOB LICOB LICOB)
      components=(structured structured velodrome_inductive codeae_inductive v0plus_inductive)
      study=${studies[$SLURM_ARRAY_TASK_ID]}
      resource=(--gpu)
    else
      components=(linear cellhit-0 cellhit-1 cellhit-2)
      study=LICOB
      resource=()
    fi
    arm=${components[$SLURM_ARRAY_TASK_ID]}
    stage=predict
    args+=(--study "$study" --component "$arm" --genomics "$run/genomics" "${resource[@]}"
           --settings "$root/v0plus-runs/final-8e5d3f4/settings.json"
           --priors "$root/v0plus-public-priors" --upstream "$root/v0plus-upstream"
           --hallmark /mnt/S1/sdata/pubdata/Virtual_Cell/MSigDB/2026.1.Hs/h.all.v2026.1.Hs.symbols.gmt
           --targets "$root/external-code-v3/drug_targets.json" --auxiliary "$root/v0plus-runs/depmap-25q3-bb8a47b"
           --patient-audit "$licob/patient-audit.json" --out "$run/$study/$arm.parquet")
    ;;
  merge)
    args+=(--study "$study" --parts "$run/$study" --archive "$root/v0plus-runs/limore-corrected-4a986af/final.parquet"
           --out "$run/$study/final.parquet")
    ;;
  score)
    if [[ "$study" == LIMORE ]]; then
      labels="$root/function/cross-v1/test_labels/outcomes.parquet"
    else
      labels="$root/licob-v0plus/NIHMS1969891-supplement-Supplementary_Tables_S1-S13.xlsx"
    fi
    args+=(--study "$study" --labels "$labels" --patient-audit "$licob/patient-audit.json"
           --forecast "$run/$study/final.parquet" --out "$run/$study/score.json")
    ;;
  *) exit 2 ;;
esac
exec "$python" -m vc_build.measured_driver_transfer "$stage" "${args[@]}"
