#!/usr/bin/env bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=18G
#SBATCH --time=00:40:00
set -euo pipefail
code=$1
root=$2
run=$3
stage=$4
storage=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
python="$storage/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
export PYTHONPATH="$code/src:$root/v0plus-model-deps:$root/function/prepare-deps:$storage/vcell_liver/vcell_r12_runtime_cu126"
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
runs="$root/v0plus-runs"
cold="$runs/licob-cold-5bf53fc"
args=(--out "$run" --cold "$cold/inputs" --dose "$runs/licob-dose-c471062")
case "$stage" in
  fit)
    args+=(--oslo "$runs/oslo-public-5187b3f/registered"
           --chemistry "$code/reports/vc_build_v1/v0plus/oslo_licob_chemistry.tsv"
           --pubchem "$root/3d-public/oslo-crlm-hr94h42xdc-v3/pubchem-parent-v1"
           --baseline "$cold/parts/fold-3.parquet"
           --priors "$root/v0plus-public-priors" --targets "$root/external-code-v3/drug_targets.json"
           --hallmark /mnt/S1/sdata/pubdata/Virtual_Cell/MSigDB/2026.1.Hs/h.all.v2026.1.Hs.symbols.gmt)
    ;;
  score)
    args+=(--supplement "$root/licob-v0plus/NIHMS1969891-supplement-Supplementary_Tables_S1-S13.xlsx")
    ;;
  *) exit 2 ;;
esac
exec "$python" -m vc_build.oslo_licob_transfer "$stage" "${args[@]}"
