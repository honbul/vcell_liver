#!/usr/bin/env bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=22G
#SBATCH --time=00:50:00
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
args=(--run "$run" --inputs "$runs/licob-dose-c471062/inputs" --cold "$runs/licob-cold-5bf53fc/inputs")
case "$stage" in
  fit)
    args+=(--previous "$runs/licob-curves-499f89e" --priors "$root/v0plus-public-priors"
           --hallmark /mnt/S1/sdata/pubdata/Virtual_Cell/MSigDB/2026.1.Hs/h.all.v2026.1.Hs.symbols.gmt
           --targets "$root/external-code-v3/drug_targets.json" --auxiliary "$runs/depmap-25q3-bb8a47b")
    ;;
  score|audit)
    args+=(--supplement "$root/licob-v0plus/NIHMS1969891-supplement-Supplementary_Tables_S1-S13.xlsx")
    ;;
  *) exit 2 ;;
esac
exec "$python" -m vc_build.licob_crossing "$stage" "${args[@]}"
