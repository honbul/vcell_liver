#!/usr/bin/env bash
#SBATCH --cpus-per-task=3
#SBATCH --mem=14G
#SBATCH --time=00:40:00
set -euo pipefail
code=$1
root=$2
run=$3
action=$4
storage=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
python="$storage/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
export PYTHONPATH="$code/src:$root/v0plus-model-deps:$root/function/prepare-deps:$storage/vcell_liver/vcell_r12_runtime_cu126"
export OPENBLAS_NUM_THREADS=3 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
runs="$root/v0plus-runs"
args=(--organoid "$runs/wong-expanded-15b1345" --tissue "$runs/wong-tissue-a4120a9")
case "$action" in
  predict)
    args+=(--priors "$root/v0plus-public-priors"
           --hallmark /mnt/S1/sdata/pubdata/Virtual_Cell/MSigDB/2026.1.Hs/h.all.v2026.1.Hs.symbols.gmt
           --targets "$root/external-code-v3/drug_targets.json" --auxiliary "$runs/depmap-25q3-bb8a47b"
           --out "$run/frozen")
    ;;
  score)
    args+=(--freeze "$run/frozen" --workbook "$root/licob-v0plus/Wong_HCC_PDO_Supplementary_v1.xlsx"
           --out "$run/score.json")
    ;;
  *) exit 2 ;;
esac
exec "$python" -m vc_build.wong_paired_rna "$action" "${args[@]}"
