#!/usr/bin/env bash
#SBATCH --cpus-per-task=5
#SBATCH --mem=28G
#SBATCH --time=01:15:00
set -euo pipefail
repository=$1
root=$2
run=$3
stage=$4
storage=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
python="$storage/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
export PYTHONPATH="$repository/src:$root/v0plus-model-deps:$root/function/prepare-deps:$storage/vcell_liver/vcell_r12_runtime_cu126"
export OPENBLAS_NUM_THREADS=5 MKL_NUM_THREADS=5 OMP_NUM_THREADS=5
runs="$root/v0plus-runs"
cold="$runs/licob-cold-5bf53fc"
supplement="$root/licob-v0plus/NIHMS1969891-supplement-Supplementary_Tables_S1-S13.xlsx"
original="$root/function/cross-v1"
module=licob_dose_sources
case "$stage" in
  register)
    args=(--original "$original" --core "$runs/source-core-92aa482/core"
          --extension "$runs/source-panel-c351b0b/extension" --gdsc-extension "$runs/improve-gdsc-stereo-347009e/extension"
          --cold "$cold/inputs" --supplement "$supplement" --out "$run/inputs")
    ;;
  extract)
    screens=(CTRPv2 GDSC2 PRISM20Q2)
    assay=${screens[${SLURM_ARRAY_TASK_ID:?}]}
    args=(--assay "$assay" --original "$original" --inputs "$run/inputs"
          --ctrp /mnt/S1/sdata/pubdata/DepMap/DrugScreens/CTRP_CTD2/v20.data.per_cpd_post_qc.txt
          --gdsc "$root/v0plus-public-priors/gdsc2-raw-8.5/GDSC2_public_raw_data_27Oct23.zip"
          --prism-treatments "$runs/dose-window-3e0e6dc/prism20q2-treatment.csv" --out "$run/$assay")
    ;;
  freeze|score)
    module=licob_endpoints
    args=(--cold "$cold/inputs" --prior "$cold/parts/fold-3.parquet" --run "$run")
    if [[ "$stage" == freeze ]]; then
      args+=(--priors "$root/v0plus-public-priors" --targets "$root/external-code-v3/drug_targets.json"
             --auxiliary "$runs/depmap-25q3-bb8a47b"
             --hallmark /mnt/S1/sdata/pubdata/Virtual_Cell/MSigDB/2026.1.Hs/h.all.v2026.1.Hs.symbols.gmt)
    else
      args+=(--supplement "$supplement")
    fi
    ;;
  *) exit 2 ;;
esac
exec "$python" -m "vc_build.$module" "$stage" "${args[@]}"
