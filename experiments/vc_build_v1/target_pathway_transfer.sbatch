#!/usr/bin/env bash
#SBATCH --cpus-per-task=5
#SBATCH --mem=25G
#SBATCH --time=01:00:00
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
limore="$root/function/cross-v1"
measured="$root/v0plus-runs/measured-drivers-4dd2872"
case "$stage" in
  predict)
    args=(--core "$root/v0plus-runs/source-core-92aa482/core" --limore "$limore"
          --archive "$measured/LIMORE/final.parquet" --genomics "$measured/genomics"
          --kernel-source "$run/registered_kernel.py"
          --priors "$root/v0plus-public-priors"
          --hallmark /mnt/S1/sdata/pubdata/Virtual_Cell/MSigDB/2026.1.Hs/h.all.v2026.1.Hs.symbols.gmt
          --targets "$root/external-code-v3/drug_targets.json"
          --auxiliary "$root/v0plus-runs/depmap-25q3-bb8a47b" --out "$run/final.parquet")
    ;;
  score)
    args=(--forecast "$run/final.parquet" --labels "$limore/test_labels/outcomes.parquet" --out "$run/score.json")
    ;;
  *) exit 2 ;;
esac
exec "$python" -m vc_build.target_pathway_transfer "$stage" "${args[@]}"
