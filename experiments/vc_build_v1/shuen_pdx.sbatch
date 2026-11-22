#!/usr/bin/env bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:25:00
set -euo pipefail
repository=$1
root=$2
run=$3
stage=$4
storage=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
python="$storage/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
runtime="$storage/vcell_liver/vcell_r12_runtime_cu126"
export PYTHONPATH="$repository/src:$root/v0plus-model-deps:$root/function/prepare-deps:$runtime"
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
inputs="$root/3d-public/shuen-hcc-pdx-rna-v1"
case "$stage" in
  predict)
    args=(--rna "$inputs/17PDX_treatment-naive_gene_counts.csv"
          --gtf "$root/3d-public/tcga-lihc-rna-format-v1/gencode.v36.annotation.gtf.gz"
          --package "$root/v0plus-runs/structured-release-6d2c8c6/V0plus")
    ;;
  score) args=(--sources "$inputs/response") ;;
  *) exit 2 ;;
esac
exec "$python" -m vc_build.shuen_pdx "$stage" --run "$run" "${args[@]}"
