#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
set -euo pipefail

repository=$1
run=$2
shift 2
storage=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
python="$storage/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
export PYTHONPATH="$repository/src"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
cd "$repository"
for track in "$@"; do
  "$python" -m vc_build.evaluate --prediction "$run/predictions.parquet" \
    --observations "$run/labels.parquet" --split "$track" --endpoint-direction higher \
    --compare selected_context selected_noninteraction \
    --compare selected_context drug_mean --compare selected_context selected_context_rna_swap \
    --compare selected_context selected_context_rna_drop \
    --out "$run/metrics-${track}.json"
done
