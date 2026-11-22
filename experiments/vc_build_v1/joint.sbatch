#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
set -euo pipefail

repository=$1
functional_run=$2
data=$3
state=$4
run=$5
step=$6
storage=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
python="$storage/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
export PYTHONPATH="$repository/src"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
cd "$repository"
case "$step" in
  predict)
    "$python" -m vc_build.joint --data "$data" --splits "$functional_run/splits.parquet" \
      --state "$state" --symbols /mnt/S1/sdata/pubdata/DepMap/25Q3/OmicsExpressionTPMLogp1HumanAllGenes.csv \
      --registry manifests/pathway_sensitivity_a1/model_registry.tsv --out "$run" ;;
  score)
    "$python" -m vc_build.evaluate --prediction "$run/predictions.parquet" \
      --observations "$functional_run/labels.parquet" --split C_joint --endpoint-direction higher \
      --compare multi_output direct --compare state_assisted direct --compare state_assisted multi_output \
      --out "$run/metrics.json" ;;
  *) exit 2 ;;
esac
