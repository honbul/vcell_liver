#!/usr/bin/env bash
#SBATCH --job-name=vc_query_annotation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --nodelist=g3
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --chdir=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver
#SBATCH --output=logs/slurm_query_annotation_%j.log
set -euo pipefail
source envs/activate.sh scgpt
query=${1:?counts.h5ad};model=${2:?model};cache_name=${3:?unique_cache_name};output=${4:?new_output_directory}
case "$model" in pca|scgpt|geneformer|geneformer_clcancer) ;; *) exit 2;; esac
version=1
if [ "$model" = pca ]; then version=2; fi
extra=()
if [ "$model" != pca ]; then
  environment=geneformer
  if [ "$model" = scgpt ]; then environment=scgpt; fi
  .venvs/vcell-runtime310/bin/python envs/portable310.py --env "$environment" scripts/research/extract_query_features.py --input "$query" --model "$model" --cache-name "$cache_name"
  extra=(--features "$VCELL_DATA_ROOT/research_phase2/query_features/${model}_${cache_name}.npz")
fi
.venvs/vcell-runtime310/bin/python envs/portable310.py --env research scripts/research/predict_annotation.py --input "$query" --bundle "$VCELL_DATA_ROOT/research_phase2/runs/F00_${model}_development12000_seed42_v${version}/annotation_bundle.joblib" --output "$output" "${extra[@]}"
