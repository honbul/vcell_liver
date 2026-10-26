#!/usr/bin/env bash
#SBATCH --job-name=vc_final_fit
#SBATCH --partition=cpu
#SBATCH --nodelist=g1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --chdir=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver
#SBATCH --output=logs/slurm_final_fit_%j.log
set -euo pipefail
export CUDA_VISIBLE_DEVICES=-1
source envs/activate.sh scgpt
for model in pca scgpt geneformer geneformer_clcancer; do
  version=1
  if [ "$model" = pca ]; then version=2; fi
  .venvs/vcell-runtime310/bin/python envs/portable310.py --env research scripts/research/fit_final_annotation.py --model "$model" --run-id F00_${model}_development12000_seed42_v${version}
done
