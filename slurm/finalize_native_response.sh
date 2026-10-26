#!/usr/bin/env bash
#SBATCH --job-name=vc_native_evaluate
#SBATCH --partition=cpu
#SBATCH --nodelist=g1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --chdir=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver
#SBATCH --output=logs/slurm_native_evaluate_%j.log
set -euo pipefail
export CUDA_VISIBLE_DEVICES=-1
source envs/activate.sh scgpt
.venvs/vcell-runtime310/bin/python envs/portable310.py --env research scripts/research/evaluate_responses.py --task native_common
python scripts/research/export_research_results.py
python scripts/research/execute_tutorials.py notebooks/07_measured_perturbation_response.ipynb
