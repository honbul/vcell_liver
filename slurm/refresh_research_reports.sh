#!/usr/bin/env bash
#SBATCH --job-name=vc_report_refresh
#SBATCH --partition=cpu
#SBATCH --nodelist=g1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --chdir=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver
#SBATCH --output=logs/slurm_report_refresh_%j.log
set -euo pipefail
export CUDA_VISIBLE_DEVICES=-1
source envs/activate.sh scgpt
python scripts/research/summarize_annotation_uncertainty.py --regime inductive
python scripts/research/summarize_annotation_uncertainty.py --regime query_adaptive
python scripts/research/resolve_run_metadata.py
python scripts/research/export_research_results.py
python scripts/research/execute_tutorials.py notebooks/06_research_development_cv.ipynb notebooks/07_measured_perturbation_response.ipynb notebooks/08_data_model_improvements.ipynb notebooks/09_new_hcc_query_annotation.ipynb
