#!/usr/bin/env bash
#SBATCH --job-name=vc_dev_diagnostics
#SBATCH --partition=cpu
#SBATCH --nodelist=g1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --chdir=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver
#SBATCH --output=logs/slurm_dev_diagnostics_%j.log
set -euo pipefail
export CUDA_VISIBLE_DEVICES=-1
source envs/activate.sh scgpt
.venvs/vcell-runtime310/bin/python envs/portable310.py --env research scripts/research/diagnose_development.py
.venvs/vcell-runtime310/bin/python envs/portable310.py --env research scripts/research/evaluate_responses.py --task hepg2
