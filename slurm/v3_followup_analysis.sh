#!/usr/bin/env bash
#SBATCH --job-name=vc_v3_followup
#SBATCH --partition=cpu
#SBATCH --nodelist=g1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --chdir=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver
#SBATCH --output=logs/slurm_v3_followup_%j.log
set -euo pipefail
export CUDA_VISIBLE_DEVICES=-1
source envs/activate.sh scgpt
.venvs/vcell-runtime310/bin/python envs/portable310.py --env research scripts/research/analyze_tissue_errors.py
.venvs/vcell-runtime310/bin/python envs/portable310.py --env research scripts/research/analyze_control_sensitivity.py
