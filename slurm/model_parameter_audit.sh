#!/usr/bin/env bash
#SBATCH --job-name=vc_parameter_audit
#SBATCH --partition=cpu
#SBATCH --nodelist=g1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --chdir=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver
#SBATCH --output=logs/slurm_parameter_audit_%j.log
set -euo pipefail
export CUDA_VISIBLE_DEVICES=-1
source envs/activate.sh scgpt
.venvs/vcell-runtime310/bin/python envs/portable310.py --env research scripts/research/audit_saved_models.py --model scvi
.venvs/vcell-liverct/bin/python scripts/research/audit_saved_models.py --model liverct
