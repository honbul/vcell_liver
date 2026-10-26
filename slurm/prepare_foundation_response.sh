#!/usr/bin/env bash
#SBATCH --job-name=vc_response_prepare
#SBATCH --partition=cpu
#SBATCH --nodelist=g1
#SBATCH --nice=1000
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --chdir=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver
#SBATCH --output=logs/slurm_response_prepare_%j.log
set -euo pipefail
export CUDA_VISIBLE_DEVICES=-1
source envs/activate.sh scgpt
python scripts/research/prepare_foundation_response.py
