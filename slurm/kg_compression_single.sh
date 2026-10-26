#!/usr/bin/env bash
#SBATCH --job-name=vc_kg_onefold
#SBATCH --partition=cpu
#SBATCH --nodelist=g1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --chdir=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver
#SBATCH --output=logs/slurm_kg_onefold_%j.log
set -euo pipefail
export CUDA_VISIBLE_DEVICES=-1
source envs/activate.sh scgpt
python scripts/research/run_kg_compression.py "$@"
