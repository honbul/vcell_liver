#!/usr/bin/env bash
#SBATCH --job-name=vc_improvement_one
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --nodelist=g1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --chdir=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver
#SBATCH --output=logs/slurm_improvement_%j.log
set -euo pipefail
source envs/activate.sh geneformer
python scripts/research/run_geneformer_adaptation.py "$@"
