#!/usr/bin/env bash
#SBATCH --job-name=vc_query_probe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --nodelist=g3
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --chdir=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver
#SBATCH --output=logs/slurm_query_probe_%j.log
set -euo pipefail
source envs/activate.sh scgpt
.venvs/vcell-runtime310/bin/python envs/portable310.py --env research scripts/research/check_annotation_interface.py
