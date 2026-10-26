#!/usr/bin/env bash
#SBATCH --job-name=vc_v3_control_probe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --nodelist=g1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --chdir=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver
#SBATCH --output=logs/slurm_v3_control_probe_%j.log
set -euo pipefail
source envs/activate.sh scgpt
.venvs/vcell-runtime310/bin/python envs/portable310.py --env scgpt scripts/research/probe_native_control_sensitivity.py
