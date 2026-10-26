#!/usr/bin/env bash
#SBATCH --job-name=vc_response_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --nodelist=g3
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --chdir=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver
#SBATCH --output=logs/slurm_response_smoke_%j.log
set -euo pipefail
source envs/activate.sh scgpt
.venvs/vcell-runtime310/bin/python envs/portable310.py --env scgpt scripts/research/run_scgpt_response.py --initialization pretrained --run-id B04_native_response_smoke_seed42_v1 --smoke
