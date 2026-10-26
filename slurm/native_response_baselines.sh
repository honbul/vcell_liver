#!/usr/bin/env bash
#SBATCH --job-name=vc_native_baselines
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --chdir=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver
#SBATCH --output=logs/slurm_native_baselines_%j.log
set -euo pipefail
export CUDA_VISIBLE_DEVICES=-1
source envs/activate.sh research
.venvs/vcell-runtime310/bin/python envs/portable310.py --env research scripts/research/run_response_baselines.py --task hepg2 --task-root hepg2_foundation_common_v1 --run-id B00_B02_native_common_seed42_v1
