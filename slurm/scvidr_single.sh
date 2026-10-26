#!/usr/bin/env bash
#SBATCH --job-name=vc_scvidr_single
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --nodelist=g3
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --chdir=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver
#SBATCH --output=logs/slurm_scvidr_%j.log
set -euo pipefail
source envs/activate.sh scgpt
run_id=${1:-C00_scvidr_mouse_TCDD_portal_seed42_v3}
recovery=${2:-C00_scvidr_mouse_TCDD_portal_seed42_v2}
extra=()
if [ "$recovery" != fresh ]; then extra=(--recover-trained-run "$recovery"); fi
.venvs/vcell-scvidr/bin/python scripts/research/run_scvidr_single.py --run-id "$run_id" "${extra[@]}"
