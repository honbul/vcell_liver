#!/usr/bin/env bash
#SBATCH --job-name=vc_v3_official_audit
#SBATCH --partition=cpu
#SBATCH --nodelist=g1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --chdir=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver
#SBATCH --output=logs/slurm_v3_official_audit_%j.log
set -euo pipefail
export CUDA_VISIBLE_DEVICES=-1
source envs/activate.sh scgpt
case "${1:-gap}" in
  gap) python scripts/research/audit_official_scgpt_gap.py ;;
  selection) python scripts/research/compare_official_scgpt_source.py ;;
  *) exit 2 ;;
esac
