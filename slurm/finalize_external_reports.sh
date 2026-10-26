#!/usr/bin/env bash
#SBATCH --job-name=vc_external_finalize
#SBATCH --partition=cpu
#SBATCH --nodelist=g1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --chdir=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver
#SBATCH --output=logs/slurm_external_finalize_%j.log
set -euo pipefail
export CUDA_VISIBLE_DEVICES=-1
source envs/activate.sh scgpt
.venvs/vcell-runtime310/bin/python envs/portable310.py --env research scripts/research/audit_external_completion.py
python scripts/research/record_external_status.py --job-id 12919
python scripts/research/resolve_run_metadata.py
python scripts/research/export_research_results.py
