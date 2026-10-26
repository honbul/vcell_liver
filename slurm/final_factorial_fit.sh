#!/usr/bin/env bash
#SBATCH --job-name=vc_final_factors
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --nodelist=g3
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --chdir=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver
#SBATCH --output=logs/slurm_final_factors_%j.log
set -euo pipefail
source envs/activate.sh scgpt
.venvs/vcell-runtime310/bin/python envs/portable310.py --env research scripts/research/summarize_factorial.py --mode all
for factor in DM00 DM10 DM01 DM11; do
  .venvs/vcell-runtime310/bin/python envs/portable310.py --env geneformer scripts/research/fit_final_factorial.py --factor "$factor" --run-id F01_${factor}_development4096_seed42_v1
done
