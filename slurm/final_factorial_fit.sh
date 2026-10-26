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
  extra=(--features "$VCELL_DATA_ROOT/research_phase2/query_features/geneformer_annotation_query_interface_v1.npz")
  if [[ "$factor" = DM01 || "$factor" = DM11 ]]; then
    extra=(--token-cache "$VCELL_DATA_ROOT/tokenized/query_annotation_query_interface_v1/cells.dataset")
  fi
  .venvs/vcell-runtime310/bin/python envs/portable310.py --env geneformer scripts/research/predict_annotation.py --input "$VCELL_DATA_ROOT/research_phase2/probes/annotation_query_interface_v1/counts_only.h5ad" --bundle "$VCELL_DATA_ROOT/research_phase2/runs/F01_${factor}_development4096_seed42_v1/annotation_bundle.joblib" --output "$VCELL_DATA_ROOT/research_phase2/probes/annotation_query_interface_v1/$factor" "${extra[@]}"
done
