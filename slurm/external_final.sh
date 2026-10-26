#!/usr/bin/env bash
#SBATCH --job-name=vc_external_final
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --nodelist=g3
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --chdir=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver
#SBATCH --output=logs/slurm_external_final_%j.log
set -euo pipefail
source envs/activate.sh scgpt
python310=.venvs/vcell-runtime310/bin/python
bootstrap=envs/portable310.py
probe=$VCELL_DATA_ROOT/research_phase2/probes/annotation_query_interface_v1
for factor in DM00 DM10 DM01 DM11; do
  if [ ! -f "$probe/$factor/inference.json" ]; then
    extra=(--features "$VCELL_DATA_ROOT/research_phase2/query_features/geneformer_annotation_query_interface_v1.npz")
    if [[ "$factor" = DM01 || "$factor" = DM11 ]]; then
      extra=(--token-cache "$VCELL_DATA_ROOT/tokenized/query_annotation_query_interface_v1/cells.dataset")
    fi
    "$python310" "$bootstrap" --env geneformer scripts/research/predict_annotation.py --input "$probe/counts_only.h5ad" --bundle "$VCELL_DATA_ROOT/research_phase2/runs/F01_${factor}_development4096_seed42_v1/annotation_bundle.joblib" --output "$probe/$factor" "${extra[@]}"
  fi
done
"$python310" "$bootstrap" --env research scripts/research/freeze_external_evaluation.py
query=$VCELL_DATA_ROOT/research_phase2/external/GSE151530_HCC/counts_query_v1.h5ad
output=$VCELL_DATA_ROOT/research_phase2/external_final_predictions_v1
"$python310" "$bootstrap" --env research scripts/research/predict_annotation.py --input "$query" --bundle "$VCELL_DATA_ROOT/research_phase2/runs/F00_pca_development12000_seed42_v1/annotation_bundle.joblib" --output "$output/pca"
for model in scgpt geneformer geneformer_clcancer; do
  environment=geneformer
  if [ "$model" = scgpt ]; then environment=scgpt; fi
  "$python310" "$bootstrap" --env "$environment" scripts/research/extract_query_features.py --input "$query" --model "$model" --cache-name external_final_v1
  "$python310" "$bootstrap" --env research scripts/research/predict_annotation.py --input "$query" --bundle "$VCELL_DATA_ROOT/research_phase2/runs/F00_${model}_development12000_seed42_v1/annotation_bundle.joblib" --features "$VCELL_DATA_ROOT/research_phase2/query_features/${model}_external_final_v1.npz" --output "$output/$model"
done
for factor in DM00 DM10 DM01 DM11; do
  extra=(--features "$VCELL_DATA_ROOT/research_phase2/query_features/geneformer_external_final_v1.npz")
  if [[ "$factor" = DM01 || "$factor" = DM11 ]]; then
    extra=(--token-cache "$VCELL_DATA_ROOT/tokenized/query_external_final_v1/cells.dataset")
  fi
  "$python310" "$bootstrap" --env geneformer scripts/research/predict_annotation.py --input "$query" --bundle "$VCELL_DATA_ROOT/research_phase2/runs/F01_${factor}_development4096_seed42_v1/annotation_bundle.joblib" --output "$output/$factor" "${extra[@]}"
done
"$python310" "$bootstrap" --env research scripts/research/evaluate_external.py
