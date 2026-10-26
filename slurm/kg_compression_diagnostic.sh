#!/usr/bin/env bash
#SBATCH --job-name=vc_kg_compression
#SBATCH --partition=cpu
#SBATCH --nodelist=g1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --chdir=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver
#SBATCH --output=logs/slurm_kg_compression_%j.log
set -euo pipefail
export CUDA_VISIBLE_DEVICES=-1
source envs/activate.sh scgpt
for fold in 0 1 2 3; do
  baseline=D04_control_full4096_fold${fold}_seed42_v1
  if [ "$fold" = 0 ]; then baseline=D00_full4096_fold0_seed42_v1; fi
  if [ ! -f reports/research_phase2/runs/$baseline/run.json ]; then
    python scripts/research/run_annotation_cv.py --config configs/research/protocol_improvement.yaml --model geneformer --fold "$fold" --run-id "$baseline"
  fi
  python scripts/research/run_kg_compression.py --fold "$fold" --baseline-run "$baseline"
done
python scripts/research/summarize_kg_compression.py
