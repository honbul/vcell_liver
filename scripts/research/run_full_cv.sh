#!/usr/bin/env bash
set -euo pipefail
cd /mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver
source envs/activate.sh scgpt
model=$1
case "$model" in pca) code=A00;; scgpt) code=A01;; geneformer) code=A02;; geneformer_clcancer) code=A05;; *) exit 2;; esac
for fold in 0 1 2 3; do
  run_id=${code}_full_fold${fold}_seed42_v2
  if [ ! -e reports/research_phase2/runs/$run_id/run.json ]; then
    python scripts/research/run_annotation_cv.py --config configs/research/protocol_annotation_full.yaml --model "$model" --fold "$fold" --run-id "$run_id"
  fi
done
