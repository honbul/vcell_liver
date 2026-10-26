#!/usr/bin/env bash
set -euo pipefail
cd /mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver
source envs/activate.sh scgpt
for model in pca scgpt geneformer; do
  case "$model" in pca) exp=A00 ;; scgpt) exp=A01 ;; geneformer) exp=A02 ;; esac
  for fold in 0 1 2 3; do
    run="${exp}_pilot_fold${fold}_seed42_v2"
    if [ -f "reports/research_phase2/runs/$run/run.json" ]; then continue; fi
    python -u scripts/research/run_annotation_cv.py --config configs/research/protocol_annotation.yaml --model "$model" --fold "$fold" --run-id "$run"
  done
done
