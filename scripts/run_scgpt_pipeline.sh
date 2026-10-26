#!/usr/bin/env bash
set -euo pipefail
cd /mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver
source envs/activate.sh scgpt
for subset in smoke pilot comparison; do
    python -u scripts/run_stage.py --name "pca_${subset}" -- python -u scripts/02_baseline_annotation.py --config "configs/pca_${subset}.yaml"
    python -u scripts/run_stage.py --name "scgpt_${subset}" -- python -u scripts/03_scgpt_annotation.py --config "configs/scgpt_${subset}.yaml"
done
