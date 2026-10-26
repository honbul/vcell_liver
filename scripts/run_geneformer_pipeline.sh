#!/usr/bin/env bash
set -euo pipefail
cd /mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver
source envs/activate.sh geneformer
for subset in smoke pilot comparison; do
    python -u scripts/run_stage.py --name "geneformer_${subset}" -- python -u scripts/04_geneformer_annotation.py --config "configs/geneformer_${subset}.yaml"
done
python -u scripts/run_stage.py --name isp_pilot -- python -u scripts/06_geneformer_virtual_perturb.py --config configs/isp_pilot.yaml
