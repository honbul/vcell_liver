#!/usr/bin/env bash
# The frozen same-head half of the registered 2x2; no adapted-model selection is needed here.
set -euo pipefail
cd /mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver
source envs/activate.sh geneformer
for seed in 42 43 44; do
  for fold in 0 1 2 3; do
    for factor in D0 D1; do
      variant=D0_natural
      run_id=DM00_full4096_fold${fold}_seed${seed}_v1
      if [ "$factor" = D1 ]; then variant=D1_donor_balanced; run_id=DM10_full4096_fold${fold}_seed${seed}_v1; fi
      # Reuse the already executed identical frozen-head screen for this one control.
      if [ "$seed" = 42 ] && [ "$fold" = 0 ] && [ "$factor" = D0 ]; then run_id=M01_full4096_fold0_seed42_v1; fi
      if [ ! -e reports/research_phase2/runs/$run_id/run.json ]; then
        python scripts/research/run_geneformer_adaptation.py --model frozen_nn --data-variant "$variant" --seed "$seed" --fold "$fold" --run-id "$run_id"
      fi
    done
  done
done
