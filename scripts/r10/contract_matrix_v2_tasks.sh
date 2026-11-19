#!/bin/bash
# Write the R10-E1a v2 evaluation task table (one line per role x line x arm group).
# usage: scripts/r10/contract_matrix_v2_tasks.sh <out_dir> <roles comma> [extra evaluate args, e.g. --max_triplets 12]
set -euo pipefail
OUT=$1; ROLES=$2; shift 2; EXTRA="$*"
V1=assets/v1_runs; R=assets/runs/r10_contract_matrix_v2
COMMON="r10_dose_composition.evaluate --config configs/r10/r10_v1.json --triplets assets/cache/r10_v1/eligible_triplets.tsv --atlas_K $V1/r10_b1_v3/atlas_K.json --directions cross,same --episodes E100,E1000 $EXTRA"
mkdir -p $OUT
for role in ${ROLES//,/ }; do
  for line in A549 K562 MCF7; do
    echo "$COMMON --role $role --line $line --methods mixture --out $OUT/${role}_${line}_D00.tsv.gz"
    echo "$COMMON --role $role --line $line --methods mixture_t --t_star_json $V1/r10_c3_tstar_v1.json --out $OUT/${role}_${line}_tstar_cross.tsv.gz"
    echo "$COMMON --role $role --line $line --methods mixture_t --t_star_json $R/tstar_same.json --out $OUT/${role}_${line}_tstar_same.tsv.gz"
    echo "$COMMON --role $role --line $line --methods D10,D01,D11,W --heads_dir $R/heads_cross --train_contract cross --seeds 0,1,2 --out $OUT/${role}_${line}_heads_cross.tsv.gz"
    echo "$COMMON --role $role --line $line --methods D10,D01,D11,W --heads_dir $R/heads_same --train_contract same --seeds 0,1,2 --out $OUT/${role}_${line}_heads_same.tsv.gz"
  done
done
