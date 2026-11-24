#!/usr/bin/env bash
set -euo pipefail
snap=$1
stage=$2
backend=$3
tree_device=${4:-cpu}
extra=()
if [[ ${5:-} == early-neural ]]; then
  extra+=(--early-neural)
fi
id=${SLURM_ARRAY_TASK_ID:?}
splits=(drug-0 drug-1 drug-2 cell-0 cell-1 cell-2 CTRPv2-0 CTRPv2-1 CTRPv2-2 GDSC2-0 GDSC2-1 GDSC2-2 PRISM20Q2-0 PRISM20Q2-1 PRISM20Q2-2)
if [[ $backend == gbdt ]]; then
  families=(gbdt hybrid)
else
  families=(film attention gnn hybrid)
fi
n=${#families[@]}
split=${splits[id / (3*n)]}
inner=$(( (id / n) % 3))
family=${families[id % n]}
if [[ $tree_device == cuda ]]; then
  export H3_CUDA_LIGHTGBM=1
fi
exec "$snap/experiments/nextgen_h3_v2/job.sbatch" "$snap" -m h3v2.fit \
  "$stage" "$split" "$family" "$inner" "$backend" --tree-device "$tree_device" "${extra[@]}"
