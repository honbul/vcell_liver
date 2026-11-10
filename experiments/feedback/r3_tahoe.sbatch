#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --exclude=g1,g2,g3,g4
#SBATCH --job-name=r3_tahoe
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
# Usage: sbatch [--cpus-per-task/--mem overrides] experiments/feedback/r3_tahoe.sbatch <step> [extra args]
#   step = pb_test | pb_all | prep | loco
# Run from the G1 worktree root. Tahoe mirror is read-only; outputs go to assets/r3 and reports/feedback/R3.
set -euo pipefail
ROOT=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_feedback_20261110
PY=$ROOT/assets/envs/r3/bin/python
A=$ROOT/assets/r3
R=$ROOT/reports/feedback/R3
cd "$ROOT"; mkdir -p "$A" "$R" logs
step=$1; shift
echo "job=$SLURM_JOB_ID node=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?} step=$step start=$(date -Is)"
TV="/usr/bin/time -v"
case "$step" in
  pb_test) $TV $PY -u src/feedback/r3_pseudobulk_all.py --out-dir $A/pb_test --procs 2 --plates plate1,plate13 --max-shards-per-plate 3 "$@" ;;
  pb_all) $TV $PY -u src/feedback/r3_pseudobulk_all.py --out-dir $A/pb --procs ${SLURM_CPUS_PER_TASK:-14} "$@" ;;
  prep) $TV $PY -u src/feedback/r3_prep.py --pb-dir $A/pb --out-dir $A/prep --report-dir $R "$@" ;;
  loco) $TV $PY -u src/feedback/r3_loco.py --prep-dir $A/prep --out-dir $A/loco --report-dir $R "$@" ;;
  *) echo "unknown step $step"; exit 2 ;;
esac
echo "end=$(date -Is)"
