#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=f4_tahoe
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
# Usage: sbatch [--cpus-per-task/--mem overrides] experiments/feedback/f4_tahoe.sbatch <step>
#   step = meta | pb_snu423 | pb_hepg2 | pbde | effects | emerald
# Run from the G1 worktree root. Source data are read-only; outputs go to assets/f4 and reports/feedback/F4.
set -euo pipefail
ROOT=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_feedback_20261110
PY=/mnt/S1/sdata/pubdata/Virtual_Cell/gssong/vcell_liver_feedback_20261110/envs/f4/bin/python
A=$ROOT/assets/f4
R=$ROOT/reports/feedback/F4
BL=/mnt/S1/sdata/pubdata/Virtual_Cell/Tahoe-100M/by_cell_line
cd "$ROOT"; mkdir -p "$A" "$R"
echo "job=$SLURM_JOB_ID node=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?} step=$1 start=$(date -Is)"
case "$1" in
  meta) $PY -u src/feedback/f4_tahoe_metadata.py --out-dir "$R" --asset-dir "$A" ;;
  pb_snu423) $PY -u src/feedback/f4_tahoe_pseudobulk.py --h5ad $BL/SNU-423_counts.h5ad --cell-qc $A/snu423_cell_qc.parquet --out $A/snu423_pseudobulk.npz ;;
  pb_hepg2) $PY -u src/feedback/f4_tahoe_pseudobulk.py --h5ad $BL/HepG2-C3A_counts.h5ad --cell-qc $A/hepg2c3a_cell_qc.parquet --out $A/hepg2c3a_pseudobulk.npz ;;
  pbde) $PY -u src/feedback/f4_tahoe_pbde.py --out-dir "$R" --asset-dir "$A" --procs ${SLURM_CPUS_PER_TASK:-8} ;;
  effects) $PY -u src/feedback/f4_tahoe_effects.py --asset-dir "$A" --out-dir "$R" ;;
  emerald) $PY -u src/feedback/f4_emeraldbay_meta.py --out-dir "$R" ;;
  *) echo "unknown step $1"; exit 2 ;;
esac
echo "end=$(date -Is)"
sstat -j "$SLURM_JOB_ID" --format=JobID,MaxRSS 2>/dev/null || true
