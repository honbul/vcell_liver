# Environment for Track B (TranSiGen low-data HCC adaptation). Sourced by every job of this study.
export TB_ROOT=${TB_ROOT:-/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_transigen_adaptation_b1}
export TB_ASSETS="$TB_ROOT/assets"
export TB_PY="$TB_ASSETS/envs/tsg/bin/python"
export TB_TSG_SRC="$TB_ASSETS/external/TranSiGen"       # upstream, pinned 8ec2218e
export TB_RAW="$TB_ASSETS/raw/cmap2020"
export TB_PROC="$TB_ASSETS/processed"
export TB_RUNS="$TB_ASSETS/runs"
cd "$TB_ROOT"
mkdir -p logs "$TB_RAW" "$TB_PROC" "$TB_RUNS" manifests/transigen_adaptation_b1 reports/transigen_adaptation_b1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONHASHSEED=0
export WANDB_MODE=offline
export WANDB_SILENT=true
