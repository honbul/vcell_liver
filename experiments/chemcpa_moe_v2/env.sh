# Environment for the chemCPA-based conditional-expert study (goal sections 1-6).
export MOE_ROOT=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_chemcpa_moe_v2
export MOE_PY="$MOE_ROOT/assets/envs/moe/bin/python"          # the small-MLP / analysis env
export CPA_PY="$MOE_ROOT/assets/envs/chemcpa/bin/python"      # torch + lightning + hydra + rdkit
export CPA_SRC="$MOE_ROOT/assets/external/chemCPA"            # pinned 43e830eb
export V2_OUT="$MOE_ROOT/assets/v2"
export V1_DATA="$MOE_ROOT/assets/v1_assets/data"
cd "$MOE_ROOT"
mkdir -p logs "$V2_OUT"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
# Nothing in this study may report to an external tracker.
export WANDB_MODE=offline
export WANDB_SILENT=true
