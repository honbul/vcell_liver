# Environment for the initialisation-fix re-runs (goal section 0).
#
# These re-use the v1 condition table byte for byte -- same truth definition, same 209/70/70 drug
# split, same 48 cell lines -- and change only the model initialisation. Nothing here rebuilds data
# and nothing here touches the v1 worktree or its outputs, which stay as the record of the first
# result.
export MOE_ROOT=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_chemcpa_moe_v2
export MOE_PY="$MOE_ROOT/assets/envs/moe/bin/python"
export MOE_DATA="$MOE_ROOT/assets/v1_assets/data"     # read-only: the v1 build
export MOE_OUT="$MOE_ROOT/assets/initfix_v2"
cd "$MOE_ROOT"
mkdir -p logs "$MOE_OUT"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
