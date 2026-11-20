# Shared environment for the conditional-expert drug-response runs. Sourced by every sbatch here.
# A dedicated venv, separate from the feedback-20261110 envs, so this study cannot perturb them.
export MOE_ROOT=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_conditional_moe_v1
export MOE_PY="$MOE_ROOT/assets/envs/moe/bin/python"
cd "$MOE_ROOT"
mkdir -p logs
# Keep threading deterministic and modest; these are small models and the node is shared.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
