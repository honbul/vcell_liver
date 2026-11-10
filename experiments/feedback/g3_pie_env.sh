# Sourced by the g3_pie_* sbatch scripts (run from the G1 worktree root).
# Dedicated PIE env (python 3.12, uv sync --frozen of ArcInstitute/pie v1.0.1); no tokens, no wandb.
G3_ROOT="$(pwd)/assets/g3_pie"
export PIE_PY="$(pwd)/assets/envs/pie/venv/bin/python"
export PIE_BIN="$(pwd)/assets/envs/pie/venv/bin/pie"
export PIE_DATA_ROOT="$G3_ROOT/data" PIE_RUNS_ROOT="$G3_ROOT/runs" PIE_CACHE_DIR="$G3_ROOT/cache"
export HF_HOME="$G3_ROOT/hf_home" HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=disabled WANDB_DISABLED=true
export WANDB_ENTITY=none WANDB_PROJECT=none PYTHONHASHSEED=0
export PIE_ENV_FILE="$G3_ROOT/common.sh"
mkdir -p "$PIE_DATA_ROOT" "$PIE_RUNS_ROOT" "$PIE_CACHE_DIR" "$HF_HOME" logs
[ -f "$PIE_ENV_FILE" ] || printf 'PIE_DATA_ROOT=%s\nPIE_RUNS_ROOT=%s\nPIE_CACHE_DIR=%s\nWANDB_ENTITY=none\nWANDB_PROJECT=none\n' \
  "$PIE_DATA_ROOT" "$PIE_RUNS_ROOT" "$PIE_CACHE_DIR" > "$PIE_ENV_FILE"
