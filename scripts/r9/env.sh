# R9 environment: source from the R9 server worktree root.
# Core pipeline reuses the earlier project's venv READ-ONLY (numpy/scipy/pandas/sklearn/anndata/scanpy/torch;
# lock: envs/vcell-scgpt.freeze.txt). External-model env (scPoli/ProtoCloud) lives under assets/envs/.
R9_ROOT="${R9_ROOT:-/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r9_rare_state}"
export R9_ROOT
export R9_ASSETS="$R9_ROOT/assets"
export R9_PY="${R9_PY:-/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver/.venvs/vcell-scgpt/bin/python}"
export HOME="${HOME:-$R9_ASSETS/home}"
[ -d "$HOME" ] || export HOME="$R9_ASSETS/home"
export XDG_CACHE_HOME="$R9_ASSETS/cache/xdg"
export MPLCONFIGDIR="$R9_ASSETS/cache/matplotlib"
export NUMBA_CACHE_DIR="$R9_ASSETS/cache/numba"
export PIP_CACHE_DIR="$R9_ASSETS/cache/pip"
export HF_HOME="$R9_ASSETS/cache/hf"
export TORCH_HOME="$R9_ASSETS/cache/torch"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
mkdir -p "$R9_ASSETS/home" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" "$PIP_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$R9_ASSETS/logs"
cd "$R9_ROOT"
