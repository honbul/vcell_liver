# source envs/activate.sh [scgpt|geneformer]
VCELL_ROOT=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver
export VCELL_DATA_ROOT=/mnt/S1/sdata/pubdata/Virtual_Cell/gssong/vcell_liver
export HF_HOME="$VCELL_DATA_ROOT/cache/huggingface"
export XDG_CACHE_HOME="$VCELL_DATA_ROOT/cache/xdg"
export NUMBA_CACHE_DIR="$VCELL_DATA_ROOT/cache/numba"
export MPLCONFIGDIR="$VCELL_DATA_ROOT/cache/matplotlib"
export PIP_CACHE_DIR="$VCELL_DATA_ROOT/cache/pip"
export JUPYTER_PATH="$VCELL_ROOT/.jupyter/share/jupyter${JUPYTER_PATH:+:$JUPYTER_PATH}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
# One GPU only; caller may choose a different device before sourcing.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
source "$VCELL_ROOT/.venvs/vcell-${1:-scgpt}/bin/activate"
cd "$VCELL_ROOT"
