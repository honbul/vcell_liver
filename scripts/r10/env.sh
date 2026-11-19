# Sourced by every R10 Slurm script: own env, caches inside assets/, src on PYTHONPATH.
ROOT=$(pwd)
export HOME=$ROOT/assets/home XDG_CACHE_HOME=$ROOT/assets/cache MPLCONFIGDIR=$ROOT/assets/cache/mpl \
       PYTHONPATH=$ROOT/src PYTHONHASHSEED=0 PY=$ROOT/assets/envs/r10/bin/python
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
mkdir -p "$HOME" "$MPLCONFIGDIR" "$ROOT/assets/logs"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} git=$(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l) start=$(date -Is)"
