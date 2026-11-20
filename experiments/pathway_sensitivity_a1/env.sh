# Track A environment. Sourced from the server worktree root. Private env; nothing shared is modified.
export PSA1_ENV="$PWD/assets/envs/psa1"
export PSA1_PY="$PSA1_ENV/bin/python"
export PSA1_RSCRIPT="$PSA1_ENV/bin/Rscript"
export PSA1_SRC="$PWD/src/pathway_sensitivity_a1"
export PSA1_RUNS="$PWD/assets/runs"
# one BLAS thread per worker; parallelism comes from Slurm array tasks
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
