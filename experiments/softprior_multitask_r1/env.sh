# Study R1 environment. Sourced from the server worktree root.
# The Track A environment (psa1) is used read-only through a private venv created with
#   <psa1>/bin/python -m venv --system-site-packages assets/envs/r1
# Extra packages (torch for the solver cross-check, tabpfn for the optional comparator) are installed
# only into assets/envs/r1 under this worktree.
export R1_A1_ENV=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_pathway_sensitivity_a1/assets/envs/psa1
export R1_PY="$PWD/assets/envs/r1/bin/python"
export R1_RSCRIPT="$R1_A1_ENV/bin/Rscript"
export R1_SRC="$PWD/src/softprior_multitask_r1"
export R1_RUNS="$PWD/assets/runs"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
