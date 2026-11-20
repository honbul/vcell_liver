#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=03:00:00
set -euo pipefail
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONHASHSEED=0
if [[ "$1" == stage ]]; then export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; fi
printf 'job=%s host=%s start=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$(date -Iseconds)"
module="$1"
shift
/usr/bin/time -v "${R12_PY:-../vcell_liver_r11_chemical_probe/assets/study-env/bin/python}" -m "src.r12_multiaxis.$module" --config configs/r12_multiaxis/config.yaml "$@"
