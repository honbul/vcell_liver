#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
set -euo pipefail
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$PWD/../vcell_liver/vcell_r12_runtime_cu126:${PYTHONPATH:-}"
printf 'job=%s host=%s start=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$(date -Iseconds)"
module="$1"
shift
"${R12_PY:-../vcell_liver_r11_chemical_probe/assets/study-env/bin/python}" -m "src.r12_multiaxis.$module" --config configs/r12_multiaxis/config.yaml "$@"
