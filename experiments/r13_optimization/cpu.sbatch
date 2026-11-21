#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
set -euo pipefail
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=0
export R13_DEVICE=cpu
python="$PWD/../vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
r12="$PWD/../vcell_liver_r12_multiaxis/assets"
printf 'job=%s task=%s host=%s start=%s\n' "$SLURM_JOB_ID" "${SLURM_ARRAY_TASK_ID:-none}" "$(hostname)" "$(date -Iseconds)"
family="$1"
protocol="$2"
run="$3"
shift 3
/usr/bin/time -v "$python" -m src.r13_optimization.tune_auxiliary --task "$SLURM_ARRAY_TASK_ID" \
  --family "$family" --protocol "$protocol" --out "runs/$run/$protocol/$family/task$SLURM_ARRAY_TASK_ID" \
  --config configs/r12_multiaxis/config.yaml --assets "$r12" "$@"
