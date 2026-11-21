#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=04:00:00
set -euo pipefail
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=0
export R13_DEVICE=cuda CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$PWD/../vcell_liver/vcell_r12_runtime_cu126:${PYTHONPATH:-}"
python="$PWD/../vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
r12="$PWD/../vcell_liver_r12_multiaxis/assets"
protocol="$1"
run="$2"
printf 'job=%s task=%s host=%s start=%s\n' "$SLURM_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$(hostname)" "$(date -Iseconds)"
/usr/bin/time -v "$python" -m src.r13_optimization.predict --config configs/r12_multiaxis/config.yaml \
  --study configs/r13_optimization/study.yaml --assets "$r12" --task "$SLURM_ARRAY_TASK_ID" --protocol "$protocol" \
  --tuned "runs/$run/$protocol" --out "runs/$run/$protocol/frozen/task$SLURM_ARRAY_TASK_ID"
