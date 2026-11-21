#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
set -euo pipefail
export OPENBLAS_NUM_THREADS=3 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTHONHASHSEED=0
python="$PWD/../vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
r12="$PWD/../vcell_liver_r12_multiaxis/assets"
protocol="$1"
work="runs/$2/$protocol"
opts=(--config configs/r12_multiaxis/config.yaml --study configs/r13_optimization/study.yaml --assets "$r12"
  --predictions "$work/frozen" --freeze "$work/freeze.json" --protocol "$protocol")
printf 'job=%s host=%s start=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$(date -Iseconds)"
"$python" -m src.r13_optimization.score freeze "${opts[@]}"
/usr/bin/time -v "$python" -m src.r13_optimization.score score "${opts[@]}" --out "$work/scored"
