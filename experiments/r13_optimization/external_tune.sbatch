#!/usr/bin/env bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
set -euo pipefail
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=0
family="$1"
if [[ "$family" == neural ]]; then
  export R13_DEVICE=cuda CUBLAS_WORKSPACE_CONFIG=:4096:8
  export PYTHONPATH="$PWD/../vcell_liver/vcell_r12_runtime_cu126:${PYTHONPATH:-}"
else
  export R13_DEVICE=cpu
fi
python="$PWD/../vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
r12="$PWD/../vcell_liver_r12_multiaxis/assets"
exec "$python" -m src.r13_optimization.external_tune evaluate \
  --config configs/r12_multiaxis/config.yaml --assets "$r12" \
  --manifest assets/external-gse306429/manifest-v3 --family "$family" \
  --task "$SLURM_ARRAY_TASK_ID" --out "runs/$2/$family/task$SLURM_ARRAY_TASK_ID"
