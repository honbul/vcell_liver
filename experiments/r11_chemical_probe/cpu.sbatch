#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -euo pipefail
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONHASHSEED=0
if [[ "$1" == prepare && "$3" != metadata ]]; then
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
fi
module_name="$1"
shift
printf 'job=%s host=%s start=%s module=%s argv=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$(date -Iseconds)" "$module_name" "$*"
python_path="${R11_PY:-$PWD/assets/study-env/bin/python}"
"$python_path" -m "src.r11_chemical_probe.$module_name" --config configs/r11_chemical_probe/config.yaml --assets assets "$@"
