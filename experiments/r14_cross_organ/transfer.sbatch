#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=26G
#SBATCH --time=03:00:00
set -euo pipefail
base="$(realpath "$PWD/../..")"
r11="$base/vcell_liver_r11_chemical_probe/assets"
stage="$base/vcell_liver_r12_multiaxis/assets"
export PYTHONPATH="$PWD:$base/vcell_liver_r13_optimization:$PWD/..:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OMP_NUM_THREADS=$OPENBLAS_NUM_THREADS MKL_NUM_THREADS=$OPENBLAS_NUM_THREADS
python="$r11/study-env/bin/python"
protocol="$PWD/reports/r14_cross_organ/EXECUTION_PROTOCOL.md"
work="$PWD/${R14_EXEC_LANE:-r14}"
args=(--assets "$stage" --r11 "$r11" --work "$work" --protocol "$protocol")
case "$1" in
  predict)
    task=$SLURM_ARRAY_TASK_ID
    roles=(A B)
    args+=(--split "$2" --fold "$((task / 2))" --role "${roles[$((task % 2))]}")
    args+=(--positions "$base/vcell_liver_position_audit/coordinates-v1" --data "$PWD/data")
    exec "$python" -m src.r14_cross_organ.transfer "${args[@]}"
    ;;
  freeze|score)
    exec "$python" -m src.r14_cross_organ.transfer_score "$1" "${args[@]}" --split "$2"
    ;;
  external-predict|external-freeze|external-score)
    args+=(--positions "$base/vcell_liver_position_audit/coordinates-v1" --data "$PWD/data")
    exec "$python" -m src.r14_cross_organ.transfer_external "${1#external-}" "${args[@]}" --name "$2"
    ;;
  *)
    exit 2
    ;;
esac
