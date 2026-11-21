#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:25:00
set -euo pipefail

study=$1
phase=$2
lab=$(dirname "$(dirname "$study")")
repo="$study/repo"
r14="$lab/vcell_liver_r14_cross_organ"
python="$lab/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
export PYTHONPATH="$repo"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
datasets=(GSE296254 GSE284321)
dataset=${datasets[SLURM_ARRAY_TASK_ID]}
opts=(--work "$study" --dataset "$dataset" --cache "$r14/execution-v1" --index "$r14/metadata/v4"
      --t3 "$lab/vcell_liver_goal_tracks/v1/t3" --t3-score "$repo/reports/goal_tracks/data/heparg/receipt.json"
      --protocol "$repo/reports/r19_plan/HEPARG_POSITION_REPEAT_PROTOCOL.md")
if [[ "$phase" == score ]]; then
  opts+=(--expected-sha256 "$(cat "$study/$dataset/freeze.sha256")")
fi
/usr/bin/time -v "$python" -m src.r19_plan.heparg_position_repeat "$phase" "${opts[@]}"
if [[ "$phase" == lock ]]; then
  sha256sum "$study/$dataset/freeze.json" | cut -d' ' -f1 > "$study/$dataset/freeze.sha256"
fi
