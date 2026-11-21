#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:10:00
set -euo pipefail
run="$1"
action="$2"
base="$(dirname "$(dirname "$run")")"
prior="$base/vcell_liver_goal_tracks/v1"
p123="$base/vcell_liver_p123/v1"
r12="$base/vcell_liver_r12_multiaxis"
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
export PYTHONPATH="$PWD:$base/vcell_liver/vcell_r12_runtime_cu126:$base/vcell_liver/vcell_r12_runtime_transformers:$prior/t2/dependencies:$prior/t2-public/state-deps:$prior/t2-public/deps"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONHASHSEED=0
options=(--protocol reports/p4/PROTOCOL.md --archive reports --prior "$prior" --p123 "$p123" --out "$run")
case "$action" in
  predict)
    task="$SLURM_ARRAY_TASK_ID"
    doses=(5 0.5 0.05)
    roles=(A B)
    counts=(20 200)
    exec "$python" -m src.p4.predict "${options[@]}" --lane "$3" --assets "$r12/assets" \
      --config "$r12/configs/r12_multiaxis/config.yaml" --dose "${doses[task/12]}" \
      --role "${roles[task%2]}" --fold "$((task%6/2))" --minimum-cells "${counts[task%12/6]}"
    ;;
  freeze|score)
    exec "$python" -m src.p4.score "$action" "${options[@]}" --dose "$3"
    ;;
  *)
    exit 2
    ;;
esac
