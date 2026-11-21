#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:10:00
set -euo pipefail
v1="$1"
action="$2"
base="$(dirname "$(dirname "$v1")")"
prior="$base/vcell_liver_goal_tracks/v1"
code="$v1/p1-code"
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
protocol="$code/reports/p123/PROTOCOL.md"
export PYTHONPATH="$code:$prior/t1-code:$prior/t2/predict-code"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
test "$(sha256sum "$protocol" | cut -d' ' -f1)" = d0680c422b41c63f895b8c1efc0fdedbcbc7965fdc1e35a40e6eb7bde6da8693
options=(--protocol "$protocol" --stage "$prior/t1-cache")
case "$action" in
  position|predict)
    task="$SLURM_ARRAY_TASK_ID"
    doses=(5 0.5 0.05)
    roles=(A B)
    counts=(20 200)
    options+=(--config "$base/vcell_liver_r13_optimization/configs/r12_multiaxis/config.yaml"
              --assets "$base/vcell_liver_r12_multiaxis/assets"
              --dose "${doses[task/12]}" --role "${roles[task%2]}"
              --fold "$((task%6/2))" --minimum-cells "${counts[task%12/6]}")
    if [[ "$action" == position ]]; then
      exec "$python" -m src.p123.p1_position "${options[@]}" --out "$v1/p1-position"
    fi
    exec "$python" -m src.p123.p1_predict "${options[@]}" --position "$v1/p1-position" \
      --prior "$prior" --out "$v1/p1-predict"
    ;;
  freeze|score)
    dose="$3"
    shift 3
    exec "$python" -m src.p123.p1_score "$action" "${options[@]}" \
      --prior "$prior" --dose "$dose" --p1-root "$v1/p1-predict" "$@"
    ;;
  *)
    exit 2
    ;;
esac
