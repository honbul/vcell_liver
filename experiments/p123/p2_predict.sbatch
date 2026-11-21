#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail
root="$1"
base=$(realpath "$root/../..")
old="$base/vcell_liver_goal_tracks/v1"
assets="$base/vcell_liver_r12_multiaxis/assets"
r11="$base/vcell_liver_r11_chemical_probe/assets"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$PWD:$old/t2-public/deps"
task="$SLURM_ARRAY_TASK_ID"
doses=(5 .5 .05)
ns=(20 200)
roles=(A B)
dose="${doses[$((task / 12))]}"
n="${ns[$((task / 6 % 2))]}"
fold="$((task / 2 % 3))"
role="${roles[$((task % 2))]}"
printf 'job=%s task=%s dose=%s n=%s fold=%s role=%s start=%s\n' "$SLURM_JOB_ID" "$task" "$dose" "$n" "$fold" "$role" "$(date -Is)"
exec "$r11/study-env/bin/python" -m src.p123.p2_predict --dose "$dose" --minimum-cells "$n" --fold "$fold" --role "$role" \
   --config "$base/vcell_liver_r13_optimization/configs/r12_multiaxis/config.yaml" --assets "$assets" \
   --protocol reports/p123/PROTOCOL.md --t1-protocol reports/goal_tracks/PROTOCOL.md --stage "$old/t1-cache" --prior "$old" \
   --positions "$root/p1-position" --lincs "$root/lincs" --out "$root/p2/predictions"
