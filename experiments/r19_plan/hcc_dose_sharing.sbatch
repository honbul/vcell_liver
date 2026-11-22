#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:25:00
set -euo pipefail
run=$1
phase=$2
base=$(dirname "$(dirname "$run")")
repo="$run/repo"
goal="$base/vcell_liver_goal_tracks/v1"
assets="$base/vcell_liver_r11_chemical_probe/assets"
python="$assets/study-env/bin/python"
export PYTHONPATH="$repo:$goal/t2/dependencies:$goal/t2-public/deps"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
opts=(--protocol "$repo/reports/p4/PROTOCOL.md" --archive "$repo/reports" --prior "$goal"
      --p123 "$base/vcell_liver_p123/v1" --out "$base/vcell_liver_p4/v1" --work "$run"
      --features "$assets/features/features.npz" --design "$repo/reports/r19_plan/HCC_DOSE_SHARING_PROTOCOL.md")
case "$phase" in
  predict)
    roles=(A B)
    opts+=(--fold "$((SLURM_ARRAY_TASK_ID/2))" --role "${roles[SLURM_ARRAY_TASK_ID%2]}")
    ;;
  score)
    opts+=(--expected-sha256 "$(cat "$run/freeze.sha256")")
    ;;
esac
/usr/bin/time -v "$python" -u -m src.r19_plan.hcc_dose_sharing "$phase" "${opts[@]}"
if [[ "$phase" == lock ]]; then
  sha256sum "$run/freeze.json" | cut -d' ' -f1 > "$run/freeze.sha256"
fi
