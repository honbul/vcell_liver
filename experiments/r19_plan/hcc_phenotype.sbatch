#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
set -euo pipefail

run=$1
phase=$2
plan=$(dirname "$run")
base=$(dirname "$plan")
repo="$run/repo"
reports="$repo/reports/r19_plan"
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
t2="$base/vcell_liver_goal_tracks/v1/t2"
export PYTHONPATH="$repo:$t2/dependencies:$t2-public/deps" OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
args=(--out "$run" --reports "$reports" --protocol "$reports/HCC_PHENOTYPE_PROTOCOL.md"
  --previous "$plan/hcc-scaffold-refit.lWd6KCZt" --support "$plan/hcc-phen-support.mmZEzxcE/audit"
  --features "$base/vcell_liver_r11_chemical_probe/assets/features/features.npz")
case "$phase" in
  predict)
    task=${SLURM_ARRAY_TASK_ID:?}
    roles=(A B)
    args+=(--fold "$((task / 2))" --role "${roles[task%2]}")
    ;;
  freeze) ;;
  score) args+=(--expected-sha256 "$(cat "$run/freeze.sha256")") ;;
  *) exit 2 ;;
esac
/usr/bin/time -v "$python" -m src.r19_plan.hcc_phenotype "$phase" "${args[@]}"
if [[ "$phase" == freeze ]]; then
  sha256sum "$run/freeze.json" | cut -d' ' -f1 > "$run/freeze.sha256"
fi
