#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
set -euo pipefail

run=$1
phase=$2
root=$(dirname "$run")
base=$(dirname "$root")
repo="$run/repo"
reports="$repo/reports/r19_plan"
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
t2="$base/vcell_liver_goal_tracks/v1/t2"
export PYTHONPATH="$repo:$t2/dependencies:$t2-public/deps"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
task=${SLURM_ARRAY_TASK_ID:?}
context=$task
if [[ "$phase" == predict ]]; then
  context=$((task / 9))
fi
cells=(PHH HEPG2)
target=${cells[context]}
out="$run/$target"
args=(--target "$target" --reports "$reports" --protocol "$reports/T4_CONTEXT_RESIDUAL_PROTOCOL.md"
  --hepg2 "$root/t4-run.fS2rpsIm" --phh "$root/t4-phh.p1AqqE6o" --previous "$root/t4-context.XlDSzix0"
  --forecasts "$out/forecasts" --freeze "$out/freeze.json" --out "$out/score")
case "$phase" in
  predict)
    roles=(A B C)
    args+=(--group "f$((task%9/3))-${roles[task%3]}")
    ;;
  score) args+=(--digest "$(cat "$out/freeze.sha256")") ;;
  freeze) ;;
  *) exit 2 ;;
esac
/usr/bin/time -v "$python" -m src.r19_plan.t4_context_residual "$phase" "${args[@]}"
if [[ "$phase" == freeze ]]; then
  sha256sum "$out/freeze.json" | cut -d' ' -f1 > "$out/freeze.sha256"
fi
