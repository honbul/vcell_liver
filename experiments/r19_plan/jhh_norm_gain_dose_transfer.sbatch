#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
set -euo pipefail

run=$1
dose=$2
phase=$3
root=$(dirname "$run")
base=$(dirname "$root")
repo="$run/repo"
report="$repo/reports/r19_plan"
public="$base/vcell_liver_goal_tracks/v1"
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
export PYTHONPATH="$repo:$public/t2/dependencies:$public/t2-public/deps"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
out="$run/$dose"
high="$report/data/jhh5_off_position"
args=(--dose "$dose" --run "$out" --protocol "$report/JHH_NORM_GAIN_DOSE_TRANSFER_PROTOCOL.md" --access "$out/access"
  --access-pin "$out/access/receipt.json" --high "$root/jhh5-off-position.0ldmuIkJ" --high-access-pin "$high/access/receipt.json"
  --high-protocol "$report/JHH5_OFF_POSITION_PROTOCOL.md" --high-native-protocol "$report/JHH5_OFF_POSITION_NATIVE_PROTOCOL.md"
  --high-pin "$high/freeze.json")
roles=(A B C)
if [[ "$phase" != access ]]; then
  args+=(--access-digest "$(cat "$out/access.sha256")")
fi
case "$phase" in
  access) args+=(--audit "$root/jhh-low-dose.gepTAIFI/audit" --audit-pin "$report/data/jhh_low_dose/audit/receipt.json"
    --cmap-pin "$report/data/preflight/r18/index/CMap2020/receipt.json"
    --landmarks "$base/vcell_liver_r15_lincs/v1/cache/GSE92742/landmarks.tsv" --out "$out/access") ;;
  donor) args+=(--role "${roles[${SLURM_ARRAY_TASK_ID:?}]}") ;;
  predict)
    task=${SLURM_ARRAY_TASK_ID:?}
    args+=(--group "f$((task/3))-${roles[task%3]}") ;;
  score) args+=(--digest "$(cat "$out/freeze.sha256")") ;;
  freeze) ;;
  *) exit 2 ;;
esac
/usr/bin/time -v "$python" -u -m src.r19_plan.jhh_norm_gain_dose_transfer "$phase" "${args[@]}"
case "$phase" in
  access) sha256sum "$out/access/receipt.json" | cut -d' ' -f1 > "$out/access.sha256" ;;
  freeze) sha256sum "$out/freeze.json" | cut -d' ' -f1 > "$out/freeze.sha256" ;;
esac
