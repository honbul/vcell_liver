#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail

phase="$1"
run="$2"
b="$3"
g="$b/vcell_liver_r19_plan"
t="$b/vcell_liver_goal_tracks/v1"
r15="$b/vcell_liver_r15_lincs/v1"
orig="$b/vcell_liver_dev_i/i-preflight.TE8L9Fqo/locked"
access="$g/d1-physical.N4qP3EO9/access"
inputs="$g/d12.gBkpbPh2/inputs"
python="$b/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
r=reports/r19_plan
control="$r/data/preflight/d1/receipt.json"
pin="$r/data/preflight/d1/access/receipt.json"
protocol="$r/D3_PROTOCOL.md"
score=(--access "$access" --access-reference "$pin" --control-reference "$control"
       --original-plan "$orig" --original-reference "$r/data/preflight/i_track/receipt.json"
       --dose-reference "$r/data/x_l_support/l1/receipt.json" --cache "$r15/cache/GSE92742"
       --protocol "$protocol" --forecasts "$run/forecasts" --freeze "$run/freeze.json")

export PYTHONPATH="$PWD:$b/vcell_liver/vcell_r12_runtime_cu126:$b/vcell_liver/vcell_r12_runtime_transformers:$t/t2/dependencies:$t/t2-public/deps"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

case "$phase" in
  predict)
    task="${SLURM_ARRAY_TASK_ID:?expected an array task}"
    [[ "$task" =~ ^[0-8]$ ]]
    roles=(A B C)
    slice="f$((task/3))-${roles[task%3]}"
    exec "$python" -m src.r19_plan.d3_predict --slice "$slice" --bundle "$inputs/$slice" --pin "$pin" \
      --assignments "$g/preflight.kwjS2aLo/d1-control-lock/assignments.tsv" --control-reference "$control" \
      --config "$b/vcell_liver_r12_multiaxis/configs/r12_multiaxis/config.yaml" --assets "$r15/model-inputs" \
      --protocol "$protocol" --out "$run/forecasts/$slice"
    ;;
  freeze)
    digest="$("$python" -m src.r19_plan.d3_workflow freeze "${score[@]}")"
    [[ "$digest" =~ ^[0-9a-f]{64}$ ]]
    (set -o noclobber; printf '%s\n' "$digest" > "$run/freeze.sha256")
    ;;
  score)
    IFS= read -r digest < "$run/freeze.sha256"
    [[ "$digest" =~ ^[0-9a-f]{64}$ ]]
    exec "$python" -m src.r19_plan.d3_workflow score "${score[@]}" --expected-sha256 "$digest" \
       --vehicles "$b/vcell_liver_r17_lincs/v1/vehicles" --out "$run/score" \
       --instances /mnt/S1/sdata/pubdata/LINCS/GSE92742/GSE92742_Broad_LINCS_inst_info.txt
    ;;
  *) printf 'Unknown D3 phase %s\n' "$phase" >&2; exit 2 ;;
esac
