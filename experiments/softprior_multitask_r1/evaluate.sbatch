#!/usr/bin/env bash
#SBATCH --job-name=r1_eval
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/r1_eval_%j.log
# Frozen final evaluation (the only step that reads outer-test and LCCL labels).
# Submit: C=<frozen code commit, 7 chars> sbatch --export=ALL experiments/softprior_multitask_r1/evaluate.sbatch
set -euo pipefail
: "${C:?}"
source experiments/softprior_multitask_r1/env.sh
echo "host=$(hostname) commit=$(git rev-parse HEAD) frozen_code=$C start=$(date -Is)"
"$R1_PY" "$R1_SRC/evaluate.py" --internal-runs "f_d1_$C,f_r2_$C,f_r3_$C" --perm-runs "fp_d1_$C,fp_r2_$C,fp_r3_$C" \
  --external-runs "fx_d1_$C,fx_r2_$C,fx_r3_$C" --cmp-run "f_cmp_$C" --tracka-run r0_tracka_cdbf857 \
  --tag final --out-dir reports/softprior_multitask_r1/tables
echo "end=$(date -Is)"
