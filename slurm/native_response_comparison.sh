#!/usr/bin/env bash
#SBATCH --job-name=vc_native_response
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --nodelist=g3
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --chdir=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver
#SBATCH --output=logs/slurm_native_response_%j.log
set -euo pipefail
source envs/activate.sh scgpt
failed=0
for initialization in pretrained random; do
  run_id=B04_${initialization}_native_common_seed42_v1
  if [ ! -f reports/research_phase2/runs/$run_id/run.json ]; then
    .venvs/vcell-runtime310/bin/python envs/portable310.py --env scgpt scripts/research/run_scgpt_response.py --initialization "$initialization" --run-id "$run_id" || failed=1
  fi
done
run_id=B03_GEARS_native_common_seed42_v1
if [ ! -f reports/research_phase2/runs/$run_id/run.json ]; then
  .venvs/vcell-runtime310/bin/python envs/portable310.py --env research scripts/research/run_gears_hepg2.py --task-root hepg2_foundation_common_v1 --run-id "$run_id" --epochs 5 --patience 2 || failed=1
fi
exit "$failed"
