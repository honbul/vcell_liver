#!/usr/bin/env bash
#SBATCH --job-name=combo-v1-report
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:45:00
set -euo pipefail

root=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/nextgen_5track
result=$root/results/combo-v1
panels=$root/prepared/combo-v1/panels
track2=${root%/nextgen_5track}/liver_dili_v1
python=$root/runtime/bin/python
cd "$root/code/vc_combo_hcc_v1"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1

if [[ ! -f "$result/scores/all/receipt.json" ]]; then
    "$python" -m vc_combo_hcc.score --panels "$panels" --predictions "$result/predictions" \
        --output "$result/scores/all"
fi

if [[ ! -f "$result/report-v1/receipt.json" ]]; then
    "$python" -m vc_combo_hcc.report --results "$result" --track2-dili "$track2" \
        --output "$result/report-v1"
fi
