#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:20:00
set -euo pipefail

workspace="$1"
stage="$2"
base="$(dirname "$(dirname "$workspace")")"
code="$workspace/code"
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
deps="$base/vcell_liver_goal_tracks/v1"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=0
export PYTHONPATH="$code:$deps/t2/dependencies:$deps/t2-public/deps"
export CUDA_VISIBLE_DEVICES=''
cd "$code"

case "$stage" in
    index)
        exec "$python" -m src.vc_build.state_data \
            --audit "$base/vcell_liver_r19_plan/hcc-lincs24.weD7cHPm/audit" \
            --audit-reference "$code/reports/r19_plan/data/hcc_lincs24/audit/receipt.json" \
            --matrix-reference "$code/reports/r19_plan/data/preflight/r18/index/CMap2020/receipt.json" \
            --chemistry /mnt/S1/sdata/pubdata/LINCS/LINCS2020/compoundinfo_beta.txt \
            --landmarks "$base/vcell_liver_r15_lincs/v1/cache/GSE92742/landmarks.tsv" \
            --depmap /mnt/S1/sdata/pubdata/DepMap/25Q3 \
            --protocol "$code/state/PROTOCOL.md" --out "$workspace/cmap-index-v1"
        ;;
    *)
        exec "$python" -m src.vc_build.state_eval "$stage" \
            --index "$workspace/cmap-index-v1" --run "$workspace/cmap-state-v1"
        ;;
esac
