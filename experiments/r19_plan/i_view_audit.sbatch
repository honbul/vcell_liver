#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:45:00
set -euo pipefail

stage="$1"
action="$2"
base="$(dirname "$(dirname "$stage")")"
t2="$base/vcell_liver_goal_tracks/v1"
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
export PYTHONPATH="$stage:$t2/t2/dependencies:$t2/t2-public/deps"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES=''
audit="$base/vcell_liver_r19_plan/audit.YBfVmbuo/r18-registered"

case "$action" in
    population)
        exec "$python" -m src.r19_plan.i_audit --base "$base" --registered "$audit" \
            --registered-reference "$stage/reports/r19_plan/data/r18-support-receipt.json" \
            --role-reference "$stage/reports/r17_lincs/data/support/receipt.json" --out "$stage/population"
        ;;
    I1)
        db="$base/vcell_liver_evidence_gated_r7/assets/external/chembl_37/chembl_37/chembl_37_sqlite/chembl_37.db"
        exec "$python" -m src.r19_plan.i_chembl --database "$db" --population "$stage/population" --out "$stage/I1"
        ;;
    I2|I4)
        exec "$python" -m src.r19_plan.i_phenotypes --view "$action" --data "$stage/data" \
            --population "$stage/population" --out "$stage/$action"
        ;;
    coverage)
        exec "$python" -m src.r19_plan.i_coverage --root "$stage" --out "$stage/coverage"
        ;;
    *) exit 2 ;;
esac
