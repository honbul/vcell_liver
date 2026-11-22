#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:24:00
#SBATCH --array=0-3
set -euo pipefail

stage="$1"
base="$(dirname "$(dirname "$stage")")"
t2="$base/vcell_liver_goal_tracks/v1"
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
db="$base/vcell_liver_evidence_gated_r7/assets/external/chembl_37/chembl_37/chembl_37_sqlite/chembl_37.db"
export PYTHONPATH="$stage:$t2/t2/dependencies:$t2/t2-public/deps"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''

exec "$python" -m src.r19_plan.i_chembl_presence --database "$db" --identities "$stage/I1/identities.tsv" \
    --chunk "$SLURM_ARRAY_TASK_ID" --chunks 4 --out "$stage/I1-presence/$SLURM_ARRAY_TASK_ID"
