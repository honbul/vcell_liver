#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
set -euo pipefail
root="$1"
out="$2"
base=$(realpath "$root/../..")
old="$base/vcell_liver_goal_tracks/v1"
r11="$base/vcell_liver_r11_chemical_probe/assets"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$PWD:$old/t2-public/deps"
echo "job=$SLURM_JOB_ID host=$(hostname) start=$(date -Is)"
exec "$r11/study-env/bin/python" -m src.p123.p2_lincs \
    --lincs /mnt/S1/sdata/pubdata/LINCS --protocol reports/p123/PROTOCOL.md \
    --tahoe-entities "$r11/features/entities.tsv" --tahoe-split "$old/t1-cache/split.json" \
    --heparg "$root/heparg/lincs_targets.tsv" --out "$out"
