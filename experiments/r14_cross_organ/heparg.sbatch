#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=18G
#SBATCH --time=01:30:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
root="$PWD"
r14="$(realpath ..)"
base="$(realpath ../..)"
r11="$base/vcell_liver_r11_chemical_probe/assets"
primary=/mnt/S1/sdata/pubdata/Virtual_Cell/GEO/GSE296254
protocol="$root/reports/r14_cross_organ/EXECUTION_PROTOCOL.md"
manifest="$root/data/heparg_compound_folds.tsv"
export PYTHONPATH="$root:$base/vcell_liver_r13_optimization:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS"
python="$r11/study-env/bin/python"
index="$r14/metadata/v4"
case "$1" in
 chemistry)
  args=(--index "$index" --dsstox "$root/references/DSSTox_01May18_Full_SelectMetaDataPlus.zip"
        --tahoe-partition "$base/vcell_liver_r12_multiaxis/assets/partition" --out "$manifest")
  exec "$python" -m src.r14_cross_organ.heparg_prepare chemistry "${args[@]}"
  ;;
 prepare)
  if [[ "$2" == GSE296254 ]]; then raw="$primary"; output=heparg; else raw="$r14/raw/heparg_bulk"; output=heparg_284321; fi
  args=(--index "$index" --manifest "$manifest" --genes "$r11/metadata/genes.tsv" --protocol "$protocol"
        --primary-file "$primary/GSE296254_tox21-counts-1.tsv.gz" --raw "$raw"
        --dataset "$2" --out "$root/data/$output")
  exec "$python" -m src.r14_cross_organ.heparg_prepare prepare "${args[@]}"
  ;;
 repeat|predict|freeze|score|cross)
  args=(--root "$root" --protocol "$protocol")
  exec "$python" -m src.r14_cross_organ.heparg_eval "$1" "${args[@]}" "${@:2}"
  ;;
 *) exit 2 ;;
esac
