#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
set -euo pipefail

base="$(realpath "$PWD/../..")"
export PYTHONPATH="$PWD:$base/vcell_liver_r13_optimization:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OMP_NUM_THREADS=$OPENBLAS_NUM_THREADS MKL_NUM_THREADS=$OPENBLAS_NUM_THREADS
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
output="$PWD/ceilings"
mode=()
if [[ ${1:-} == moa ]]; then
  output="$PWD/ceilings-moa"
  mode+=(--moa_only)
fi

exec "$python" -m src.tahoe_position.ceilings \
  --assets "$base/vcell_liver_r12_multiaxis/assets" \
  --coordinates "$base/vcell_liver_position_audit/coordinates-v1" \
  --common "$PWD/common" --out "$output" \
  --config "$PWD/configs/r12_multiaxis/config.yaml" \
  --protocol "$PWD/reports/tahoe_position/STEPS23_PROTOCOL.md" \
  --halves "$base/vcell_liver_r6_followup_v1/assets/cache/halves_v1" \
  --halves_receipt "$PWD/manifests/r6_followup_v1/halves_receipt.json" \
  --half_genes "$PWD/manifests/chemcpa_moe_v2/genes.tsv" \
  --strata "$PWD/common/f4_labels.tsv" \
  --moa /mnt/S1/sdata/pubdata/Tahoe/Tahoe-100M/metadata/drug_metadata.parquet "${mode[@]}"
