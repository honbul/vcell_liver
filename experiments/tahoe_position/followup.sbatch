#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:35:00
set -euo pipefail

base="$(realpath "$PWD/../..")"
review="$base/review_claude_20261121"
previous="$review/position_steps23"
r11="$base/vcell_liver_r11_chemical_probe/assets"
export PYTHONPATH="$PWD:$base/vcell_liver_r13_optimization:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OMP_NUM_THREADS=$OPENBLAS_NUM_THREADS MKL_NUM_THREADS=$OPENBLAS_NUM_THREADS
python="$r11/study-env/bin/python"
protocol="$PWD/reports/tahoe_position/FOLLOWUP_PROTOCOL.md"

if [[ "$1" == data ]]; then
  args=(
    --assets "$base/vcell_liver_r12_multiaxis/assets" --primary "$r11/source"
    --previous "$previous/common" --coordinates "$base/vcell_liver_position_audit/coordinates-v1"
    --genes "$r11/metadata/genes.tsv" --gtf "$previous/references/Homo_sapiens.GRCh38.109.gtf.gz"
    --protocol "$protocol" --out "$PWD/cache"
  )
  exec "$python" -m src.tahoe_position.followup_data "${args[@]}"
elif [[ "$1" == a ]]; then
  exec "$python" -m src.tahoe_position.followup_a --cache "$PWD/cache" --protocol "$protocol" --out "$PWD/A"
elif [[ "$1" == d ]]; then
  args=(
    --cache "$PWD/cache" --protocol "$protocol" --out "$PWD/D"
    --r12 "$base/vcell_liver_r12_multiaxis/assets" --r13 "$base/vcell_liver_r13_optimization/runs/r13-round1/compound"
    --previous "$previous/common" --gmt "$PWD/references/h.all.v2024.1.Hs.symbols.gmt"
  )
  exec "$python" -m src.tahoe_position.followup_d "${args[@]}"
else
  exit 2
fi
