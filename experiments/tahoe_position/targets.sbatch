#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=00:45:00
set -euo pipefail
base="$PWD/../.."
export PYTHONPATH="$PWD:$base/vcell_liver_r13_optimization:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
gtf="$PWD/references/Homo_sapiens.GRCh38.109.gtf.gz"
mkdir -p references
curl --fail --silent --show-error --retry 3 --connect-timeout 20 --location \
  https://ftp.ensembl.org/pub/release-109/gtf/homo_sapiens/Homo_sapiens.GRCh38.109.gtf.gz \
  --output "$gtf.partial"
gzip -t "$gtf.partial"
mv "$gtf.partial" "$gtf"
exec "$python" -m src.tahoe_position.targets \
  --assets "$base/vcell_liver_r12_multiaxis/assets" \
  --history "$base/vcell_liver_r13_optimization/runs/r13-round1/compound" \
  --coordinates "$base/vcell_liver_position_audit/coordinates-v1" \
  --config configs/r12_multiaxis/config.yaml --protocol reports/tahoe_position/STEPS23_PROTOCOL.md \
  --gtf "$gtf" --f4 reports/feedback/F4 --common "$PWD/common"
