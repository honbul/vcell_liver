#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
set -euo pipefail
base="/mnt/S1/sdata/workbench/Virtual_Cell/gssong"
supplement=${2:-"$PWD/${1:-E}_metadata_supplements.json"}
if [[ ! -f "$supplement" ]]; then
  python3 -s -m src.tahoe_position.followup_e_supplements --raw "$base/vcell_liver_r14_cross_organ/raw" --out "$supplement"
fi
exec "$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python" \
  -m src.tahoe_position.followup_e --protocol reports/tahoe_position/FOLLOWUP_PROTOCOL.md \
  --supplements "$supplement" --out "$PWD/${1:-E}"
