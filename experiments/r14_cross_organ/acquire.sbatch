#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=3G
#SBATCH --time=01:30:00
set -euo pipefail
source="$1"
python="$PWD/../vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
shift
exec "$python" -m src.r14_cross_organ.acquire --source "$source" --out "$PWD/raw/$source" "$@"
