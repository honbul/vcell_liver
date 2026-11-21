#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
python3 -m src.r14_cross_organ.audit_metadata --raw "$PWD/raw" \
  --tox21 /mnt/S1/sdata/pubdata/Virtual_Cell/GEO/GSE296254 \
  --tahoe-genes "$PWD/../vcell_liver_r11_chemical_probe/assets/metadata/genes.tsv" --out "$PWD/$1"
