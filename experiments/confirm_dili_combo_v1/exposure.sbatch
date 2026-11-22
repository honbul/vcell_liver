#!/usr/bin/env bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=5G
#SBATCH --time=01:30:00
set -euo pipefail
root=$1
mkdir -p "$root"
target="$root/pubchem18.zip"
if [[ -e "$target" ]]; then
    unzip -Z -1 "$target"
    exit
fi
incoming="$root/pubchem18.download"
curl --fail --silent --show-error --location --retry 3 --retry-all-errors --max-time 2000 \
     --output "$incoming" https://cloud.ml.jku.at/s/2ybfLRXWSYb4DZN/download
unzip -Z -1 "$incoming"
mv -n "$incoming" "$target"
sha256sum "$target" > "$root/pubchem18.sha256"
