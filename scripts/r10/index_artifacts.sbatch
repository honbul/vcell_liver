#!/bin/bash
#SBATCH --job-name=r10_index
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=2:00:00
#SBATCH --output=assets/logs/%x_%j.out
# sha256 index of the server-only R10 artifacts (cache, atlas, runs; raw files are in the GEO receipt).
set -euo pipefail
source scripts/r10/env.sh
OUT=assets/server_artifacts.tsv
printf "path\tbytes\tsha256\n" > $OUT.tmp
find assets/cache/r10_v1 assets/runs -type f ! -name "*.lock" | sort | while read f; do
  printf "%s\t%s\t%s\n" "$f" "$(stat -c %s "$f")" "$(sha256sum "$f" | cut -d" " -f1)" >> $OUT.tmp
done
mv $OUT.tmp $OUT
du -sh assets assets/* | tee assets/disk_usage.txt
echo "end=$(date -Is)"
