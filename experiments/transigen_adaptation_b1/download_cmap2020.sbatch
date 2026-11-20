#!/usr/bin/env bash
#SBATCH -J tb_dl_cmap2020
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem 16G
#SBATCH -t 1-00:00:00
#SBATCH -o logs/tb_dl_cmap2020_%j.out
# One-time download of the official CMap LINCS 2020 Level 3 matrices and instance metadata.
# Resumable parallel range requests, written to <name>.part and renamed atomically, guarded by a lock.
set -euo pipefail
source experiments/transigen_adaptation_b1/env.sh
BASE=https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020
FILES=(
  instinfo_beta.txt
  level3/level3_beta_ctl_n188708x12328.gctx
  level3/level3_beta_trt_cp_n1805898x12328.gctx
)
exec 9>"$TB_RAW/.download.lock"
flock -n 9 || { echo "another download holds the lock"; exit 1; }
echo "start=$(date -Is) commit=$(git rev-parse --short HEAD) host=$(hostname)"
for rel in "${FILES[@]}"; do
  name=$(basename "$rel")
  if [ -s "$TB_RAW/$name" ]; then echo "present, skipping: $name"; continue; fi
  echo "downloading $rel at $(date -Is)"
  python3 src/transigen_adaptation_b1/parallel_download.py --threads 24 "$BASE/$rel" "$TB_RAW/$name"
done
python3 -m src.transigen_adaptation_b1.record_download --raw "$TB_RAW" --base "$BASE" \
  --out manifests/transigen_adaptation_b1/download_manifest.json "${FILES[@]}"
chmod a-w "$TB_RAW"/*.gctx "$TB_RAW"/instinfo_beta.txt
echo "end=$(date -Is)"
