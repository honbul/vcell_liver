#!/usr/bin/env bash
#SBATCH -J tb_dl_cmap2020
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem 16G
#SBATCH -t 1-00:00:00
#SBATCH -o logs/tb_dl_cmap2020_%j.out
# One-time download of the official CMap LINCS 2020 Level 3 matrices and instance metadata.
# Resumable (curl -C -), written to <name>.part and renamed atomically, guarded by a lock.
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
  tries=0
  until curl -fL -sS -C - -A "curl/8" --retry 5 --retry-delay 30 -o "$TB_RAW/$name.part" "$BASE/$rel"; do
    tries=$((tries+1)); [ $tries -lt 200 ] || { echo "giving up on $name"; exit 1; }
    echo "curl stopped ($tries), resuming at $(stat -c %s "$TB_RAW/$name.part" 2>/dev/null || echo 0) bytes"; sleep 30
  done
  expected=$(curl -sI -A "curl/8" "$BASE/$rel" | awk 'tolower($1)=="content-length:"{print $2}' | tr -d '\r')
  actual=$(stat -c %s "$TB_RAW/$name.part")
  [ "$expected" = "$actual" ] || { echo "size mismatch for $name: $actual vs $expected"; exit 1; }
  mv "$TB_RAW/$name.part" "$TB_RAW/$name"
done
"$TB_PY" -I -m src.transigen_adaptation_b1.record_download --raw "$TB_RAW" --base "$BASE" \
  --out manifests/transigen_adaptation_b1/download_manifest.json "${FILES[@]}" \
  || python3 -m src.transigen_adaptation_b1.record_download --raw "$TB_RAW" --base "$BASE" \
  --out manifests/transigen_adaptation_b1/download_manifest.json "${FILES[@]}"
chmod a-w "$TB_RAW"/*.gctx "$TB_RAW"/instinfo_beta.txt
echo "end=$(date -Is)"
