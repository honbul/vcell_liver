#!/bin/bash
#SBATCH --job-name=r10_dl
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=assets/logs/%x_%j.out
# Download the original public Sci-Plex3 files (GEO GSE139944 / GSM4150378) once into assets/raw/.
# NCBI public HTTPS endpoint only, generic User-Agent, no credentials. Verifies byte size against the
# GEO filelist.txt, gzip integrity, and records sha256 in assets/raw/raw_data_receipt.json.
set -euo pipefail
ROOT=$(pwd)
RAW=$ROOT/assets/raw/GSE139944_GSM4150378
mkdir -p "$RAW" "$ROOT/assets/logs"
exec 9>"$ROOT/assets/raw/.download.lock"
flock -n 9 || { echo "another download holds the lock"; exit 1; }

BASE=https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4150nnn/GSM4150378/suppl
LIST=https://ftp.ncbi.nlm.nih.gov/geo/series/GSE139nnn/GSE139944/suppl/filelist.txt
UA="Mozilla/5.0 (compatible; generic-data-fetch)"
FILES="GSM4150378_sciPlex3_pData.txt.gz
GSM4150378_sciPlex3_A549_MCF7_K562_screen_cell.annotations.txt.gz
GSM4150378_sciPlex3_A549_MCF7_K562_screen_gene.annotations.txt.gz
GSM4150378_sciPlex3_A549_MCF7_K562_hashTable_metadata.txt.gz
GSM4150378_sciPlex3_hashSampleSheet.txt.gz
GSM4150378_sciPlex3_A549_MCF7_K562_screen_hashTable.out.txt.gz
GSM4150378_sciPlex3_A549_MCF7_K562_screen_UMI.count.matrix.gz"

curl -sS -A "$UA" -o "$RAW/filelist.txt" "$LIST"
for f in $FILES; do
  want=$(awk -F'\t' -v n="$f" '$2==n{print $4}' "$RAW/filelist.txt")
  have=$(stat -c %s "$RAW/$f" 2>/dev/null || echo 0)
  if [ "$have" != "$want" ]; then
    echo "$(date -Is) fetching $f (expected $want bytes)"
    curl -sS -f -L -C - --retry 5 --retry-delay 20 -A "$UA" -o "$RAW/$f" "$BASE/$f"
  fi
  have=$(stat -c %s "$RAW/$f")
  [ "$have" = "$want" ] || { echo "SIZE MISMATCH $f $have != $want"; exit 2; }
  gzip -t "$RAW/$f"
  echo "$(date -Is) ok $f $have"
done
cd "$RAW"
sha256sum filelist.txt $FILES > SHA256SUMS
python3 -I - "$RAW" "$BASE" "$LIST" <<'PY'
import json, os, sys, datetime
raw, base, lst = sys.argv[1:4]
rows = [l.split() for l in open(os.path.join(raw, "SHA256SUMS"))]
meta = {}
for l in open(os.path.join(raw, "filelist.txt")):
    p = l.rstrip("\n").split("\t")
    if len(p) >= 4 and p[0] == "File":
        meta[p[1]] = dict(geo_time=p[2], geo_size=int(p[3]), geo_type=p[4] if len(p) > 4 else "")
out = dict(
    accession=dict(series="GSE139944", sample="GSM4150378", screen="sciPlex3 A549/K562/MCF7"),
    endpoint="NCBI GEO public HTTPS (ftp.ncbi.nlm.nih.gov), generic User-Agent, no credentials",
    filelist_url=lst,
    access_date_utc=datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    not_downloaded=["GSM4150378_sciPlex3_*_24hrs.RDS.gz", "GSM4150378_sciPlex3_cds_all_cells.RDS.gz",
                    "GSE139944_RAW.tar (other sci-Plex samples)"],
    files=[dict(name=n, url=(lst if n == "filelist.txt" else f"{base}/{n}"),
                bytes=os.path.getsize(os.path.join(raw, n)), sha256=h, **meta.get(n, {}))
           for h, n in rows],
)
json.dump(out, open(os.path.join(raw, "raw_data_receipt.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
PY
echo "$(date -Is) done"
