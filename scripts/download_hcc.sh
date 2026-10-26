#!/usr/bin/env bash
set -euo pipefail
D=/mnt/S1/sdata/pubdata/Virtual_Cell/GEO/GSE149614
mkdir -p "$D"
for f in GSE149614_HCC.metadata.updated.txt.gz GSE149614_HCC.scRNAseq.S71915.count.txt.gz; do
  if [ ! -f "$D/$f" ]; then
    curl --fail --location --retry 2 --connect-timeout 20 -C - -o "$D/$f.part" "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE149nnn/GSE149614/suppl/$f"
    gzip -t "$D/$f.part"
    mv "$D/$f.part" "$D/$f"
  fi
done
sha256sum "$D"/*.gz
