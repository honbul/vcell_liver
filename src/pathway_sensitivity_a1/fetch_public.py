#!/usr/bin/env python3
"""Fetch the public files Track A needs into the private asset area.

Each file goes into its own study directory, is written to a temporary name and renamed atomically,
and is recorded with URL, UTC date, byte size and SHA-256. A file that already exists is hashed and
recorded but not downloaded again. Shared read-only copies on the server are only hashed.

Usage: fetch_public.py <assets_root> <manifest_tsv>
"""
from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import os
import sys
import urllib.request

UA = "Mozilla/5.0 (compatible; research-data-fetch/1.0)"

LCCL = "https://lccl.zucmanlab.com/downloads/catalogue/"
DOWNLOADS = [
    # (study, file name, url)
    ("LCCL", "LCCLdescription.txt", LCCL + "LCCLdescription.txt"),
    ("LCCL", "RNAseqFPKM.txt", LCCL + "RNAseqFPKM.txt"),
    ("LCCL", "AUC.txt", LCCL + "AUC.txt"),
    ("LCCL", "GI50.txt", LCCL + "GI50.txt"),
    ("LCCL", "Screenedcompounds.txt", LCCL + "Screenedcompounds.txt"),
    ("GSE97098", "GSE97098_HCC81.genes.fpkm_table.log2.txt.gz",
     "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE97nnn/GSE97098/suppl/"
     "GSE97098_HCC81.genes.fpkm_table.log2.txt.gz"),
]

# Server-held read-only copies (hash only, never written).
SHARED = [
    ("LIMORE", "/mnt/S1/sdata/pubdata/LIMORE/drug.txt"),
    ("LIMORE", "/mnt/S1/sdata/pubdata/LIMORE/HCC81.proteinGenes.fpkm_table.log2.txt"),
    ("LIMORE", "/mnt/S1/sdata/pubdata/LIMORE/81clinic_information.txt"),
    ("MSigDB", "/mnt/S1/sdata/pubdata/Virtual_Cell/MSigDB/2026.1.Hs/h.all.v2026.1.Hs.symbols.gmt"),
    ("DepMap", "/mnt/S1/sdata/pubdata/DepMap/25Q3/Model.csv"),
]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: str) -> str:
    if os.path.exists(dest):
        return "already_present"
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=300) as resp, open(tmp, "wb") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    os.replace(tmp, dest)
    return "downloaded"


def main() -> None:
    assets, manifest = sys.argv[1], sys.argv[2]
    raw = os.path.join(assets, "raw")
    os.makedirs(raw, exist_ok=True)
    rows = []
    with open(os.path.join(raw, ".fetch.lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for study, name, url in DOWNLOADS:
            d = os.path.join(raw, study)
            os.makedirs(d, exist_ok=True)
            dest = os.path.join(d, name)
            now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            try:
                status = fetch(url, dest)
                rows.append((study, name, url, status, now, os.path.getsize(dest), sha256(dest),
                             f"assets/raw/{study}/{name}"))
            except Exception as exc:  # recorded, not hidden
                rows.append((study, name, url, f"FAILED:{type(exc).__name__}:{exc}", now, "", "", ""))
    for study, path in SHARED:
        now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if os.path.exists(path):
            rows.append((study, os.path.basename(path), "server_shared_readonly", "hashed_in_place",
                         now, os.path.getsize(path), sha256(path), path))
        else:
            rows.append((study, os.path.basename(path), "server_shared_readonly", "MISSING", now, "", "", path))
    # Keep the first-download date of files recorded earlier.
    old = {}
    if os.path.exists(manifest):
        with open(manifest) as fh:
            next(fh)
            for line in fh:
                p = line.rstrip("\n").split("\t")
                old[(p[0], p[1])] = p
    with open(manifest + ".tmp", "w") as fh:
        fh.write("study\tfile\turl\tstatus\tutc_time\tbytes\tsha256\tlocation\n")
        for r in rows:
            key = (r[0], r[1])
            if r[3] == "already_present" and key in old and old[key][6] == r[6]:
                r = tuple(old[key])
            fh.write("\t".join(str(x) for x in r) + "\n")
    os.replace(manifest + ".tmp", manifest)
    for r in rows:
        print(*r[:4], r[5], sep="\t")


if __name__ == "__main__":
    main()
