#!/usr/bin/env python3
"""Fetch the Cellosaurus fields used to decide diagnosis and family for Track A cell models.

Public reference data only. Queries contain nothing but Cellosaurus accessions listed in
configs/pathway_sensitivity_a1/model_aliases.tsv. Parents (HI) are followed so that derived lines
can be grouped; same-individual links (OI) are recorded.

Usage: fetch_cellosaurus.py <model_aliases.tsv> <out.tsv>
"""
from __future__ import annotations

import csv
import json
import sys
import time
import urllib.request

API = "https://api.cellosaurus.org"
FIELDS = "id,ac,sy,di,hi,oi,ag,sx,ca,cc"
UA = "Mozilla/5.0 (compatible; research-data-fetch/1.0)"


def get(ac: str) -> dict:
    url = f"{API}/cell-line/{ac}?format=txt&fields={FIELDS}"
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=60) as r:
        txt = r.read().decode()
    rec = dict(ac=ac, name="", synonyms="", diseases=[], age="", sex="", category="", parent="",
               same_individual=[], problematic=[], caution=[])
    for line in txt.splitlines():
        tag, _, val = line.partition("   ")
        if tag == "ID":
            rec["name"] = val
        elif tag == "SY":
            rec["synonyms"] = val
        elif tag == "DI" and val.startswith("NCIt"):
            rec["diseases"].append(val.split("; ", 1)[1])
        elif tag == "AG":
            rec["age"] = val
        elif tag == "SX":
            rec["sex"] = val
        elif tag == "CA":
            rec["category"] = val
        elif tag == "HI":
            rec["parent"] = val.split(" ! ")[0]
        elif tag == "OI":
            rec["same_individual"].append(val.split(" ! ")[0])
        elif tag == "CC" and val.startswith("Problematic cell line:"):
            rec["problematic"].append(val[len("Problematic cell line: "):])
        elif tag == "CC" and val.startswith("Caution:"):
            rec["caution"].append(val[len("Caution: "):])
    return rec


def main() -> None:
    aliases, out = sys.argv[1], sys.argv[2]
    acs = []
    with open(aliases) as fh:
        for row in csv.DictReader((l for l in fh if not l.startswith("#")), delimiter="\t"):
            if row["cellosaurus_id"] and row["cellosaurus_id"] not in acs:
                acs.append(row["cellosaurus_id"])
    req = urllib.request.Request(f"{API}/release-info?format=json", headers={"User-Agent": UA})
    rel = json.load(urllib.request.urlopen(req, timeout=60))["Cellosaurus"]["header"]["release"]
    seen: dict[str, dict] = {}
    todo = list(acs)
    while todo:
        ac = todo.pop(0)
        if ac in seen:
            continue
        seen[ac] = get(ac)
        time.sleep(0.25)
        if seen[ac]["parent"]:
            todo.append(seen[ac]["parent"])

    def root(ac: str) -> str:
        while seen[ac]["parent"]:
            ac = seen[ac]["parent"]
        return ac

    with open(out, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["cellosaurus_id", "name", "synonyms", "ncit_diseases", "age", "sex", "category",
                    "parent", "root_ancestor", "same_individual", "problematic_note", "caution_note",
                    "cellosaurus_release", "cellosaurus_updated"])
        for ac in sorted(seen):
            r = seen[ac]
            w.writerow([ac, r["name"], r["synonyms"], " | ".join(r["diseases"]), r["age"], r["sex"],
                        r["category"], r["parent"], root(ac), ",".join(sorted(r["same_individual"])),
                        " | ".join(r["problematic"]), " | ".join(r["caution"]), rel["version"],
                        rel["updated"]])
    print(f"wrote {len(seen)} records, Cellosaurus {rel['version']} ({rel['updated']})")


if __name__ == "__main__":
    main()
