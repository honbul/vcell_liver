"""Fetch the Cellosaurus fields the audit needs for a fixed list of accessions.

Public reference data only: accession, name, disease, age, category, parent (HI), same-individual
links (OI) and the 'Problematic cell line' comment. Parents are followed to the root so that derived
clones can be grouped. The query contains nothing but Cellosaurus accessions.
"""
from __future__ import annotations

import csv
import json
import sys
import time
import urllib.request

API = "https://api.cellosaurus.org"
FIELDS = "id,ac,di,hi,oi,ag,sx,ca,cc"


def get(ac: str) -> dict:
    url = f"{API}/cell-line/{ac}?format=txt&fields={FIELDS}"
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "audit-script"}),
                                timeout=60) as r:
        txt = r.read().decode()
    rec = dict(ac=ac, name="", diseases=[], age="", sex="", category="", parent="", same_individual=[],
               problematic="")
    for line in txt.splitlines():
        tag, _, val = line.partition("   ")
        if tag == "ID":
            rec["name"] = val
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
            rec["problematic"] = val[len("Problematic cell line: "):]
    return rec


def main() -> None:
    acs, out = sys.argv[1].split(","), sys.argv[2]
    rel = json.load(urllib.request.urlopen(f"{API}/release-info?format=json", timeout=60))
    rel = rel["Cellosaurus"]["header"]["release"]
    seen: dict[str, dict] = {}
    todo = list(acs)
    while todo:
        ac = todo.pop(0)
        if ac in seen:
            continue
        seen[ac] = get(ac)
        time.sleep(0.3)
        if seen[ac]["parent"]:
            todo.append(seen[ac]["parent"])

    def root(ac: str) -> str:
        while seen[ac]["parent"]:
            ac = seen[ac]["parent"]
        return ac

    with open(out, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["cellosaurus_id", "name", "ncit_diseases", "age", "sex", "category", "parent",
                    "root_ancestor", "same_individual", "problematic_note", "cellosaurus_release",
                    "cellosaurus_updated"])
        for ac in sorted(seen):
            r = seen[ac]
            w.writerow([ac, r["name"], " | ".join(r["diseases"]), r["age"], r["sex"], r["category"],
                        r["parent"], root(ac), ",".join(sorted(r["same_individual"])),
                        r["problematic"], rel["version"], rel["updated"]])
    print(f"wrote {len(seen)} records, Cellosaurus {rel['version']} ({rel['updated']})")


if __name__ == "__main__":
    main()
