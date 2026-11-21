"""Reconcile three DEV I-track feature views and joint parent/fold coverage."""

import argparse
from collections import Counter
from itertools import combinations
from pathlib import Path

from src.r11_chemical_probe.common import sha256
from src.r15_lincs.data import table
from src.r19_plan.i_audit import CAP_BYTES, audit_views, checked_sources, write_receipt
from src.r19_plan.r18_preflight import checked_tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    source = checked_sources(root / "data")
    meta = checked_tables(root / "population", root / "population/receipt.json")
    pop = table(root / "population/rows.tsv")
    sets, receipts = {}, {}
    for name in ("I1", "I2", "I4"):
        folder = root / name
        checked_tables(folder, folder / "receipt.json")
        available = table(folder / "parents.tsv")
        if available.parent.duplicated().any():
            raise ValueError(f"{name} reports duplicate parent identities")
        sets[name] = set(available.parent)
        if not sets[name] <= set(pop.parent):
            raise ValueError(f"{name} reports parents outside the locked LINCS population")
        receipts[name] = sha256(folder / "receipt.json")
    joint = {"&".join(pair): set.intersection(*(sets[item] for item in pair))
             for pair in combinations(sets, 2)}
    joint["I1&I2&I4"] = set.intersection(*sets.values())
    counter = Counter(parent for view in sets.values() for parent in view)
    joint["at_least_two_views"] = {parent for parent, count in counter.items() if count >= 2}
    support = audit_views(pop, sets | joint)
    parents = pop[["parent", "outer_fold", "r18_identity_overlap"]].drop_duplicates()
    for name, present in sets.items():
        parents[name] = parents.parent.isin(present)

    total = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    if total >= CAP_BYTES:
        raise ValueError("I-track storage exceeds the approved cap")
    args.out.mkdir(exist_ok=False)
    support.to_csv(args.out / "modality_coverage.tsv", sep="\t", index=False)
    parents.sort_values("parent").to_csv(args.out / "parent_availability.tsv", sep="\t", index=False)
    write_receipt(args.out, dict(
        population_sha256=sha256(root / "population/receipt.json"), excluded_R18_assay_wells=meta["excluded_assay_wells"],
        feature_view_receipts=receipts, total_new_server_file_bytes=total,
        public_file_bytes=sum(entry["size"] for entry in source), expression_values_read=False,
    ))
    print(support.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
