"""Measure metadata-only Sci-Plex3 coverage of the Tahoe compound splits."""

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

from src.moe_drug_v1.prepare import canonical_parent
from .acquire import checksums


def table(path, compressed=False):
    reader = gzip.open if compressed else Path.open
    with reader(path, "rt") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def parent(smiles):
    result = canonical_parent(smiles) if smiles not in {"NA", ""} else None
    return result[1] if result else None


def diagnose(args):
    catalog = {r["drug"]: parent(r["canonical_smiles"]) for r in table(args.tahoe)}
    sciplex = defaultdict(set)
    for row in table(args.sciplex, compressed=True):
        if row["time_point"] == "24" and row["vehicle"] == "FALSE":
            sciplex[row["catalog_number"]].add(row["SMILES"])
    if any(len(v) != 1 for v in sciplex.values()):
        raise ValueError("A Sci-Plex catalog entry has conflicting published SMILES")
    source = {key: parent(next(iter(values))) for key, values in sciplex.items()}
    observed_tahoe = set(catalog.values()) - {None}
    additional = sorted({v for v in source.values() if v} - observed_tahoe)
    split = table(args.split)
    query = defaultdict(lambda: defaultdict(set))
    for row in split:
        if row["usage"] == "outer_query" and row["role"] == "A":
            name = row["compound_entity_id"]
            identity = catalog[name]
            if identity:
                query[int(row["outer_fold"])][row["compound_group_id"]].add(identity)
    if set(query) != {0, 1, 2}:
        raise ValueError("Expected the three historical folds")

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)
    fingerprint = {p: generator.GetFingerprint(Chem.MolFromSmiles(p)) for p in observed_tahoe | set(additional)}
    supplemental = [fingerprint[p] for p in additional]
    folds = []
    for fold, rows in sorted(query.items()):
        training = {p for key, groups in query.items() if key != fold for values in groups.values() for p in values}
        validation = set().union(*rows.values())
        local = [fingerprint[p] for p in sorted(training - validation)]
        before = [max(max(DataStructs.BulkTanimotoSimilarity(fingerprint[p], local)) for p in v) for v in rows.values()]
        other = [max(max(DataStructs.BulkTanimotoSimilarity(fingerprint[p], supplemental)) for p in v) for v in rows.values()]
        after = [max(a, b) for a, b in zip(before, other)]
        folds.append(dict(fold=fold, mapped_groups=len(rows), nearer_with_sciplex=sum(b > a for a, b in zip(before, after)),
                          below_025_before=sum(x < 0.25 for x in before), below_025_after=sum(x < 0.25 for x in after),
                          at_least_050_before=sum(x >= 0.5 for x in before), at_least_050_after=sum(x >= 0.5 for x in after)))
    return dict(catalog_names=len(catalog), sciplex3_catalog_compounds=len(source),
                sciplex3_mapped_compounds=sum(bool(p) for p in source.values()),
                sciplex3_exact_parent_matching_tahoe=sum(p in observed_tahoe for p in source.values()),
                sciplex3_distinct_nonmatching_parents=len(additional), fold_preview=folds,
                input_sha256={p.name: checksums(p)[0] for p in (args.tahoe, args.sciplex, args.split)},
                rdkit_version=rdBase.rdkitVersion, treated_expression_read=False,
                scope="Parent-structure screening, before full name/CAS/CID/tautomer alias closure and expression QC.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("tahoe", "sciplex", "split", "out"):
        parser.add_argument(f"--{flag}", required=True, type=Path)
    args = parser.parse_args()
    result = diagnose(args)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
