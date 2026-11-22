"""Reconcile R17 support, selection and scores against its pinned receipts on CPU."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r15_lincs.data import genes
from src.r17_lincs import score, vehicles

CACHED_STUDIES = ("GSE92742", "GSE70138")
SHARED = ["support.tsv", "fold_support.tsv", "plate_support.tsv", "assignments.tsv"]


def matches(source, replay):
    if sha256(source) != sha256(replay):
        raise ValueError(f"R17 reconciliation mismatch: {source} vs {replay}")
    return sha256(source)


def main(args):
    args.out.mkdir(parents=True, exist_ok=False)
    role_out = args.out / "support"
    vehicles.register(SimpleNamespace(out=role_out, cache=args.cache, lincs=args.lincs))
    role_receipt = read_json(args.original / "support/receipt.json")
    for name in SHARED:
        original = args.original / "support" / name
        if sha256(original) != role_receipt["files"][name]:
            raise ValueError(f"Changed original R17 role registry: {name}")
    role_file_hashes = {name: matches(args.original / "support" / name, role_out / name) for name in SHARED}
    registered = pd.read_csv(role_out / "assignments.tsv", sep="\t")
    actual = genes(args.lincs, CACHED_STUDIES[0])
    axes = {}
    for study in CACHED_STUDIES:
        registered_axis = vehicles.checked(args.cache / study / "landmarks.tsv")
        cached = pd.read_csv(registered_axis, sep="\t")
        if not actual.equals(cached) or not cached.equals(genes(args.lincs, study)):
            raise ValueError(f"R17 landmark identities or order changed: {study}")
        axes[study] = sha256(registered_axis)

    for name in ("predict", "freeze.json"):
        (args.out / name).symlink_to(args.original / name)
    receipt = read_json(args.reference / "score/receipt.json")
    if sha256(args.original / "freeze.json") != receipt["freeze_sha256"]:
        raise ValueError("Original R17 freeze differs from repository receipt")
    score.run(SimpleNamespace(action="score", out=args.out, protocol=args.protocol, r15_protocol=args.r15_protocol,
                              cache=args.cache, vehicles=args.original / "vehicles"))
    score_hashes = {}
    for name in ("points.tsv", "paired.tsv", "selection.tsv", "receipt.json"):
        digest = matches(args.reference / "score" / name, args.out / "score" / name)
        score_hashes[name] = matches(args.original / "score" / name, args.out / "score" / name)
        if score_hashes[name] != digest:
            raise ValueError(f"Unexpected score copy digest: {name}")
    if registered.duplicated(["rna_plate", "rna_well"]).any():
        raise ValueError("Duplicate R17 physical control coordinate")
    support = pd.read_csv(role_out / "support.tsv", sep="\t")
    write_json(args.out / "reconciliation.json", {
        "lane": "D0_CPU_RECONCILIATION", "model_fit": False,
        "freeze_sha256": receipt["freeze_sha256"],
        "role_files_sha256": role_file_hashes, "score_files_sha256": score_hashes,
        "gene_axis_sha256": axes, "landmarks": len(actual), "physical_control_wells": len(registered),
        "treatment_support": support.to_dict("records"),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original", "reference", "out", "cache", "lincs", "protocol", "r15-protocol"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    main(parser.parse_args())
