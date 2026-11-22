"""Find exact candidate HCC drug support using structure and roster metadata only."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .expanded_panel_cv import RECEIPTS, registered_inputs
from .io import publish_json, sha256
from .pdo_transfer_cv import inputs, PDO_HASH

GDSC_RECEIPT = "7fc3f92b576273f71a8ef41df83353471d65d98d6d53d4b9f8f6a3ddb9acf20b"


def read_registered(path, checksum, columns, sep=None):
    if sha256(path) != checksum:
        raise ValueError(f"Expected locked structure metadata: {path.name}")
    if sep is not None:
        return pd.read_csv(path, sep=sep, usecols=columns)
    return pd.read_parquet(path, columns=columns)


def coverage(panel, previously_scored, base, augmented, with_gdsc, donors):
    exact = set(panel.compound_id.dropna())
    if not set(previously_scored) <= exact:
        raise ValueError("An already scored compound was not in the registered panel")
    new = (exact & augmented) - previously_scored
    later = (exact & with_gdsc) - augmented - previously_scored
    data = {
        "registered_catalogue_drugs": len(panel), "exact_structures": len(exact),
        "previously_scored_drugs": len(previously_scored), "registered_hcc_model_count": donors,
        "match_with_updated_core": len(exact & base),
        "extra_unscored_supported_by_1129_panel": sorted(new),
        "extra_unscored_supported_by_gdsc_candidate": sorted(later),
    }
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("root", "out"):
        parser.add_argument("--" + field, type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    runs = root / "v0plus-runs"
    args.original = root / "function/cross-v1"
    args.core = runs / "source-core-92aa482/core"
    args.extension = runs / "source-panel-c351b0b/extension"
    _, core, _, extra = registered_inputs(args, include_extension=True)
    pdo, _ = inputs(runs / "improve-pdo-347009e/development")
    gdsc = runs / "improve-gdsc-stereo-347009e/extension"
    if sha256(gdsc / "receipt.json") != GDSC_RECEIPT:
        raise ValueError("GDSC source candidate registry changed")
    g_art = json.loads((gdsc / "receipt.json").read_text())["artifacts"]
    add_gdsc = read_registered(gdsc / "additional_rows.parquet", g_art["additional_rows.parquet"], ["compound_id"])
    base = set(core.compound_id)
    all2d = base | set(extra.compound_id)
    with_gdsc = all2d | set(add_gdsc.compound_id)
    source_metadata = {}
    tables = {}
    for tag, directory, flag in (
        ("LICOB", "licob-author-2f1328b", "isomeric_smiles"),
        ("Wong", "wong-84d021d", "in_training"),
    ):
        cohort = runs / directory / "cohort"
        receipt = json.loads((cohort / "receipt.json").read_text())
        digest = receipt["files"]["chemistry.parquet"]
        panel = read_registered(cohort / "chemistry.parquet", digest, ["compound_id", flag])
        included = panel[flag].notna() if flag == "isomeric_smiles" else panel[flag]
        min_donors = 29 if tag == "LICOB" else int(receipt["lineages"])
        tables[tag] = coverage(panel, set(panel.loc[included, "compound_id"]), base, all2d, with_gdsc, min_donors)
        source_metadata[tag] = digest
    limore = root / "function/limore-v1"
    receipt = json.loads((limore / "receipt.json").read_text())
    digest = receipt["artifacts"]["compound_registry.tsv"]["sha256"]
    panel = read_registered(limore / "compound_registry.tsv", digest, ["compound_id"], sep="\t")
    cross = json.loads((args.original / "receipt.json").read_text())
    scored = read_registered(args.original / "test_inputs/rows.parquet",
                             cross["artifacts"]["test_inputs/rows.parquet"]["sha256"], ["compound_id", "cell_id"])
    source_metadata["LIMORE"] = digest
    tables["LIMORE"] = coverage(panel, set(scored.compound_id), base, all2d, with_gdsc, scored.cell_id.nunique())
    for support in tables.values():
        new = set(support["extra_unscored_supported_by_1129_panel"])
        future = set(support["extra_unscored_supported_by_gdsc_candidate"])
        support["unscored_compounds_with_non_hcc_pdo_anchor"] = sorted((new | future) & set(pdo.compound_id))
    publish_json(args.out, {"cohorts": tables, "registry_hashes": source_metadata, "source_receipts": RECEIPTS,
                            "pdo_receipt": PDO_HASH, "gdsc_candidate_receipt": GDSC_RECEIPT})


if __name__ == "__main__":
    main()
