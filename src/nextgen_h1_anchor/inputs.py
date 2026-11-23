"""Registered response sources and identity-only support audit."""
import argparse
from hashlib import sha256 as digest
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vc_build.io import sha256

BASE = Path("/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vc_build_v1")
RUNS = BASE / "v0plus-runs"
CORE = RUNS / "source-core-92aa482/core"
CROSS = BASE / "function/cross-v1"
COLD = RUNS / "licob-cold-5bf53fc/inputs"
WONG = RUNS / "wong-expanded-15b1345/inputs"
EXTENSIONS = [RUNS / "source-panel-c351b0b/extension", RUNS / "improve-gdsc-stereo-347009e/extension"]
RESTORED = RUNS / "prism-parent-75c35e8/extension"
LICOB_LABEL = BASE / "licob-v0plus/NIHMS1969891-supplement-Supplementary_Tables_S1-S13.xlsx"
LIMORE_LABEL = CROSS / "test_labels/outcomes.parquet"
PAN = ("GDSC2", "CTRPv2", "PRISM20Q2")
DRUG_META = ["compound_id", "source_compound_label", "isomeric_smiles", "drug_target_source_labels"]
TISSUE = "tissue_origin"
NAMES = (*PAN, "LIMORE", "LICOB")
DENOM = {"GDSC2": 1., "CTRPv2": 24., "PRISM20Q2": 1., "LICOB": 1.}


def json_write(path, payload):
    path = Path(path)
    with path.open("x") as file:
        json.dump(payload, file, sort_keys=True, indent=2, allow_nan=False)
        file.write("\n")


def source_registry():
    return [
        (CORE, "train/rows.parquet", "cell_group"),
        *((root, "additional_rows.parquet", "lineage_group") for root in EXTENSIONS),
    ]


def source_rows(labels=False):
    result = []
    for root, rel, group_key in source_registry():
        columns = [*DRUG_META, "source_id", "cell_id", group_key, TISSUE, "endpoint_id"]
        if labels:
            columns += ["endpoint_value", "endpoint_direction"]
        result.append(pd.read_parquet(root / rel, columns=columns).rename(columns={group_key: "lineage_group"}))
    current = pd.concat(result, ignore_index=True)
    cols = [*DRUG_META, "source_id", "cell_id", "lineage_group", TISSUE, "endpoint_id"]
    if labels:
        cols += ["endpoint_value", "endpoint_direction"]
    additional = pd.read_parquet(RESTORED / "additional_rows.parquet", columns=cols)
    registered = set(current.loc[current.source_id.eq("PRISM20Q2"), "compound_id"])
    additional = additional[additional.source_id.eq("PRISM20Q2") & ~additional.compound_id.isin(registered)]
    return pd.concat([current, additional], ignore_index=True)


def rosters():
    limore = pd.read_parquet(CROSS / "test_inputs/rows.parquet")
    limore = limore.rename(columns={"cell_group": "lineage_group", "record_id": "row_id"})
    licob = pd.read_parquet(COLD / "query.parquet")
    licob = licob[licob.adult_age_confirmed].assign(lineage_group=lambda x: x.donor_id, tissue_origin="Liver")
    wong = pd.read_parquet(WONG / "query.parquet")
    return limore, licob, wong


def fold_order(groups, salt="904877"):
    return sorted(set(groups), key=lambda g: digest(f"{salt}:{g}".encode()).hexdigest())


def density(frame):
    groups = frame[["lineage_group", "compound_id"]].drop_duplicates()
    return groups.groupby("compound_id").size() / groups.lineage_group.nunique()


def metadata(frame):
    rows = []
    for compound, block in frame[DRUG_META].drop_duplicates().groupby("compound_id"):
        # Prefer the most detailed public annotation with a deterministic lexical tiebreak.
        labels = sorted(set(block.source_compound_label.dropna().astype(str)), key=lambda s: (len(s), s))
        smiles = sorted(set(block.isomeric_smiles.dropna().astype(str)))
        mechanisms = sorted(set(block.drug_target_source_labels.dropna().astype(str)))
        rows.append((compound, labels[0] if labels else compound, smiles[0] if smiles else "",
                     ";".join(mechanisms)))
    return pd.DataFrame(rows, columns=DRUG_META).set_index("compound_id")


def verify_artifact(root, rel):
    info = json.loads((root / "receipt.json").read_text())
    entry = info.get("artifacts", info.get("files", {})).get(rel)
    expected = entry["sha256"] if isinstance(entry, dict) else entry
    value = sha256(root / rel)
    if expected is None or value != expected:
        raise ValueError(f"Registered SHA-256 mismatch: {root / rel}")
    return value


def audit(out):
    sources = source_rows()
    limore, licob, wong = rosters()
    sets = {name: block for name, block in sources.groupby("source_id", sort=True)}
    sets.update(LIMORE=limore, LICOB=licob, Wong=wong)
    coverage = {name: density(frame) for name, frame in sets.items()}
    panels = {}
    for threshold in (.95, .90, .85, .80, .70, .60, .50, .25, 0.):
        common = set.intersection(*(set(coverage[s][coverage[s] >= threshold].index) for s in PAN))
        panels[str(threshold)] = sorted(common)
    core = set(sources.iloc[:65678].compound_id)
    support = {}
    for name, frame in sets.items():
        cover = coverage[name]
        if name in PAN:
            dense = [drug for drug in cover.index if cover[drug] >= .80]
            panel = sorted(sorted(dense, key=lambda d: (d not in core, -cover[d], d))[:68])
            if len(panel) < 40:
                raise ValueError(f"{name} has fewer than 40 drugs with 80% coverage")
        else:
            panel = sorted(cover.index)
        support[name] = {
            "registered_rows": len(frame), "cells": frame.cell_id.nunique(),
            "groups": frame.lineage_group.nunique(), "registered_drugs": len(cover),
            "panel": panel, "panel_missing_fraction": float(1 - cover.loc[panel].mean()),
            "panel_min_drug_coverage": float(cover.loc[panel].min()),
            "pan_tissue_groups": frame.drop_duplicates("lineage_group")[TISSUE].fillna("Unspecified").value_counts().to_dict()
            if name in PAN else {},
            "endpoints": sorted(map(str, frame.endpoint_id.unique())) if "endpoint_id" in frame else [],
        }
    pairs = {}
    for name in (*PAN, "LIMORE"):
        overlap = coverage[name][coverage[name] >= .80].index.intersection(coverage["LICOB"].index)
        pairs[name] = sorted(overlap)
    file_hashes = {}
    for root, rel in [(r, f) for r, f, _ in source_registry()] + [
        (CORE, "train/rna.parquet"), (CROSS, "test_inputs/rows.parquet"), (CROSS, "test_inputs/rna.parquet"),
        (CROSS, "test_labels/outcomes.parquet"), (COLD, "query.parquet"), (COLD, "rna.parquet"),
        (WONG, "query.parquet"), (WONG, "rna.parquet"), (RESTORED, "additional_rows.parquet"),
    ]:
        file_hashes[str(root / rel)] = verify_artifact(root, rel)
    from vc_build.v0plus_licob import ASSET_HASHES

    workbook = sha256(LICOB_LABEL)
    if workbook != ASSET_HASHES["supplement"]:
        raise ValueError("LICOB workbook SHA-256 mismatch")
    file_hashes[str(LICOB_LABEL)] = workbook
    payload = {
        "file_sha256": file_hashes, "cohorts": support, "pan_density_threshold": .80,
        "common_pan_n_by_density": {k: len(v) for k, v in panels.items()},
        "core_drugs": sorted(core),
        "pan_common_chemistry": panels,
        "wong_limore_panel": sorted(set(limore.compound_id) & set(wong.compound_id)),
        "licob_transfer_panels": pairs,
        "labels_decoded": False,
    }
    json_write(out, payload)
    print({name: (entry["groups"], len(entry["panel"]), entry["panel_missing_fraction"])
           for name, entry in support.items()})
    print({name: len(panel) for name, panel in pairs.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    audit(parser.parse_args().out)
