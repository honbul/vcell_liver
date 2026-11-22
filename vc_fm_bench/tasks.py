"""Register evaluation panels and materialize separate training and query roles."""
import argparse
from pathlib import Path
import json

import pandas as pd

from .artifacts import file_record, new_directory, verify_record, write_json


def freeze_panel(root, name, train, query, labels, task, split_group, rna_panel=None):
    categorical = task == "classification"
    if set(train["id"]) & set(query["id"]) or not len(train) or not len(query):
        raise ValueError(f"Empty role or shared observations in {name}")
    if split_group and set(train[split_group]) & set(query[split_group]):
        raise ValueError(f"Cold group overlap: {name}")
    if split_group in {"scaffold", "connectivity"} and set(train["connectivity"]) & set(query["connectivity"]):
        raise ValueError(f"Chemical connectivity overlap in {name}")
    if "outcome" in query or (labels is not None and set(query.id) != set(labels.id)):
        raise ValueError(f"Query/outcome role mismatch: {name}")
    if categorical and train.outcome.nunique() != 2:
        raise ValueError(f"Missing training class: {name}")
    with new_directory(root / name) as stage:
        train.to_parquet(stage / "train.parquet", index=False)
        query.to_parquet(stage / "query.parquet", index=False)
        if labels is not None:
            labels.to_parquet(stage / "outcomes.parquet", index=False)
        write_json(stage / "receipt.json", {
            "task": task, "split_group": split_group, "rna_panel": rna_panel,
            "train_rows": len(train), "query_rows": len(query),
            "train_groups": train[split_group].nunique() if split_group else None,
            "query_groups": query[split_group].nunique() if split_group else None,
            "train_compounds": train.drug_id.nunique(), "query_compounds": query.drug_id.nunique(),
            "files": {p.name: file_record(p) for p in sorted(stage.iterdir())},
        })


def held_out_by_scaffold(root, name, data, folds, task, rna_panel=None):
    membership = data.scaffold.map(folds)
    if membership.isna().any():
        raise ValueError(f"Unregistered scaffold in {name}")
    for fold in sorted(membership.unique()):
        captured = data.loc[membership == fold]
        train = data.loc[membership != fold]
        freeze_panel(root, f"{name}-{fold}", train, captured.drop(columns="outcome"),
                     captured[["id", "outcome"]], task, "scaffold", rna_panel)


def class_labels(data, label):
    known = data.loc[data[label].isin(("Most", "Less", "No"))].copy()
    # Release-increment detects either class of known concern. Exclude ambiguous
    # entries and chemically identical parents whose binary labels contradict.
    known["outcome"] = known[label].isin(("Most", "Less")).astype(int)
    conflicting = known.groupby("drug_id").outcome.transform("nunique").ne(1)
    return known.loc[~conflicting].drop_duplicates("drug_id")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    option_names = ["pan_train", "limore_cell_cold", "limore_drug_cold", "limore_drug_labels",
                    "dili_inputs", "dili_labels"]
    original = json.loads((a.input / "receipt.json").read_text())
    sources = {}
    for key in option_names:
        path = a.input / f"{key}.parquet"
        verify_record(path, original["files"][path.name])
        sources[key] = pd.read_parquet(path)
    path = a.input / "split_groups.json"
    verify_record(path, original["files"][path.name])
    registry = json.loads(path.read_text())
    pan = sources["pan_train"]
    # All source AUCs are lower-is-sensitive, including the test folds from GDSC.
    # Rank/calibration is performed inside fitting, using each training role only.
    pan["outcome"] = 1 - pan["outcome"]
    limore = sources["limore_drug_cold"].merge(
        sources["limore_drug_labels"], on="id", validate="one_to_one",
    )
    gdsc = pan.loc[pan.source_id.str.contains("gdsc", case=False)].copy()
    dili_full = sources["dili_inputs"].merge(sources["dili_labels"], on="source_drug", validate="one_to_one")
    dili_full = dili_full.rename(columns={"source_drug": "id"})
    dili_full = dili_full[["id", "drug_id", "connectivity", "scaffold", "release_increment",
                          "concern", "v1_concern"]]
    dili = class_labels(dili_full, "concern").drop(columns=["concern", "v1_concern"])
    historic = class_labels(dili_full.loc[~dili_full.release_increment], "v1_concern")
    new = class_labels(dili_full.loc[dili_full.release_increment], "concern")
    historic = historic.loc[~historic.connectivity.isin(new.connectivity)]
    historic, new = [x.drop(columns=["concern", "v1_concern"]) for x in (historic, new)]
    with new_directory(a.output) as stage:
        freeze_panel(stage, "limore-cell-cold", pan, sources["limore_cell_cold"],
                     None, "regression", "cell_group", "cross")
        held_out_by_scaffold(stage, "limore-drug-cold", limore, registry["limore_scaffold"],
                             "regression", "limore")
        held_out_by_scaffold(stage, "gdsc2-drug-cold", gdsc, registry["gdsc2_scaffold"],
                             "regression", "cross")
        held_out_by_scaffold(stage, "dili-scaffold", dili, registry["dili_scaffold"], "classification")
        freeze_panel(stage, "dili-release-increment", historic, new.drop(columns="outcome"),
                     new[["id", "outcome"]], "classification", "connectivity")
        write_json(stage / "receipt.json", {
            "prepared_receipt": file_record(a.input / "receipt.json"),
            "fda_raw_resolved": len(dili_full), "fda_v2_eligible_unique_parents": len(dili),
            "licob": "Unusable publisher RNA checksum; LIMORE cell-cold is the external fallback",
            "pan_audit_train_sources": {str(k): int(v) for k, v in pan.source_id.value_counts().items()},
            "pan_has_external_labels": False,
            "panel_receipts": {p.name: file_record(p / "receipt.json") for p in sorted(stage.iterdir())},
        })


if __name__ == "__main__":
    main()
