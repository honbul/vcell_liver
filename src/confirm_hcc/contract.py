"""Verified corrected source and the detached LIMORE evaluation contract."""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold

from vc_build.external import KEY, SOURCE
from vc_build.io import sha256
from vc_build.limore_corrected import OBSERVATIONS, registered
from vc_build.v0plus_cv import training_inputs

BASE = Path("/mnt/S1/sdata/workbench/Virtual_Cell/gssong")
BUILD = BASE / "vc_build_v1"
RUNS = BUILD / "v0plus-runs"
CORE = RUNS / "source-core-92aa482/core"
ORIGINAL = BUILD / "function/cross-v1"
NEXTGEN = BASE / "nextgen_5track"
OLD_V2 = BASE / "top3_tuning_v2/a/runs"
FM = NEXTGEN / "fm_store_v1"
CMAP = BASE / "top3_tuning_v2/a/input/cmap24-01"
HALLMARK = Path("/mnt/S1/sdata/pubdata/Virtual_Cell/MSigDB/2026.1.Hs/h.all.v2026.1.Hs.symbols.gmt")
TARGETS = BUILD / "external-code-v3/drug_targets.json"
SETTINGS = RUNS / "final-8e5d3f4/settings.json"
TRAIN_SHA = "1edd32ef38be60a742efe0dcc376e15910623c1da8cc27d923291bee8a4a38cc"
RNA_SHA = "ced61eceb5aac91defcd592819d01a67ef5d6ed89c3ac76cada348f8d319588e"
OLD_SETTINGS = {
    "round1-02": (
        "6002d05e25ef6a220d6801735988167bc5823b35ab3e95a0f953bcbfda5cfe2c",
        "b5fca4b9a4651bf45a6cfeba8dc4f138425e5abdfb277e3d651160be816f71dd",
        "9d4016880d50518b5cb2790cac23537bf9b9fbf21864abc0c4238b7c7bb20c52",
    ),
    "round2-02": (
        "99dd2f49918f95d79edb3586310ef2c61689577381931fa90196b0a7120f03df",
        "3d93a0cec26fa4a3ed6c91e426652bbf78e98265b21c2ecadfb9470973fb19d4",
        "996a652ff509836f1c1b2cfbdc813227d9dd37953edb1f595d812cc4b7a3a25e",
    ),
    "round4-01": (
        "fbbce28452290dff8afd519c34910710137557087abdbf255438e25f0bf177bd",
        "e56748b65560d387aa40c033d85ab7f309482b57d00f7c3df23ad805d5aab59b",
        "d94448bb5d6fe816c2a53afcc27b50d31bbef7cec4294df0afd5d0a43fd683b8",
    ),
}
TRACK = "LIMORE50"
EXPOSURE = "previously_exposed_retrospective_locked_comparison"
MODELS = ("drug_mean", "additive", "drug_ridge", "cellhit", "structured",
          "ridge_structured_50_50", "pairwise_rank", "pairwise_top5", "ecfp4_path_elasticnet")


def verified(path, digest):
    if sha256(path) != digest:
        raise ValueError(f"Registered hash mismatch: {path}")
    return path


def read_json(path):
    return json.loads(path.read_text())


def corrected():
    verified(CORE / "train/rows.parquet", TRAIN_SHA)
    verified(CORE / "train/rna.parquet", RNA_SHA)
    rows, rna, receipt = training_inputs(CORE)
    if (len(rows), rows.cell_id.nunique(), rows.lineage_group.nunique(),
        rows.compound_id.nunique()) != (65678, 1014, 991, 68):
        raise ValueError("Corrected training roster changed")
    if (set(rows.source_id) != {"GDSC2", "CTRPv2", "PRISM20Q2"}
        or rows.row_id.duplicated().any()
        or rows.duplicated(["lineage_group", "cell_id", "compound_id", *SOURCE]).any()):
        raise ValueError("Corrected native source keys changed")
    return rows, rna.sort_index(axis=1), receipt


def inputs():
    from types import SimpleNamespace

    train, _, _ = corrected()
    _, query, rna = registered(SimpleNamespace(core=CORE, original=ORIGINAL), "corrected")
    rna = rna.sort_index(axis=1)
    fit_rna, query_rna = rna.loc[rna.index.isin(train.cell_id)], rna.loc[rna.index.isin(query.cell_id)]
    if (query.cell_id.nunique() != 50 or query.duplicated(["lineage_group", "compound_id"]).any()
        or set(query.lineage_group) & set(train.lineage_group)):
        raise ValueError("LIMORE must be adult HCC with 50 independent complete cases")
    return train, fit_rna, query, query_rna


def fold_indices(clean):
    # Preserve the exact V2 outer groups associated with its pre-existing hyperparameters.
    # Read only historical group keys. No historical assay outcomes enter any fit.
    root = ORIGINAL / "train/rows.parquet"
    verified(root, read_json(ORIGINAL / "receipt.json")["artifacts"]["train/rows.parquet"]["sha256"])
    old = pd.read_parquet(root, columns=["cell_group"])
    if len(old) != 65685 or set(old.cell_group) != set(clean.lineage_group):
        raise ValueError("Cannot recover the three historical setting-specific lineage folds")
    for _, indices in GroupKFold(3).split(old, groups=old.cell_group):
        held = clean.lineage_group.isin(set(old.cell_group.iloc[indices]))
        yield clean.loc[~held], clean.loc[held]


def v2_settings(fold):
    result = {}
    for name, hashes in OLD_SETTINGS.items():
        source = OLD_V2 / name / f"fold{fold}/receipt.json"
        result[name] = read_json(verified(source, hashes[fold]))
    return result


def native_keys(frame):
    return frame[[*KEY, *SOURCE]].copy()
