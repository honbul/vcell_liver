"""Verified H2 registrations and detached H3-v2 modeling partitions."""
from hashlib import sha256 as digest_bytes
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from vc_build.external import native_sensitivity
from vc_build.io import _publish, publish_json, sha256
from nextgen_h2.assets import COLD, STORAGE, KEY, LIMORE, parquet_lock
from nextgen_h2.run import META, read_json, subfold

ROOT = STORAGE / "nextgen_v2/h3_v2"
H2 = STORAGE / "nextgen_h123/h2_multiview"
ASSETS = H2 / "assets-v3"
TARGETS = H2 / "targets-v1"
STUDY = H2 / "study-v2"
V1 = STORAGE / "nextgen_h123/h3_pcgrad"
CORE = STORAGE / "vc_build_v1/v0plus-runs/source-core-92aa482/core/train"
FM = STORAGE / "nextgen_5track/fm_store_v1/chemistry-v1"
SCREENS = ("CTRPv2", "GDSC2", "PRISM20Q2")
SPLITS = [f"{track}-{fold}" for track in ("drug", "cell", *SCREENS) for fold in range(3)]
INPUT_SHA = {
    ASSETS / "data_receipt.json": "60602ebba5ecbc0e54e92e0a761f5e87fe3719760a5a1a2d086c4946b6b7a5ca",
    COLD / "receipt.json": "67c11aba490282e30be9ef6d3d4d28ad192a2f5620a9e0b84c978019a0ccdb9e",
    STUDY / "champion.json": "02d9c3e8cac5a1b6829e889286f12ad659fda3b8815735d9b4600c9bb30d7f91",
    V1 / "data/manifest.json": "c187394659dac68c12a3c246a9da3a39675a0c84dd57c364af594bc00222a914",
    CORE / "rna.parquet": "ced61eceb5aac91defcd592819d01a67ef5d6ed89c3ac76cada348f8d319588e",
}
AS_COLUMNS = ["endpoint_direction", "endpoint_value"]


def register():
    return parquet_lock(ASSETS / "folds.parquet", read_json(ASSETS / "data_receipt.json")["files"]["folds.parquet"])


def store(df, path):
    _publish(path, lambda p: df.to_parquet(p, index=False))


def table_hash(frame):
    keys = "\n".join(frame.row_id.astype(str))
    return digest_bytes(keys.encode()).hexdigest()


def lineage_folds(rows):
    groups = rows.lineage_group
    if groups.nunique() < 3:
        raise ValueError("Insufficient independent lineages")
    output = np.empty(len(rows), dtype=np.int8)
    for fold, (_, check) in enumerate(GroupKFold(3).split(groups, groups=groups)):
        output[check] = fold
    return output


def grouped_split(rows, mode, registry):
    if mode == "drug":
        return rows[KEY].map(registry.loc[rows[KEY].unique(), "scaffold_block"]).to_numpy()
    return rows.lineage_group.to_numpy()


def nested(rows, split, registry, inner):
    mode, idx = split.rsplit("-", 1)
    if mode == "drug":
        groups = rows[KEY].map(registry[f"inner{idx}"])
        if (groups < 0).any():
            raise ValueError("Outer held drug entered the inner fit")
        return rows.loc[groups.ne(inner)], rows.loc[groups.eq(inner)]
    groups = lineage_folds(rows)
    return rows.loc[groups != inner], rows.loc[groups == inner]


def detach(fit, check, query, group_name, registry):
    target = ROOT / "splits" / group_name
    mode = group_name.rsplit("-", 1)[0]
    overlap = set(grouped_split(fit, mode, registry)) & set(grouped_split(query, mode, registry))
    if overlap or fit.row_id.duplicated().any() or query.row_id.duplicated().any():
        raise ValueError(f"Leakage or duplicates in {group_name}")
    if mode in SCREENS and (mode in set(fit.source_id) or set(query.source_id) != {mode}):
        raise ValueError(f"Held screen leaked into {group_name}")
    store(fit, target / "fit.parquet")
    store(query.drop(columns=AS_COLUMNS, errors="ignore"), target / "query.parquet")
    labels = ROOT / "outcomes" / f"{group_name}.parquet"
    store(check, labels)
    return {
        "fit_rows": len(fit), "query_rows": len(query), "train_parents": fit[KEY].nunique(),
        "test_parents": query[KEY].nunique(), "fit_groups": len(set(grouped_split(fit, mode, registry))),
        "query_groups": len(set(grouped_split(query, mode, registry))), "blocked_group_intersection": len(overlap),
        "fit_id_sha256": table_hash(fit), "query_id_sha256": table_hash(query),
        "fit_sha256": sha256(target / "fit.parquet"), "query_sha256": sha256(target / "query.parquet"),
        "label_sha256": sha256(labels),
    }


def inherited_v1(split, role):
    path = V1 / "data" / ("outer-labels" if role == "label" else f"splits/{split}")
    path = path / (f"{split}.parquet" if role == "label" else f"{role}.parquet")
    receipt = read_json(V1 / "data/manifest.json")
    return parquet_lock(path, receipt["files"][str(path.relative_to(V1))])


def prepare():
    for path, expected in INPUT_SHA.items():
        if sha256(path) != expected:
            raise ValueError(f"Registered file changed: {path}")
    asset = read_json(ASSETS / "data_receipt.json")
    cold = read_json(COLD / "receipt.json")
    data = parquet_lock(COLD / "train.parquet", cold["files"]["train.parquet"]).sort_values("row_id")
    fit_rna = parquet_lock(CORE / "rna.parquet", INPUT_SHA[CORE / "rna.parquet"])
    legacy_rna = parquet_lock(COLD / "rna.parquet", cold["files"]["rna.parquet"])
    registry = register()
    cells = parquet_lock(ASSETS / "cells.parquet", asset["files"]["cells.parquet"])
    if not isinstance(cells.index, pd.RangeIndex):
        cells = cells.reset_index()
    coverage = parquet_lock(ASSETS / "coverage.parquet", asset["files"]["coverage.parquet"])
    if (len(data), data[KEY].nunique(), registry.scaffold.nunique(), len(cells),
            int(coverage.reindex(registry.index).lincs_modz24.sum())) != (670830, 1186, 1001, 1014, 724):
        raise ValueError("Extended screening support differs from H2")
    if not set(data.cell_id) <= set(fit_rna.index) or not set(data.cell_id) <= set(legacy_rna.index):
        raise ValueError("Both RNA axes need exact source IDs")
    if (fit_rna.shape[1], legacy_rna.shape[1]) != (17871, 14421):
        raise ValueError("Registered two RNA axes changed")
    if data.row_id.duplicated().any() or not np.isfinite(native_sensitivity(data)).all():
        raise ValueError("Primary table must contain unique, observed response rows")
    cell_fold = lineage_folds(data)
    meta = cells.set_index("cell_id")
    if not data.cell_id.map(meta.lineage_group).eq(data.lineage_group).all():
        raise ValueError("Inconsistent global lineage IDs")
    writeup = {}
    for name in SPLITS:
        mode, number = name.rsplit("-", 1)
        idx = int(number)
        if mode in ("drug", "cell"):
            if mode == "drug":
                fitting, holding = subfold(registry, idx)
                train, held = data.loc[data[KEY].isin(fitting)], data.loc[data[KEY].isin(holding)]
            else:
                train, held = data.loc[cell_fold != idx], data.loc[cell_fold == idx]
            roster = held[META + ["isomeric_smiles"]]
            check = held[META].assign(truth=native_sensitivity(held))
        else:
            historical = inherited_v1(name, "fit")
            roster = inherited_v1(name, "query")
            check = inherited_v1(name, "label").rename(columns={"observed": "truth"})
            # Match the old physical fit lineage block, including lineages without rows in the selected test screen.
            allow = set(historical.lineage_group)
            train = data.loc[data.lineage_group.isin(allow) & data.source_id.ne(mode)]
            if not set(roster[KEY]) <= set(train[KEY]) or not set(roster.row_id) == set(check.row_id):
                raise ValueError("The expanded training panel cannot cover registered H3-v1 rows")
        writeup[name] = detach(train, check, roster, name, registry)
    for name in ("drug", "cell"):
        if sum(row["query_rows"] for key, row in writeup.items() if key.startswith(name)) != len(data):
            raise ValueError("Full panel coverage is incomplete")
    if sum(row["query_rows"] for key, row in writeup.items() if key.rsplit("-",1)[0] in SCREENS) != 45428:
        raise ValueError("H3-v1 core support differs")
    source_set = set(data.cell_id)
    marks = {
        "rows": len(data), "parents": len(registry), "source_cell_count": len(source_set),
        "linages": data.lineage_group.nunique(), "tissues": meta.tissue.value_counts(dropna=False).to_dict(),
        "rna_full_core": fit_rna.shape[1], "rna_h2": legacy_rna.shape[1],
        "intersection_rna": len(fit_rna.columns.intersection(legacy_rna.columns)),
        "lincs_observed": int(coverage.reindex(registry.index).lincs_modz24.sum()),
        "split_support": writeup, "inputs": {str(p): h for p, h in INPUT_SHA.items()},
        "python": sys.version,
    }
    publish_json(ROOT / "splits/manifest.json", marks)
    print({k: v for k,v in marks.items() if k not in ("split_support","inputs")}, flush=True)


def read_data(split, role):
    manifest = read_json(ROOT / "splits/manifest.json")["split_support"][split]
    return parquet_lock(ROOT / "splits" / split / f"{role}.parquet", manifest[role + "_sha256"])


if __name__ == "__main__":
    prepare()
