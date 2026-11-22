"""Prepare a hashed FDA evaluation split with separate fit and scoring labels."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from sklearn.model_selection import StratifiedGroupKFold

from .chemistry import features


def digest(path):
    with path.open("rb") as source:
        value = sha256()
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def receipt(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def butina_groups(fingerprints, cutoff=.60):
    distances = []
    for i, fp in enumerate(fingerprints):
        distances.extend(1. - np.asarray(DataStructs.BulkTanimotoSimilarity(fp, fingerprints[:i])))
    clusters = Butina.ClusterData(distances, len(fingerprints), 1-cutoff, isDistData=True, reordering=True)
    grouping = np.empty(len(fingerprints), dtype=int)
    for label, indices in enumerate(clusters):
        grouping[list(indices)] = label
    return grouping


def data_groups(table, fingerprints, kind):
    local = table.scaffold if kind == "murcko" else pd.Series(butina_groups(fingerprints), index=table.index)
    roots = dict.fromkeys(local.unique())

    def root(key):
        while roots[key] is not None:
            key = roots[key]
        return key

    # Connectivity parents are always inseparable, including stereo variants in a Butina split.
    for _, members in local.groupby(table.connectivity, sort=True):
        anchor = root(members.iloc[0])
        for member in members.iloc[1:]:
            other = root(member)
            if other != anchor:
                roots[other] = anchor
    return local.map(root).astype(str)


def known_table(path):
    rows = pd.read_csv(path, sep="\t").set_index("fda_id", verify_integrity=True)
    binary = ["Most", "Less", "No"]

    def release(column):
        eligible = rows.loc[rows.status.eq("resolved") & rows[column].isin(binary)].copy()
        eligible["label"] = eligible[column].isin(["Most", "Less"]).astype(int)
        conflict = eligible.groupby("inchikey").label.transform("nunique").gt(1)
        return eligible.loc[~conflict].sort_index(), int(conflict.sum())

    records, conflict = release("concern")
    history, historic_conflict = release("v1_concern")
    counts = dict(fda_records=len(rows), structure_resolved=int(rows.status.eq("resolved").sum()),
                  ambiguous=int(rows.concern.eq("Ambiguous").sum()), binary_resolved=len(records) + conflict,
                  conflicting_records=conflict, conflict_free_records=len(records),
                  historical_binary_records=len(history), historical_conflicting_records=historic_conflict,
                  exact_stereo_duplicates=int(records.duplicated("inchikey").sum()))
    return records, history, set(rows.connectivity[rows.release_increment].dropna()), counts


def nearest(train, query):
    return np.asarray([max(DataStructs.BulkTanimotoSimilarity(fp, train)) for fp in query])


def prepare(source, output):
    table, history, release_parents, counts = known_table(source)
    fingerprints, _ = zip(*(features(smiles) for smiles in table.smiles))
    groups = {kind: data_groups(table, fingerprints, kind) for kind in ("murcko", "butina")}
    folds = {}
    for kind, labels in groups.items():
        splitter = StratifiedGroupKFold(5, shuffle=True, random_state=20261122)
        folds[kind] = list(splitter.split(table, table.label, labels))
    history = history.loc[~history.connectivity.isin(release_parents)].copy()
    increment = table.loc[table.release_increment].copy()
    if history.empty or increment.empty:
        raise ValueError("No independent FDA release evaluation")
    partitions = [(kind, fold, table.iloc[train].copy(), table.iloc[test].copy(),
                   groups[kind].iloc[train], groups[kind].iloc[test])
                  for kind, pairs in folds.items() for fold, (train, test) in enumerate(pairs)]
    partitions.append(("release", 0, history, increment, history.scaffold, increment.scaffold))

    output.mkdir(parents=True, exist_ok=False)
    sizes, rosters = [], []
    label_free_columns = ["smiles", "inchikey", "connectivity", "scaffold"]
    for kind, fold, dev, query, fit_group, test_group in partitions:
        destination = output / f"{kind}-{fold}"
        destination.mkdir()
        if set(dev.connectivity) & set(query.connectivity):
            raise ValueError("Train–test connectivity leakage")
        if kind != "release" and set(fit_group) & set(test_group):
            raise ValueError(f"Train–test {kind} leakage")
        dev["group"] = fit_group
        dev[label_free_columns + ["group", "label"]].to_parquet(destination / "train.parquet")
        query[label_free_columns].to_parquet(destination / "query.parquet")
        query[["label", "concern"]].to_parquet(destination / "score_labels.parquet")
        distances = nearest([features(s)[0] for s in dev.smiles], [features(s)[0] for s in query.smiles])
        pd.Series(distances, index=query.index, name="nearest_tanimoto").to_csv(
            destination / "nearest.tsv", sep="\t")
        size = dict(split=kind, fold=fold, train=len(dev), test=len(query),
                    train_positive=int(dev.label.sum()), test_positive=int(query.label.sum()),
                    train_connectivity=dev.connectivity.nunique(), test_connectivity=query.connectivity.nunique(),
                    exact_overlap=0, scaffold_overlap=len(set(dev.scaffold) & set(query.scaffold)))
        size.update({f"tanimoto_q{round(q*100):02}": float(np.quantile(distances, q))
                     for q in [0., .05, .25, .5, .75, .95, 1.]})
        sizes.append(size)
        rosters.extend(dict(split=kind, fold=fold, role=role, fda_id=i)
                       for role, rows in (("fit", dev), ("query", query)) for i in rows.index)
    pd.DataFrame(sizes).to_csv(output / "support.tsv", sep="\t", index=False)
    pd.DataFrame(rosters).to_parquet(output / "roster.parquet", index=False)
    files = sorted(p for p in output.rglob("*") if p.is_file())
    receipt(output / "receipt.json", {"source_sha256": digest(source), "counts": counts,
        "files": {str(p.relative_to(output)): digest(p) for p in files}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.source, args.output)
