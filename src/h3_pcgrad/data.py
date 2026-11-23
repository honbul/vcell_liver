"""Historical outer split keys, detached response labels, and fit-only molecular views."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import GroupKFold

from vc_build.expanded_panel_cv import RECEIPTS, same_split
from vc_build.external import native_sensitivity
from vc_build.io import _publish, publish_json, sha256, verify_forecast
from vc_build.v0plus_cv import panel_targets, training_inputs

BASE = Path("/mnt/S1/sdata/workbench/Virtual_Cell/gssong")
ROOT = BASE / "nextgen_h123/h3_pcgrad"
BUILD = BASE / "vc_build_v1"
RUNS = BUILD / "v0plus-runs"
ORIGINAL = BUILD / "function/cross-v1"
CORE = RUNS / "source-core-92aa482/core"
ARCHIVE = RUNS / "expanded-panel-81119c8"
HALLMARK = Path("/mnt/S1/sdata/pubdata/Virtual_Cell/MSigDB/2026.1.Hs/h.all.v2026.1.Hs.symbols.gmt")
IDENTITY = ["row_id", "cell_id", "lineage_group", "compound_id", "source_id", "endpoint_id"]
ROSTER = [*IDENTITY, "tissue_origin", "isomeric_smiles"]


def read_json(path):
    return json.loads(Path(path).read_text())


def contract():
    path = Path(os.environ.get("H3_CONTRACT", "experiments/nextgen_h3_pcgrad/contract.json"))
    return read_json(path)


def save_table(frame, path):
    _publish(path, lambda target: frame.to_parquet(target, index=False))


def split_id(index):
    return f"{contract()['screens'][index // 3]}-{index % 3}"


def prior_structure():
    from vc_build.v0plus_cv import molecular_inputs
    from vc_build.v0plus_neural import load_auxiliary

    edges, targets, prior = molecular_inputs(BUILD / "v0plus-public-priors", HALLMARK,
                                              BUILD / "external-code-v3/drug_targets.json")
    auxiliary, aux_receipt = load_auxiliary(RUNS / "depmap-25q3-bb8a47b")
    return (edges, targets, auxiliary), {"priors": prior, "auxiliary": aux_receipt}


def outer_reference(query, split):
    path = ARCHIVE / "forecasts" / f"{split}.parquet"
    previous, receipt = verify_forecast(path)
    key = ["cell_id", "lineage_group", "compound_id", "track"]
    registered = query.assign(track=query.source_id + "/" + query.endpoint_id)
    index = pd.MultiIndex.from_frame(registered[key])
    values = {}
    for arm, name in (("corrected_core", "structured"), ("corrected_drug_mean", "drug_mean"),
                      ("corrected_ridge", "drug_ridge")):
        ref = previous.loc[previous.method.eq(arm)].set_index(key).predicted_value
        if not ref.index.is_unique or len(ref) != len(index) or set(ref.index) != set(index):
            raise ValueError(f"Historical corrected {arm} support differs: {split}")
        values[name] = ref.reindex(index).to_numpy()
    return values, receipt["sha256"]


def prepare():
    from sklearn import __version__ as sklearn_version

    freeze = contract()
    for bundle, name in ((ORIGINAL, "original"), (CORE, "core")):
        if sha256(bundle / "receipt.json") != RECEIPTS[name]:
            raise ValueError(f"Registered source receipt changed: {name}")
    old, _, _ = training_inputs(ORIGINAL)
    clean, rna, _ = training_inputs(CORE)
    if sha256(CORE / "train/rows.parquet") != freeze["source_rows_sha256"]:
        raise ValueError("Corrected observation hash differs")
    if sha256(CORE / "train/rna.parquet") != freeze["source_rna_sha256"]:
        raise ValueError("Corrected RNA hash differs")
    if (len(clean), clean.cell_id.nunique(), clean.lineage_group.nunique(),
            clean.compound_id.nunique()) != (65678, 1014, 991, 68):
        raise ValueError("Corrected support differs")
    if clean.duplicated(IDENTITY[1:]).any() or clean.row_id.duplicated().any():
        raise ValueError("Duplicate native observation identity")
    if not np.isfinite(native_sensitivity(clean)).all():
        raise ValueError("Registered table contains non-observations")
    support, artifacts = [], {}
    for index in range(9):
        name = split_id(index)
        screen, fold = name.rsplit("-", 1)
        _, fit, check = same_split(old, clean, int(fold), screen)
        hold = check[ROSTER].reset_index(drop=True)
        _, ref_hash = outer_reference(hold, name)
        store = ROOT / "data/splits" / name
        labels = ROOT / "data/outer-labels" / f"{name}.parquet"
        save_table(fit, store / "fit.parquet")
        save_table(hold, store / "query.parquet")
        save_table(check[IDENTITY].assign(observed=native_sensitivity(check)), labels)
        pairs = GroupKFold(n_splits=2).split(fit, groups=fit.lineage_group)
        for inner, (tr, va) in enumerate(pairs):
            train, valid = fit.iloc[tr], fit.iloc[va]
            valid = valid.loc[valid.compound_id.isin(train.compound_id)]
            if set(train.lineage_group) & set(valid.lineage_group) or screen in set(valid.source_id):
                raise ValueError("Inner group or assay leaked")
            save_table(train, store / f"inner{inner}-fit.parquet")
            save_table(valid, store / f"inner{inner}-valid.parquet")
            support.append(describe(name, inner, train, valid))
        support.append(describe(name, "outer", fit, hold) | {"structure_forecast_sha256": ref_hash})
        artifacts[str(labels.relative_to(ROOT))] = sha256(labels)
        for path in sorted(store.glob("*.parquet")):
            artifacts[str(path.relative_to(ROOT))] = sha256(path)
    publish_json(ROOT / "data/manifest.json", {
        "split_support": support, "files": artifacts, "sklearn": sklearn_version,
        "native_rows": clean.source_id.value_counts().to_dict(),
        "native_endpoints": clean.groupby(["source_id", "endpoint_id"]).size().to_dict().__str__(),
        "rna_genes": len(rna.columns), "contract": freeze,
    })


def describe(name, inner, train, check):
    overlap = set(train.lineage_group) & set(check.lineage_group)
    if overlap or len(check) == 0:
        raise ValueError("Leaked or empty lineage fold")
    return {"split": name, "inner": inner, "fit_rows": len(train), "check_rows": len(check),
            "fit_lineages": train.lineage_group.nunique(), "check_lineages": check.lineage_group.nunique(),
            "check_drugs": check.compound_id.nunique(), "overlap": len(overlap),
            "fit_screens": sorted(train.source_id.unique()), "check_screens": sorted(check.source_id.unique())}


def read_part(path):
    expected = read_json(ROOT / "data/manifest.json")["files"][str(path.relative_to(ROOT))]
    if sha256(path) != expected:
        raise ValueError(f"Detached asset changed: {path}")
    return pd.read_parquet(path)


def expression():
    file = CORE / "train/rna.parquet"
    if sha256(file) != contract()["source_rna_sha256"]:
        raise ValueError("Unexpected RNA input")
    from vc_build.external import _rna
    return _rna(file).sort_index(axis=1)


def sample_rank(matrix):
    return ((rankdata(matrix, axis=1) - .5) / matrix.shape[1] - .5).astype(np.float32)


def molecular_view(fit, query, rna, *, ranked=None):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    panel, anchor, aligned, _ = panel_targets(fit)
    cells = pd.Index(sorted(fit.cell_id.unique()))
    drugs = pd.Index(sorted(fit.compound_id.unique()))
    hold = pd.Index(sorted(query.cell_id.unique()))
    if ranked is None:
        ranked = pd.DataFrame(sample_rank(rna.to_numpy(np.float32)), index=rna.index, columns=rna.columns)
    train = ranked.loc[cells].to_numpy()
    valid = ranked.loc[hold].to_numpy()
    selected = np.argsort(-train.std(axis=0), kind="stable")[:512]
    low, center, high = np.quantile(train[:, selected], [.1, .5, .9], axis=0).astype(np.float32)
    spread = np.maximum((high - low) / 2.56, .03)
    transform = lambda x: np.clip((x[:, selected] - center) / spread, -8, 8)
    chemicals = fit[["compound_id", "isomeric_smiles"]].drop_duplicates().set_index("compound_id")
    if not chemicals.index.is_unique:
        raise ValueError("Compound registered with multiple structures")
    morgan = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)
    molecules = np.stack([morgan.GetFingerprintAsNumPy(Chem.MolFromSmiles(smiles))
                          for smiles in chemicals.loc[drugs].isomeric_smiles]).astype(np.float32)
    y = panel.pivot(index="cell_id", columns="compound_id", values="residual").loc[cells, drugs].to_numpy()
    mask = np.isfinite(y)
    scale = max(.05, float(y[mask].std()))
    mapping = hold.get_indexer(query.cell_id), drugs.get_indexer(query.compound_id)
    if min(axis.min() for axis in mapping) < 0:
        raise ValueError("Outside fitted compound or held RNA support")
    return {"x": transform(train), "q": transform(valid), "drug": molecules, "y": y / scale,
            "anchor": anchor.loc[drugs].to_numpy(), "scale": scale, "mapping": mapping,
            "genes": rna.columns[selected].tolist(), "center": center, "spread": spread,
            "drugs": list(drugs), "cells": list(cells), "aligned": aligned, "anchor_series": anchor}


if __name__ == "__main__":
    prepare()
