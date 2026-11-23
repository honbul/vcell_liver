"""Materialize outcome-free LINCS and DepMap views from frozen public releases."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nextgen_h2.assets import COLD, GCTX, KEY, LINCS, MODEL, STORAGE, modz_weights, parquet_lock
from vc_build.io import publish_json, sha256

V1 = STORAGE / "nextgen_h123/h2_multiview"
V1_ASSETS = V1 / "assets-v3"
V1_TARGETS = V1 / "targets-v1"
V1_RUN = V1 / "study-v2"
ROOT = STORAGE / "nextgen_v2/h2_v2"
TAHOE = Path("/mnt/S1/sdata/pubdata/Tahoe/Tahoe-100M")
FOLDS_SHA = "98873d0954444add1719c65d3e478ff6b159af8f7a72f55eb063f80a4daf3800"
GDSC_SHA = "7346d53fdeea31e867444b3bf2206a8c87f321b247209b1d6ca980993e739768"
LINCS_SIGNATURE_SHA = "9813b620973ea984144cf797938e4e22315c798df2f001714a42de739b336204"
OMICS_FILES = {
    "mut": "OmicsSomaticMutationsMatrixHotspot.csv", "cnv": "OmicsCNGeneWGS.csv",
    "fusion": "OmicsFusionFiltered.csv", "sig": "OmicsGlobalSignatures.csv",
    "crispr": "CRISPRGeneEffect.csv",
}


def registries():
    registry = parquet_lock(V1_ASSETS / "folds.parquet", FOLDS_SHA)
    gdsc = parquet_lock(V1_ASSETS / "gdsc_query.parquet", GDSC_SHA)
    if len(registry) != 1186 or gdsc[KEY].nunique() != 66:
        raise ValueError("V1 freeze differs from H2-v2 registration")
    return registry, gdsc


def publish(out, receipt):
    publish_json(out / "receipt.json", receipt | {
        "artifacts": {path.name: sha256(path) for path in sorted(out.iterdir())}})


def lincs_block(signatures, genes):
    import h5py

    with h5py.File(GCTX, "r") as handle:
        row_axis = pd.Index(handle["0/META/ROW/id"][:].astype(str))
        column_axis = pd.Index(handle["0/META/COL/id"][:].astype(str))
        rows = column_axis.get_indexer(signatures.sig_id)
        cols = row_axis.get_indexer(genes.index)
        if (rows < 0).any() or (cols < 0).any():
            raise ValueError("Missing pinned Level 5 signature or gene")
        order = rows.argsort()
        matrix = handle["0/DATA/0/matrix"]
        result = np.empty((len(rows), len(cols)), dtype=np.float32)
        for start in range(0, len(order), 1000):
            ix = order[start:start + 1000]
            result[ix] = matrix[rows[ix], :][:, cols]
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite Level 5 value")
    return result


def group_reduce(keys, values, calibration, function):
    indexed, rows = [], []
    for key, index in keys.items():
        indexed.append(key)
        rows.append(function(values[index], calibration, len(index)))
    return pd.MultiIndex.from_tuples(indexed), np.asarray(rows, dtype=np.float32)


def by_groups(index, levels):
    return pd.Series(np.arange(len(index)), index=index).groupby(level=levels, sort=True).indices


def cell_consensus(sig, values, landmarks):
    grouping = sig.groupby([KEY, "cell_iname", "time_h", "dose_um"], sort=True).indices
    index, context = group_reduce(grouping, values, landmarks, lambda part, mask, n:
                                  part[0] if n == 1 else modz_weights(part[:, mask]) @ part)
    keys = index.to_frame(index=False)
    keys.columns = [KEY, "cell_iname", "time_h", "dose_um"]
    return keys, context


def summarized(keys, context, recipe):
    if recipe == "multitime":
        windows = [keys.time_h.eq(6), keys.time_h.eq(24)]
    elif recipe == "dose":
        windows = [keys.time_h.eq(24) & keys.dose_um.le(5), keys.time_h.eq(24) & keys.dose_um.gt(5)]
    else:
        windows = [keys.time_h.eq(24)]
    parts = []
    for allowed in windows:
        subindex = pd.MultiIndex.from_frame(keys.loc[allowed, [KEY, "cell_iname"]])
        sub, cells = group_reduce(by_groups(subindex, [0, 1]), context[allowed], None,
                                  lambda part, _mask, _n: part.mean(axis=0))
        groups = by_groups(sub, [0])
        blocks = [cells[index] for index in groups.values()]
        part = np.stack([block.mean(axis=0) for block in blocks])
        if recipe in ("sd", "iqr"):
            eligible = [i for i, block in enumerate(blocks) if len(block) >= 2]
            variability = [(blocks[i].std(axis=0) if recipe == "sd" else
                            np.subtract(*np.quantile(blocks[i], [.75, .25], axis=0))) for i in eligible]
            part = np.concatenate([part[eligible], np.asarray(variability, np.float32)], axis=1)
            parents = pd.Index(list(groups), name=KEY)[eligible]
        else:
            parents = pd.Index(list(groups), name=KEY)
        parts.append(pd.DataFrame(part, index=parents))
    return pd.concat(parts, axis=1, join="inner", ignore_index=True)


def lincs(out):
    registry, _ = registries()
    sig = parquet_lock(V1_ASSETS / "lincs_selected_signatures.parquet", LINCS_SIGNATURE_SHA).reset_index(drop=True)
    if set(sig.cell_iname) & {"HEPG2", "HUH7", "JHH5", "JHH7"}:
        raise ValueError("Liver signature present")
    genes = pd.read_csv(LINCS / "geneinfo_beta.txt", sep="\t", dtype={"gene_id": str}).set_index("gene_id")
    landmarks = genes.feature_space.eq("landmark").to_numpy()
    bing = genes.feature_space.isin(["landmark", "best inferred"]).to_numpy()
    axes = {"landmark": landmarks, "bing": bing, "full": np.ones(len(genes), dtype=bool)}
    if [int(x.sum()) for x in axes.values()] != [978, 10174, 12328]:
        raise ValueError("Wrong L1000 gene space")
    out.mkdir(parents=True, exist_ok=False)
    value = lincs_block(sig, genes)
    keys, contexts = cell_consensus(sig, value, landmarks)
    del value
    info = {}
    for recipe in ("mean", "sd", "iqr", "multitime", "dose"):
        frame = summarized(keys, contexts, recipe)
        for axis, use in axes.items():
            columns = np.tile(use, frame.shape[1] // len(use))
            data = frame.loc[:, columns].copy()
            data.columns = [f"x{i}" for i in range(data.shape[1])]
            name = f"{axis}_{recipe}"
            data.to_parquet(out / (name + ".parquet"))
            eligible = registry.loc[registry.index.intersection(data.index)]
            info[name] = {"parents": len(data), "drp_parents": len(eligible),
                          "drp_scaffolds": int(eligible.scaffold.nunique()), "features": data.shape[1]}
        print(recipe, len(frame), flush=True)
    # Delta due to summation order is explicitly bounded against the v1 namespace.
    exact = pd.read_parquet(V1_ASSETS / "lincs_modz24.parquet")
    new = pd.read_parquet(out / "landmark_mean.parquet")
    aligned_symbols = genes.loc[landmarks, "gene_symbol"].to_list()
    if len(set(aligned_symbols)) != 978 or set(new.index) != set(exact.index):
        raise ValueError("24h LINCS support departed from v1")
    delta = (new.loc[exact.index].to_numpy() - exact[aligned_symbols].to_numpy())
    publish(out, {"signatures_sha256": LINCS_SIGNATURE_SHA, "sampled_signatures": len(sig),
                  "contexts": len(keys), "cells": sig.cell_iname.nunique(),
                  "gene_metadata_sha256": sha256(LINCS / "geneinfo_beta.txt"),
                  "raw_matrix_bytes": GCTX.stat().st_size,
                  "structures_with_summation_order_error": float(np.max(np.abs(delta))),
                  "views": info})


def omics_wide(path, mode, wanted):
    if mode == "crispr":
        features = pd.read_csv(path, nrows=0).columns[1:]
        frame = pd.read_csv(path, index_col=0, dtype=dict.fromkeys(features, np.float32))
        frame = frame.loc[frame.index.intersection(wanted)]
    else:
        cols = pd.read_csv(path, nrows=0).columns
        metadata = ["ModelID", "IsDefaultEntryForModel", "IsDefaultEntryForMC", "SequencingID"]
        signatures = ["MSIScore", "LoHFraction", "WGD", "CIN", "Ploidy", "Aneuploidy"]
        features = [c for c in cols if c in signatures or c.endswith(")")]
        columns = [*metadata, *features]
        frame = pd.read_csv(path, usecols=columns, dtype={key: np.float32 for key in features}, low_memory=False)
        frame = frame.loc[frame.ModelID.isin(wanted) & frame.IsDefaultEntryForModel.eq("Yes")
                          & frame.IsDefaultEntryForMC.eq("Yes")]
        frame = frame.drop(columns=[c for c in metadata if c != "ModelID"])
        frame = frame.groupby("ModelID", sort=True).median()
    frame.columns = [x.split(" (", 1)[0] for x in frame.columns]
    frame = frame.T.groupby(level=0, sort=True).mean().T
    into = np.isfinite(frame).mean(axis=1).ge(.75)
    return frame.loc[into].astype(np.float32)


def omics(out):
    registry, gdsc = registries()
    cells = pd.read_parquet(V1_ASSETS / "cells.parquet")
    roster = pd.read_csv(MODEL, usecols=["ModelID"]).ModelID
    incoming = set(cells.index) & set(roster)
    paths = {mode: MODEL.parent / name for mode, name in OMICS_FILES.items()}
    rna_profiles = pd.read_csv(MODEL.parent / "OmicsProfiles.csv",
                               usecols=["ModelID", "IsDefaultEntryForModel", "IsDefaultEntryForMC",
                                        "SequencingID", "DataType"])
    rna_profiles = rna_profiles[rna_profiles.ModelID.isin(incoming) &
        rna_profiles.IsDefaultEntryForModel.eq("Yes") & rna_profiles.IsDefaultEntryForMC.eq("Yes") &
        rna_profiles.DataType.eq("rna")]
    out.mkdir(parents=True, exist_ok=False)
    info = {}
    for mode, path in paths.items():
        if mode != "fusion":
            frame = omics_wide(path, mode, incoming)
        else:
            fusion = pd.read_csv(path, usecols=["ModelID", "SequencingID", "CanonicalFusionName", "FFPM",
                                                "IsDefaultEntryForModel", "IsDefaultEntryForMC"])
            fusion = fusion.loc[fusion.IsDefaultEntryForModel.eq("Yes") & fusion.IsDefaultEntryForMC.eq("Yes")]
            found = fusion.merge(rna_profiles[["ModelID", "SequencingID"]].drop_duplicates(),
                                 on=["ModelID", "SequencingID"], validate="many_to_one")
            # Only models with a registered fusion-file sequencing identity establish coverage.
            frame = found.pivot_table(index="ModelID", columns="CanonicalFusionName", values="FFPM",
                                      aggfunc="max", fill_value=0).astype(np.float32)
        frame.to_parquet(out / (mode + ".parquet"))
        info[mode] = {"cells": len(frame), "features": len(frame.columns),
                      "GDSC1_cells": len(set(frame.index) & set(gdsc.cell_id)),
                      "source_sha256": sha256(path)}
        print(mode, len(frame), len(frame.columns), flush=True)
    publish(out, {"source_cells": len(cells), "assembly": "DepMap Public 25Q3",
                  "model_sha256": sha256(MODEL),
                  "profiles_sha256": sha256(MODEL.parent / "OmicsProfiles.csv"),
                  "fusion_coverage": "default RNA sequencing identity represented by the fusion release",
                  "views": info})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["lincs", "omics"])
    parser.add_argument("out", type=Path)
    options = parser.parse_args()
    {"lincs": lincs, "omics": omics}[options.stage](options.out)
