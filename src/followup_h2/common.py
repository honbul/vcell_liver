"""Read the pinned H2 study assets and isolate the follow-up execution namespace."""
import hashlib
from pathlib import Path

import pandas as pd
import torch

from nextgen_h2.assets import AUXILIARY, COLD, HALLMARK, PRIOR, STORAGE, TARGET, parquet_lock
from nextgen_h2.model import Inputs, molecular_features
from nextgen_h2.run import META, TRAIN, read_json
from nextgen_h2_v2.assets import ROOT as V2, V1_ASSETS, V1_RUN, V1_TARGETS, registries
from vc_build.io import sha256
from vc_build.v0plus_cv import molecular_inputs
from vc_build.v0plus_neural import load_auxiliary

ROOT = STORAGE / "followup_20261124/h2_e1e3"
CELLOSAURUS = COLD.parents[2] / "function/sources/Cellosaurus-57.0.txt.gz"
ARCHIVE = COLD.parent
V2_FORECAST = V2 / "study/forecasts"


def device():
    torch.set_num_threads(8)
    name = torch.cuda.get_device_name(0)
    if not any(x in name for x in ("A30", "A40")):
        raise ValueError(f"The follow-up requires A30 or A40, received {name}")
    print(f"GPU {name}", flush=True)
    return torch.device("cuda")


def registered():
    reg, gdsc = registries()
    assets = read_json(V1_ASSETS / "data_receipt.json")
    cells = parquet_lock(V1_ASSETS / "cells.parquet", assets["files"]["cells.parquet"])
    source = read_json(COLD / "receipt.json")
    if sha256(TRAIN) != source["files"]["train.parquet"]:
        raise ValueError("Changed native source panel")
    return reg, gdsc, cells, assets


def context(on, *, external=False):
    # Load only the RNA, chemistry, and 24h LINCS inputs used by the contracted arms.
    from nextgen_h2.assets import LIMORE, canonical_registry

    reg, gdsc, _, assets = registered()
    target_receipt = read_json(V1_TARGETS / "receipt.json")
    if sha256(V1_TARGETS / "targets.json") != target_receipt["targets_sha256"]:
        raise ValueError("The target registry changed")
    targets = read_json(V1_TARGETS / "targets.json")
    source = read_json(COLD / "receipt.json")
    rna = parquet_lock(COLD / "rna.parquet", source["files"]["rna.parquet"])
    library, policy = V2 / "assets-lincs", V2 / "assets-lincs/receipt.json"
    old = parquet_lock(V1_ASSETS / "lincs_modz24.parquet", assets["files"]["lincs_modz24.parquet"])
    new = parquet_lock(library / "landmark_mean.parquet", read_json(policy)["artifacts"]["landmark_mean.parquet"])
    query = None
    all_drugs = reg
    if external:
        note = read_json(LIMORE / "receipt.json")["artifacts"]
        lim = parquet_lock(LIMORE / "test_inputs/rows.parquet", note["test_inputs/rows.parquet"]["sha256"])
        lic = parquet_lock(COLD / "query.parquet", source["files"]["query.parquet"])
        erna = parquet_lock(LIMORE / "test_inputs/rna.parquet", note["test_inputs/rna.parquet"]["sha256"])
        rna = pd.concat([rna, erna.reindex(columns=rna.columns)], verify_integrity=True)
        lim = lim.rename(columns={"cell_group": "lineage_group", "record_id": "row_id"}).assign(source_id="LIMORE")
        lic = lic.assign(source_id="LICOB", endpoint_id="AUC")
        query = pd.concat([lim, lic, gdsc], ignore_index=True).drop_duplicates("row_id")
        ext_reg = canonical_registry(query)
        all_drugs = pd.concat([reg, ext_reg.loc[ext_reg.index.difference(reg.index)]])
    edges, _, _ = molecular_inputs(PRIOR, HALLMARK, TARGET)
    auxiliary, _ = load_auxiliary(AUXILIARY)
    profiles = {"modz24": old, "landmark_mean": new}
    data = Inputs(V1_ASSETS, molecular_features(all_drugs), profiles, None, rna, edges, targets, auxiliary, on)
    return data, reg, query


def native_rows(keys, lineages=None, *, labels=False):
    filters = [("compound_id", "in", list(keys))]
    if lineages is not None:
        filters.append(("cell_id", "in", list(lineages)))
    return pd.read_parquet(TRAIN, columns=None if labels else META, filters=filters).sort_values(
        "row_id", ignore_index=True)


def panel_hash(frame):
    text = frame[META].to_csv(index=False, sep="\t", lineterminator="\n")
    return hashlib.sha256(text.encode()).hexdigest()


def keys_hash(keys):
    return hashlib.sha256("\n".join(sorted(map(str, keys))).encode()).hexdigest()
