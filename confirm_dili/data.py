"""Verified historical and registered query access."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vc_fm_bench.artifacts import digest, verify_record
from .molecular import arrays, extract, molecule

G1 = Path("/mnt/S1/sdata/workbench/Virtual_Cell/gssong")
WORKER = G1 / "confirm_20261123/worker2_dc"
CLAMP = G1 / "nextgen_5track/checkpoints/clamp"
STORE = G1 / "nextgen_5track/fm_store_v1/chemistry-v1"
SPLITS = G1 / "liver_dili_v1/splits-v1"
V2 = G1 / "top3_tuning_v2/b/runs/release-01"
TRACK6 = G1 / "nextgen_5track"
CHEMBERTA = G1 / "liver_dili_v1/embeddings-full-v1/chemberta.parquet"
CLAMP_VECTOR_SHA = "eea3e355933ed96b6bdf72ede73cace50f2f1589cbaaf47fac85fc9f04a723ef"
SCOPES = [f"{kind}-{fold}" for kind in ("murcko", "butina") for fold in range(5)] + ["release-0"]


def read(path):
    return json.loads(path.read_text())


class Registered:
    def __init__(self):
        self.split_hash = digest(SPLITS / "receipt.json")
        self.manifest = read(SPLITS / "receipt.json")["files"]

    def table(self, name, **kwargs):
        target = SPLITS / name
        if digest(target) != self.manifest[name]:
            raise ValueError(f"Registered split bytes changed: {name}")
        return pd.read_parquet(target, **kwargs)

    def historical(self):
        source = self.table("release-0/train.parquet")
        if len(source) != 702 or source.connectivity.nunique() != 681 or not source.label.isin([0, 1]).all():
            raise ValueError("FDA v1 training roster changed")
        return source

    def query(self, scope):
        return self.table(f"{scope}/query.parquet")

    def fit(self, scope):
        query = self.query(scope)
        if scope == "release-0":
            rows = self.historical()
        else:
            increment = self.query("release-0")
            rows = self.table(f"{scope}/train.parquet", filters=[("fda_id", "not in", increment.index.tolist())])
        if set(rows.connectivity) & set(query.connectivity) or set(rows.index) & set(query.index):
            raise ValueError(f"Train–query identity overlap: {scope}")
        return rows

    def labels(self, scope):
        return self.table(f"{scope}/score_labels.parquet")


class FeatureBank:
    def __init__(self, checkpoint=CLAMP, batch_size=64):
        root = STORE / "clamp"
        record = read(root / "receipt.json")["output"]
        if record["sha256"] != CLAMP_VECTOR_SHA:
            raise ValueError("CLAMP extraction receipt changed")
        verify_record(root / "vectors.npz", record)
        with np.load(root / "vectors.npz", allow_pickle=False) as archive:
            vectors = archive["vectors"]
            keys = archive["id"].tolist()
        if len(set(keys)) != len(keys) or vectors.shape != (len(keys), 768) or not np.isfinite(vectors).all():
            raise ValueError("Invalid frozen CLAMP feature bank")
        self.lookup = dict(zip(keys, vectors))
        self.checkpoint = checkpoint
        self.batch_size = batch_size

    def get(self, rows):
        names = [molecule(s)[:2] for s in rows.smiles]
        absent = {key: smiles for smiles, key in names if key not in self.lookup}
        if absent:
            vectors = extract(self.checkpoint, list(absent.values()), batch_size=self.batch_size)
            self.lookup.update(zip(absent, vectors))
        return arrays(rows.smiles, [self.lookup[key] for _, key in names])


def legacy_rosters():
    prepared = TRACK6 / "prepared/track6-panels-v2"
    saved = TRACK6 / "results/track6-v1"
    subsets = {}
    for kind, scopes in {
        "murcko": [f"dili-scaffold-{i}" for i in range(5)],
        "release": ["dili-release-increment"],
    }.items():
        raw, retained, missing = [], [], {}
        for scope in scopes:
            panel = prepared / scope
            rec = read(panel / "receipt.json")
            verify_record(panel / "query.parquet", rec["files"]["query.parquet"])
            r = pd.read_parquet(panel / "query.parquet")
            model = saved / "predictions" / scope
            scored = saved / "scores" / scope
            rec = read(scored / "receipt.json")
            verify_record(model / "predictions.parquet", rec["frozen_predictions"])
            p = pd.read_parquet(model / "predictions.parquet", columns=["id", "drug_id"])
            if not set(p.id) <= set(r.id) or r.id.duplicated().any() or p.id.duplicated().any():
                raise ValueError(f"Legacy FM coverage integrity error in {scope}")
            raw.append(r)
            retained.append(p)
        raw, retained = pd.concat(raw).set_index("id"), pd.concat(retained).set_index("id")
        for model in ("chemberta2", "molformer", "unimol", "clamp", "ecfp4"):
            folder = STORE / model
            rec = read(folder / "receipt.json")
            verify_record(folder / "vectors.npz", rec["output"])
            with np.load(folder / "vectors.npz", allow_pickle=False) as archive:
                missing[model] = set(raw.index[~raw.drug_id.isin(archive["id"])])
        subsets[kind] = raw, retained, missing
    return subsets
