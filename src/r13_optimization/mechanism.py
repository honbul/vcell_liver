"""Public pretreatment mechanisms as priors for fitted-neighbor response transfer."""

from functools import cache
from pathlib import Path
import re

import numpy as np
import pandas as pd

from src.r11_chemical_probe.dataset import ENTITY, GROUP, LINE

NULL = {"", "NAN", "NONE", "UNKNOWN", "UNCLEAR", "N/A", "NA"}


@cache
def annotations(root):
    path = Path(root) / "metadata/drug_metadata.parquet"
    rows = pd.read_parquet(path, columns=["drug", "targets", "moa-fine", "moa-broad"])
    if rows.drug.duplicated().any():
        raise ValueError("Ambiguous compound annotation")
    return rows.set_index("drug")


def tokens(row, column):
    text = str(row[column]).strip().upper()
    if column == "targets":
        return {tok.strip() for tok in re.split(r"[,;|/]", text) if tok.strip() not in NULL}
    return {text} if text not in NULL else set()


def annotation_similarity(reference, query, column, idf=False):
    vocabulary = sorted(set().union(*(tokens(row, column) for _, row in reference.iterrows())))
    if not vocabulary:
        return np.zeros((len(query), len(reference)))
    r = np.stack([[t in tokens(row, column) for t in vocabulary] for _, row in reference.iterrows()]).astype(float)
    q = np.stack([[t in tokens(row, column) for t in vocabulary] for _, row in query.iterrows()]).astype(float)
    weights = np.log1p(len(r) / (1 + r.sum(axis=0))) if idf else np.ones(r.shape[1])
    cross = (q * weights) @ r.T
    union = q @ weights
    union = union[:, None] + (r @ weights)[None, :] - cross
    return np.divide(cross, union, out=np.zeros_like(cross), where=union > 0)


def bit_tanimoto(a, b):
    cross = a @ b.T
    union = a.sum(axis=1)[:, None] + b.sum(axis=1)[None, :] - cross
    return np.divide(cross, union, out=np.zeros_like(cross), where=union > 0)


def candidates():
    rows = []
    for source in ["targets", "target-idf", "fine", "hybrid-target", "hybrid-full", "signed"]:
        for exponent in [1, 3]:
            for blend in [0.0, 0.5, 1.0]:
                name = f"{source}-p{exponent}-line{blend:g}"
                rows.append(dict(name=name, source=source, exponent=exponent, line=blend, neighbors=12))
    for k in [3, 6, 24]:
        for source in ["target-idf", "hybrid-full"]:
            rows.append(dict(name=f"{source}-k{k}", source=source, exponent=2, line=0.5, neighbors=k))
    for label, weight in [("moa-null", 0.0), ("moa-half", 0.5)]:
        rows.append(dict(name=label, source="hybrid-full", exponent=2, line=weight, neighbors=12, permute=True))
    return rows


class Learner:
    def __init__(self, fit):
        self.fit = fit
        tr, te = fit.train.rows, fit.query.rows
        self.groups, self.tr_group = np.unique(tr[GROUP], return_inverse=True)
        self.lines = sorted(tr[LINE].unique())
        self.tr_line = pd.Index(self.lines).get_indexer(tr[LINE])
        self.te_line = pd.Index(self.lines).get_indexer(te[LINE])
        self.query_entities, self.qi = np.unique(te[ENTITY], return_inverse=True)
        train_names = pd.Index(tr[ENTITY])
        _, first = np.unique(self.tr_group, return_index=True)
        ref_names = train_names[first]
        meta = annotations(fit.source.cfg["tahoe_root"])
        self.ref = meta.reindex(ref_names)
        self.query = meta.reindex(self.query_entities)
        ref_bits = fit.train.chemical["E"][first].astype(np.float32)
        _, te_first = np.unique(self.qi, return_index=True)
        query_bits = fit.query.chemical["E"][te_first].astype(np.float32)
        self.chemistry = bit_tanimoto(query_bits, ref_bits)
        self.sim = {
            col: annotation_similarity(self.ref, self.query, field, idf)
            for col, field, idf in [
                ("targets", "targets", False),
                ("target-idf", "targets", True),
                ("fine", "moa-fine", False),
            ]
        }
        r = self.ref["moa-broad"].fillna("unclear").to_numpy()
        q = self.query["moa-broad"].fillna("unclear").to_numpy()
        sign = q[:, None] == r[None, :]
        self.sim["signed"] = self.sim["target-idf"] * (0.25 + 0.75 * sign)
        self.sim["hybrid-target"] = 0.7 * self.sim["target-idf"] + 0.3 * self.chemistry
        self.sim["hybrid-full"] = (
            0.55 * self.sim["target-idf"] + 0.25 * self.sim["fine"] + 0.2 * self.chemistry
        )
        width = fit.residual.shape[1]
        ix = self.tr_group * len(self.lines) + self.tr_line
        count = np.bincount(ix, weights=fit.weight, minlength=len(self.groups) * len(self.lines))
        totals = np.zeros((len(count), width), dtype=np.float32)
        np.add.at(totals, ix, fit.residual * fit.weight.astype(np.float32)[:, None])
        cells = totals / np.maximum(count[:, None], 1e-20).astype(np.float32)
        self.valid = (count > 0).reshape((len(self.groups), len(self.lines)))
        self.cells = cells.reshape((len(self.groups), len(self.lines), width))
        self.mean = self.cells.sum(axis=1) / self.valid.sum(axis=1)[:, None]
        self.mean = self.mean.astype(np.float32)

    def predict(self, spec):
        sim = self.sim[spec["source"]]
        if spec.get("permute"):
            order = np.random.default_rng(413).permutation(len(self.groups))
            sim = sim[:, order]
        k = min(spec["neighbors"], sim.shape[1])
        # Stable ties share the same score, so threshold ties are kept together.
        cutoff = np.partition(sim, -k, axis=1)[:, -k, None]
        weights = np.where(sim >= cutoff, sim ** spec["exponent"], 0).astype(np.float32)
        standard = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        shared = standard @ self.mean
        out = shared[self.qi].copy()
        rate = spec["line"]
        if rate:
            for col in np.unique(self.te_line):
                if col < 0:
                    continue
                ix = np.flatnonzero(self.te_line == col)
                entities, inverse = np.unique(self.qi[ix], return_inverse=True)
                local = weights[entities] * self.valid[:, col]
                sums = local.sum(axis=1, keepdims=True)
                local /= np.maximum(sums, 1e-12)
                difference = self.cells[:, col] - self.mean
                local_pred = local @ difference
                out[ix] += rate * local_pred[inverse]
        return self.fit.offset + out


def predict(fit, spec):
    if not hasattr(fit, "_r13_mechanism"):
        fit._r13_mechanism = Learner(fit)
    return fit._r13_mechanism.predict(spec)
