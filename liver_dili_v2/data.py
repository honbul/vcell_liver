"""Reconstruct registered partitions without admitting any increment labels during tuning."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from liver_dili.splits import digest

SEED = 20261122
KINDS = ("murcko", "butina")


class RegisteredData:
    def __init__(self, root: Path):
        self.root = root
        self.manifest = json.loads((root / "receipt.json").read_text())
        # Only frozen identifiers and roles are required to construct the development mask.
        roles = self.read("roster.parquet")
        self.increment = frozenset(roles.loc[roles.split.eq("release") & roles.role.eq("query"), "fda_id"])
        if len(self.increment) != 155:
            raise ValueError("The registered 155-compound increment changed")
        self.development = frozenset(roles.loc[roles.split.eq("murcko"), "fda_id"]) - self.increment
        if len(self.development) != 732:
            raise ValueError("The registered source-only development support changed")
        self.roles = roles

    def read(self, relative, **kwargs):
        path = self.root / relative
        if digest(path) != self.manifest["files"][relative]:
            raise ValueError(f"Registered source changed: {relative}")
        return pd.read_parquet(path, **kwargs)

    def fit_table(self, kind, fold):
        if kind == "release":
            table = self.read("release-0/train.parquet")
        else:
            path = f"{kind}-{fold}/train.parquet"
            table = self.read(path, filters=[(self.index_column(path), "in", sorted(self.development))])
        if set(table.index) & self.increment:
            raise ValueError("FDA increment entered a fit role")
        return table

    def index_column(self, path):
        import pyarrow.parquet as pq

        meta = pq.read_schema(self.root / path).metadata
        index, = json.loads(meta[b"pandas"])["index_columns"]
        return index

    def outer_query(self, kind, fold, *, development):
        path = f"{kind}-{fold}/query.parquet"
        filters = [(self.index_column(path), "in", sorted(self.development))] if development else None
        table = self.read(path, filters=filters)
        if development and set(table.index) & self.increment:
            raise ValueError("FDA increment entered model selection")
        return table

    def outcomes(self, kind, fold, *, development):
        path = f"{kind}-{fold}/score_labels.parquet"
        filters = [(self.index_column(path), "in", sorted(self.development))] if development else None
        return self.read(path, filters=filters)

    def release_exclusions(self):
        # A split-only holdout registry is used for purging auxiliary assays. No FDA endpoint is read.
        return self.outer_query("release", 0, development=False)[["connectivity", "scaffold"]]


def inner_groups(fit):
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    for a, b in cv.split(fit, fit.label, fit.group):
        check_disjoint(fit.iloc[a], fit.iloc[b], group=True)
        yield a, b


def check_disjoint(fit, query, *, group=False):
    if set(fit.index) & set(query.index) or set(fit.connectivity) & set(query.connectivity):
        raise ValueError("Fit/validation drug identity leakage")
    if group and set(fit.group) & set(query.group):
        raise ValueError("Fit/validation split group leakage")
    if not np.isin(fit.label.to_numpy(), [0, 1]).all():
        raise ValueError("Expected binary, unambiguous FDA labels")
