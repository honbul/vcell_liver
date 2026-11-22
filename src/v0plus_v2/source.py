"""Selected V0+ v2 expression heads, refitted under grouped source splits."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from vc_build.v0plus_cv import panel_targets

from .adaptive import FAMILY, NORMS, Projection, fit_candidates

def grouped(fit, n):
    for left, right in GroupKFold(n).split(fit, groups=fit.lineage_group):
        develop, held = fit.iloc[left], fit.iloc[right]
        if set(develop.lineage_group) & set(held.lineage_group):
            raise ValueError("Source cell/patient lineage cross-over")
        yield develop, held


def setup(fit, rna):
    unique = rna.loc[rna.index.intersection(fit.cell_id)]
    projections = {norm: Projection.fit(unique, norm) for norm in NORMS}
    views = {name: obj.transform(rna) for name, obj in projections.items()}
    return projections, views


@dataclass
class DrugModels:
    projections: dict
    heads: dict
    anchor: pd.Series

    def predict(self, query, rna):
        transformed = {norm: view.transform(rna) for norm, view in self.projections.items()}
        base = query.compound_id.map(self.anchor).to_numpy()
        result = {method: base.copy() for method in self.heads}
        for method, drugs in self.heads.items():
            for drug, positions in query.groupby("compound_id").indices.items():
                if drug in drugs:
                    cell = rna.index.get_indexer(query.iloc[positions].cell_id)
                    if np.any(cell < 0):
                        raise ValueError("Missing expression for registered query cell")
                    head = drugs[drug]
                    result[method][positions] += head.apply(transformed[head.norm][cell])
        return result


def known_genes(rna, targets, drug):
    a = rna.columns.get_indexer(targets.get(drug, []))
    return a[a >= 0]


def fit_models(fit, rna, targets, selected):
    panel, anchor, _, _ = panel_targets(fit)
    projections, views = setup(fit, rna)
    heads = {name: {} for name in (*FAMILY, "adaptive")}
    for drug, rows in panel.groupby("compound_id"):
        options = selected.get(drug, {})
        if not options:
            continue
        ids = rna.index.get_indexer(rows.cell_id)
        cfgs = list(dict.fromkeys(options.values()))
        found = {setting: model for setting, model, _ in fit_candidates(
            views, projections, ids, np.array([], dtype=int),
            rows.residual.to_numpy(), known_genes(rna, targets, drug), configs=cfgs)}
        for name, setting in options.items():
            heads[name][drug] = found[setting]
    return DrugModels(projections, heads, anchor)
