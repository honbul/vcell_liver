"""Persistent inference classes for the frozen sensitivity ensemble."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .lean import forecast as neural_predict


@dataclass
class SourceMember:
    adaptive: object
    boosted: object
    neural: dict
    meta: dict

    def predict(self, query, expression, banks, *, device="cpu"):
        selected = set().union(*(fit.columns for fit in self.meta.values()))
        options = self.adaptive.predict(query, expression) | self.boosted.predict(query, expression)
        for name, model in self.neural.items():
            if name in selected:
                options[name] = neural_predict(model, query, expression, banks, device=device)
        return pd.DataFrame(options)


@dataclass
class Release:
    genes: list
    median: np.ndarray
    models: list
    methods: list
    compounds: pd.DataFrame
    selected_genes: np.ndarray

    def align(self, expression, minimum=.85):
        matching = expression.reindex(columns=self.genes)
        observed = matching.notna()
        coverage = pd.DataFrame({"rna_gene_coverage": observed.mean(axis=1),
                                 "selected_gene_coverage": observed.iloc[:, self.selected_genes].mean(axis=1)})
        valid = coverage.min(axis=1).ge(minimum) & matching.max(axis=1).gt(0)
        matching = matching.fillna(pd.Series(self.median, index=self.genes))
        return matching, coverage, valid

    def predict(self, query, expression, banks, *, device="cpu"):
        result = {name: np.zeros(len(query), dtype=float) for name in self.methods}
        anchors = np.zeros(len(query), dtype=float)
        for model in self.models:
            values = model.predict(query, expression, banks, device=device)
            mean = query.compound_id.map(model.adaptive.anchor).to_numpy()
            anchors += mean / len(self.models)
            for recipe in self.methods:
                result[recipe] += model.meta[recipe].predict(
                    values, mean, query.compound_id) / len(self.models)
        if any(not np.isfinite(val).all() for val in result.values()):
            raise ValueError("A supported source sensitivity score is nonfinite")
        return result, anchors
