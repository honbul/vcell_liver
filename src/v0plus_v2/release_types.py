"""Persistent inference classes for the frozen sensitivity ensemble."""

from dataclasses import dataclass

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
