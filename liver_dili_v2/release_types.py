"""Persistent calibrated DILI ensemble and verified input-only confidence."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import tanimoto


@dataclass
class ReleaseFit:
    components: dict
    graphs: dict
    metadata: object
    calibrator: object
    support_fingerprints: np.ndarray
    mechanism_projection: object
    mechanism_names: list

    def forecast(self, query, features, *, device="cpu", cohort=False):
        x = features.arrays(query)
        required = set(self.metadata.transform.features)
        raw = {name: ranker.raw(x) for name, ranker in self.components.items() if name in required}
        mechanism_required = bool(required.intersection(self.mechanism_names) or "signed_mechanism" in required)
        for name, members in self.graphs.items():
            if name in required or (name == "mechanism_dmpnn" and mechanism_required):
                scores = np.mean([model.raw(query.smiles, device=device) for model in members], axis=0)
                raw[name] = scores[:, 0]
                for i, mechanism in enumerate(members[0].names):
                    raw[mechanism] = scores[:, i+1]
        table = pd.DataFrame(raw, index=query.index)
        if self.mechanism_projection is not None:
            table["signed_mechanism"] = self.mechanism_projection.decision_function(table[self.mechanism_names])
        value = self.metadata.raw(table)
        prob, prior = self.calibrator.probabilities(value, estimate_prior=cohort)
        support = tanimoto(x["base"], self.support_fingerprints, mode="bits").max(axis=1)
        return pd.DataFrame({"raw": value, "probability": prob,
                             "decision": prob >= self.calibrator.threshold,
                             "nearest_tanimoto": support, "prior_input_only": prior}, index=query.index)
