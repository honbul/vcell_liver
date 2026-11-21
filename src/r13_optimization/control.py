"""Pure baselines and frozen R12 anchor for a matched follow-up."""

import numpy as np


def candidates():
    return [{"name": "no-change", "pool": None}] + [
        {"name": f"pool-{k}", "pool": k} for k in [0, 2, 8, 32]
    ]


def predict(fit, spec):
    if spec["pool"] is None:
        return np.zeros_like(fit.offset)
    return fit.means.predict(fit.query.rows, spec["pool"])[0]
