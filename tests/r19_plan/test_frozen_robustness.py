"""Check that crossed reweighting preserves an identical relative advantage."""

import numpy as np
import pandas as pd

from src.r19_plan.frozen_robustness import crossed_weights


def test_crossed_bootstrap_preserves_constant_relative_error_reduction():
    observations = pd.DataFrame({"parent": ["a", "a", "b", "b", "c"],
                                 "rna_plate": ["p", "q", "q", "q", "r"]})
    original = np.array([2.0, 10.0, 4.0, 12.0, 39.0])
    decreased = 0.64*original
    result = crossed_weights(observations, 10, original, decreased, original, draws=64)
    np.testing.assert_allclose(result, [[.2,.2],[.36,.36]], atol=1e-14)
