import numpy as np
import pandas as pd
import pytest

from src.r19_plan.x_b import matched_donors


def test_matched_donors_exclude_self_and_hepatic_response_from_anchor():
    class Bank:
        fit = "A"

        def response(self, rows, role):
            assert role == self.fit
            return rows[["response"]].to_numpy(dtype=np.float32)

    source = pd.DataFrame({"parent": ["one"] * 4, "cell_id": ["A549", "A549", "A375", "HEPG2"],
                           "response": [4., 8., 2., 100.]})
    wanted = pd.DataFrame({"parent": ["one"] * 3, "cell_id": ["A549", "A375", "HEPG2"]})
    np.testing.assert_array_equal(matched_donors(Bank(), source, wanted).ravel(), [2., 6., 4.])
    with pytest.raises(ValueError, match="lacks a measured B donor"):
        matched_donors(Bank(), source, wanted.assign(parent="missing"))
