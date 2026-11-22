"""Check the exact score vehicle and direction in the HepaRG technical replay."""

import numpy as np
import pandas as pd
import pytest

from src.r19_plan.t4_heparg_compare import cosine, hep_truth


def test_heparg_reconstructs_same_row_and_subtracts_only_selected_vehicle():
    rows = pd.DataFrame({"library": ["l1", "l2"], "plate": ["p1", "p1"]}).set_index("library", drop=False)
    vehicles = pd.DataFrame({"plate": ["p1", "p1", "p1"], "role": ["A", "B", "C"]})
    arrays = {"delta": np.array([[1, 0], [3, 5]]), "control": np.array([[2, 4], [2, 4]]),
              "vehicles": np.array([[1, 2], [3, 5], [2, 5]])}
    asset = (rows, vehicles, arrays, np.array([1, 0]))
    target = pd.DataFrame({"library": ["l2"], "plate": ["p1"]})
    np.testing.assert_array_equal(hep_truth(asset, target, "B"), [[4, 2]])
    np.testing.assert_array_equal(hep_truth(asset, target, "C"), [[4, 3]])
    with pytest.raises(ValueError, match="physical|coordinates"):
        hep_truth(asset, target.assign(plate="p2"), "B")


def test_cosine_uses_direction_and_keeps_zero_contrasts_out_of_direction_average():
    truth = np.array([[1., 2.], [-1., -2.], [0., 0.]])
    actual = cosine(truth, np.array([3., 6.]))
    np.testing.assert_allclose(actual[:2], [1, -1])
    assert np.isnan(actual[2])
