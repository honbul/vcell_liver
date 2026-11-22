import math

import numpy as np

from vc_build.wong_assay_bridge import fit_transfer, paired_permutation


def test_calibration_recovers_signed_gain_and_constant_drug():
    x = np.array([[1, 4, 7], [2, 4, 10], [4, 4, 12]], dtype=float)
    offsets = np.array([2, 7, -3])
    y = offsets - 0.5 * x
    query = np.array([[10, 4, 5], [0, 4, 8]], dtype=float)
    values, slope = fit_transfer(x, y, query)
    assert slope == -0.5
    np.testing.assert_allclose(values[-1], offsets - 0.5 * query)
    np.testing.assert_allclose(values[2], values[-1])


def test_permutation_uses_all_eight_patients_in_one_block():
    x = np.random.default_rng(712).normal(size=(8, 20))
    result = paired_permutation(x, x, repeats=40)
    for row in result.values():
        np.testing.assert_allclose(row["matched"], 1)
        np.testing.assert_allclose(row["permutation_mean"], 0, atol=1e-13)
        assert row["one_sided_exact_p"] == 1 / math.factorial(8)
