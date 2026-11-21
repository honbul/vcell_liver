import numpy as np
import pandas as pd

from src.r11_chemical_probe.dataset import ENTITY, GROUP, LINE, group_weights
from src.r11_chemical_probe.metrics import Contrast
from src.r19_plan.hcc_scaffold import LINE_ID
from src.r19_plan.snu423_donor import expand, gain, ridge4, shuffle


def test_fixed_operator_and_scalar_keep_missing_donor_at_zero():
    fit = pd.DataFrame({GROUP: list(range(7)), ENTITY: [f"e{i}" for i in range(7)], LINE: LINE_ID,
                        "physical_well_id": [f"p{i}" for i in range(7)], "plate_id": "plate", "dose_uM": 5, "time_h": 24})
    query = pd.DataFrame({GROUP: [7, 8]})
    rng = np.random.default_rng(710)
    training = pd.DataFrame(rng.normal(size=(6, 5)))
    test = pd.DataFrame(rng.normal(size=(1, 5)), index=[7])
    y = rng.normal(size=(len(fit), 5))
    model = ridge4(fit, query, training, test, y, Contrast(fit, 3), group_weights(fit))
    assert np.isfinite(model).all() and np.any(model[0] != 0)
    np.testing.assert_array_equal(model[1], np.zeros(5))
    np.testing.assert_array_equal(expand(query, test)[1], np.zeros(5))
    np.testing.assert_allclose(gain(np.ones((3, 5)), np.full((3, 5), .4), np.ones(3)), .4)


def test_donor_shuffle_keeps_support_and_never_copies_own_parent():
    data = pd.DataFrame(np.arange(12).reshape(6, 2), index=np.arange(10, 16))
    switched = shuffle(data, np.random.default_rng(5))
    assert switched.index.equals(data.index)
    assert set(switched[0]) == set(data[0])
    assert not switched[0].eq(data[0]).any()
