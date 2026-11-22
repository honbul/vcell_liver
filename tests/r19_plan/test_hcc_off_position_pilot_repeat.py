import numpy as np
import pandas as pd
import pytest

from src.r19_plan.hcc_off_position_pilot_repeat import directed_pairs, evaluate
from src.r19_plan.jhh5_off_position import CALIBRATED
from src.r19_plan.jhh7_off_position_native import REFERENCES


def test_noiseless_repeats_recover_gain_and_do_not_count_roles_as_replicates():
    parents = np.repeat([f"P{i:03d}" for i in range(1, 41)], 2)
    records = pd.DataFrame(dict(cell="SNU-423", canonical_parent=parents, parent_slot=parents,
        plate_id=["p1", "p2"]*40, culture_id=["c1", "c2"]*40, well=["A01", "B02"]*40,
        independent_start_id=["s1", "s2"]*40, batch_id=["b1", "b2"]*40))
    values = np.arange(1, 81, dtype=float).reshape(-1, 2)
    truth = np.tile(np.repeat(values, 2, axis=0)[None], (3, 1, 1))
    predictions = {arm: truth*.6 for arm in (CALIBRATED, *REFERENCES)}
    results = evaluate(records, truth, predictions, np.ones(80, dtype=bool), draws=25)
    assert tuple(results["support"].iloc[0][["wells", "undirected_pairs", "roles"]]) == (80, 40, 3)
    np.testing.assert_allclose(results["fitted_gains"].gain, 1.)
    agreement = results["repeat_agreement"].set_index("metric").estimate
    np.testing.assert_allclose(agreement.loc["repeat_cosine"], 1.)
    np.testing.assert_allclose(agreement.loc["repeat_disagreement_rms"], 0.)


def test_repeat_requires_distinct_positions_even_with_distinct_cultures():
    rows = pd.DataFrame(dict(cell=["SNU-423"]*2, canonical_parent=["P001"]*2,
        plate_id=["p1", "p2"], culture_id=["c1", "c2"], independent_start_id=["s1", "s2"],
        well=["A01", "A01"]))
    with pytest.raises(ValueError, match="physical and culture repeats"):
        directed_pairs(rows)
