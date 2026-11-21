"""Exercise I5's held-parent endpoints and modality fallback on synthetic arrays."""

import numpy as np
import pandas as pd
import pytest

from src.r19_plan.i5_score import score_i5
from src.r19_plan.i_score import BOOTSTRAPS


def synthetic():
    rows = pd.DataFrame({
        "study": "S",
        "inst_id": ["a1", "a2", "b1", "c1", "d1", "e1"],
        "rna_plate": ["P1", "P2", "P1", "P2", "P3", "P3"],
        "rna_well": ["A01", "A01", "A02", "A02", "A01", "A02"],
        "parent": ["a", "a", "b", "c", "d", "e"],
        "fold": [0, 0, 1, 2, 0, 2],
    })
    rows = rows.loc[rows.index.repeat(3)].reset_index(drop=True)
    rows["role"] = ["A", "B", "C"] * 6
    truth = np.repeat([1., 3., 2., 7., 4., 4.], 3)[:, None]
    views = {
        "I2": np.repeat([True, True, False, False, True, True], 3),
        "I4": np.repeat([True, True, True, False, False, True], 3),
    }
    measured = truth * (views["I2"] | views["I4"])[:, None]
    forecasts = dict(
        B_R17=truth * .5, I0=np.zeros_like(truth), I5_kernel=measured * .25, I5_neural=measured,
        I5_neural_mask_only=measured * .1, I5_neural_parent_shuffle=measured * .2,
    )
    return rows, truth, forecasts, views


def comparison(table, scope, reference, metric="rmse", arm="I5_neural"):
    found = table.loc[
        table.scope.eq(scope) & table.arm.eq(arm) & table.reference.eq(reference) & table.metric.eq(metric)
    ]
    assert len(found) == 1
    return found.iloc[0]


def test_paired_metrics_cluster_roles_by_parent_on_registered_support():
    rows, truth, forecasts, views = synthetic()
    fit = {fold: set(rows.loc[rows.fold.ne(fold), "parent"]) for fold in range(3)}
    scores = score_i5(rows, truth, forecasts, views, fit_parents_by_fold=fit,
                      expected_support={"both_phenotypes": 2, "prism_available": 3, "whole_fallback": 5})
    assert len(scores) == 60

    matched = comparison(scores, "both_phenotypes", "I5_kernel")
    assert (matched.parents, matched.rows, matched.physical_wells, matched.roles) == (2, 9, 3, 3)
    assert (matched.fold0_parents, matched.fold1_parents, matched.fold2_parents) == (1, 0, 1)
    assert (matched.arm_value, matched.reference_value, matched.difference) == pytest.approx((0, 2.25, -2.25))
    assert (matched.low99, matched.high99) == pytest.approx((-3.0, -1.5))
    assert not matched.eligible
    assert matched.valid_bootstrap_draws == BOOTSTRAPS
    assert comparison(scores, "both_phenotypes", "I5_kernel", "r2_pooled").difference == pytest.approx(.5625)

    prism = comparison(scores, "prism_available", "B_R17")
    assert (prism.parents, prism.physical_wells, prism.rows) == (3, 4, 12)
    whole = comparison(scores, "whole_fallback", "B_R17", arm="I5_kernel")
    assert (whole.parents, whole.physical_wells, whole.rows) == (5, 6, 18)
    assert comparison(scores, "absent", "I5_kernel").difference == 0
    assert comparison(scores, "i2_only", "I5_neural_parent_shuffle").parents == 1


def test_validates_neural_fallback_and_the_structure_only_path():
    rows, truth, forecasts, views = synthetic()
    absent = ~(views["I2"] | views["I4"])
    forecasts["I5_neural_mask_only"][absent] += 1
    with pytest.raises(ValueError, match="I5_neural_mask_only must use I5_kernel"):
        score_i5(rows, truth, forecasts, views)
    forecasts["I5_kernel"][absent] += 1
    with pytest.raises(ValueError, match="I5_kernel must use I0"):
        score_i5(rows, truth, forecasts, views)


def test_rejects_wrong_support_view_lineage_or_missing_comparator():
    rows, truth, forecasts, views = synthetic()
    with pytest.raises(ValueError, match="registered count: prism_available"):
        score_i5(rows, truth, forecasts, views, expected_support={"prism_available": 4})
    wrong = views["I2"].copy()
    wrong[0] = False
    with pytest.raises(ValueError, match="I2 availability.*constant by parent"):
        score_i5(rows, truth, forecasts, dict(views, I2=wrong))
    del forecasts["B_R17"]
    with pytest.raises(ValueError, match="registered candidate and reference"):
        score_i5(rows, truth, forecasts, views)


def test_retains_empty_scope_and_undefined_r2():
    rows, truth, forecasts, views = synthetic()
    views["I2"] &= ~views["I4"]
    all_zero = score_i5(rows, np.zeros_like(truth), forecasts, views)
    empty = comparison(all_zero, "both_phenotypes", "I5_kernel")
    assert empty.parents == 0 and empty.valid_bootstrap_draws == 0
    assert np.isnan(empty.difference)
    undefined_r2 = comparison(all_zero, "whole_fallback", "I5_kernel", "r2_pooled")
    assert undefined_r2.valid_bootstrap_draws == 0 and np.isnan(undefined_r2.high99)
