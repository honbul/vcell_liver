"""Exercise I-track support and paired outcome statistics with synthetic wells."""

import numpy as np
import pandas as pd
import pytest

from src.r19_plan.i_score import BOOTSTRAPS, ViewCoverage, score_i_track


def synthetic():
    rows = pd.DataFrame({
        "study": "S",
        "inst_id": ["a1", "a2", "b1", "c1", "d1"],
        "rna_plate": ["P1", "P2", "P1", "P3", "P2"],
        "rna_well": ["A01", "B01", "A02", "A01", "B02"],
        "parent": ["a", "a", "b", "c", "d"],
        "fold": [0, 0, 1, 2, 1],
        "known_scaffold": [True, True, False, False, False],
    })
    rows = rows.loc[rows.index.repeat(2)].reset_index(drop=True)
    rows["role"] = ["A", "B"] * 5
    truth = np.repeat([1., 3., 2., 5., 3.], 2)[:, None]
    forecasts = {"I0": np.zeros_like(truth), "B_R17": np.full_like(truth, 0.25)}
    coverage = {}
    config = {
        "I1": (1, [1, 1, 1, 0, 1]),
        "I2": (1, [1, 1, 0, 0, 1]),
        "I3": (2, [2, 2, 1, 0, 2]),
        "I4": (1, [0, 0, 1, 0, 1]),
        "I3_mask_only": (2, [2, 2, 1, 0, 2]),
        "I3_parent_shuffle": (2, [2, 2, 1, 0, 2]),
    }
    for arm, (total, values) in config.items():
        count = np.repeat(values, 2)
        coverage[arm] = ViewCoverage(total, count)
        multiplier = 0.25 if arm.endswith(("_mask_only", "_parent_shuffle")) else 1.
        absent = np.ones_like(count) if arm.endswith("_mask_only") else count > 0
        forecasts[arm] = (multiplier * truth) * absent[:, None]
    return rows, truth, forecasts, coverage


def row(table, scope, arm, reference, metric="rmse"):
    match = table.loc[
        table.scope.eq(scope) & table.arm.eq(arm) & table.reference.eq(reference) & table.metric.eq(metric)
    ]
    assert len(match) == 1
    return match.iloc[0]


def test_scores_matched_support_and_full_fallback_on_different_denominators():
    rows, truth, forecasts, coverage = synthetic()
    held = {fold: set(rows.loc[rows.fold.ne(fold), "parent"]) for fold in (0, 1, 2)}
    scores = score_i_track(rows, truth, forecasts, coverage, fit_parents_by_fold=held)

    matched = row(scores["same_support"], "complete_shared", "I3", "I0")
    assert (matched.parents, matched.rows, matched.physical_wells, matched.plates, matched.roles) == (2, 6, 3, 2, 2)
    assert matched.arm_value == 0
    assert matched.reference_value == pytest.approx(2.5)
    assert matched.low99 == pytest.approx(-3.)
    assert matched.high99 == pytest.approx(-2.)
    assert not matched.eligible
    strong_reference = row(scores["same_support"], "complete_shared", "I3", "B_R17")
    assert strong_reference.reference_value == pytest.approx(2.25)
    common_i4 = row(scores["same_support"], "complete_shared", "I4", "I3")
    assert (common_i4.parents, common_i4.physical_wells) == (1, 1)

    full = row(scores["full_fallback"], "all", "I3", "I0")
    assert (full.parents, full.rows, full.physical_wells, full.plates) == (4, 10, 5, 3)
    assert (full.arm_value, full.reference_value) == pytest.approx((1.25, 3.))
    assert row(scores["full_fallback"], "all", "I3", "I0", "r2_pooled").arm_value == pytest.approx(23 / 48)

    strata = scores["strata"]
    assert row(strata, "views_absent", "I3", "I0").difference == 0
    assert row(strata, "views_partial", "I3", "I0").parents == 1
    assert row(strata, "views_complete", "I3", "I0").parents == 2
    assert row(strata, "views_partial", "I2", "I0").parents == 0
    assert row(strata, "unseen_parent", "I3", "I0").parents == 4
    assert row(strata, "scaffold_seen_in_fit", "I3", "I0").parents == 1
    assert row(strata, "scaffold_unseen_in_fit", "I3", "I0").parents == 3
    sensitivity = scores["plate_sensitivity"]
    held = sensitivity.loc[sensitivity.scope.eq("all") & sensitivity.arm.eq("I3")
                             & sensitivity.reference.eq("I0") & sensitivity.removed_rna_plate.eq("P3")]
    assert len(held) == 1
    assert held.iloc[0].remaining_parents == 3
    assert held.iloc[0].delta_rmse == pytest.approx(-7 / 3)

    controls = scores["controls"]
    masked = row(controls, "all", "I3_mask_only", "I0")
    assert masked.parents == 4
    assert masked.difference < 0
    shuffled = row(controls, "complete_shared", "I3", "I3_parent_shuffle")
    assert shuffled.parents == 2
    assert shuffled.difference < 0


def test_kernel_only_uses_its_own_matched_i0_reference():
    rows, truth, forecasts, coverage = synthetic()
    diagnostic = {key: array + 0.5 for key, array in forecasts.items() if key != "B_R17"}
    scores = score_i_track(rows, truth, forecasts, coverage, kernel_only=diagnostic)
    kernel_i3 = row(scores["kernel_only"], "complete_shared", "I3", "I0")
    assert (kernel_i3.parents, kernel_i3.arm_value, kernel_i3.reference_value) == pytest.approx((2, .5, 2.))
    assert "B_R17" not in scores["kernel_only"].reference.unique()


def test_requires_total_and_per_fold_support_before_claiming_eligibility():
    n = 120
    rows = pd.DataFrame(dict(
        study="S", parent=[f"p{i}" for i in range(n)], inst_id=[f"i{i}" for i in range(n)],
        rna_plate="P", rna_well=[f"A{i + 1:02d}" for i in range(n)], role="A",
        fold=np.repeat([0, 1, 2], [20, 50, 50]),
    ))
    truth = np.ones((n, 1))
    counts = np.r_[0, np.ones(n - 1, dtype=int)]
    forecasts = {"B_R17": truth / 2, "I0": truth * 0, "I1": truth * counts[:, None]}
    tables = score_i_track(rows, truth, forecasts, {"I1": ViewCoverage(1, counts)})
    complete = row(tables["same_support"], "complete_shared", "I1", "I0")
    whole = row(tables["full_fallback"], "all", "I1", "I0")
    assert (complete.parents, complete.fold0_parents, complete.valid_bootstrap_draws) == (119, 19, BOOTSTRAPS)
    assert not complete.eligible
    assert whole.eligible


def test_offers_rmse_but_suppresses_r2_intervals_for_undefined_resamples():
    rows, truth, forecasts, coverage = synthetic()
    truth = np.where(rows.parent.eq("a").to_numpy()[:, None], truth, 0)
    tables = score_i_track(rows, truth, forecasts, coverage)
    partial_energy = row(tables["full_fallback"], "all", "I3", "I0", "r2_pooled")
    assert np.isfinite(partial_energy.difference)
    assert 0 < partial_energy.valid_bootstrap_draws < BOOTSTRAPS
    assert np.isnan(partial_energy.low99)
    assert np.isnan(partial_energy.high99)
    assert row(tables["full_fallback"], "all", "I3", "I0").valid_bootstrap_draws == BOOTSTRAPS

    all_zero = score_i_track(rows, np.zeros_like(truth), forecasts, coverage)
    zero_r2 = row(all_zero["full_fallback"], "all", "I3", "I0", "r2_pooled")
    assert zero_r2.valid_bootstrap_draws == 0
    assert np.isnan(zero_r2.arm_value)
    assert np.isnan(zero_r2.reference_value)


def test_rejects_changed_fallback_role_or_view_lineage():
    rows, truth, forecasts, coverage = synthetic()
    wrong_fallback = forecasts["I3"].copy()
    wrong_fallback[6] = 1.
    with pytest.raises(ValueError, match="frozen I0"):
        score_i_track(rows, truth, dict(forecasts, I3=wrong_fallback), coverage)
    other_control_counts = ViewCoverage(2, np.where(coverage["I3"].observed_counts > 0, 2, 0))
    with pytest.raises(ValueError, match="changed the view availability"):
        score_i_track(rows, truth, forecasts, dict(coverage, I3_mask_only=other_control_counts))
    with pytest.raises(ValueError, match="crossed into the fit registry"):
        score_i_track(rows, truth, forecasts, coverage, fit_parents_by_fold={0: {"a"}, 1: set(), 2: set()})
    rows.loc[1, "inst_id"] = "well_alias"
    with pytest.raises(ValueError, match="well identifiers"):
        score_i_track(rows, truth, forecasts, coverage)
