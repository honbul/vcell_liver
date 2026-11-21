import numpy as np
import pandas as pd
import pytest

from src.r19_plan.i_operational import PRIMARY, SUPPLEMENTAL, operational_points
from src.r19_plan.i_score import BOOTSTRAPS, ViewCoverage


def synthetic_score(folds):
    n = len(folds)
    physical = pd.DataFrame(dict(
        study="synthetic", inst_id=[f"sample{i}" for i in range(n)], rna_plate="plate",
        rna_well=[f"A{i + 1:02d}" for i in range(n)], parent=[f"parent{i:03d}" for i in range(n)],
        fold=folds,
    ))
    rows = physical.loc[physical.index.repeat(3)].assign(role=list("ABC") * n).reset_index(drop=True)
    truth = np.full((len(rows), 2), 10.0)
    counts = {arm: ViewCoverage(1, np.ones(len(rows), dtype=int)) for arm in (*PRIMARY, *SUPPLEMENTAL)}
    forecasts = {arm: truth - 1.0 for arm in (*PRIMARY, *SUPPLEMENTAL)}
    forecasts.update(B_R17=truth - 2.0, I0=truth - 3.0)
    return rows, truth, forecasts, counts


def test_relative_rmse_uses_each_paired_reference_draw_and_reports_fold_directions():
    rows, truth, forecasts, coverage = synthetic_score([0, 1, 2, 2])
    reference = np.array([1.0, 2.0, 4.0, 6.0])
    candidate = np.array([0.5, 3.0, 4.5, 6.0])
    forecasts["B_R17"] = truth - reference.repeat(3)[:, None]
    forecasts["I1"] = truth - candidate.repeat(3)[:, None]
    coverage["I1"] = ViewCoverage(1, np.array([1, 1, 1, 0]).repeat(3))

    scores = operational_points(rows, truth, forecasts, coverage)
    assert len(scores) == 40
    got = scores.loc[(scores.arm == "I1") & (scores.reference == "B_R17")].set_index("fold")
    ref, actual = reference[:3], candidate[:3]
    draw = np.random.default_rng(150104).integers(len(ref), size=(BOOTSTRAPS, len(ref)))
    relative = (ref[draw] - actual[draw]).sum(axis=1) / ref[draw].sum(axis=1)
    relative_bounds = np.quantile(relative, [.005, .995])
    assert got.loc["all", "relative_rmse"] == pytest.approx(-1 / 7)
    np.testing.assert_allclose(got.loc["all", ["relative_rmse_low99", "relative_rmse_high99"]].astype(float),
                               relative_bounds)
    assert got.loc["all", "parents"] == 3
    assert got.loc["all", "rows"] == 9
    assert got.loc["all", "valid_bootstrap_draws"] == BOOTSTRAPS
    assert got.loc["all", "delta_r2"] == pytest.approx(((ref ** 2 - actual ** 2).sum()) / 300)
    assert got.loc[[0, 1, 2], "jointly_improves"].tolist() == [True, False, False]


def test_support_is_the_same_for_both_references_and_checks_every_fold():
    rows, truth, forecasts, coverage = synthetic_score([0] * 20 + [1] * 20 + [2] * 61)
    counts = coverage["I1"].observed_counts.copy()
    counts[:3] = 0
    coverage["I1"] = ViewCoverage(1, counts)

    scores = operational_points(rows, truth, forecasts, coverage)
    pooled = scores.loc[scores.fold == "all"].set_index(["arm", "reference"])
    for reference in ("B_R17", "I0"):
        assert pooled.loc[("I1", reference), "parents"] == 100
        assert not pooled.loc[("I1", reference), "eligible"]
        assert pooled.loc[("I2", reference), "eligible"]
    assert set(scores.loc[scores.family == "supplemental", "arm"]) == {"I3_all"}


def test_zero_reference_rmse_draws_leave_interval_undefined():
    rows, truth, forecasts, coverage = synthetic_score([0, 1, 2])
    forecasts["B_R17"] = truth - np.array([0.0, 0.0, 1.0]).repeat(3)[:, None]
    scores = operational_points(rows, truth, forecasts, coverage)
    effect = scores.loc[(scores.arm == "I1") & (scores.reference == "B_R17") & (scores.fold == "all")].iloc[0]
    assert 0 < effect.valid_bootstrap_draws < BOOTSTRAPS
    assert np.isnan(effect.relative_rmse_low99)
    assert np.isnan(effect.relative_rmse_high99)
