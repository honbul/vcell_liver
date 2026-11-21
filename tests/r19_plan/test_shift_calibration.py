"""Test the source-only T1–T3 estimators without opening any assay data."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.r19_plan.dev_selection import PhysicalControls, ResponseAxis
from src.r19_plan.shift_calibration import (
    ControlSupportError, EffectDeltas, RegisteredControls, calibrate, fit_diagonal,
    fit_held_source_scalar, fit_source_scalar, refit_composed_scalar, source_holdout,
)


def deltas(values, genes=(1, 2), rows=None, coordinate="matched-log2-well"):
    values = np.array(values, dtype=float)
    axis = ResponseAxis(rows or tuple(("study", f"well{i}") for i in range(len(values))), genes)
    return EffectDeltas(axis, coordinate, values)


def controls(prefix, values, *, per_plate=4, scale="matched-log2-well"):
    wells = tuple((f"{prefix}{i // per_plate}", f"A{i + 1:02d}") for i in range(len(values)))
    physical = PhysicalControls(wells, (1, 2, 3), np.asarray(values, dtype=float))
    return RegisteredControls(physical, wells, scale)


def test_t1_is_the_clipped_pooled_energy_optimum_not_macro_rmse():
    forecast = deltas([[1.], [10.]], genes=(1,))
    observed = deltas([[1.], [0.]], genes=(1,))
    scalar = fit_source_scalar(forecast, observed)
    assert scalar.gain == pytest.approx(1 / 101)
    assert abs(calibrate(forecast, scalar=scalar).values - observed.values).mean() > .5
    assert fit_source_scalar(forecast, replace(observed, values=3 * forecast.values)).gain == 2.
    assert fit_source_scalar(forecast, replace(observed, values=-forecast.values)).gain == 0.
    zero_prediction = replace(forecast, values=np.zeros_like(forecast.values))
    assert fit_source_scalar(zero_prediction, observed).gain == 0.


def test_source_fit_and_query_must_share_coordinate_and_gene_order():
    forecast = deltas([[2., 3.], [1., 4.]])
    reverse = replace(forecast, axis=ResponseAxis(forecast.axis.rows[::-1], forecast.axis.genes))
    with pytest.raises(ValueError, match="row, gene or coordinate"):
        fit_source_scalar(forecast, reverse)
    with pytest.raises(ValueError, match="finite"):
        fit_source_scalar(forecast, replace(forecast, values=np.array([[np.nan, 3.], [1., 4.]])))
    scalar = fit_source_scalar(forecast, forecast)
    with pytest.raises(ValueError, match="effect coordinate"):
        calibrate(replace(forecast, coordinate="absolute"), scalar=scalar)
    reordered = replace(forecast, axis=ResponseAxis(forecast.axis.rows, (2, 1)))
    with pytest.raises(ValueError, match="gene"):
        calibrate(reordered, scalar=scalar)


def test_t2_uses_only_matched_physical_fit_controls_and_respects_clip():
    x = np.linspace(-1., 1., 12)
    source = controls("S", np.column_stack((x, 0 * x, 4 * x)))
    target = controls("T", np.column_stack((4 * x, 0 * x, x)))
    calibration = fit_diagonal(source, target, alpha=1., floor=1e-12, clip=np.log(2))
    np.testing.assert_allclose(calibration.factors, [2., 1., .5])
    expected = deltas([[3., -2., 4.]], genes=(1, 2, 3))
    np.testing.assert_allclose(calibrate(expected, diagonal=calibration).values, [[6., -2., 2.]])
    np.testing.assert_allclose(fit_diagonal(source, target, alpha=0., floor=1., clip=np.log(2)).factors,
                               np.ones(3))
    with pytest.raises(ValueError, match="effect coordinate"):
        calibrate(replace(expected, coordinate="unmatched"), diagonal=calibration)
    with pytest.raises(ValueError, match="assay/aggregation"):
        fit_diagonal(source, replace(target, scale_id="cell-level-pseudobulk"), alpha=1., floor=1., clip=1.)
    with pytest.raises(ValueError, match="registered physical fit wells"):
        fit_diagonal(source, replace(target, fit_wells=target.fit_wells[1:]), alpha=1., floor=1., clip=1.)
    with pytest.raises(ControlSupportError, match="12 target-fit"):
        fit_diagonal(source, controls("T", target.physical.values[:11]), alpha=1., floor=1., clip=1.)
    with pytest.raises(ControlSupportError, match="3 plates"):
        fit_diagonal(source, controls("T", target.physical.values, per_plate=6), alpha=1., floor=1., clip=1.)
    corrupt = replace(target.physical, wells=target.physical.wells[:-1] + (target.physical.wells[0],))
    with pytest.raises(ValueError, match="duplicate"):
        fit_diagonal(source, replace(target, physical=corrupt), alpha=1., floor=1., clip=1.)


def test_t3_refits_the_source_gain_after_per_row_diagonal_transform():
    source = deltas([[1., 2.], [3., 4.]])
    factors = np.array([[2., .5], [.5, 2.]])
    observed = replace(source, values=.75 * source.values * factors)
    assert refit_composed_scalar(source, observed, factors, factor_axis=source.axis).gain == pytest.approx(.75)
    assert fit_source_scalar(source, observed).gain != pytest.approx(.75)
    with pytest.raises(ValueError, match="strictly positive"):
        refit_composed_scalar(source, observed, -factors, factor_axis=source.axis)
    with pytest.raises(ValueError, match="ordered source"):
        refit_composed_scalar(source, observed, factors,
                              factor_axis=ResponseAxis(source.axis.rows[::-1], source.axis.genes))


def test_leave_domain_and_parent_out_requests_only_permitted_source_labels():
    rows = pd.DataFrame({
        "study": ["s"] * 7,
        "inst_id": list("abcdefg"),
        "domain": ["held", "held", "d1", "d1", "d2", "d2", "d3"],
        "parent": ["same", "secondary", "same", "fit1", "same", "fit2", "secondary"],
    })
    registry = tuple(rows[["study", "inst_id"]].itertuples(index=False, name=None))
    forecast = deltas(np.ones((7, 2)), rows=registry)
    called = []

    def allowed_reader(ids):
        assert ids == (("s", "d"), ("s", "f"))
        called.append(ids)
        return deltas(.75 * np.ones((2, 2)), rows=ids)

    scalar, partition = fit_held_source_scalar(forecast, rows, "held", allowed_reader)
    assert scalar.gain == pytest.approx(.75)
    assert len(called) == 1
    np.testing.assert_array_equal(partition.held, [0, 1])
    np.testing.assert_array_equal(partition.fit, [3, 5])
    one_held = source_holdout(rows, "held", outer_parents={"same"})
    np.testing.assert_array_equal(one_held.held, [0])
    np.testing.assert_array_equal(one_held.fit, [3, 5, 6])
    with pytest.raises(ValueError, match="fewer than two"):
        source_holdout(rows.iloc[:5], "held")
    with pytest.raises(ValueError, match="ordered OOF rows"):
        fit_held_source_scalar(forecast, rows.iloc[::-1], "held", allowed_reader)
    assert len(called) == 1
