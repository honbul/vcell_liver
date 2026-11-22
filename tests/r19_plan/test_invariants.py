import numpy as np
import pandas as pd
import pytest

from src.r11_chemical_probe.dataset import ENTITY, GROUP, KEY, POOL
from src.r11_chemical_probe.metrics import Contrast, correlations
from src.r13_optimization.objectives import GainScores, center
from src.r19_plan.contracts import assert_dev_allowed, validate_forecast


@pytest.fixture
def example():
    values = {column: ["pool_1"] * 3 + ["pool_2"] * 3 for column in POOL}
    values[KEY[0]] = list(range(6))
    values[GROUP] = values[ENTITY] = list("ABCABC")
    rows = pd.DataFrame(values)
    truth = np.array([[1., 2., 0.], [-3., 1., 0.], [2., -3., 1.],
                      [2., 2., 0.], [-1., 1., 1.], [3., -3., 1.]])
    forecast = truth + np.array([[.2, .3, .5], [.1, -.2, .1], [-.2, .1, -.3],
                                [-.2, .4, .1], [.1, -.1, .5], [.2, -.4, .3]])
    return rows, forecast, truth


def test_positive_scalar_preserves_cosine_and_pool_rank(example):
    rows, prediction, truth = example
    metric = Contrast(rows, 3)
    expected = correlations(prediction, truth)[0]
    retrieval = metric.retrieval(prediction, truth)
    for gain in (.125, .5, 2.):
        np.testing.assert_allclose(correlations(gain * prediction, truth)[0], expected)
        for actual, original in zip(metric.retrieval(gain * prediction, truth), retrieval, strict=True):
            np.testing.assert_array_equal(actual, original)


def test_zero_scalar_yields_ties_and_undefined_cosine(example):
    rows, prediction, truth = example
    zero = np.zeros_like(prediction)
    assert np.isnan(correlations(zero, truth)[0]).all()
    rank, top, support, same = Contrast(rows, 3).retrieval(zero, truth)
    np.testing.assert_allclose(rank, .5)
    np.testing.assert_allclose(top, 1 / 3)
    np.testing.assert_array_equal(support, 3)
    assert same.all()


def test_common_pool_shift_preserves_contrast_including_fixed_diagonal(example):
    rows, prediction, truth = example
    metric = Contrast(rows, 3)
    shifted = prediction.copy()
    diagonal = np.array([.5, 1., 3.])
    for idx, shift in zip(metric.pools, ([1., -2., 3.], [-4., 3., -1.]), strict=True):
        shifted[idx] -= 1.25 * np.asarray(shift)
    np.testing.assert_allclose(center(metric, shifted), center(metric, prediction), atol=1e-12)
    np.testing.assert_allclose(center(metric, shifted * diagonal), center(metric, prediction) * diagonal, atol=1e-12)
    assert metric.point(shifted, truth) == pytest.approx(metric.point(prediction, truth))
    assert metric.points(shifted, truth)[1] != pytest.approx(metric.points(prediction, truth)[1])


def test_rmse_r2_gain_identities_and_zero_perfect_predictions(example):
    rows, prediction, truth = example
    metric = Contrast(rows, 3)
    gains = np.array([0., .25, .5, 1., 2.])
    gain_score = GainScores(rows, truth, np.zeros_like(truth), 3)
    actual_ctr, actual_abs = gain_score.points(prediction, gains)
    normalized_energy = np.mean(truth ** 2)
    predicted_energy = np.mean(prediction ** 2)
    cross = np.mean(prediction * truth)
    for i, gain in enumerate(gains):
        ctr, absolute, *_ = metric.points(gain * prediction, truth)
        assert actual_ctr[i] == pytest.approx(ctr)
        assert actual_abs[i] == pytest.approx(absolute)
        mse = np.mean((gain * prediction - truth) ** 2)
        assert mse == pytest.approx(normalized_energy - 2 * gain * cross + gain ** 2 * predicted_energy)
        assert 1 - mse / normalized_energy == pytest.approx(
            (2 * gain * cross - gain ** 2 * predicted_energy) / normalized_energy)
    assert metric.point(truth, truth) == 0
    assert metric.point(np.zeros_like(truth), truth) > 0
    assert metric.point(truth[[2, 0, 1, 5, 3, 4]], truth) > 0


def test_prediction_axis_requires_shape_finite_values_and_gene_order(example):
    _, prediction, truth = example
    genes = ["1", "2", "3"]
    validate_forecast(prediction, truth, genes, genes)
    bad = prediction.copy()
    bad[0, 0] = np.nan
    for values, got in [(bad, genes), (prediction[:, 0], genes), (prediction, genes[::-1]),
                        (prediction, ["1", "1", "3"])]:
        with pytest.raises(ValueError):
            validate_forecast(values, truth, got, genes)


def test_dev_guard_blocks_well_signature_coordinate_and_B_exception():
    fields = ["sig_id", "inst_id", "rna_plate", "rna_well", "det_plate", "det_well"]
    reserved = pd.DataFrame([["SIG", "well", "rna1", "A01", "det1", "B01"]], columns=fields)
    exceptions = pd.DataFrame([["source", "donor", "rna2", "A01", "", ""]], columns=fields)
    clean = ["new", "id", "rna3", "A03", "", ""]
    assert_dev_allowed(pd.DataFrame([clean], columns=fields), reserved, exceptions)
    for row in [["sig", "alias", "", "", "", ""], ["", "WELL", "", "", "", ""],
                ["", "alias", "", "", "RNA1", "a1"], ["", "donor", "", "", "", ""],
                ["", "", "", "", "", ""]]:
        with pytest.raises(ValueError):
            assert_dev_allowed(pd.DataFrame([row], columns=fields), reserved, exceptions)
