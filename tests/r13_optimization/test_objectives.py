import numpy as np
import pandas as pd

from src.r11_chemical_probe.dataset import GROUP, KEY
from src.r11_chemical_probe.metrics import Contrast
from src.r6_followup_v1.metrics import cond_rmse
from src.r13_optimization.objectives import GainScores
from src.r13_optimization.predict import choose
from src.r13_optimization.score import draw_contrast
from src.r13_optimization.tune import select


def frame():
    return pd.DataFrame({
        GROUP: [1, 1, 2, 3, 4, 1, 2, 3],
        "compound_entity_id": ["x", "y", "z", "u", "v", "x", "z", "u"],
        KEY[0]: list("abcdefgh"), KEY[1]: ["a"] * 5 + ["b"] * 3,
        "plate_id": 1, "dose_uM": 0.5, "time_h": 24,
    })


def test_gain_and_bootstrap_match_existing_rmse_contract():
    rng = np.random.default_rng(0)
    rows = frame()
    truth, anchor, correction = rng.normal(size=(3, len(rows), 11))
    gains = np.array([0, 0.5, 1, 2, 4])
    loss = GainScores(rows, truth, anchor, 3)
    reference = Contrast(rows, 3)
    ctr, absolute = loss.points(correction, gains)
    for i, gain in enumerate(gains):
        pred = anchor + gain * correction
        assert np.isclose(ctr[i], reference.point(pred, truth))
        assert np.isclose(absolute[i], np.average(cond_rmse(pred, truth), weights=reference.all_weight))

    draws = rng.exponential(size=(40, len(reference.group_ids)))
    baseline = reference.bootstrap_values(pred, truth, draws)
    expect = np.sum(baseline * reference.primary_weight * draws[:, reference.inverse], axis=1)
    np.testing.assert_allclose(draw_contrast(reference, pred, truth, draws, reference.inverse), expect, atol=1e-12)


def test_selector_requires_improvement_in_both_inner_folds():
    baseline = dict(contrast_by_fold={"0": 0.1, "1": 0.2})
    rows = []
    for inner, scores in enumerate([[0.09, 0.08], [0.19, 0.21]]):
        for name, score in zip(["stable", "unbalanced"], scores):
            rows.append(dict(inner=inner, name=name, gain=1, contrast=score, absolute=score, contrast_weight=1,
                             absolute_weight=1, specification='{"enabled":true}'))
    picked = select(rows)
    stable = picked["best_nonzero"]
    assert stable["name"] == "stable"
    choices = dict.fromkeys(("mechanism", "kernels", "transfer", "neural"), {
        "gain": 1, "specification": {}, "contrast": 0.145, "contrast_by_fold": {"0": 0.08, "1": 0.21},
    })
    choices["transfer"] = stable
    assert choose(choices, baseline, {"positive_tolerance": 1e-8}) == [("transfer", stable)]
