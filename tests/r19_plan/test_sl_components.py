from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from src.r11_chemical_probe.dataset import ENTITY, GROUP, KEY, LINE
from src.r12_multiaxis.data import Design
from src.r12_multiaxis.network import ResidualHead, fit_network
from src.r19_plan.sl_components import exact_pools, fit_variant, directional, huber, matched_contrast, separate_magnitude
from src.r19_plan import sl_predict


def design_rows():
    return pd.DataFrame({KEY[0]: list("abcd"), GROUP: ["p", "p", "q", "r"], ENTITY: ["p", "p", "q", "r"],
                         "plate_id": "P", LINE: "HEPG2", "dose_uM": 10., "time_h": 6., "inst_id": list("abcd")})


def test_l1_pair_term_uses_parent_means_and_measured_dose():
    rows = design_rows()
    exact = exact_pools(rows, pd.Series([10., 10., 10., 10.001], index=list("abcd")))
    y = torch.tensor([[2.], [0.], [3.], [8.]])
    value = matched_contrast(.1)(exact)(y, torch.zeros_like(y))
    torch.testing.assert_close(value, torch.tensor([4.4, .4, 9.4, 64.]))
    with pytest.raises(ValueError, match="measured µM"):
        exact_pools(rows, pd.Series([10.], index=["a"]))


def test_l2_keeps_exact_zero_target_finite_and_l3_matches_small_error_mse():
    y = torch.tensor([[0., 0.], [1., 0.], [1., 0.]])
    prediction = torch.tensor([[0., 0.], [.9, 0.], [-.9, 0.]], requires_grad=True)
    objective, floor = directional(y.numpy(), .1)
    result = objective(None)(prediction, y)
    assert floor == 1 and result[0] == 0 and result[1] < result[2]
    result.sum().backward()
    assert torch.isfinite(prediction.grad).all()
    got = huber(.5)(None)(prediction.detach(), y)
    torch.testing.assert_close(got, torch.tensor([0., .005, .825]))


def test_s2_keeps_baseline_backbone_and_one_magnitude_head():
    model = ResidualHead([2, 2], 8, "film", 0., None)
    old = dict(model.named_parameters())
    out = separate_magnitude(model)
    assert out.drug.weight is old["drug.weight"] and out.decoder.vector.weight is old["decoder.weight"]
    assert sum(x.numel() for x in out.parameters()) - sum(x.numel() for x in old.values()) == 129
    y = out(torch.zeros((4, 2)), torch.zeros((4, 2)))
    assert y.shape == (4, 8) and torch.isfinite(y).all()
    assert (torch.linalg.vector_norm(y, dim=1) <= .1 * np.sqrt(8.)).all()


def test_blind_checkpoint_alignment_and_default_training_equivalence():
    rows = design_rows()
    design = Design(train=(np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)),
                    query=(np.eye(4, dtype=np.float32)[:2], np.eye(4, dtype=np.float32)[:2]))
    fit = SimpleNamespace(train=SimpleNamespace(rows=rows), query=SimpleNamespace(treated=None),
                          cfg=dict(dropout=.1, weight_decay=.01, batch_rows=8), offset=np.zeros((2, 2)),
                          residual=np.ones((4, 2), dtype=np.float32), weight=np.ones(4), scale=1., panel=[1, 2],
                          inputs=lambda kind, rank=None: design)
    expected, _, _, _ = fit_network(fit, design, "film", "L0", .001, 42, "cpu", 1, patience=False)
    forecasts, _ = fit_variant(fit, "S0", None, .001, (1,), rows)
    np.testing.assert_array_equal(forecasts[1], expected["contrast"][2])
    with pytest.raises(ValueError, match="order or group"):
        fit_variant(fit, "S0", None, .001, (1,), rows.iloc[::-1])
    fit.query.treated = np.ones((2, 2))
    with pytest.raises(ValueError, match="fit-only"):
        fit_variant(fit, "S0", None, .001, (1,), rows)


def test_variant_jointly_selects_kernel_and_neural_gains_before_refitting(monkeypatch):
    calls = []

    def checkpoints(fit, strategy, value, lr, epochs, pools, *, device):
        calls.append((fit, strategy, lr, epochs))
        amplitude = 8. if fit == "outer" else (2. if lr == .0003 else 4.)
        return {epoch: np.full((2, 2), amplitude) for epoch in epochs}, {}

    monkeypatch.setattr(sl_predict, "fit_variant", checkpoints)
    rows = pd.DataFrame({"parent": ["p", "q"]})
    zeros = np.zeros((2, 2))
    fitted = sl_predict.choose_variant(SimpleNamespace(device="cpu"), "S2", ("inner", None), ("outer", None),
                                       rows, np.full((2, 2), 2.), (zeros, zeros),
                                       (np.ones((2, 2)), np.full((2, 2), 3.)))
    assert fitted.selection["neural_gain"] == .25 and fitted.selection["kernel_gain"] == 1.
    assert calls[-1][0] == "outer" and calls[-1][2:] == (.001, (1,))
    np.testing.assert_array_equal(fitted.full, np.full((2, 2), 5.))
    np.testing.assert_array_equal(fitted.neural_only, np.full((2, 2), 2.))
