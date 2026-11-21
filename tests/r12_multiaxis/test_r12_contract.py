from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import sparse
import torch

from src.r11_chemical_probe.dataset import GROUP, KEY, LINE
from src.r11_chemical_probe.metrics import Contrast

from src.r12_multiaxis.calibration import Calibration, shrink_fit
from src.r12_multiaxis.data import select_panel
from src.r12_multiaxis.network import ResidualHead, center, packs, parameter_count, row_loss, vicreg
from src.r12_multiaxis.split import assignments, balanced_partition


def rows():
    return pd.DataFrame([
        {GROUP: i, "compound_entity_id": f"d{i}", KEY[0]: f"p{p}-w{i}", LINE: line,
         "plate_id": f"p{p}", "dose_uM": 0.5, "time_h": 24,
         "control_A": f"p{p}-A", "control_B": f"p{p}-B"}
        for p in range(2) for line in ("a", "b") for i in range(12)
    ])


def test_split_protects_group_and_well_and_each_role_has_own_control():
    frame = rows()
    outer, inner, manifest = assignments(frame, 42, 3)
    assert frame.groupby(GROUP).size().shape[0] == len(outer)
    for k in range(3):
        subset = manifest[manifest.evaluation_outer.eq(k)]
        assert len(set(inner[k]).intersection(subset.loc[subset.usage.eq("outer_query"), GROUP])) == 0
        for role in ("A", "B"):
            one = subset[subset.role.eq(role)]
            assert one.own_control.str.endswith(role).all()
            assert set(one.loc[one.usage.eq("outer_query"), KEY[0]]).isdisjoint(
                one.loc[one.usage.eq("outer_train"), KEY[0]]
            )
            assert one.query_pool_eligible.all()
    blocks = {i: i // 2 for i in range(12)}
    split = balanced_partition(frame, 3, 42, blocks)
    assert all(split[i] == split[i + 1] for i in range(0, 12, 2))


def test_fit_gene_panel_ignores_columns_that_only_vary_in_heldout():
    x = sparse.csr_matrix([[0, 0, 0, 1, 0, 1], [0, 0, 0, 5, 0, 2], [0, 0, 0, 7, 0, 3]])
    assert select_panel(x[:2], 2).tolist() == [3, 5]
    changed = x.toarray()
    changed[2] = [0, 0, 0, 1, 1e4, 7]
    assert select_panel(sparse.csr_matrix(changed[:2]), 2).tolist() == [3, 5]


def test_pool_loss_centering_and_gradient_reach_the_context_head():
    frame = rows()
    by_pack = packs(frame, 20)
    ix = np.concatenate([p[0] for p in by_pack])
    assert np.array_equal(np.sort(ix), np.arange(len(frame)))
    model = ResidualHead((32, 16), 7, "concat", 0.0, None)
    idx, pool, weight, count = by_pack[0]
    pool = torch.tensor(pool)
    weight = torch.tensor(weight, dtype=torch.float32)
    t = model(torch.randn(len(idx), 32), torch.randn(len(idx), 16))
    common = torch.randn(count, 7)
    assert torch.allclose(center(t + common[pool], pool, weight, count),
                          center(t, pool, weight, count), atol=1e-6)
    v = row_loss(t - torch.randn_like(t), "L2", pool, weight, count)
    v.mean().backward()
    assert model.context.weight.grad.norm().item() > 0
    a = torch.randn(9, 16, requires_grad=True)
    b = torch.randn(9, 16)
    vicreg(a, b).backward()
    assert a.grad.norm().item() > 0


def test_calibration_recovers_shift_and_positive_draw_recomputes_center():
    frame = rows()
    rng = np.random.default_rng(1)
    pred = rng.normal(size=(len(frame), 8))
    offsets = np.full_like(pred, 0.6)
    true = offsets + 1.8 * (pred - offsets) + 0.3
    a, b = shrink_fit(pred - offsets, true - offsets, np.ones(len(pred)), 1.0, 1e-8,
                      (1.0, np.zeros(8)), "S3")
    assert abs(a - 1.8) < 1e-6
    assert np.allclose(b, 0.3)
    uncal = Calibration(frame, pred, offsets, true, 1.0, "identity", "S3")
    assert np.allclose(uncal.predict(frame, pred, offsets), pred)
    model = Calibration(frame, pred, offsets, true, 1.0, 1e-8, "S3")
    cal = model.predict(frame, pred, offsets)
    assert np.allclose(cal, true, atol=1e-6)
    metric = Contrast(frame, 3)
    draws = rng.exponential(size=(3, len(metric.group_ids)))
    sampled = metric.bootstrap_values(pred, true, draws)
    positions = metric.pools[0]
    w = metric.centers[0] * draws[0, metric.inverse[positions]]
    err = pred[positions] - true[positions]
    manual = np.sqrt(np.square(err - np.average(err, weights=w, axis=0)).mean(axis=1))
    assert np.allclose(sampled[0, positions], manual)
    pool_constant = pred.copy()
    for idx in metric.pools:
        pool_constant[idx] = offsets[idx] * 0.5
    assert np.isclose(metric.point(pool_constant, true), metric.point(np.zeros_like(true), true))


def test_parameters_match_at_registered_output_width_and_seed_repeats():
    groups = (("raw", 2000), ("pca", 16))
    for _, context in groups:
        models = []
        for head in ("concat", "film", "bilinear"):
            torch.manual_seed(42)
            model = ResidualHead((32, context), 2000, head, 0.1, None)
            models.append(parameter_count(model))
        assert max(models) / min(models) < 1.1
    options = SimpleNamespace(dims=(32, 16), genes=2000, head="film", dropout=0.1, projection=None)
    torch.manual_seed(42)
    a = ResidualHead(**vars(options)).eval()
    torch.manual_seed(42)
    b = ResidualHead(**vars(options)).eval()
    x = (torch.ones(2, 32), torch.ones(2, 16))
    assert torch.equal(a(*x), b(*x))


def test_scoring_joins_multiple_treated_wells_to_one_physical_control():
    from src.r12_multiaxis.scoring import match_rows

    left = pd.DataFrame({"plate_id": ["p1", "p1", "p2"], LINE: ["a", "a", "a"],
                         KEY[0]: ["treated1", "treated2", "treated3"]})
    right = pd.DataFrame({"plate_id": ["p2", "p1"], LINE: ["a", "a"],
                          KEY[0]: ["control2", "control1"]})
    pos = match_rows(left, right, ["plate_id", LINE])
    assert right.iloc[pos][KEY[0]].tolist() == ["control1", "control1", "control2"]
