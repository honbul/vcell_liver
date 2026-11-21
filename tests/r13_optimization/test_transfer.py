from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.r11_chemical_probe.dataset import Block, CONTROL, ENTITY, GROUP, KEY, LINE, group_weights
from src.r13_optimization.dose_transfer import candidates as dose_candidates
from src.r13_optimization.dose_transfer import leave_compound_out_scale, predict as predict_dose
from src.r13_optimization.transfer import candidates, predict, reproducible_gene_basis


def sample_fit():
    rng = np.random.default_rng(657)
    fingerprints = np.stack([(np.arange(32) >> k) & 1 for k in range(5)], axis=1)
    line_control = rng.normal(size=(6, 40)).astype(np.float32)
    direction = rng.normal(size=(5, 40))
    chemical = fingerprints @ direction
    interaction = np.linspace(-1, 1, 6)[:, None] * rng.normal(size=40)
    response = chemical[:, None] + fingerprints[:, 1, None, None] * interaction[None]

    def block(groups):
        dg, ln = map(np.ravel, np.meshgrid(groups, np.arange(6), indexing="ij"))
        plate = dg % 3
        rows = pd.DataFrame({GROUP: dg, ENTITY: dg, KEY[0]: dg + 100, LINE: ln,
                             "plate_id": plate, CONTROL: 10 * plate + ln})
        return SimpleNamespace(rows=rows, chemical={"E": fingerprints[dg], "C": chemical[dg]},
                               control=line_control[ln]), response[dg, ln].astype(np.float32)

    train, residual = block(np.arange(24))
    query, observed = block(np.arange(24, 32))
    fit = SimpleNamespace(train=train, query=query, residual=residual,
                          weight=group_weights(train.rows), offset=np.full(observed.shape, 0.1, np.float32))
    return fit, observed


def test_transfer_protocol_grid_and_shape():
    fit, observed = sample_fit()
    specs = candidates()
    assert len(specs) == len({s["name"] for s in specs})
    for spec in specs:
        result = predict(fit, spec)
        assert result.shape == fit.offset.shape
        assert np.isfinite(result).all()
    raw = predict(fit, next(s for s in specs if s["name"] == "tr_ecfp_a0.3_full_none0_g0.0_mean"))
    assert np.mean((raw - fit.offset - observed) ** 2) < np.mean(observed ** 2)


def test_query_subset_does_not_change_predictions():
    full, _ = sample_fit()
    reduced, _ = sample_fit()
    take = [12, 2, 4, 7, 0]
    for field in ("rows", "control"):
        obj = getattr(reduced.query, field)
        setattr(reduced.query, field, obj.iloc[take] if field == "rows" else obj[take])
    reduced.query.chemical = {k: v[take] for k, v in reduced.query.chemical.items()}
    reduced.offset = reduced.offset[take]
    spec = next(s for s in candidates() if s["line_basis"] == "empirical")
    np.testing.assert_allclose(predict(reduced, spec), predict(full, spec)[take], atol=1e-6)


def test_overlap_is_rejected():
    fit, _ = sample_fit()
    fit.query.rows.loc[0, GROUP] = fit.train.rows.loc[0, GROUP]
    with pytest.raises(ValueError, match="disjoint"):
        predict(fit, candidates()[0])


def test_negative_cross_view_direction_is_discarded():
    x = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=np.float32)
    result = reproducible_gene_basis(x, x * [1, -1], 2)
    assert result.shape == (2, 1)
    np.testing.assert_allclose(np.abs(result[:, 0]), [1, 0], atol=1e-6)


def test_dose_scale_excludes_current_compound_outcome():
    aux = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    labels = aux * 0.5
    original = leave_compound_out_scale(aux, labels)
    labels[0] = 100
    edited = leave_compound_out_scale(aux, labels)
    np.testing.assert_allclose(original[0], edited[0])
    assert not np.allclose(original[1:], edited[1:])


def test_dose_grid_fits_only_compound_overlaps():
    fit, _ = sample_fit()
    train = fit.train
    primary = fit.residual + train.control
    low = train.rows.assign(dose_uM=0.05, time_h=24)
    high = train.rows.assign(dose_uM=5.0, time_h=24)
    # Auxiliary responses carry known amplitude shifts and separate physical wells.
    low[KEY[0]] += 1000
    high[KEY[0]] += 2000
    block = Block(pd.concat([low, high], ignore_index=True), np.concatenate([train.control] * 2),
                  {k: np.concatenate([v] * 2) for k, v in train.chemical.items()},
                  np.concatenate([0.25 * primary, 2.0 * primary]))

    class Cache:
        def block(self, _):
            return block

    fit.source = SimpleNamespace(dose_cache=Cache())
    fit.cfg = dict(offset_shrinkage=2)
    for spec in dose_candidates():
        assert np.isfinite(predict_dose(fit, spec)).all()
