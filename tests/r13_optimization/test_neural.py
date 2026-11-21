"""Contract and fit-only invariants for the neural optimization family."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.r11_chemical_probe.dataset import ENTITY, GROUP, KEY, LINE, group_weights
from src.r13_optimization.neural import _coordinates, candidates, predict, predict_many


def _stub():
    rng = np.random.default_rng(1742)

    class SealedQuery:
        @property
        def delta(self):
            raise AssertionError("Optimization opened query outcomes")

    def rows(first, count):
        return pd.DataFrame({
            KEY[0]: [f"p{plate}_{group}" for plate in range(2) for group in range(first, first + count)],
            LINE: [f"l{plate}" for plate in range(2) for _ in range(count)],
            GROUP: list(range(first, first + count)) * 2,
            ENTITY: [f"d{i}" for i in range(first, first + count)] * 2,
            "plate_id": [f"p{plate}" for plate in range(2) for _ in range(count)],
            "dose_uM": 0.5,
            "time_h": 24,
        })

    def tiles(array):
        return np.vstack([array, array])

    train, query = SimpleNamespace(rows=rows(0, 10)), SealedQuery()
    query.rows = rows(10, 4)
    fp = rng.normal(size=(14, 8)).astype(np.float32)
    signs = np.concatenate((np.ones((10, 1)), -np.ones((10, 1)) * 0.25))
    train.delta = (tiles(fp[:10]) @ rng.normal(size=(8, 12))) * signs
    train.chemical = {"E": tiles(fp[:10])}
    query.chemical = {"E": tiles(fp[10:])}
    offset = np.full((len(query.rows), 12), 0.2, dtype=np.float32)
    vectors = rng.normal(size=(2, 6)).astype(np.float32)
    keys = [("w0", "l0"), ("w1", "l1")]
    fit = SimpleNamespace(
        train=train, query=query, offset=offset, residual=train.delta - 0.2,
        scale=1.0, weight=group_weights(train.rows),
        drug_blocks=(tiles(fp[:10]), tiles(fp[10:])),
        fit_units=keys, train_units=[keys[0]] * 10 + [keys[1]] * 10,
        query_units=[keys[0]] * 4 + [keys[1]] * 4,
        context_bank=lambda _: dict(zip(keys, vectors)),
    )
    return fit


def test_projection_preserves_full_gene_loss():
    rng = np.random.default_rng(74)
    values = rng.normal(size=(7, 12))
    basis, _ = np.linalg.qr(rng.normal(size=(12, 4)))
    pred = rng.normal(size=(7, 4))
    coords, omitted = _coordinates(values, basis.T)
    actual = np.square(pred @ basis.T - values).sum(axis=1)
    compressed = np.square(pred - coords).sum(axis=1) + omitted
    np.testing.assert_allclose(actual, compressed, rtol=1e-6)


def test_nested_runner_checkpoints_keep_query_outcomes_sealed(monkeypatch):
    monkeypatch.setenv("R13_DEVICE", "cpu")
    fit = _stub()
    grid = candidates()
    assert len({x["name"] for x in grid}) == len(grid)
    initial = {**grid[0], "epochs": 2}
    pair = [initial, {**initial, "name": "later", "epochs": 3}]
    saved = list(predict_many(fit, pair))
    assert [spec["epochs"] for spec, _ in saved] == [2, 3]
    assert all(pred.shape == fit.offset.shape and np.isfinite(pred).all() for _, pred in saved)
    assert not np.allclose(saved[0][1], fit.offset)
    np.testing.assert_allclose(saved[0][1], predict(fit, initial), atol=1e-7)
