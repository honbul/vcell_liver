"""External-cohort safeguards without access to the assay outcome file."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from src.r13_optimization import external


def test_only_requested_rows_are_scaled_by_all_released_genes(monkeypatch, tmp_path):
    counts = np.array([[2, 25, 3], [10, 80, 10], [9, 90, 1]], dtype=np.float32)

    class Layer:
        shape = counts.shape

        def __getitem__(self, key):
            rows, columns = key
            assert rows.tolist() == [0, 2]
            return counts[rows, columns]

    class Handle:
        def __enter__(self):
            return {external.COUNT_LAYER: Layer()}

        def __exit__(self, *args):
            return False

    monkeypatch.setitem(sys.modules, "h5py", SimpleNamespace(File=lambda *_: Handle()))
    values = external.read_count_rows(tmp_path / "artificial.h5", counts.shape, [2, 0, 2], [2, 0])
    expected = np.log1p(np.array([[100, 900], [1000, 20000 / 30], [100, 900]]))
    np.testing.assert_allclose(values, expected, rtol=1e-6)


def test_scoring_requires_a_label_free_receipt_before_opening_outcomes(monkeypatch, tmp_path):
    external.write_json(tmp_path / "freeze.json", {"treated_expression_accessed": True})

    def forbidden(*_):
        raise AssertionError("Opened treatments before verifying the prediction freeze")

    monkeypatch.setattr(external, "read_count_rows", forbidden)
    with pytest.raises(ValueError, match="label-free inference"):
        external.score_frozen(SimpleNamespace(out=tmp_path), None, None, None, None)
