"""Verify rank support and parent weights for the source-only time operator."""

import numpy as np
import pandas as pd
import pytest

from src.r19_plan.time_operator import fit_time_operator


def example():
    n = 4
    x = np.vstack([np.eye(n), -np.eye(n)])
    rows = pd.DataFrame(dict(study="lincs", cell_id="HEPG2", parent=[f"p{i}" for i in range(len(x))],
                             dose_uM=10.))
    scale = np.array([.5, .6, 1.5, 2.])
    intercept = np.array([1., -2., 0., 1.])
    return rows, x, x * scale + intercept, scale, intercept


def test_time_operator_applies_fit_only_centered_low_rank_ridge():
    rows, x, y, scale, intercept = example()
    genes = (21, 31, 41, 51)
    fit = fit_time_operator(rows, x, y, genes, rank=4, ridge=.1, held_parents={"new"})
    held = np.array([[1., 2., 3., 4.]])
    np.testing.assert_allclose(fit.predict(held, genes), held * scale * (2. / 2.1) + intercept)
    with pytest.raises(ValueError, match="gene order"):
        fit.predict(held, tuple(reversed(genes)))
    with pytest.raises(ValueError, match="held parent"):
        fit_time_operator(rows, x, y, genes, rank=4, ridge=.1, held_parents={"p0"})


def test_parent_weights_are_preserved_when_an_identical_condition_is_added():
    rows, x, y, _, _ = example()
    genes = (1, 2, 3, 4)
    choice = dict(rank=4, ridge=1., held_parents=set())
    original = fit_time_operator(rows, x, y, genes, **choice)
    expanded = pd.concat([rows, rows.iloc[:1].assign(dose_uM=.5)], ignore_index=True)
    copied = fit_time_operator(expanded, np.vstack((x, x[:1])), np.vstack((y, y[:1])), genes, **choice)
    np.testing.assert_allclose(original.weights, copied.weights, atol=1e-12)
    assert original.fit_parent_count == copied.fit_parent_count == 8


def test_time_operator_rejects_duplicate_pairs_and_unsupported_rank():
    rows, x, y, _, _ = example()
    genes = (1, 2, 3, 4)
    with pytest.raises(ValueError, match="unique measured dose"):
        fit_time_operator(rows.assign(parent="same"), x, y, genes, rank=4, ridge=.1, held_parents=set())
    with pytest.raises(ValueError, match="independent source parents"):
        fit_time_operator(rows, x, y, genes, rank=16, ridge=.1, held_parents=set())
    with pytest.raises(ValueError, match="numerical support"):
        fit_time_operator(rows, np.zeros_like(x), y, genes, rank=4, ridge=.1, held_parents=set())
