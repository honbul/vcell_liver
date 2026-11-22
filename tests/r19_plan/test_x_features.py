import numpy as np
import pytest

from src.r19_plan.x_features import CONTEXT_WIDTH, fit_basal_context, fit_cell_identity


def test_unseen_context_id_uses_equal_fit_context_mean():
    model = fit_cell_identity(["A", "B", "B", "B"])
    enc = model.transform(["B", "new", "A"])
    assert enc.shape == (3, CONTEXT_WIDTH)
    np.testing.assert_array_equal(enc[1], (enc[0] + enc[2]) / 2.)


def test_basal_scaler_and_basis_depend_only_on_fit_contexts():
    contexts = ["A", "B", "C", "query"]
    genes = [3, 2, 6, 4]
    profiles = np.array([[1., 0., 3., 5.], [2., 5., 3., 1.], [4., 6., 3., 4.], [3., 0., 2., 1.]])
    encoder = fit_basal_context(contexts[:3], contexts, genes, profiles)
    changed = profiles.copy()
    changed[-1] = 1000
    new = fit_basal_context(contexts[:3], contexts, genes, changed)
    np.testing.assert_array_equal(encoder.mean, new.mean)
    np.testing.assert_array_equal(encoder.basis.components_, new.basis.components_)
    score = encoder.transform(genes, profiles)
    assert score.shape == (4, CONTEXT_WIDTH) and not score[:, 2:].any()
    with pytest.raises(ValueError, match="gene order"):
        encoder.transform(genes[::-1], profiles)
