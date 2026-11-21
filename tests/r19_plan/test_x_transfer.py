import numpy as np
import pytest

from src.r19_plan.x_transfer import donor_mean, fit_basal_response


def test_basal_similarity_uses_only_non_target_donors():
    labels = ["A", "B"]
    donor = np.array([[[1.], [5.]], [[2.], [7.]]])
    means = {"A": np.zeros(2), "B": np.full(2, 3.), "held": np.ones(2)*.1}
    np.testing.assert_array_equal(donor_mean(donor, labels, "held"), np.array([[3.], [4.5]]))
    assert np.linalg.norm(donor_mean(donor, labels, "held", means, .5) - donor[:, 0]) < .001
    with pytest.raises(ValueError, match="non-target"):
        donor_mean(donor, labels, "A")


def test_basal_linear_correction_fits_source_pairs_and_uses_new_context_feature():
    cells = ["A", "B", "C"]
    contexts = np.array([[-1., 0.], [0., 0.], [1., 0.]])
    common = np.linspace(-1, 1, 10)[:, None, None]
    response = common + 2 * contexts[None, :, :1]
    parents = [f"p{i}" for i in range(10)]
    model = fit_basal_response(response, parents, cells, contexts, ridge=.1, held_parents=["q"], held_context="new")
    means = dict(zip(cells, contexts, strict=True)) | {"new": np.array([.5, 0.])}
    guess = model.predict(np.zeros((1, 1)), "new", cells, means)
    assert abs(guess[0, 0] - 1.) < .1
    with pytest.raises(ValueError, match="protected"):
        fit_basal_response(response, parents, cells, contexts, ridge=.1, held_parents=["p1"], held_context="new")
