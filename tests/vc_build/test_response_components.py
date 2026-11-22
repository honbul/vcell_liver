import numpy as np
import pandas as pd

from vc_build.response_components import component_scores, orthogonal_parts


def test_weighted_projection_matches_least_squares_on_disconnected_observed_support():
    value = np.array([[1., 2., np.nan, np.nan], [3., 9., np.nan, np.nan],
                      [np.nan, np.nan, 5., 7.], [np.nan, np.nan, 4., np.nan]])
    weights = np.array([[1., 2., 0., 0.], [4., 1., 0., 0.],
                        [0., 0., 3., 5.], [0., 0., 1., 0.]])
    sample, drug, interaction = orthogonal_parts(value, weights)
    rows, columns = np.where(weights > 0)
    design = np.column_stack([np.eye(4)[rows], np.eye(4)[columns]])
    root = np.sqrt(weights[rows, columns])
    measured = value[rows, columns]
    coefficient = np.linalg.lstsq(design * root[:, None], measured * root, rcond=None)[0]
    expected = measured - design @ coefficient
    np.testing.assert_allclose(interaction[rows, columns], expected, atol=1e-12)
    np.testing.assert_allclose((weights * interaction).sum(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose((weights * interaction).sum(axis=1), 0, atol=1e-12)
    np.testing.assert_allclose((sample[:, None] + drug + interaction)[rows, columns], measured)


def test_report_separates_drug_mean_from_real_interaction():
    rng = np.random.default_rng(2)
    rows, cols = 5, 6
    observed = rng.normal(size=(rows, cols))
    anchor = np.broadcast_to(observed.mean(axis=0), observed.shape)
    identity = pd.DataFrame(np.indices(observed.shape).reshape(2, -1).T, columns=["lineage_group", "compound_id"])
    trial = pd.concat([identity.assign(method="fit", endpoint_value=observed.ravel(), predicted_value=observed.ravel()),
                       identity.assign(method="drug_mean", endpoint_value=observed.ravel(), predicted_value=anchor.ravel())])
    metrics = component_scores(trial, cell_axis="lineage_group")
    assert metrics["scores"]["fit"]["raw_scale_interaction_r2"] == 1
    assert metrics["scores"]["drug_mean"]["interaction_pearson"] is None
    assert abs(metrics["scores"]["drug_mean"]["raw_scale_interaction_r2"]) < 1e-12
