"""Verify the I3 Gram and ensure held profiles cannot affect fit-only transforms."""

import numpy as np

from src.r19_plan.i_kernel import FitProfile, masked_gram


def test_held_rows_cannot_change_standardization_or_pca():
    training = np.array([[1., 3., np.nan], [2., 2., np.nan], [3., 1., 6.]])
    profile = FitProfile.learn(training, rank=1)
    np.testing.assert_equal(profile.columns, [True, True, False])
    mean, components = profile.mean.copy(), profile.basis.components_.copy()
    profile.transform([[100., 200., 300.]])
    profile.transform([[-100., -200., -300.]])
    np.testing.assert_equal(profile.mean, mean)
    np.testing.assert_equal(profile.basis.components_, components)


def test_masked_gram_psd_and_structure_only_fallback():
    bits = np.array([[1., 0.], [1., 1.], [0., 1.]])
    structure = bits @ bits.T
    x = np.array([[1., 0.], [10., 5.], [0., 1.]])
    m = np.array([True, False, True])
    y = np.array([[0., 0.], [10., 3.], [5., 1.]])
    gram = masked_gram(structure, [(x, m), (y, np.zeros(3, dtype=bool))])
    assert np.linalg.eigvalsh(gram).min() >= -1e-12
    assert gram[1, 1] == structure[1, 1]
    np.testing.assert_allclose(masked_gram(structure, [(x, np.zeros_like(m))]), structure)
