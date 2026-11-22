"""Verify DEV D3 with synthetic fit-control and treatment arrays."""

from dataclasses import replace
from importlib.util import find_spec

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not all(find_spec(name) for name in ("h5py", "rdkit")),
                                reason="requires the R17 LINCS estimators environment")


@pytest.fixture
def fit_controls():
    from src.r19_plan.dev_selection import PhysicalControls

    genes = (2, 5, 7)
    wells = (("p1", "B01"), ("p1", "B02"), ("p1", "B03"), ("p2", "B01"), ("p2", "B02"))
    values = np.array([[96., 240., -55.], [100., 240., -55.], [104., 240., -55.],
                       [-52., 297., 120.], [-52., 303., 120.]])
    roles = {
        "fit": wells,
        "select": (("p1", "C01"), ("p2", "C01")),
        "score": (("p1", "D01"), ("p2", "D01")),
    }
    return PhysicalControls(wells, genes, values), roles


def test_augmentation_keeps_each_well_weight_and_expected_residual(fit_controls):
    from src.r19_plan.dev_vehicle import augment_fit

    fit, roles = fit_controls
    wells = (("p2", "A1"), ("p1", "A2"), ("p1", "A1"))
    treated = np.arange(9).reshape(3, 3) * 1.
    anchor = np.array([[1., 3., 5.], [3., 5., 7.], [5., 7., 9.]])
    weights = np.array([.2, .3, .5])
    augmented = augment_fit(wells, fit.genes, treated, anchor, weights, fit, roles)
    np.testing.assert_array_equal(augmented.row_indices, [0, 0, 1, 1, 2, 2])
    np.testing.assert_array_equal(augmented.view_ids, [0, 1, 0, 1, 0, 1])
    for plate, halves in augmented.view_wells.items():
        assert not set(halves[0]) & set(halves[1])
        assert set(halves[0]) | set(halves[1]) == {
            (p.upper(), well) for p, well in roles["fit"] if p.upper() == plate
        }
    restored_weight = augmented.weights.reshape(3, 2).sum(axis=1)
    np.testing.assert_allclose(restored_weight, weights)
    mean = (augmented.residual * augmented.weights[:, None]).reshape(3, 2, 3).sum(axis=1)
    unsubdivided = np.stack([fit.values[3:].mean(axis=0)] + [fit.values[:3].mean(axis=0)] * 2)
    np.testing.assert_allclose(mean / weights[:, None], treated - anchor - unsubdivided)
    np.testing.assert_allclose(augmented.residual, treated[augmented.row_indices] -
                               augmented.controls - anchor[augmented.row_indices])

    order = [4, 1, 2, 3, 0]
    reordered = replace(fit, wells=tuple(fit.wells[i] for i in order), values=fit.values[order])
    second = augment_fit(wells, fit.genes, treated, anchor, weights, reordered, roles)
    assert augmented.view_wells == second.view_wells
    np.testing.assert_array_equal(augmented.residual, second.residual)
    np.testing.assert_array_equal(augmented.weights, second.weights)


def test_augmented_training_view_reuses_position_pool_offset_and_query(fit_controls):
    from types import SimpleNamespace

    import pandas as pd

    from src.r12_multiaxis.data import Design
    from src.r19_plan.d3_predict import augment_design

    control, roles = fit_controls
    rows = pd.DataFrame({"rna_plate": ["p1", "p2"], "rna_well": ["A01", "A01"]})
    treated = np.array([[102., 235., -48.], [-50., 308., 125.]])
    chrom = np.array([[1., 2., 3.], [4., 5., 6.]])
    mean = np.stack([control.values[:3].mean(axis=0), control.values[3:].mean(axis=0)])
    delta = treated - mean - chrom
    residual = delta - 2.
    fit = SimpleNamespace(train=SimpleNamespace(rows=rows, delta=delta), residual=residual,
                          weight=np.array([.4, .6]), panel=control.genes, project=lambda _: (None, lambda x: x))
    design = Design((np.array([[1.], [2.]]), mean), (np.ones((1, 1)), np.zeros((1, 3))))
    duplicate, inputs = augment_design(fit, design, treated, chrom, control, roles)
    assert len(duplicate.train.rows) == 4 and inputs.query is design.query
    np.testing.assert_allclose(duplicate.weight.reshape(2, 2).sum(axis=1), fit.weight)
    np.testing.assert_allclose((duplicate.residual * duplicate.weight[:, None]).reshape(2, 2, 3).sum(axis=1)
                               / fit.weight[:, None], residual, atol=1e-6)


def test_nuisance_is_within_plate_and_leaves_anchor_and_orthogonal_signal_intact(fit_controls):
    from src.r19_plan.dev_vehicle import learn_nuisance_basis, shrink_neural_residual

    fit, roles = fit_controls
    basis = learn_nuisance_basis(fit, roles, fit.genes)
    assert basis.repeated_fit_plates == ("P1", "P2")
    assert (basis.physical_wells, basis.contrast_df, basis.supported_ranks) == (5, 3, (1, 2))
    np.testing.assert_allclose(basis.vectors @ basis.vectors.T, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(basis.vectors[0], [0., 1., 0.], atol=1e-12)
    np.testing.assert_allclose(basis.vectors.T @ basis.vectors, np.diag([1., 1., 0.]), atol=1e-12)

    offsets = np.array([[1000., 2000., -1500.]] * 3 + [[-400., -300., 1800.]] * 2)
    shifted = learn_nuisance_basis(replace(fit, values=fit.values + offsets), roles, fit.genes)
    np.testing.assert_allclose(basis.vectors, shifted.vectors, atol=1e-12)
    residual = np.array([[1., 2., 3.], [5., 7., 9.]], dtype=np.float32)
    anchor = np.array([[4., 3., 2.], [1., 2., 3.]], dtype=np.float32)
    np.testing.assert_array_equal(shrink_neural_residual(residual, anchor, fit.genes, basis, 1, 0.), anchor + residual)
    np.testing.assert_allclose(shrink_neural_residual(residual, anchor, fit.genes, basis, 2, .5),
                               anchor + residual * np.array([.5, .5, 1.]))
    np.testing.assert_allclose(shrink_neural_residual(residual, anchor, fit.genes, basis, 1, .25),
                               anchor + residual * np.array([1., .75, 1.]))


def test_role_disjointness_gene_axis_and_plate_support_are_required(fit_controls):
    from src.r19_plan.dev_vehicle import ControlSupportError, augment_fit, learn_nuisance_basis

    fit, roles = fit_controls
    with pytest.raises(ValueError, match="roles must be disjoint"):
        learn_nuisance_basis(fit, dict(roles, score=(("P1", "B1"),)), fit.genes)
    with pytest.raises(ValueError, match="exactly the registered fit controls"):
        source = replace(fit, wells=roles["score"], values=fit.values[:2])
        learn_nuisance_basis(source, roles, fit.genes)
    with pytest.raises(ValueError, match="registered gene order"):
        learn_nuisance_basis(fit, roles, fit.genes[::-1])

    thin = replace(fit, wells=fit.wells[:-1], values=fit.values[:-1])
    thin_roles = dict(roles, fit=thin.wells)
    with pytest.raises(ControlSupportError, match="two fit wells per plate"):
        augment_fit((("p2", "A01"),), fit.genes, np.ones((1, 3)), np.zeros((1, 3)), [1.], thin, thin_roles)
    with pytest.raises(ControlSupportError, match="two independently repeated"):
        learn_nuisance_basis(thin, thin_roles, fit.genes)


def test_unavailable_rank_is_blocked_and_grid_is_fixed(fit_controls):
    from src.r19_plan.dev_vehicle import ControlSupportError, learn_nuisance_basis, shrink_neural_residual

    fit, roles = fit_controls
    values = fit.values.copy()
    values[3:] = values[3]
    basis = learn_nuisance_basis(replace(fit, values=values), roles, fit.genes)
    assert basis.supported_ranks == (1,)
    residual, anchor = np.ones((1, 3)), np.zeros((1, 3))
    with pytest.raises(ControlSupportError, match="rank 2"):
        shrink_neural_residual(residual, anchor, fit.genes, basis, 2, .5)
    with pytest.raises(ValueError, match="outside the registered"):
        shrink_neural_residual(residual, anchor, fit.genes, basis, 1, .75)
    with pytest.raises(ValueError, match="gene order"):
        shrink_neural_residual(residual, anchor, fit.genes[::-1], basis, 1, .25)
    with pytest.raises(ControlSupportError, match="no nonzero"):
        learn_nuisance_basis(replace(fit, values=np.zeros_like(values)), roles, fit.genes)
