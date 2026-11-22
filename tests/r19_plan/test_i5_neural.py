"""Synthetic split, feature-isolation and checkpoint guards for the I5 neural arm."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.r12_multiaxis.data import Design
from src.r19_plan import i5_neural as model


def feature_fixture():
    parents = pd.Index(list("abcdefghijkl"))
    fit = ["a", "a", "b", "c", "d"]
    query = ["e", "f", "g", "h", "i", "j", "e"]
    domains = ["one", "one", "one", "two", "two", "two", "one"]
    rng = np.random.default_rng(27)
    views = {name: rng.normal(size=(len(parents), 4)) for name in model.VIEWS}
    views["I2"][parents.get_loc("f")] = np.nan
    views["I2"][parents.get_loc("i")] = np.nan
    views["I4"][parents.get_loc("f")] = np.nan
    views["I4"][parents.get_loc("j")] = np.nan
    base = Design(
        train=(rng.normal(size=(len(fit), 3)), rng.normal(size=(len(fit), 2))),
        query=(rng.normal(size=(len(query), 3)), rng.normal(size=(len(query), 2))),
    )
    return parents, views, fit, query, domains, base


def test_modes_keep_structure_context_and_masks_on_exact_parent_well_axis():
    parents, views, fit, query, domains, base = feature_fixture()
    designs, observed_fit, observed_query = model.augment_designs(base, parents, views, fit, query, domains)
    np.testing.assert_array_equal(observed_fit, np.ones((len(fit), 2), dtype=bool))
    np.testing.assert_array_equal(
        observed_query, [[1, 1], [0, 0], [1, 1], [1, 1], [0, 1], [1, 0], [1, 1]],
    )
    width = designs["full"].query[0].shape[1]
    for design in designs.values():
        assert design.train[0].shape == (len(fit), width)
        assert design.query[0].shape == (len(query), width)
        assert design.train[1] is base.train[1]
        assert design.query[1] is base.query[1]
        np.testing.assert_allclose(design.train[0][:, :3], base.train[0])
        np.testing.assert_allclose(design.query[0][:, :3], base.query[0])
        np.testing.assert_array_equal(design.train[0][:, -2:], observed_fit)
        np.testing.assert_array_equal(design.query[0][:, -2:], observed_query)
        np.testing.assert_array_equal(design.query[0][1, 3:-2], 0.)
    np.testing.assert_array_equal(designs["mask_only"].train[0][:, 3:-2], 0.)
    np.testing.assert_array_equal(designs["mask_only"].query[0][:, 3:-2], 0.)
    np.testing.assert_array_equal(designs["full"].train[0][0, 3:], designs["full"].train[0][1, 3:])
    np.testing.assert_array_equal(designs["parent_shuffle"].query[0][0, 3:],
                                  designs["parent_shuffle"].query[0][6, 3:])


def test_shuffles_preserve_per_view_availability_and_role_without_fitting_query_values():
    parents, views, fit, query, domains, _ = feature_fixture()
    fit_views, query_views = model.view_blocks(parents, views, fit, query, domains)
    changed = {name: values.copy() for name, values in views.items()}
    for values in changed.values():
        values[4:] = -20 * values[4:]
    changed_fit, changed_query = model.view_blocks(parents, changed, fit, query, domains)
    for original, updated in zip(fit_views, changed_fit, strict=True):
        for old, new in zip(original, updated, strict=True):
            np.testing.assert_array_equal(old, new)
    assert not np.array_equal(query_views[0][0], changed_query[0][0])
    for values, shuffled, available in query_views:
        np.testing.assert_array_equal(shuffled[~available], 0.)
        for positions in ([0, 1, 2], [3, 4, 5]):
            observed = [i for i in positions if available[i]]
            assert sorted(map(tuple, values[observed])) == sorted(map(tuple, shuffled[observed]))
    with pytest.raises(ValueError, match="reached the fit partition"):
        model.view_blocks(parents, views, fit, ["a"], ["one"])
    with pytest.raises(ValueError, match="explicit query domain"):
        model.view_blocks(parents, views, fit, ["e"], [""])


def selection_fixture():
    fit = SimpleNamespace(query=SimpleNamespace(rows=pd.DataFrame({"parent": ["q0", "q1"]}), treated=None),
                          panel=[901, 902], cfg={})
    return model.NeuralSplit(fit, {}, np.ones((2, 2), dtype=bool), np.array([[True, False], [False, False]]))


def test_fixed_kernel_checkpoint_selection_and_common_no_view_fallback(monkeypatch):
    split = selection_fixture()
    fixed = np.array([[1., 1.], [20., 20.]])
    fallback = np.array([[50., 50.], [4., 4.]])
    truth = np.array([[2., 2.], [4., 4.]])
    winning = model.R12_LRS[-1], model.R12_EPOCHS[-1]
    calls = []

    def checkpoint(source, mode, lr, epoch):
        assert source is split and mode == "full"
        calls.append((lr, epoch))
        observed = 4. if (lr, epoch) == winning else .25
        return np.array([[observed, observed], [999., 999.]])

    monkeypatch.setattr(model, "checkpoint_prediction", checkpoint)
    choice, prediction = model.select_neural(split, truth, fixed, fallback, "full")
    assert calls == [(lr, epoch) for lr in model.R12_LRS for epoch in model.R12_EPOCHS]
    assert (choice.lr, choice.epoch) == winning
    assert choice.gain == .25
    assert choice.loss == 0.
    np.testing.assert_array_equal(prediction, truth)
    np.testing.assert_array_equal(model.predict_neural(split, fixed, fallback, "full", choice), prediction)
    np.testing.assert_array_equal(fixed, [[1., 1.], [20., 20.]])


def test_checkpoint_calls_locked_training_without_held_treatment(monkeypatch):
    split = selection_fixture()
    design = object()
    split.designs["mask_only"] = design
    seen = []

    def network(fit, inputs, arch, loss, lr, seed, device, epochs, *, patience):
        assert fit.query.treated is None
        assert inputs is design and (arch, loss, device, patience) == ("film", "L0", "cpu", False)
        assert seed == model.SEED and fit.cfg["min_epochs"] == epochs + 1
        seen.append((lr, epochs))
        return {"contrast": (0., epochs, np.zeros((2, 2)))}, {}, (), {}

    monkeypatch.setattr(model, "fit_network", network)
    lr, epoch = model.R12_LRS[0], model.R12_EPOCHS[0]
    np.testing.assert_array_equal(model.checkpoint_prediction(split, "mask_only", lr, epoch), np.zeros((2, 2)))
    split.fit.query.treated = np.zeros((2, 2))
    with pytest.raises(ValueError, match="must receive no held treatment"):
        model.checkpoint_prediction(split, "mask_only", lr, epoch)
    assert seen == [(lr, epoch)]
