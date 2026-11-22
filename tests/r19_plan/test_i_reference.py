"""Verify the I-track R17-style bank and common FiLM reference on synthetic wells."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.r15_lincs.design import fold_id
from src.r19_plan import i_reference
from src.r19_plan.i_run import IModelBank


def test_matched_reference_uses_select_truth_and_a_common_film_fit():
    fold = 1
    parents = ["C" * n for n in range(1, 50)]
    outer = next(p for p in parents if fold_id(p, "R15-outer", 3) == fold)
    train = [p for p in parents if fold_id(p, "R15-outer", 3) != fold]
    select = next(p for p in train if fold_id(p, f"R15-inner-{fold}", 5) == 0)
    inner = next(p for p in train if fold_id(p, f"R15-inner-{fold}", 5) != 0)

    def rows(parents, plates, wells):
        return pd.DataFrame(dict(study="S", cell_id="HEPG2", hours="6", dose="10",
                                 parent=parents, rna_plate=plates, inst_id=wells, rna_well="A01"))

    labels = rows([inner, select], ["P", "Q"], ["Y1", "Y2"]).assign(inner_phase=["inner_fit", "select"])
    query = rows([outer], ["R"], ["Y3"])
    fit_ctl = pd.DataFrame(dict(study="S", rna_plate=["P", "Q", "R"], role="A"))
    select_ctl = pd.DataFrame(dict(study="S", rna_plate=["Q"], role="B"))
    staged = SimpleNamespace(
        tables=dict(inner_and_outer_labels=labels, query_features=query, fit_controls=fit_ctl, select_controls=select_ctl),
        arrays=dict(inner_and_outer_labels=np.array([[2., 3.], [3., 4.]]),
                    fit_controls=np.zeros((3, 2)), select_controls=np.ones((1, 2))),
        landmarks=pd.DataFrame(dict(pr_gene_id=[1, 2])),
    )
    bank = IModelBank(staged, "f1-A")
    fit, validation, training, held = bank.split()
    with pytest.raises(ValueError, match="query treatment"):
        bank.response(held, "A")
    with pytest.raises(ValueError, match="score-role"):
        bank.control(held, "C")
    with pytest.raises(ValueError, match="validation treatments"):
        bank.control(fit, "B")

    def kernel(_bank, _fit, requested, _labels, rules):
        return {rule: np.zeros((len(requested), 2)) for rule in rules}

    def film(_args, _bank, _fit, requested, labels, truth, _anchor, model, _work, params):
        assert truth is None
        assert model == "FiLM_R12"
        assert labels.shape[1] == 2
        return np.tile([2., 3.], (len(requested), 1))

    with (
        patch.object(i_reference, "choose_proposed", return_value=((("tanimoto", .1), .25), 1.)),
        patch.object(i_reference, "grid_chemical", side_effect=kernel),
        patch.object(i_reference, "fit_known", side_effect=film) as candidate,
    ):
        result = i_reference.fit_reference(SimpleNamespace(role="A"), bank, fit, validation, training, held)
    np.testing.assert_array_equal(result.select_truth, [[2., 3.]])
    np.testing.assert_array_equal(result.query.film, [[2., 3.]])
    np.testing.assert_array_equal(result.forecast, result.query.film)
    assert result.settings["neural_gain"] == 1.
    assert result.settings["kernel_gain"] == 0.
    assert candidate.call_count == len(i_reference.R12_LRS) * len(i_reference.R12_EPOCHS) + 1
