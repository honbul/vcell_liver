from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.r11_chemical_probe.dataset import GROUP, LINE
from src.r15_lincs.design import fold_id
from src.r19_plan.x_access import split_scope
from src.r19_plan.x_inputs import validate_predict
from src.r19_plan.x_predict import make_design
from src.r19_plan.x_workflow import center_by_context, interaction_axes


def test_outer_and_inner_splits_hold_cell_and_parent_globally():
    names = ["HEPG2", "A375", "A549", "HT29", "MCF7"]
    rows = pd.DataFrame([dict(inst_id=f"{cell}:{parent}", study="S", cell_id=cell, parent=f"smiles-{parent}",
                              rna_plate=cell, rna_well=str(parent), level3_row_index=i*493+parent)
                         for i, cell in enumerate(names) for parent in range(493)])
    rows["parent_fold"] = rows.parent.map(lambda p: fold_id(p, "R15-outer", 3))
    controls = pd.DataFrame([dict(study="S", inst_id=f"control-{cell}-{role}", cell_id=cell, rna_plate=cell,
                                  rna_well=role, role=role) for cell in names for role in "ABC"])
    plans, summary = split_scope(rows, controls, names, "HEPG2", 0, "A")
    predict = plans["predict"]
    validate_predict(predict, "HEPG2-f0-A", names)
    assert summary["training_contexts"] == 3 and summary["outer_contexts"] == 4
    assert not set(predict["labels"].parent) & set(predict["query"].parent)
    assert "level3_row_index" not in predict["query"] and plans["score"]["controls"].role.eq("C").all()
    changed = {**predict, "query": predict["query"].assign(level3_row_index=0)}
    with pytest.raises(ValueError, match="held-context"):
        validate_predict(changed, "HEPG2-f0-A", names)


def test_x2_ablation_only_permutates_fit_context_and_never_changing_a_held_profile():
    train = pd.DataFrame({LINE: ["A", "B", "C"], GROUP: ["g1", "g2", "g3"]})
    query = pd.DataFrame({LINE: ["D"], GROUP: ["g4"]})
    fit = SimpleNamespace(train=SimpleNamespace(rows=train), query=SimpleNamespace(rows=query),
                          drug_blocks=(np.ones((3, 4)), np.ones((1, 4))))
    data = (list("ABCD"), [1, 2], np.array([[0., 5.], [1., 0.], [3., 2.], [6., 4.]]))
    full, _ = make_design(fit, data, "X2")
    permuted, _ = make_design(fit, data, "X2_context_shuffle")
    assert not np.array_equal(full.train[1], permuted.train[1])
    np.testing.assert_array_equal(full.query[1], permuted.query[1])
    assert not make_design(fit, data, "X2_drug_off")[0].query[0].any()


def test_interaction_centers_across_cell_means_not_repeat_wells():
    rows = pd.DataFrame({"cell_id": ["A", "A", "B", "C"], "parent": "p", "role": "A"})
    frame, groups, counts, index = interaction_axes(rows, ["A", "B", "C"])
    got = center_by_context(np.array([[1.], [3.], [4.], [6.]]), groups, counts, index, 3)
    assert list(frame.cell_id) == ["A", "B", "C"]
    np.testing.assert_array_equal(got.ravel(), [-2., 0., 2.])
