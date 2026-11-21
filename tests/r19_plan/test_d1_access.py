import pandas as pd
import pytest

from src.r19_plan.d1_access import CONTROL_KEYS, check_split, equal_roles, matched
from src.r19_plan.r18_preflight import GEO


def fixtures():
    wells = pd.DataFrame([
        [GEO, "fit", "P1", "A01", "fit-parent", "1", "inner_fit", "10"],
        [GEO, "select", "P1", "A02", "select-parent", "2", "select", "20"],
        [GEO, "held", "P1", "A03", "held-parent", "0", "", "30"],
    ], columns=["study", "inst_id", "rna_plate", "rna_well", "parent", "outer_fold", "inner_phase", "cache_index"])
    controls = pd.DataFrame([
        [GEO, "A", "P1", "B01", "A", 101], [GEO, "B", "P1", "B02", "B", 102],
        [GEO, "C", "P1", "B03", "C", 103],
    ], columns=[*CONTROL_KEYS, "level3_row_index"])
    held = wells.iloc[2:].drop(columns="inner_phase")
    return controls, {
        "predict": {"inner_and_outer_labels": wells.iloc[:2], "query_features": held.drop(columns="cache_index"),
                    "fit_controls": controls.iloc[:1], "select_controls": controls.iloc[1:2]},
        "score": {"score_labels": held, "score_controls": controls.iloc[2:]},
    }


def test_keeps_equal_separate_physical_controls_and_unchanged_query():
    controls, split = fixtures()
    assert equal_roles(controls).loc["P1"].tolist() == [1, 1, 1]
    assert matched(controls[CONTROL_KEYS], controls).equals(controls)
    evidence = check_split(split, "f0-A")
    assert [evidence[f"{role}_controls"] for role in ("fit", "select", "score")] == [1, 1, 1]


def test_rejects_role_count_or_held_axis_changes():
    controls, split = fixtures()
    with pytest.raises(ValueError, match="equal independent"):
        equal_roles(controls.iloc[:2])
    split["score"]["score_labels"] = split["score"]["score_labels"].assign(parent="other-parent")
    with pytest.raises(ValueError, match="outer query axis"):
        check_split(split, "f0-A")


def test_rejects_score_control_overlap_even_if_role_labels_differ():
    _, split = fixtures()
    split["score"]["score_controls"] = split["score"]["score_controls"].assign(rna_well="B01")
    with pytest.raises(ValueError, match="physical RNA well"):
        check_split(split, "f0-A")
