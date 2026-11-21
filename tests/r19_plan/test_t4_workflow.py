"""Check exact-dose lookup and the held plate and parent scope for T4."""

import numpy as np
import pandas as pd
import pytest

from src.r15_lincs.design import fold_id
from src.r19_plan.t4_workflow import average_conditions, logdose, lookup, scalar, validate_predict


def test_exact_dose_lookup_and_fixed_logdose_weighting():
    info = pd.DataFrame({"parent": ["p", "p", "p"], "actual_dose_uM": ["10", "5", "10"]})
    candidate = average_conditions(info, np.array([[1., 2.], [3., 8.], [3., 4.]]))
    query = info.iloc[:1]
    np.testing.assert_allclose(lookup(candidate, query), [[2, 3]])
    expected = np.average([[2., 3.], [3., 8.]], axis=0, weights=[1, np.exp(-.5)])
    np.testing.assert_allclose(logdose(candidate, query), expected[None, :])
    with pytest.raises(ValueError, match="exact dose"):
        lookup(candidate, query.assign(actual_dose_uM="7"))
    np.testing.assert_allclose(scalar(info, np.ones((3, 2), "float32"), np.full((3, 2), .7, "float32")), .7)


def test_prediction_plan_rejects_same_plate_donor_and_held_source_parent():
    options = [f"parent-{i}" for i in range(50)]
    query = next(p for p in options if fold_id(p, "R15-outer", 3) == 0)
    sub = [p for p in options if fold_id(p, "R15-outer", 3) != 0]
    inner = next(p for p in sub if fold_id(p, "R15-inner-0", 5) != 0)
    select = next(p for p in sub if fold_id(p, "R15-inner-0", 5) == 0)

    def row(name, plate, well, parent, hours, **kwargs):
        return dict(inst_id=name, study="GSE92742", cell_id="HEPG2", rna_plate=plate, rna_well=well,
                    parent=parent, hours=hours, **kwargs)

    t = {
        "source6": [row("t1", "early", "A01", inner, "6"), row("t2", "early", "A02", select, "6")],
        "source24": [row("t3", "late", "A01", inner, "24", phase="inner_fit"),
                     row("t4", "late", "A02", select, "24", phase="select")],
        "donor6": [row("t5", "early", "A03", query, "6")],
        "direct24": [row("t6", "late", "A03", query, "24")],
        "query": [row("t7", "held", "A01", query, "24")],
        "fit_controls": [row("v1", "early", "H01", "", "6", role="A"),
                         row("v2", "late", "H01", "", "24", role="A")],
        "select_controls": [row("v3", "late", "H02", "", "24", role="B")],
    }
    t = {name: pd.DataFrame(rows) for name, rows in t.items()}
    validate_predict(t, "f0-A-held")
    with pytest.raises(ValueError, match="physical plate"):
        validate_predict(dict(t, direct24=t["direct24"].assign(rna_plate="held")), "f0-A-held")
    with pytest.raises(ValueError, match="independent parent"):
        validate_predict(dict(t, source6=t["source6"].assign(parent=query)), "f0-A-held")
