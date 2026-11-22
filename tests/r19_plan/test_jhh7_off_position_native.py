import numpy as np
import pandas as pd

from src.r19_plan.i_score import BOOTSTRAPS
from src.r19_plan.jhh7_off_position_native import CALIBRATED, REFERENCES, best_comparison


def test_best_vc_is_reselected_per_parent_draw():
    rows = pd.DataFrame(dict(parent=["A", "B", "C"], inst_id=["a", "b", "c"]))
    y = np.ones((3, 2))
    negative = np.zeros_like(y)
    arrays = {"B0": negative, CALIBRATED: y, **{arm: negative for arm in REFERENCES}}
    arrays[REFERENCES[0]] = np.array([[.9, .9], [.9, .9], [0., 0.]])
    arrays[REFERENCES[1]] = np.array([[0., 0.], [0., 0.], [.9, .9]])

    summary, frequencies = best_comparison(rows, y, arrays, "all")

    assert all(row["best_at_point"] == REFERENCES[0] for row in summary)
    for metric in ("rmse", "r2_pooled"):
        tallies = {row["model"]: row["wins"] for row in frequencies if row["metric"] == metric}
        assert 0 < tallies[REFERENCES[0]] < BOOTSTRAPS
        assert 0 < tallies[REFERENCES[1]] < BOOTSTRAPS
        assert sum(tallies.values()) == BOOTSTRAPS
