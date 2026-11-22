import numpy as np
import pandas as pd
import pytest

from vc_build.shift import calibrate_public
from vc_build.shift_roster import roster


def test_calibration_fit_never_gets_score_or_sentinel_drugs():
    rows = [
        {"row_id": f"{lineage}:{compound}", "lineage_group": str(lineage), "cell_id": str(lineage),
         "compound_id": str(compound), "predicted_value": compound / 15, "track": "X", "fold": 0,
         "method": "direct_structure_rna"}
        for lineage in range(36) for compound in range(15)
    ]
    frozen, report = roster(pd.DataFrame(rows))
    assert len(report["score_lineages"]) == 11
    assert len(report["sentinel_compounds"]) == 5

    def values(role):
        data = frozen[frozen.role == role]
        return data[["row_id"]].assign(endpoint_value=data.predicted_value * .5 + .2)

    fit, interval, sentinel = (values(role) for role in ("shift_fit", "interval", "sentinel"))
    result = calibrate_public(frozen, fit, interval, sentinel)
    for name, arm in result.groupby("method"):
        assert len(arm) == 110 and np.isfinite(arm.predicted_value).all()
        assert arm.interval_lower.notna().all() == name.endswith("_source_scale")
    with pytest.raises(ValueError, match="must be disjoint"):
        calibrate_public(frozen, fit, interval, pd.concat([sentinel, values("score")]))
