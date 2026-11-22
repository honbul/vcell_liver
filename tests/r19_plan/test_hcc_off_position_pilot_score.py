import numpy as np
import pandas as pd
import pytest

from src.r11_chemical_probe.common import sha256
from src.r19_plan.hcc_off_position_pilot_score import MODELS, evaluate, measurements


def test_pilot_aligns_measurement_ids_and_rejects_unpinned_bytes(tmp_path):
    path = tmp_path / "measurements.npz"
    accepted = pd.DataFrame({"specimen_id": ["S1", "S2"]})
    genes = np.array(["1", "2", "3"])
    matrix = np.arange(6, dtype=float).reshape(2, 3)
    np.savez(path, values=matrix[::-1, ::-1], specimen_id=["S2", "S1"], genes=genes[::-1])
    digest = sha256(path)

    np.testing.assert_array_equal(measurements(path, digest, accepted, genes), matrix)
    np.savez(path, values=matrix, specimen_id=["S1", "S1"], genes=genes)
    with pytest.raises(ValueError, match="approved hash"):
        measurements(path, digest, accepted, genes)
    with pytest.raises(ValueError, match="Measured IDs"):
        measurements(path, sha256(path), accepted, genes)


def test_pilot_crossed_uncertainty_requires_multiple_cultures():
    rows = pd.DataFrame(dict(cell=["SNU-423", "HUH7", "HUH7"], canonical_parent=["p1", "p2", "p2"],
        culture_id=["c1", "c2", "c3"], plate_id=["a", "b", "c"], batch_id=["x", "y", "z"], well=["A01", "B02", "C03"]))
    y = np.ones((3, 3, 2))
    predictions = {arm: y.copy() if arm == "calibrated_shrink" else np.zeros_like(y) for arm in MODELS}
    scored = evaluate(rows, y, np.array([False, True, True]), predictions, draws=30)

    assert set(scored["points"].loc[lambda r:r.cell.eq("SNU-423"), "resampling"]) == {"parent"}
    assert set(scored["points"].loc[lambda r:r.cell.eq("HUH7"), "resampling"]) == {"parent", "parent_and_culture"}
    pair = scored["paired"].loc[lambda r: r.arm.eq("calibrated_shrink") & r.reference.eq("B0") & r.metric.eq("RMSE")]
    assert pair.relative_gain.eq(1.).all()
