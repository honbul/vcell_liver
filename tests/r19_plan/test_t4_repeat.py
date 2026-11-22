import numpy as np
import pandas as pd

from src.r19_plan.t4_repeat import fit_pairs


def test_calibration_sides_have_disjoint_physical_plates():
    rows = pd.DataFrame([dict(parent=f"p{i}", actual_dose_uM="10", rna_plate=plate, study="source", cell_id="PHH")
                         for i in range(5) for plate in ("a", "b", "b")])
    values = np.tile([1., 3., 5.], 5)[:, None]
    cases, repeat, result = fit_pairs(rows, values)
    assert len(cases) == 5
    for left, right in zip(repeat[:, 0], result[:, 0]):
        assert {left, right} == {1., 4.}
