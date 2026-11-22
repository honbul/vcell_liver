import numpy as np
import pandas as pd

from src.r19_plan import hcc_lincs_24h_transfer
from src.r19_plan.hcc_lincs_24h_access import assigned_controls


def test_vehicle_roles_and_means_distinguish_cells_sharing_an_rna_plate(monkeypatch):
    controls = pd.DataFrame([
        dict(study="CMap2020", rna_plate="P", cell_id=cell, rna_well=f"{row}{i:02}", inst_id=f"{cell}_vehicle_{i}")
        for cell, row, size in (("A549", "A", 4), ("MCF7", "B", 3))
        for i in range(1, size + 1)
    ])
    rows = pd.DataFrame([
        dict(study="CMap2020", rna_plate="P", rna_well=f"C{i:02}", inst_id=cell, cell_id=cell)
        for i, cell in enumerate(("A549", "MCF7"), 1)
    ])
    assigned = assigned_controls(controls)
    assert assigned.groupby("cell_id").role.agg(set).tolist() == [set("ABC"), set("ABC")]

    def gene_values(record, frame, cols, genes):
        if record == "treated":
            return np.full((len(frame), 1), 10, dtype=np.float32)
        return frame.cell_id.map({"A549": 2, "MCF7": 7}).to_numpy(dtype=np.float32)[:, None]

    monkeypatch.setattr(hcc_lincs_24h_transfer, "extract_matrix", gene_values)
    for role in "ABC":
        result = hcc_lincs_24h_transfer.effect(rows, assigned.loc[assigned.role.eq(role)],
            {"matrices": {name: name for name in ("treated", "controls")}}, np.array([100]),
            {name: np.array([0]) for name in ("treated", "controls")}, role)
        np.testing.assert_array_equal(result, np.array([[8.], [3.]]))
