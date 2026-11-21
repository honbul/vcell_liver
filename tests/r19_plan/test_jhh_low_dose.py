from pathlib import Path

import pandas as pd

from src.r16_lincs.audit import INST
from src.r19_plan.hcc_lincs_24h_support import instance_rows
from src.r19_plan.jhh_low_dose_support import summaries


def test_instance_doses_are_measured_exactly(tmp_path: Path):
    metadata = pd.DataFrame([
        ("high", "10", "uM", "trt_cp"),
        ("exact", "3.33333", "uM", "trt_cp"),
        ("approximate", "3.333", "uM", "trt_cp"),
        ("converted", "100", "nM", "trt_cp"),
        ("vehicle", "", "", "ctl_vehicle"),
    ], columns=["sample_id", "pert_dose", "pert_dose_unit", "pert_type"])
    metadata = metadata.assign(rna_plate="TEST", rna_well=[f"A{i:02}" for i in range(1, 6)],
        cell_iname="JHH5", qc_pass="1", pert_time="24", pert_time_unit="h", cmap_name="DMSO")
    path = tmp_path / "instance.tsv"
    metadata.reindex(columns=INST, fill_value="").to_csv(path, sep="\t", index=False)

    original, _, _ = instance_rows(path, {"JHH5"})
    low, controls, _ = instance_rows(path, {"JHH5"}, ("3.33333", "0.1"))

    assert original.inst_id.tolist() == ["high"]
    assert dict(zip(low.inst_id, low.actual_dose_uM, strict=True)) == {"exact": "3.33333", "converted": "0.1"}
    assert controls.inst_id.tolist() == ["vehicle"]


def test_donor_support_does_not_pool_doses_for_the_same_parent():
    target = pd.DataFrame([dict(cell_id="JHH5", actual_dose_uM="3.33333", parent="A", fold=0,
        scaffold="acyclic", inst_id="target", rna_plate="plate", rna_well="A01", b1_seen=False, previous_10uM_plate=True)])
    doses = ["3.33333"] * 7 + ["1.11111"] * 8
    donor = pd.DataFrame([dict(parent="A", actual_dose_uM=dose, cell_id=f"cell{i}",
        rna_plate=f"p{i}", inst_id=f"well{i}") for i, dose in enumerate(doses)])

    support, _ = summaries(target, donor)
    counts = {row["minimum_donor_cells_and_plates"]: row["parents"] for row in support
              if row["cell"] == "JHH5" and row["dose"] == "3.33333"}

    assert counts[4] == 1
    assert counts[8] == 0
