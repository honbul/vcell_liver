import pandas as pd

from src.r19_plan.jhh7_position_attribution import paired_locations


def test_wrong_parent_control_requires_same_source_plate_and_exact_dose():
    query = pd.DataFrame([("target", "A", "T", "B01", "1")],
                         columns=["inst_id", "parent", "rna_plate", "rna_well", "actual_dose_uM"])
    source = pd.DataFrame([
        ("exact", "A", "P", "C01", "1"),
        ("valid_wrong", "B", "P", "B01", "1"),
        ("other_plate_exact", "A", "Q", "C01", "1"),
        ("wrong_dose", "B", "Q", "B01", "0.1"),
        ("same_parent_same_position", "A", "R", "B01", "1"),
    ], columns=query.columns)

    links = paired_locations(query, source)

    assert links[["target_id", "off_id", "same_id", "source_plate"]].values.tolist() == [
        ["target", "exact", "valid_wrong", "P"],
    ]
