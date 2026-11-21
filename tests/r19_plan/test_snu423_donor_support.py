import pandas as pd

from src.r11_chemical_probe.dataset import GROUP
from src.r19_plan.snu423_donor_support import mapping, strata


def test_canonical_collisions_remove_all_linked_tahoe_groups():
    structure = pd.DataFrame({GROUP: range(203)})
    query = structure.assign(parent=[f"p{i}" for i in range(203)])
    query = pd.concat((query, pd.DataFrame({GROUP: [0, 1], "parent": ["p1", "p2"]})), ignore_index=True)
    roster, collisions = mapping(query, structure)
    assert collisions == 3
    assert roster.parent.isna().tolist() == [True] * 3 + [False] * 200


def test_metadata_keeps_exact_dose_and_same_cell_plate_support_separate():
    roster = pd.DataFrame({GROUP: [0, 1], "parent": ["p0", "p1"], "min_tanimoto": [.2, .8], "fold": [0, 2]})
    rows = pd.DataFrame({"parent": ["p0", "p0", "p1"], "inst_id": ["a", "b", "c"], "cell_id": ["A549", "MCF7", "A549"],
                         "rna_plate": ["m", "n", "p"], "actual_dose_uM": ["10", "10", "5"]})
    result = strata(roster, rows).loc[lambda t: t.structure.eq("all")].set_index("scope")
    assert result.loc["measured_5uM", "fold2"] == 1
    assert result.loc["5_to_20uM_at_least_2_RNA_plates", "fold0"] == 1
    assert result.loc["5_to_20uM_at_least_2_RNA_plates_same_cell", "parents"] == 0
    assert not result.eligible.any()
