import pandas as pd

from src.r19_plan import l1_pool_support, x_basal_support, x_context_support


def test_context_search_requires_five_contexts_and_excludes_shared_physical_plates():
    cells = ["HEPG2", "A", "B", "C", "D", "E"]
    rows = pd.DataFrame([
        (cell, f"P{parent}", f"{cell}_PLATE{parent % 3}", f"A{parent}", f"{cell}:{parent}")
        for cell in cells for parent in range(150 if cell != "E" else 115)
    ], columns=["cell_id", "parent", "plate", "well", "inst_id"])
    rows.loc[len(rows)] = ["A", "outside", "shared", "B01", "bad1"]
    rows.loc[len(rows)] = ["B", "outside", "shared", "B02", "bad2"]
    eligible, counts, parents, excluded = x_context_support.inventory(rows)
    search, chosen, matched = x_context_support.choose_loco(eligible, counts, parents)
    assert excluded == 1 and "shared" not in set(eligible.plate)
    assert len(search) == 5 and search.eligible.all()
    assert chosen == ("HEPG2", "A", "B", "C", "D")
    assert matched.parent.nunique() == 150 and len(matched) == 750


def test_l1_support_counts_canonical_parents_not_wells_or_neighboring_doses():
    rows = pd.DataFrame({"study": "GSE92742", "rna_plate": "P", "cell_id": "HEPG2", "hours": 6,
                         "inst_id": list("abcde"), "parent": ["1", "1", "2", "3", "4"],
                         "actual_dose_uM": ["10", "10", "10", "10.001", ""]})
    summary, inventory = l1_pool_support.pools(rows)
    assert len(inventory) == 2
    assert [row["eligible_wells"] for row in summary] == [3, 0, 0]
    assert summary[0]["eligible_parents"] == 2
    assert summary[0]["known_dose_wells"] == 4


def test_basal_identity_requires_one_profile_without_catalog_conflicts():
    source = pd.DataFrame({"cell_id": ["C1", "C2", "C3"], "provider_catalog_id": ["HB1", "HB2", "-666"],
                           "subtype": ["a", "b", "c"]})
    models = pd.DataFrame([
        ["M1", "C 1", "C1", "RR1", "HB-1", "A"], ["M2", "C2", "C2", "RR2", "HB-5", "B"],
        ["M3", "C3", "C3", "RR3", "", "C"],
    ], columns=x_basal_support.IDENTITY)
    profiles = pd.DataFrame([
        ["M1", "seq-1", "cond-1", "yes"], ["M2", "seq-2", "cond-2", "yes"],
        ["M3", "seq-3", "cond-3", "yes"], ["M3", "seq-4", "cond-4", "yes"],
    ], columns=x_basal_support.PROFILE)
    result = x_basal_support.resolve_contexts(set(source.cell_id), source, models, profiles).set_index("cell")
    assert result.eligible.tolist() == [True, False, False]
    assert result.catalog_conflict.tolist() == [False, True, False]
    assert result.loc["C1", "sequence_id"] == "seq-1"
