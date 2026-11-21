import pandas as pd

from src.r19_plan import l1_pool_support, x_context_support


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
