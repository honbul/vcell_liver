import gzip

import pandas as pd

from vc_build.cross_source_data import (
    MODEL_COLS, parse_cellosaurus_identity, prism_parent, protected_adult_families, source_structures,
)
from vc_build.function_data import canonical_chemistry


def test_isomeric_parent_preserves_stereochemistry_and_groups_connectivity():
    left = canonical_chemistry("C[C@H](O)C(O)=O")
    right = canonical_chemistry("C[C@@H](O)C(O)=O")
    assert left[1] != right[1]
    assert left[2] == right[2]
    assert left == canonical_chemistry("C[C@H](O)C([O-])=O.[Na+]")


def test_prism_repeated_annotations_restore_only_one_verified_parent():
    drug = "C[C@H](O)C(O)=O"
    alternative = "C[C@H](O)C([O-])=O.[Na+]"
    opposite = "C[C@@H](O)C(O)=O"
    exact = canonical_chemistry(drug)[1]
    assert prism_parent(f"{drug}, {alternative}, {drug}")[1] == exact
    assert prism_parent(f"{drug}, {opposite}") is None
    assert prism_parent(f"{drug}, ") is None
    metadata = pd.DataFrame([
        ("same", "single", f"{drug}, {alternative}"),
        ("ambiguous", "single", f"{drug}, {opposite}"),
        ("marked", "one + two", f"{drug}, {drug}"),
        ("conflict", "single", f"{drug}, {drug}"),
        ("conflict", "single", opposite),
    ], columns=["id", "name", "smiles"])
    selected = source_structures(metadata, "id", "name", "smiles", pd.DataFrame({"compound_id": [exact]}),
                                 resolve=prism_parent)
    assert selected.native_compound_id.tolist() == ["same"]


def test_query_roster_excludes_parent_alias_and_holds_same_donor_together(tmp_path):
    registry = tmp_path / "cellosaurus.txt.gz"
    with gzip.open(registry, "wt") as handle:
        handle.write(
            "ID   OldLine\nAC   CVCL_Q111\nSY   Parent line alias\nDR   Cosmic; 4040\n//\n"
            "ID   Daughter\nAC   CVCL_Q222\nHI   CVCL_Q111 ! OldLine\n//\n"
            "ID   NewLine\nAC   CVCL_Q333\n//\n"
            "ID   RelatedNewLine\nAC   CVCL_Q444\nOI   CVCL_Q333 ! NewLine\n//\n"
        )

    def cell(label, family, accession):
        return {
            "model_id": label, "family_id": family, "source_name": label, "cellosaurus_id": accession,
            "root_ancestor": accession, "same_individual": "", "depmap_model_id": "", "depmap_patient_id": "",
            "cellosaurus_name": label, "table_s1_age_years": "44", "included": "True",
        }

    cells = pd.DataFrame([
        cell("Daughter", "old", "CVCL_Q222"),
        cell("NewLine", "new", "CVCL_Q333"),
        cell("RelatedNewLine", "new", "CVCL_Q444"),
    ])
    roster = pd.DataFrame([{"source_id": "GDSC2", "name": "parent-line-alias",
                            "source_key": "SIDM9999", "source_key_2": ""}])
    audit = protected_adult_families(cells, parse_cellosaurus_identity(registry),
                                   pd.DataFrame(columns=MODEL_COLS), roster, 1)
    assert audit.set_index("cell_id").source_new.to_dict() == {
        "Daughter": False, "NewLine": True, "RelatedNewLine": True
    }
    assert audit.loc[audit.confirmation_query, "cell_id"].tolist() == ["NewLine"]
