import numpy as np
import pandas as pd

from src.r13_optimization.lincs_transfer import aggregate_donor, exclusion_audit, gene_map


META = ("pert_id", "cmap_name", "compound_aliases", "canonical_smiles", "inchi_key", "pubchem_cid")
DONORS = ("pert_id", "cmap_name", "smiles_used", "inchi_keys", "drug_group")


def test_exclude_same_parent_alias_chain_cid_and_unresolved_whole_group():
    catalog = pd.DataFrame([("Held-Out", "CC(=O)O", 123.0)],
                           columns=["drug", "canonical_smiles", "pubchem_cid"])
    metadata = pd.DataFrame([
        ("salt", "salt compound", "Hidden bridge", "CC(=O)[O-].[Na+]", "", ""),
        ("bridge", "hidden-bridge", "second member", "CCCC", "", ""),
        ("cid", "different string", "", "CCCCC", "", "123"),
        ("name", "different string", "extra | Held-Out", "CCCCCC", "", ""),
    ], columns=META)
    donors = pd.DataFrame([
        ("group-hit", "second member", "CCCC", "", "g1"),
        ("group-link", "other", "OC", "", "g1"),
        ("cid", "source cid", "CCCCC", "", "g2"),
        ("name", "source name", "CCCCCC", "", "g3"),
        ("unresolved", "zero structure", "", "", "g4"),
        ("otherwise-known", "source structure", "CCO", "", "g4"),
        ("clean", "retained source", "CCN", "", "g5"),
    ], columns=DONORS)
    audited, receipt = exclusion_audit(catalog, metadata, donors)
    assert audited.loc[audited.retained, "pert_id"].tolist() == ["clean"]
    exclusions = audited.set_index("pert_id").exclusion
    assert exclusions["group-link"] == "transitive_alias"
    assert exclusions["cid"] == "cid"
    assert exclusions["name"] == "name"
    assert exclusions["otherwise-known"] == "ambiguous_or_unresolved_group"
    assert receipt["tahoe_catalog_cids"] == 1


def test_ensembl_mapping_excludes_symbols_and_duplicate_ensembl_ids():
    donor = pd.DataFrame({"gene_symbol": ["A", "B", "C", "D", "E", "F"],
                          "ensembl_id": ["ENSG1", "ENSG2", "ENSG2", "", "ENSG4", "ENSG6"]})
    target = pd.DataFrame({"gene_symbol": list("XBCDEEF"), "token_id": range(10, 17),
                           "ensembl_id": ["ENSG1", "ENSG2", "ENSG3", "", "ENSG4", "ENSG4", "ENSG8"]})
    result = gene_map(donor, target)
    assert result[["donor_col", "token_id", "ensembl_id"]].to_dict("records") == [
        {"donor_col": 0, "token_id": 10, "ensembl_id": "ENSG1"}
    ]


def test_phase1_cid_extends_id_matching_even_without_a_lincs2020_cid_field():
    catalog = pd.DataFrame([("held-out", "CCO", 42)], columns=["drug", "canonical_smiles", "pubchem_cid"])
    metadata = pd.DataFrame([("link", "donor", "", "CO", "")], columns=META[:-1])
    donors = pd.DataFrame([("link", "new name", "CO", "", "g1"),
                           ("stay", "unrelated", "C#N", "", "g2")], columns=DONORS)
    phase1 = pd.DataFrame([("link", "older name", "", "42")],
                          columns=["pert_id", "pert_iname", "inchi_key", "pubchem_cid"])
    audited, receipt = exclusion_audit(catalog, metadata, donors, phase1)
    assert audited.loc[audited.retained, "pert_id"].tolist() == ["stay"]
    assert receipt["direct_lincs2020_cid_field"] is None
    assert receipt["donor_with_direct_cid"] == receipt["donor_direct_cid_matches"] == 1


def test_donor_signatures_balance_repeats_and_center_among_eligible_plate_groups():
    rows = pd.DataFrame({"row_id": range(10), "cell": ["A"] * 7 + ["B"] * 3,
                         "det_plate": ["p1"] * 5 + ["p2"] * 2 + ["p3"] * 3,
                         "drug_group": ["x", "x", "x", "y", "z", "x", "y", "x", "y", "z"]})
    control = np.full((10, 2), 10.)
    delta = np.array([0., 1., 2., 3., 5., 10., 2., 4., 4., 4.])
    treated = control + delta[:, None]
    groups, signature, n, views, receipt = aggregate_donor(rows, control, treated, np.array([0]))
    np.testing.assert_array_equal(groups, ["x", "y", "z"])
    np.testing.assert_allclose(signature[:, 0], [-1., 0., 1.])
    np.testing.assert_array_equal(n, [2., 2., 2.])
    np.testing.assert_allclose(views[0][0].ravel(), [-2., 0., 2.])
    np.testing.assert_allclose(views[1][0].ravel(), [0., 0., 0.])
    assert receipt["plate_condition_groups"] == 6
