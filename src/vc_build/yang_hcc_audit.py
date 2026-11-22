"""Register Yang HCC organoid RNA and assay keys without loading response values."""

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from .expanded_panel_cv import registered_inputs
from .external import KEY, _validate
from .function_data import canonical_chemistry
from .io import publish_json, sha256
from .v0plus_broutier import writer

COHORT = "YANG2024_HCC"
ARTIFACTS = {
    "RNA.meta.txt": "b747faca3a2475d2fbf695eac7c63109fa394a7406dfea1a106e5c1b0969e449",
    "RNA.rds": "e17107ff49e5fde79777f0d1f4e871996221f3be38f3f5ee2215dea11b9f42fc",
    "mmc3.xlsx": "8ec282c45c864d55af813bbd819a8cbd04c3130b89be55d00d1e15dcc5571d9b",
    "mmc8.xlsx": "e22cf8151c415645c7aa2d6967d43f90e54725e18e1639768327d3022392a696",
}
REFERENCE_SMILES = {
    "Lenvatinib": (9823820, "COC1=CC2=NC=CC(=C2C=C1C(=O)N)OC3=CC(=C(C=C3)NC(=O)NC4CC4)Cl"),
    "Sorafenib": (216239, "CNC(=O)C1=NC=CC(=C1)OC2=CC=C(C=C2)NC(=O)NC3=CC(=C(C=C3)Cl)C(F)(F)F"),
    "Regorafenib": (11167602, "CNC(=O)C1=NC=CC(=C1)OC2=CC(=C(C=C2)NC(=O)NC3=CC(=C(C=C3)Cl)C(F)(F)F)F"),
    "Apatinib": (45139106, "CS(=O)(=O)O.C1CCC(C1)(C#N)C2=CC=C(C=C2)NC(=O)C3=C(N=CC=C3)NCC4=CC=NC=C4"),
    "Pemigatinib": (86705695, "CCN1C2=C3C=C(NC3=NC=C2CN(C1=O)C4=C(C(=CC(=C4F)OC)OC)F)CN5CCOCC5"),
    "Ivosidenib": (71657455, "C1CC(=O)N([C@@H]1C(=O)N(C2=CC(=CN=C2)F)[C@@H](C3=CC=CC=C3Cl)C(=O)NC4CC(C4)(F)F)C5=NC=CC(=C5)C#N"),
}


def worksheet(path, name):
    return load_workbook(path, read_only=True, data_only=True)[name]


def registry(assets, rna, source_genes):
    meta = pd.read_csv(assets / "RNA.meta.txt", sep="\t", dtype=str).set_index("SampleID", verify_integrity=True)
    s1 = worksheet(assets / "mmc3.xlsx", "Table S1")
    data = s1.values
    columns = next(data)
    donors = pd.DataFrame(data, columns=columns)[["Patient number", "Age", "Pathology"]]
    donors = donors.rename(columns={"Patient number": "CaseID"}).set_index("CaseID", verify_integrity=True)
    if len(donors) != 144 or not set(meta.CaseID) <= set(donors.index):
        raise ValueError("Unmapped Yang donor")
    s6 = worksheet(assets / "mmc8.xlsx", "Table S6")
    drugs, metrics = tuple(s6.iter_rows(max_row=2, values_only=True))
    columns = {name.strip(): i + 2 for i, name in enumerate(drugs) if isinstance(name, str)}
    if set(columns) != ({"Bevacizumab"} | set(REFERENCE_SMILES)) or any(
            metrics[col - 1].strip() != "Normalized AUC" for col in columns.values()):
        raise ValueError("Unexpected Yang normalized AUC header")
    row_ids = {value: row for row, (value,) in enumerate(
        s6.iter_rows(min_row=3, max_col=1, values_only=True), 3,
    )}
    if len(row_ids) != 376 or any(not re.fullmatch(r"P\d+C\d+", key) for key in row_ids):
        raise ValueError("Unexpected Yang PDO roster")
    matched = meta.join(donors, on="CaseID")
    matched["pdo_id"] = matched.index.str.removesuffix("_O")
    matched = matched[matched.index.str.endswith("_O") & matched.pdo_id.isin(row_ids)]
    cohort = matched.loc[matched.Pathology.eq("HCC") & matched.Age.ge(18)].copy()
    if not all(re.fullmatch(rf"{donor}C\d+", pdo) for pdo, donor in zip(cohort.pdo_id, cohort.CaseID)):
        raise ValueError("Conflicting Yang sample and donor identifiers")
    cohort["response_row"] = cohort.pdo_id.map(row_ids)
    cohort["cell_id"] = COHORT + ":" + cohort.pdo_id
    cohort["lineage_group"] = COHORT + ":" + cohort.CaseID
    data = pd.read_csv(rna, sep="\t", low_memory=False)
    if set(data.columns) - {"symbol"} != set(meta.index):
        raise ValueError("RNA matrix columns differ from author sample metadata")
    by_gene = data.loc[data.symbol.isin(source_genes) & ~data.symbol.duplicated(keep=False),
                       ["symbol", *cohort.index]]
    matrix = by_gene.set_index("symbol").sort_index().T
    if matrix.shape[1] < 10000 or not np.isfinite(matrix.to_numpy()).all() or matrix.lt(0).any().any():
        raise ValueError("Insufficient or invalid Yang log2 TPM")
    matrix.index = pd.Index(cohort.cell_id, name="cell_id")
    return cohort, matrix, matched, columns


def run(args):
    for name, digest in ARTIFACTS.items():
        if sha256(args.assets / name) != digest:
            raise ValueError(f"Yang asset changed: {name}")
    _, core, fit_rna, expanded = registered_inputs(args, include_extension=True)
    cohorts, rna, common, columns = registry(args.assets, args.rna, fit_rna.columns)
    sources = pd.concat([core, expanded], ignore_index=True)
    chemicals, parents = [], []
    for name, col in columns.items():
        if name == "Bevacizumab":
            chemicals.append({"response_label": name, "reason": "monoclonal antibody"})
            continue
        cid, source_smiles = REFERENCE_SMILES[name]
        smiles, parent, *_ = canonical_chemistry(source_smiles)
        matched = sources[sources.compound_id.eq(parent)]
        counts = matched.groupby("source_id").cell_id.nunique().to_dict()
        entry = {"response_label": name, "pubchem_cid": cid, "parent": parent, "source_cell_counts": counts,
                 "in_core": parent in set(core.compound_id)}
        chemicals.append(entry)
        if counts:
            parents.append({"response_label": name, "compound_id": parent, "isomeric_smiles": smiles,
                            "response_column": col, "in_core": entry["in_core"]})
    if len(parents) < 3:
        raise ValueError("Insufficient exact-source support for the Yang assay")
    query = cohorts[["cell_id", "lineage_group", "response_row"]].merge(pd.DataFrame(parents), how="cross")
    query["row_id"] = query.cell_id + "|" + query.compound_id
    _validate(sources, query)
    if query.duplicated(KEY).any():
        raise ValueError("Unexpected duplicate organoid-drug query rows")
    receipt = {"cohort": COHORT, "mendeley_doi": "10.17632/rv2w3dv9rs.2", "rna_license": "CC BY 4.0",
               "download_sha256": ARTIFACTS, "converted_rna_sha256": sha256(args.rna),
               "rna_response_overlap_by_histology": common.Pathology.value_counts().to_dict(),
               "rna_response_models": len(common), "rna_response_donors": common.CaseID.nunique(),
               "hcc_pdos": len(cohorts), "hcc_donors": cohorts.CaseID.nunique(), "common_genes": rna.shape[1],
               "compound_support": chemicals, "expected_pdo_drug_keys": len(query),
               "artifacts": writer(args.out, {"query.parquet": query, "rna.parquet": rna})}
    publish_json(args.out / "receipt.json", receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original", "core", "extension", "assets", "rna", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
