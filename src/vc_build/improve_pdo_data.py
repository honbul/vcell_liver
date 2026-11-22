"""Register the public IMPROVE non-HCC organoids as a distinct adaptation cohort."""

import argparse
from collections import Counter
import csv
from io import TextIOWrapper
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from rdkit import RDLogger

from .cross_source_data import MIXTURE, unambiguous_parent
from .expanded_panel_cv import RECEIPTS, registered_inputs
from .function_data import new_output
from .io import publish_json, sha256

VERSION = "v-1.0.0"
MEMBERS = {
    "ReadMe.txt": (3343, 0xfec59d2a),
    "y_data/response.tsv": (46326769, 0x2a0bd044),
    "x_data/cancer_gene_expression.tsv": (451714317, 0x183aa2cc),
    "x_data/drug_SMILES.tsv": (198552, 0x04751c58),
    "x_data/drug_info.tsv": (338284, 0xc85d8c46),
}
STUDIES = {"IMPROVE dataset 1", "IMPROVE dataset 2", "IMPROVE dataset 3", "Pilot 1"}


def study_data(archive):
    with archive.open(f"{VERSION}/y_data/response.tsv") as f:
        selected = pd.read_csv(f, sep="\t", dtype=str, usecols=[
            "source", "study", "improve_sample_id", "improve_chem_id", "auc",
        ], keep_default_na=False)
    selected = selected[selected.source.isin(STUDIES)].rename(columns={
        "improve_sample_id": "cell_id", "study": "experiment_id",
    })
    ids = selected.cell_id
    is_pdo = ids.str.startswith(("CO-", "CR-", "PDM-", "HN-", "SARC"))
    if (~is_pdo & ~ids.str.startswith("ACH-")).any() or set(selected.source) != STUDIES:
        raise ValueError("Unknown sample type or missing IMPROVE study")
    pdo = selected[is_pdo].copy()
    pdo["cancer_type"] = "colorectal"
    pdo.loc[pdo.cell_id.str.startswith("HN-"), "cancer_type"] = "head_and_neck"
    pdo.loc[pdo.cell_id.str.startswith("SARC"), "cancer_type"] = "sarcoma"
    pdo["endpoint_value"] = pd.to_numeric(pdo.auc, errors="coerce")
    return pdo.drop(columns="auc"), selected[~is_pdo]


def map_drugs(archive, wanted):
    with archive.open(f"{VERSION}/x_data/drug_SMILES.tsv") as f:
        structures = pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False)
    with archive.open(f"{VERSION}/x_data/drug_info.tsv") as f:
        metadata = pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False)
    mixtures = set(metadata.loc[metadata.NAME.str.contains(MIXTURE), "improve_chem_id"])
    structures = structures[structures.improve_chem_id.isin(set(wanted) - mixtures)]
    records = []
    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")
    try:
        for drug, group in structures.groupby("improve_chem_id", sort=True):
            smiles = set(group.SMILES) | set(group.canSMILES)
            entries = [unambiguous_parent(item) for item in sorted(smiles - {""})]
            if not entries or any(x is None for x in entries) or len({x[1] for x in entries}) != 1:
                continue
            structure, exact, _, _ = min(entries)
            records.append({"improve_chem_id": drug, "compound_id": exact, "isomeric_smiles": structure})
    finally:
        RDLogger.EnableLog("rdApp.warning")
        RDLogger.EnableLog("rdApp.error")
    return pd.DataFrame(records)


def expression(archive, wanted, core_genes):
    with archive.open(f"{VERSION}/x_data/cancer_gene_expression.tsv") as f:
        reader = csv.reader(TextIOWrapper(f, encoding="utf-8-sig"), delimiter="\t")
        ensembl, entrez, names = [next(reader) for _ in range(3)]
        if not (len(ensembl) == len(entrez) == len(names)) or any(row[0] for row in (ensembl, entrez, names)):
            raise ValueError("Unrecognized IMPROVE transcript identifier header")
        counts = Counter(names)
        positions = [i for i, gene in enumerate(names) if gene in core_genes and counts[gene] == 1
                     and ensembl[i].startswith("ENSG") and entrez[i].isdigit()]
        values = {}
        for line in reader:
            if line and line[0] in wanted:
                if line[0] in values or len(line) != len(names):
                    raise ValueError("Ambiguous IMPROVE baseline profile")
                values[line[0]] = [float(line[i]) if line[i] not in {"", "NA", "NaN"} else np.nan for i in positions]
    frame = pd.DataFrame.from_dict(values, orient="index", columns=[names[i] for i in positions])
    retained = frame.columns[np.isfinite(frame.to_numpy()).all(axis=0)]
    if len(retained) < 1000:
        raise ValueError("Insufficient named RNA features overlapping the training core")
    return frame.loc[:, retained].sort_index().rename_axis("cell_id").astype("float32")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "original", "core", "extension", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    _, clean, core_rna, extra = registered_inputs(args, include_extension=True)
    with ZipFile(args.archive) as archive:
        for member, expected in MEMBERS.items():
            available = archive.getinfo(f"{VERSION}/{member}")
            if (available.file_size, available.CRC) != expected:
                raise ValueError(f"Unexpected Zenodo release member: {member}")
        readme = archive.read(f"{VERSION}/ReadMe.txt").decode()
        if "Cancer type information of PDO" not in readme or "Use the \"auc\" column" not in readme:
            raise ValueError("Unexpected author description of the endpoint")
        response, reference_cells = study_data(archive)
        rna = expression(archive, set(response.cell_id), set(core_rna.columns))
        chemistry = map_drugs(archive, response.improve_chem_id)
    rows = response.merge(chemistry, on="improve_chem_id", validate="many_to_one")
    key = ["source", "cell_id", "compound_id"]
    ambiguous = rows.duplicated(key, keep=False)
    admitted = rows[np.isfinite(rows.endpoint_value) & rows.cell_id.isin(rna.index) & ~ambiguous].copy()
    admitted["lineage_group"] = admitted.cell_id
    admitted["row_id"] = admitted.source + "|" + admitted.cell_id + "|" + admitted.compound_id
    admitted["endpoint_direction"] = "lower_is_more_sensitive"
    admitted["endpoint_id"] = "normalized_organoid_AUC"
    admitted["culture_format"] = "3D"
    available = set(clean.compound_id) | set(extra.compound_id)
    studies = {}
    for source, group in response.groupby("source", sort=True):
        prepared = admitted[admitted.source.eq(source)]
        core_only = prepared[prepared.compound_id.isin(clean.compound_id)]
        expanded = prepared[prepared.compound_id.isin(available)]
        studies[source] = {
            "pdo_rows": len(group), "pdo_models": int(group.cell_id.nunique()),
            "rna_matched_models": int(group.loc[group.cell_id.isin(rna.index), "cell_id"].nunique()),
            "exact_drug_matched_rows": int(rows.source.eq(source).sum()),
            "pdo_rows_after_validation": len(prepared), "validated_drugs": int(prepared.compound_id.nunique()),
            "core_overlap_rows": len(core_only), "core_overlap_drugs": int(core_only.compound_id.nunique()),
            "expanded_overlap_rows": len(expanded), "expanded_overlap_drugs": int(expanded.compound_id.nunique()),
            "separate_reference_ccl_rows": int(reference_cells.source.eq(source).sum()),
        }
    with new_output(args.out) as tmp:
        rna.loc[sorted(admitted.cell_id.unique())].to_parquet(tmp / "rna.parquet")
        admitted.sort_values("row_id", ignore_index=True).to_parquet(tmp / "observations.parquet", index=False)
        files = {p.name: sha256(p) for p in sorted(tmp.glob("*.parquet"))}
        publish_json(tmp / "receipt.json", {
            "doi": "10.5281/zenodo.22920452", "release": VERSION, "license": "CC-BY-4.0",
            "selected_archive_sha256": sha256(args.archive), "publisher_full_archive_md5": "3b66f33df039f4ff306019f73ea9efe9",
            "zip_members": {name: {"bytes": size, "crc32": f"{crc:08x}"} for name, (size, crc) in MEMBERS.items()},
            "studies": studies, "pdo_models": int(admitted.cell_id.nunique()), "rna_axis_genes": len(rna.columns),
            "validated_rows": len(admitted), "same_model_multi_study": int(
                admitted.groupby("cell_id").source.nunique().ge(2).sum()),
            "canonical_parent_collision_rows": int(ambiguous.sum()),
            "excluded_no_structure_rows": len(response) - len(rows), "original_cell_panel_receipts": RECEIPTS,
            "organoid_types": admitted.groupby("cancer_type").cell_id.nunique().to_dict(),
            "artifacts": files, "endpoint": "publisher normalized AUC, lower is more sensitive",
        })


if __name__ == "__main__":
    main()
