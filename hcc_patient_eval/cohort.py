"""Prepare one expression profile per adult TCGA-LIHC primary tumor."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path):
    with path.open("rb") as source:
        result = sha256()
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_receipt(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def prepare(gdc, cdr, output):
    sheet = gdc / "gdc_sample_sheet.2022-04-18.tsv"
    clinical = pd.read_excel(cdr, sheet_name=0).dropna(subset=["bcr_patient_barcode"])
    clinical = clinical.loc[clinical.type.eq("LIHC")].set_index("bcr_patient_barcode", verify_integrity=True)
    clinical = clinical[["age_at_initial_pathologic_diagnosis", "ajcc_pathologic_tumor_stage",
                         "OS", "OS.time", "PFI", "PFI.time"]]
    sample = pd.read_csv(sheet, sep="\t")
    sample = sample.loc[sample["Project ID"].eq("TCGA-LIHC") & sample["Sample Type"].eq("Primary Tumor")].copy()
    sample = sample.loc[sample["Sample ID"].str.match(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-01[A-Z]$")]
    if not sample["Sample ID"].str.slice(0, 12).equals(sample["Case ID"]):
        raise ValueError("GDC sample and participant identifiers disagree")
    sample = sample.sort_values(["Case ID", "Sample ID", "File ID"])
    primary = sample.drop_duplicates("Case ID").set_index("Case ID", verify_integrity=True)
    adult = pd.to_numeric(clinical.age_at_initial_pathologic_diagnosis, errors="coerce").ge(18)
    primary = primary.join(clinical).loc[lambda rows: rows.index.isin(clinical.index[adult])]
    files = {}
    columns = ["gene_id", "gene_name", "gene_type", "fpkm_unstranded", "tpm_unstranded"]
    expression = {kind: [] for kind in ("fpkm_unstranded", "tpm_unstranded")}
    genes = None
    for _, row in primary.iterrows():
        path = gdc / row["File ID"] / row["File Name"]
        matrix = pd.read_csv(path, sep="\t", comment="#", usecols=columns)
        matrix = matrix.loc[matrix.gene_id.str.startswith("ENSG")].reset_index(drop=True)
        annotation = matrix[columns[:3]]
        if genes is None:
            genes = annotation
        if not genes.equals(annotation):
            raise ValueError(f"GDC gene axis differs in {path.name}")
        for kind, rows in expression.items():
            values = matrix[kind].to_numpy(dtype=np.float32)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"Invalid expression values in {path.name}")
            rows.append(values)
        files[str(path.relative_to(gdc))] = digest(path)
    if not len(primary):
        raise ValueError("No evaluable LIHC participants")
    output.mkdir(parents=True, exist_ok=False)
    genes.to_csv(output / "genes.tsv", sep="\t", index=False)
    primary.to_csv(output / "participants.tsv", sep="\t", index_label="participant")
    for kind, rows in expression.items():
        np.save(output / f"{kind}.npy", np.stack(rows), allow_pickle=False)
    write_receipt(output / "receipt.json", {
        "sample_sheet_sha256": digest(sheet),
        "cdr_sha256": digest(cdr),
        "gdc_source": "GDC TCGA-LIHC public STAR augmented counts, April 2022, GENCODE v36",
        "eligible_gdc_primary_samples": len(sample),
        "unique_gdc_primary_participants": len(sample["Case ID"].unique()),
        "adult_participants": len(primary),
        "genes": len(genes),
        "gdc_files": files,
        "outputs": {p.name: digest(p) for p in sorted(output.iterdir())},
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gdc", required=True, type=Path)
    parser.add_argument("--cdr", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.gdc, args.cdr, args.output)
