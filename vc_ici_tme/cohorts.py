"""Register baseline pan-cancer response assays and sealed HCC queries."""
import argparse
import csv
import gzip
from hashlib import sha256
import json
from pathlib import Path
import re
import tempfile

import numpy as np
import pandas as pd


RESPONSES = {
    "CR": 1, "PR": 1, "PRCR": 1, "COMPLETE RESPONSE": 1, "PARTIAL RESPONSE": 1,
    "PD": 0, "SD": 0, "PROGRESSIVE DISEASE": 0, "STABLE DISEASE": 0,
    "RESPONDERS": 1, "NON-RESPONDERS": 0,
}


def record(path):
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def response(series):
    return series.fillna("").astype(str).str.strip().str.upper().map(RESPONSES)


def geo_soft(path):
    results, current = [], None
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("^SAMPLE = "):
                current = {"gsm": line.removeprefix("^SAMPLE = ").strip()}
                results.append(current)
            elif current is not None and " = " in line and line.startswith("!Sample_"):
                field, value = line.removeprefix("!Sample_").strip().split(" = ", 1)
                if field == "characteristics_ch1" and ": " in value:
                    field, value = value.split(": ", 1)
                elif field != "title":
                    continue
                if field in current and current[field] != value:
                    raise ValueError(f"Conflict in {current['gsm']} for {field}")
                current[field] = value.strip()
    frame = pd.DataFrame(results)
    if len(frame) == 0 or not frame.gsm.is_unique:
        raise ValueError(f"Invalid GEO family file {path}")
    return frame


def strict_gene_map(hgnc):
    x = pd.read_csv(hgnc, usecols=["entrez_id", "ensembl_gene_id", "symbol"], dtype=str)
    x["entrez_id"] = x.entrez_id.str.extract(r"^(\d+)(?:\.0+)?$", expand=False)
    tables = {}
    for column in ("entrez_id", "ensembl_gene_id"):
        names = x[[column, "symbol"]].dropna().drop_duplicates()
        ok = ~names[column].duplicated(keep=False) & ~names.symbol.duplicated(keep=False)
        tables[column] = names[ok].set_index(column).symbol
    return tables


def gene_symbols(matrix, crosswalk, id_kind):
    matrix = matrix.copy()
    matrix.index = matrix.index.astype(str).str.strip()
    if id_kind == "symbol":
        chosen = matrix.index.str.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./#@-]*")
        names = matrix.index[chosen]
    else:
        mapped = crosswalk[id_kind].reindex(matrix.index.str.split(".").str[0])
        chosen, names = mapped.notna().to_numpy(), mapped.dropna().to_numpy()
    tissue = matrix[chosen].apply(pd.to_numeric, errors="raise")
    tissue.index = names
    # Ambiguous multi-locus symbols do not enter clinical comparison.
    return tissue[~tissue.index.duplicated(keep=False)].replace([np.inf, -np.inf], np.nan)


def frame_from_metadata(source, meta, cohort, matrix, id_col, patient_col, endpoint_col):
    meta = meta[[id_col, patient_col, endpoint_col]].copy()
    rows = meta.rename(columns={id_col: "matrix_id", patient_col: "donor", endpoint_col: "recist"})
    rows = rows.dropna(subset=["matrix_id", "donor"])
    rows["outcome"] = response(rows.recist)
    if not rows.matrix_id.is_unique:
        raise ValueError(f"{source} has repeated assay IDs")
    mapped = rows[rows.outcome.notna() & rows.matrix_id.isin(matrix.columns)].copy()
    for _, patient in mapped.groupby("donor"):
        if patient.outcome.nunique() > 1:
            raise ValueError(f"Conflicting label for a repeated patient in {source}")
    # Predetermine one sample per donor without looking at expression or class.
    mapped = mapped.sort_values("matrix_id").drop_duplicates("donor").reset_index(drop=True)
    if len(mapped) < 12 or mapped.outcome.nunique() != 2:
        raise ValueError(f"{source} lacks the required registered objective response classes")
    mapped["id"] = cohort + ":" + mapped.matrix_id.astype(str)
    mapped["donor"] = cohort + ":" + mapped.donor.astype(str)
    mapped["cohort"] = cohort
    mapped["outcome"] = mapped.outcome.astype(int)
    expr = matrix.loc[:, mapped.matrix_id].T
    expr.index = mapped.id
    return expr.astype(np.float32), mapped[["id", "donor", "cohort", "outcome"]], {
        "metadata_rows": len(rows), "scorable_patients": len(mapped), "responders": int(mapped.outcome.sum()),
        "measured_symbols": len(matrix), "excluded": len(rows) - len(mapped),
    }


def log_abundance(table, kind):
    values = table.to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("Missing/nonfinite abundance values in a registered RNA assay")
    if kind == "rlog":
        return table
    if np.any(values < 0):
        raise ValueError(f"{kind} assay has negative values")
    if kind == "counts":
        library = values.sum(axis=0)
        if np.any(library <= 0):
            raise ValueError("Source count library is empty")
        values *= 1e6 / library
    return pd.DataFrame(np.log2(1 + values), index=table.index, columns=table.columns)


def register_pan(a, genes):
    imv = pd.read_csv(a.imv_counts, sep="\t", index_col=0)
    imv = gene_symbols(log_abundance(imv, "counts"), genes, "entrez_id")
    clinical = pd.read_csv(a.imv_metadata, sep="\t", dtype=str, index_col=0).rename_axis("sample_id").reset_index()
    yield "IMvigor210", frame_from_metadata(
        "IMvigor210 Core Biologies", clinical, "IMvigor210", imv, "sample_id", "ANONPT_ID", "Best Confirmed Overall Response",
    )

    gide = pd.read_csv(a.gide_rna, sep="\t", index_col="Index").drop(columns="cancer_code").T
    gide = gene_symbols(log_abundance(gide, "tpm"), genes, "symbol")
    meta = pd.read_csv(a.gide_meta, sep="\t", dtype=str)
    meta = meta[meta.Timing.eq("Pre") & meta.ICI_target_map.isin(["PD1", "PD1+CTLA4"])]
    yield "Gide", frame_from_metadata("Gide COMPASS baseline", meta, "Gide", gide,
                                      "Index", "patient_id", "RECIST")

    hugo_meta = geo_soft(a.hugo_soft).rename(columns={"patient id": "patient"})
    hugo_meta = hugo_meta[hugo_meta["biopsy time"].eq("pre-treatment")].copy()
    hugo_meta["matrix_key"] = hugo_meta.title + ".baseline"
    hugo = pd.read_excel(a.hugo_rna, sheet_name="FPKM", index_col="Gene", engine="openpyxl")
    hugo = gene_symbols(log_abundance(hugo, "fpkm"), genes, "symbol")
    yield "Hugo", frame_from_metadata("GSE78220", hugo_meta, "Hugo", hugo,
                                      "matrix_key", "patient", "anti-pd-1 response")

    riaz_meta = geo_soft(a.riaz_soft)
    riaz_meta = riaz_meta[riaz_meta["visit (pre or on treatment)"].eq("Pre")].copy()
    riaz_meta["patient"] = riaz_meta.title.str.extract(r"^(Pt[0-9]+)_Pre", expand=False)
    riaz = pd.read_csv(a.riaz_rna, index_col=0)
    riaz = gene_symbols(log_abundance(riaz, "rlog"), genes, "entrez_id")
    yield "Riaz", frame_from_metadata("GSE91061", riaz_meta, "Riaz", riaz, "title", "patient", "response")


def register_hcc(a, genes):
    supplement = pd.read_excel(a.hcc_clinical, engine="openpyxl", sheet_name="Table S1", header=1, dtype=str)
    supplement = supplement[supplement["PD1.therapy"].eq("Yes")].copy()
    supplement["patient"] = supplement.Label
    hcc = pd.read_csv(a.hcc_tpm, sep="\t", index_col=0)
    if not set(supplement.Label).issubset(hcc.columns) or len(supplement) != 17:
        raise ValueError("Original GSE202069 treatment supplement disagrees with the public expression roster")
    hcc = gene_symbols(log_abundance(hcc, "tpm"), genes, "symbol")
    tissue, response_info, audit = frame_from_metadata(
        "GSE202069 S1", supplement, "GSE202069", hcc, "Label", "patient", "RECIST.Response",
    )
    if audit["scorable_patients"] != 17 or audit["responders"] != 8:
        raise ValueError("GSE202069 publication response roster mismatch")
    yield "GSE202069", (tissue, response_info, audit)

    meta = geo_soft(a.nanostring_soft).rename(columns={"best_response": "recist"})
    meta["assay_id"] = meta.title.str.extract(r"^HCC (S\d+)$", expand=False)
    meta["patient"] = meta.assay_id
    with gzip.open(a.nanostring_rna, "rt", newline="") as stream:
        header, values = None, []
        for row in csv.reader(stream, delimiter="\t"):
            if row and row[0] == "ID_REF":
                header = [field for field in row if field]
            elif header and row and re.fullmatch(r"[A-Z0-9][A-Z0-9_./#@-]*", row[0]):
                values.append(row[:len(header)])
    if not header:
        raise ValueError("Could not identify the official NanoString count axes")
    physical = pd.DataFrame(values, columns=header).set_index("ID_REF").apply(pd.to_numeric, errors="raise")
    physical = gene_symbols(physical, genes, "symbol")
    yield "GSE140901", frame_from_metadata(
        "GSE140901 NanoString", meta, "GSE140901", physical, "assay_id", "patient", "recist",
    )

    exploratory = pd.read_csv(a.exploratory_hcc, sep="\t", index_col="GeneID")
    exploratory = gene_symbols(log_abundance(exploratory, "counts"), genes, "ensembl_gene_id").T
    exploratory.index = "GSE141198:" + exploratory.index.astype(str)
    exploratory.index.name = "id"
    yield "GSE141198", (exploratory.astype(np.float32), None, {
        "profiles": len(exploratory), "measured_symbols": exploratory.shape[1],
        "clinical_role": "Exploratory HCC expression without registered immunotherapy response",
    })


def prepare(a):
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    genes = strict_gene_map(a.hgnc)
    with tempfile.TemporaryDirectory(prefix=f".{a.output.name}.", dir=a.output.parent) as temporary:
        stage = Path(temporary)
        audit = {}
        for scope, datasets in (("sources", register_pan(a, genes)), ("external", register_hcc(a, genes))):
            for cohort, (abundance, outcomes, stats) in datasets:
                directory = stage / scope / cohort
                directory.mkdir(parents=True)
                abundance.to_parquet(directory / "expression.parquet")
                if outcomes is not None:
                    outcomes.loc[:, ["id", "outcome"]].to_parquet(directory / "labels.parquet", index=False)
                    outcomes.loc[:, ["id", "donor", "cohort"]].to_parquet(directory / "keys.parquet", index=False)
                audit[cohort] = {"role": scope, **stats}
                print(cohort, stats, flush=True)
        receipt = {
            "cohorts": audit,
            "inputs": {name: record(value) for name, value in vars(a).items() if name != "output"},
            "files": {str(f.relative_to(stage)): record(f) for f in sorted(stage.rglob("*")) if f.is_file()},
            "source_endpoint": "CR and PR versus SD and PD, one pretreatment measurement per patient",
            "external_hcc_endpoint": "GSE202069 treated subset and GSE140901 NanoString objective response",
        }
        (stage / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        stage.rename(a.output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "hgnc", "imv-counts", "imv-metadata", "gide-rna", "gide-meta", "hugo-rna",
                 "hugo-soft", "riaz-rna", "riaz-soft", "hcc-tpm", "hcc-clinical", "nanostring-soft",
                 "nanostring-rna", "exploratory-hcc"):
        p.add_argument("--" + name, type=Path, required=True)
    prepare(p.parse_args())


if __name__ == "__main__":
    main()
