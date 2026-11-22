"""Register pharmacogenomic and paired organoid transfer roles."""
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

from .cohorts import file_record


DRUG_NAMES = ("Sorafenib", "Lenvatinib", "Regorafenib", "Cabozantinib", "Trametinib")


def frozen(root, name, receipt):
    path = root / name
    if file_record(path) != receipt["files"][name]:
        raise ValueError(f"Input does not match registered bytes: {path}")
    return path


def partition(name):
    return int.from_bytes(sha256(f"scrna-subclones-51122:{name}".encode()).digest()[:8], "big") % 5


def series_matrix(path):
    allowed = {"!Sample_title", "!Sample_geo_accession", "!Sample_characteristics_ch1", "!Sample_source_name_ch1"}
    records = {}
    with gzip.open(path, "rt", newline="") as stream:
        for row in csv.reader(stream, delimiter="\t"):
            if row and row[0] in allowed:
                records.setdefault(row[0], []).append(row[1:])
    title, accession, descriptions = [
        records["!Sample_" + label][0] for label in ("title", "geo_accession", "source_name_ch1")
    ]
    patient, phenotype = records["!Sample_characteristics_ch1"]
    meta = []
    for label, code, desc, who, status in zip(title, accession, descriptions, patient, phenotype):
        match = re.fullmatch(r"Sorafenib-(sensitive|resistant) organoid of patient (\d+), replicate (\d+)", desc)
        if not match or (who, status) != ("patient: " + match[2], "drug phenotype: " + match[1]):
            raise ValueError(f"Conflicting organoid source annotation for {code}")
        meta.append({"id": code, "donor": "GSE182593:" + match[2], "replicate": int(match[3]),
                     "outcome": int(match[1] == "resistant"), "matrix_sample": label})
    output = pd.DataFrame(meta)
    if len(output) != 24 or len(output.groupby("donor")) != 4 or not output.id.is_unique:
        raise ValueError("GSE182593 must contain 24 independent assay rows across four donors")
    if not output.groupby(["donor", "outcome"]).replicate.agg(set).map(lambda x: x == {1, 2, 3}).all():
        raise ValueError("GSE182593 has a missing or duplicated paired sample")
    return output


def prepare_organoids(counts, series, output):
    output.mkdir()
    meta = series_matrix(series)
    raw = pd.read_csv(counts, sep="\t", usecols=["gene_id", *meta.matrix_sample], dtype={"gene_id": str})
    raw = raw.set_index("gene_id").groupby(level=0, sort=True).sum()
    expression = raw.loc[:, meta.matrix_sample].to_numpy(np.float32).T
    if (expression < 0).any() or not np.isfinite(expression).all() or np.any(expression.sum(axis=1) == 0):
        raise ValueError("Invalid organoid counts")
    log_tpm_proxy = np.log2(1 + expression / expression.sum(axis=1, keepdims=True) * 1e6)
    pd.DataFrame(log_tpm_proxy, index=meta.id, columns=raw.index).to_parquet(output / "rna.parquet")
    # The query IDs and donor keys are independent of the phenotype. The
    # matrix sample IDs encode class and belong only in the source manifest.
    meta[["id", "donor"]].to_parquet(output / "query.parquet", index=False)
    meta[["id", "outcome"]].to_parquet(output / "labels.parquet", index=False)
    return {"donors": 4, "conditions": 2, "replicates_per_condition_and_donor": 3,
            "genes": len(raw), "inputs": {"counts": file_record(counts), "series": file_record(series)}}


def reconcile_profiles(rna_paths):
    files = [pd.read_parquet(path) for path in rna_paths]
    columns = sorted(set.intersection(*(set(table.columns) for table in files)))
    if len(columns) < 4000:
        raise ValueError("Sparse common basal RNA gene assay")
    profiles = pd.concat([file[columns] for file in files])
    # A repeated canonical key is allowed only if the registered values
    # are identical and we can coalesce without source outcome information.
    duplicate = profiles[profiles.index.duplicated(keep=False)]
    for key, values in duplicate.groupby(level=0):
        if not np.allclose(values.to_numpy(), values.iloc[0].to_numpy(), rtol=0, atol=1e-6):
            raise ValueError(f"Conflicting measured basal RNA for {key}")
    return profiles[~profiles.index.duplicated(keep="first")]


def prepare(a):
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    pack = json.loads((a.prepared / "receipt.json").read_text())
    pan, limore, limore_truth = [
        pd.read_parquet(frozen(a.prepared, name + ".parquet", pack))
        for name in ("pan_train", "limore_drug_cold", "limore_drug_labels")
    ]
    limore = limore.merge(limore_truth, on="id", validate="1:1")
    pan["resistance_native"] = pan.pop("outcome")
    limore["resistance_native"] = -limore.pop("outcome")
    profiles = reconcile_profiles(a.rna)
    roster = pd.read_csv(a.drugs, sep="\t", dtype=str)
    drug_map = roster[roster.source_compound_label.isin(DRUG_NAMES)].set_index("source_compound_label").compound_id
    if set(drug_map.index) != set(DRUG_NAMES) or not drug_map.is_unique:
        raise ValueError("Unresolved exact identifiers for targeted therapies")
    therapies = {v: k for k, v in drug_map.items()}
    source = pd.concat([pan, limore], ignore_index=True)
    source = source[source.drug_id.isin(therapies) & source.cell_id.isin(profiles.index)]
    source = source[["id", "cell_id", "cell_group", "source_id", "drug_id", "resistance_native"]].copy()
    source["drug"] = source.drug_id.map(therapies)
    if source.id.duplicated().any() or not np.isfinite(source.resistance_native).all():
        raise ValueError("Duplicate or nonfinite registered pharmacology outcomes")
    source["split"] = source.cell_group.map(partition)
    query_rna = profiles.loc[source.cell_id.unique()]
    queries = source[source.source_id.eq("LIMORE")].drop(columns="resistance_native")
    if queries.cell_group.nunique() < 40:
        raise ValueError("No useful independent HCC cell-line response strata")
    with tempfile.TemporaryDirectory(prefix=f".{a.output.name}.", dir=a.output.parent) as d:
        staged = Path(d)
        organoids = prepare_organoids(a.organoid_counts, a.organoid_series, staged / "organoids")
        query_rna.to_parquet(staged / "cell_lines_rna.parquet")
        summaries = {}
        for fold in range(5):
            folder = staged / f"fold-{fold}"
            folder.mkdir()
            external = queries[queries.split.eq(fold)].copy()
            # Hold out lineages across all screens including any siblings.
            fit = source[~source.split.isin((fold, (fold + 1) % 5))]
            valid = source[source.split.eq((fold + 1) % 5)]
            groups = [set(t.cell_group) for t in (fit, valid, external)]
            if (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]
                    or any(len(g) < 5 for g in groups)):
                raise ValueError(f"Cell line cold roles differ from registered families in fold {fold}")
            for role, rows in (
                ("fit", fit), ("validation", valid), ("query", external),
                ("labels", source.loc[source.id.isin(external.id), ["id", "resistance_native"]]),
            ):
                rows.to_parquet(folder / f"{role}.parquet", index=False)
            summaries[str(fold)] = {
                "training_lineages": len(groups[0]), "validation_lineages": len(groups[1]),
                "test_limore_lineages": len(groups[2]),
                "fitting_drugs": sorted(set(fit.drug)),
                "validation_drugs": sorted(set(valid.drug)),
                "external_drugs": sorted(set(external.drug)),
            }
        receipt = {
            "sources": {str(k): int(v) for k, v in source.source_id.value_counts().items()},
            "drugs": drug_map.to_dict(), "organoids": organoids, "panels": summaries,
            "fit_labels": "Native resistance direction, to be fit-only normalized within each assay and drug",
            "inputs": {"rna": [file_record(f) for f in a.rna], "drug_registry": file_record(a.drugs),
                       "pharmacology": file_record(a.prepared / "receipt.json")},
            "files": {str(f.relative_to(staged)): file_record(f) for f in sorted(staged.rglob("*")) if f.is_file()},
        }
        (staged / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        staged.rename(a.output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "drugs", "organoid-counts", "organoid-series", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--rna", type=Path, nargs="+", required=True)
    prepare(p.parse_args())


if __name__ == "__main__":
    main()
