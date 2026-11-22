"""Explore TCGA-LIHC expression and published immune subtypes off-line."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import md5
import json
from pathlib import Path
import re
import tempfile

import numpy as np
import pandas as pd

from .cohorts import record
from .features import construct, published_markers
from .reference import TYPES


PRIMARY = re.compile(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})-01[A-Z]-")
SUBTYPES = {
    "1": "C1 wound healing", "2": "C2 IFN-gamma dominant", "3": "C3 inflammatory",
    "4": "C4 lymphocyte depleted", "5": "C5 immunologically quiet", "6": "C6 TGF-beta dominant",
}


def inputs(root, listing):
    source = json.loads(listing.read_text())["data"]
    expected = source["pagination"]["total"]
    if expected != len(source["hits"]):
        raise ValueError("Incomplete GDC STAR file listing")
    records = []
    for row in source["hits"]:
        file = root / row["file_id"] / row["file_name"]
        aliquots = {entry["entity_submitter_id"] for entry in row["associated_entities"]
                    if entry["entity_type"] == "aliquot"}
        if len(aliquots) != 1:
            continue
        aliquot = next(iter(aliquots))
        match = PRIMARY.match(aliquot)
        if not match or not file.is_file():
            continue
        donor = match[1]
        if {c["submitter_id"] for c in row["cases"]} != {donor}:
            raise ValueError("Invalid TCGA sample to patient crosswalk")
        records.append((donor, aliquot[:16], aliquot, file, row["md5sum"]))
    frame = pd.DataFrame(records, columns=["donor", "sample", "aliquot", "file", "md5"])
    frame = frame.sort_values(["donor", "sample", "aliquot"]).drop_duplicates("donor")
    if len(frame) < 300:
        raise ValueError("Too few public HCC primary tumors with exact GDC STAR provenance")
    return frame


def rna_file(row):
    with row.file.open("rb") as f:
        checksum = md5()
        for block in iter(lambda: f.read(1 << 20), b""):
            checksum.update(block)
    if checksum.hexdigest() != row.md5:
        raise ValueError(f"GDC checksum mismatch for {row.file.name}")
    t = pd.read_csv(row.file, sep="\t", comment="#", usecols=["gene_id", "gene_name", "tpm_unstranded"],
                    dtype={"gene_id": str, "gene_name": str})
    genes = t[t.gene_id.str.startswith("ENSG")].set_index("gene_name")
    genes = genes.loc[~genes.index.duplicated(keep=False)].sort_index()
    s = np.log2(genes.tpm_unstranded + 1).astype(np.float32)
    if not np.isfinite(s).all():
        raise ValueError("Nonfinite TCGA RNA measurement")
    return s.rename(row.sample)


def subtype_matches(path, queries):
    records = pd.read_csv(path, sep="\t", compression="gzip", usecols=["SampleBarcode", "ClusterModel1"], dtype=str)
    records = records.drop_duplicates().rename(columns={"SampleBarcode": "sample", "ClusterModel1": "class"})
    conflict = records.groupby("sample")["class"].nunique().ne(1)
    registered = records[~records["sample"].map(conflict)].rename(columns={"class": "immune_class"})
    registered["subtype"] = registered.immune_class.map(SUBTYPES)
    joined = queries.merge(registered.dropna(subset=["subtype"]), on="sample", validate="one_to_one", how="inner")
    if joined.empty:
        raise ValueError("No study-author immune subtype matches a registered STAR sample")
    return joined


def quantify(a):
    if a.output.exists():
        raise FileExistsError(a.output)
    roster = inputs(a.expression, a.listing)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        values = list(pool.map(rna_file, roster.itertuples(index=False)))
    genes = values[0].index
    if any(not gene.index.equals(genes) for gene in values):
        raise ValueError("GDC annotation axes differ")
    rna = pd.DataFrame(values)
    profiles = construct(rna, a.reference, a.hallmark)
    biomarkers, _ = published_markers(rna)
    output = pd.DataFrame(profiles.fractions, columns=TYPES, index=rna.index)
    output = output.join(biomarkers)
    output.index.name = "sample"
    # Freeze a receipt for the exploratory scores before joining subtype labels.
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{a.output.name}.", dir=a.output.parent) as handle:
        d = Path(handle)
        output.to_parquet(d / "expression_scores.parquet")
        partial = record(d / "expression_scores.parquet")
        (d / "expression_receipt.json").write_text(json.dumps({
            "gdc_listing": record(a.listing), "reference": record(a.reference / "receipt.json"),
            "hallmark": record(a.hallmark), "expression_scores": partial, "public_primary_tumors": len(roster),
        }, indent=2, sort_keys=True) + "\n")
        labels = subtype_matches(a.subtypes, roster[["sample", "donor"]])
        measured = pd.read_parquet(d / "expression_scores.parquet")
        if record(d / "expression_scores.parquet") != partial:
            raise ValueError("TCGA scores changed before subtype reporting")
        linked = measured.merge(labels, on="sample", validate="one_to_one")
        rows = []
        for name, group in linked.groupby("subtype"):
            for feature in output.columns:
                rows.append({"subtype": name, "patients": len(group), "feature": feature,
                             "median": group[feature].median(), "q25": group[feature].quantile(.25),
                             "q75": group[feature].quantile(.75)})
        pd.DataFrame(rows).to_csv(d / "immune_subtypes.tsv", sep="\t", index=False)
        (d / "receipt.json").write_text(json.dumps({
            "frozen_expression_scores": record(d / "expression_receipt.json"),
            "n_primary_tumors": len(roster), "n_published_subtypes": len(linked),
            "immune_subtype_table": record(a.subtypes), "role": "exploratory without immunotherapy outcomes",
            "files": {p.name: record(p) for p in sorted(d.iterdir())},
        }, indent=2, sort_keys=True) + "\n")
        d.rename(a.output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("expression", "listing", "subtypes", "reference", "hallmark", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--workers", type=int, default=5)
    a = p.parse_args()
    if a.workers < 1:
        p.error("Provide at least one reader")
    quantify(a)


if __name__ == "__main__":
    main()
