"""Register sparse public HCC count matrices against audited biopsy identities."""
import argparse
import gzip
import json
from pathlib import Path
import re
import tempfile

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread

from .cnv import broad_cnv, by_donor_reference, genome_projection, normalized
from .cohorts import file_record


S_PHASE = frozenset("MCM5 PCNA TYMS FEN1 MCM2 MCM4 RRM1 UNG GINS2 MCM6 CDCA7 DTL PRIM1 UHRF1 "
                    "RFC2 RPA2 NASP RAD51AP1 GMNN WDR76 SLBP CCNE2 UBR7 POLD3 MSH2 ATAD2 "
                    "RAD51 RRM2 CDC45 CDC6 EXO1 TIPIN DSCC1 BLM CASP8AP2 USP1 CLSPN POLA1 CHAF1B".split())
G2M_PHASE = frozenset("HMGB2 CDK1 NUSAP1 UBE2C BIRC5 TPX2 TOP2A NDC80 CKS2 NUF2 CKS1B MKI67 "
                      "TMPO CENPF TACC3 FAM64A SMC4 CCNB2 CKAP2L CKAP2 AURKB BUB1 KIF11 "
                      "ANP32E TUBB4B GTSE1 KIF20B HJURP CDCA3 HN1 CDCA8 ANLN LBR CKAP5 "
                      "CENPE CTCF NEK2 G2E3 GAS2L3 CBX5 CENPA".split())
EPITHELIAL = frozenset("ALB APOA1 APOA2 TTR KRT8 KRT18 KRT19 ASGR1 FABP1 EPCAM GPC3 AFP".split())


def read_gtf(path):
    columns = ["gene_id", "chromosome", "start", "end", "symbol"]
    result = []
    open_file = gzip.open if path.suffix == ".gz" else open
    with open_file(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if fields[2] != "gene":
                continue
            gene = re.search(r'\bgene_id "([^"]+)"', fields[8])
            symbol = re.search(r'\bgene_name "([^"]+)"', fields[8])
            if gene and symbol and re.match(r"ENSG[0-9]+(?:\.[0-9]+)?$", gene[1]):
                result.append((gene[1].split(".")[0], fields[0], int(fields[3]), int(fields[4]), symbol[1]))
    if len(result) < 10000:
        raise ValueError("Reference GTF has too few annotated human genes")
    return pd.DataFrame(result, columns=columns).drop_duplicates("gene_id", keep=False)


def strings(group, name):
    node = group[name]
    if isinstance(node, h5py.Dataset):
        return node.asstr()[:]
    values = node["categories"].asstr()[:]
    index = node["codes"][:]
    return np.where(index >= 0, values[index], "")


def read_liver(path, metadata):
    with h5py.File(path, "r") as handle:
        obs = handle["obs"]
        barcodes = strings(obs, obs.attrs["_index"])
        var = handle["var"]
        genes = pd.DataFrame({"gene_id": strings(var, "ensembl_id")})
        counts = handle["layers/counts"]
        if counts.attrs["encoding-type"] != "csr_matrix":
            raise ValueError("GSE149614 counts must be stored as a row-indexed CSR layer")
        expression = sparse.csr_matrix(
            (counts["data"][:], counts["indices"][:], counts["indptr"][:]),
            shape=tuple(counts.attrs["shape"]),
        )
    if set(barcodes) != set(metadata.barcode):
        raise ValueError("GSE149614 count barcodes differ from the registered cohort")
    matched = metadata.set_index("barcode", verify_integrity=True).loc[barcodes].reset_index()
    return expression, genes, matched


def read_mtx(matrix, barcodes, genes_file, info, metadata, companion):
    axis = pd.read_csv(barcodes, sep="\t", header=None, dtype=str).iloc[:, 0].to_numpy()
    description = pd.read_csv(info, sep="\t", dtype=str)
    if companion:
        label = description.Sample.to_numpy()
        cell = description["Cell Barcode"].to_numpy()
        registered_sample = metadata.source_sample
    else:
        label = description.S_ID.to_numpy()
        cell = description.Cell.to_numpy()
        registered_sample = metadata.id.str.split(":").str[1]
    # Single-cell barcodes repeat across libraries. Require the full barcodes
    # file to agree with the authors' row order, then match sample+barcode.
    if not np.array_equal(axis, cell):
        raise ValueError(f"Author assay columns and biopsy metadata are misaligned for {matrix}")
    source_ids = pd.MultiIndex.from_arrays([label, cell])
    by_key = pd.MultiIndex.from_arrays([registered_sample, metadata.barcode])
    if not by_key.is_unique or not source_ids.is_unique or set(by_key) != set(source_ids):
        raise ValueError(f"Assay sample+barcode identities mismatch for {matrix}")
    matched = metadata.iloc[by_key.get_indexer(source_ids)].reset_index(drop=True)
    genes = pd.read_csv(genes_file, sep="\t", header=None, dtype=str).iloc[:, :1]
    genes.columns = ["gene_id"]
    expression = mmread(str(matrix)).T.tocsr().astype(np.float32)
    return expression, genes, matched


def technical_metrics(counts, genes, metadata, argument, coord):
    if counts.shape != (len(metadata), len(genes)) or not np.isfinite(counts.data).all() or (counts.data < 0).any():
        raise ValueError("Invalid count matrix or axis")
    counts.eliminate_zeros()
    genes["gene_id"] = genes.gene_id.fillna("").str.split(".").str[0]
    unique = coord.set_index("gene_id")
    mapping = unique.reindex(genes.gene_id)
    genes.insert(1, "symbol", mapping.symbol.fillna("").to_numpy())
    mitochondria = mapping.chromosome.eq("chrM").to_numpy()
    total = np.asarray(counts.sum(axis=1)).ravel()
    if not mitochondria.any():
        raise ValueError("No mitochondrial annotation overlap for technical diagnostics")
    attributes = metadata.assign(
        umi=total, detected_genes=counts.getnnz(axis=1),
        mitochondrial_fraction=np.asarray(counts[:, mitochondria].sum(axis=1)).ravel() / np.maximum(total, 1),
    )
    log = normalized(counts)
    for name, marker in (("s_phase", S_PHASE), ("g2m_phase", G2M_PHASE), ("epithelial_marker", EPITHELIAL)):
        mask = genes.symbol.isin(marker).to_numpy()
        if mask.sum() < 8:
            raise ValueError(f"Inadequate {name} marker overlap")
        attributes[name] = np.asarray(log[:, mask].mean(axis=1)).ravel()
    attributes["epithelial_marker_count"] = counts[:, genes.symbol.isin(EPITHELIAL)].getnnz(axis=1)
    if argument == "GSE149614":
        operator, windows = genome_projection(genes.gene_id, coord)
        attributes = cnv_by_fold(attributes, (log @ operator).toarray())
    else:
        windows = []
        attributes["malignant_candidate"] = attributes.malignancy.eq("author_malignant")
        attributes["cnv_proxy"] = np.nan
    return attributes, windows


def cnv_by_fold(attributes, blocks):
    results = attributes.copy().reset_index(drop=True)
    results["cnv_proxy"] = np.nan
    results["cnv_cutoff"] = np.nan
    results["reference_donors"] = 0
    for fold in sorted(set(results.fold)):
        query = results.fold.eq(fold).to_numpy()
        refs = results.malignancy.eq("normal_hepatocyte").to_numpy() & ~query & (results.umi.to_numpy() > 0)
        ids = results.loc[refs, "donor"].to_numpy()
        groups = sorted(set(ids))
        if len(groups) < 3 or set(groups) & set(results.loc[query, "donor"]):
            raise ValueError(f"No independent training-donor CNV reference for fold {fold}")
        profile = blocks[refs]
        reference = by_donor_reference(profile, ids)
        thresholds = []
        for donor in groups:
            own = ids == donor
            previous = by_donor_reference(profile[~own], ids[~own])
            thresholds.append(np.quantile(broad_cnv(profile[own], previous), 0.99))
        results.loc[query, "cnv_proxy"] = broad_cnv(blocks[query], reference)
        results.loc[query, "cnv_cutoff"] = max(thresholds)
        results.loc[query, "reference_donors"] = len(groups)
    results["malignant_candidate"] = (results.malignancy.eq("needs_cnv")
                                       & results.cnv_proxy.gt(results.cnv_cutoff)
                                       & results.epithelial_marker_count.ge(2))
    return results


def stage_cohort(name, raw, coords, destination):
    counts, genes, metadata = raw
    attributes, windows = technical_metrics(counts, genes, metadata, name, coords)
    destination.mkdir()
    # Remove unannotated or ambiguous axes after calculating library metrics
    # across all expressed genes. Keep raw counts and publish the exact axis.
    annotated = genes.symbol.ne("") & ~genes.gene_id.duplicated(keep=False)
    sparse.save_npz(destination / "counts.npz", counts[:, annotated.to_numpy()].astype(np.float32))
    genes[annotated].to_csv(destination / "genes.tsv", sep="\t", index=False)
    attributes.to_parquet(destination / "metadata.parquet", index=False)
    print(name, "cells", len(attributes), "malignant epithelial candidates",
          int(attributes.malignant_candidate.sum()), flush=True)
    return {
        "cells": len(attributes), "hg38_annotated_genes": int(annotated.sum()),
        "malignant_hcc_cells": int((attributes.malignant_candidate & attributes.histology.eq("HCC")).sum()),
        "donors": int(attributes.donor.nunique()), "cnv_autosomal_windows": len(windows),
    }


def prepare(a):
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    registry_path = a.registry / "cohorts.parquet"
    audit = json.loads((a.registry / "receipt.json").read_text())
    if audit["registry"] != file_record(registry_path):
        raise ValueError("Cohort identity receipt differs")
    registry = pd.read_parquet(registry_path)
    coordinates = read_gtf(a.gtf)
    inputs = {"registry": registry_path, "gencode": a.gtf, "GSE149614": a.liver_counts}
    with tempfile.TemporaryDirectory(prefix=f".{a.output.name}.", dir=a.output.parent) as temp:
        stage = Path(temp)
        cohort = {}
        cohort["GSE149614"] = stage_cohort(
            "GSE149614", read_liver(a.liver_counts, registry[registry.cohort.eq("GSE149614")]),
            coordinates, stage / "GSE149614",
        )
        for name, files, companion in (
            ("GSE151530", a.longitudinal, False), ("GSE125449-set1", a.earlier_set1, True),
            ("GSE125449-set2", a.earlier_set2, True),
        ):
            if companion:
                stem, split = name.split("-")
                metadata = registry[registry.id.str.startswith(stem + ":" + split + ":")]
            else:
                metadata = registry[registry.cohort.eq(name)]
            for path in files:
                inputs[path.name] = path
            cohort[name] = stage_cohort(name, read_mtx(*files, metadata, companion), coordinates, stage / name)
        (stage / "receipt.json").write_text(json.dumps({
            "inputs": {name: file_record(path) for name, path in inputs.items()},
            "cohorts": cohort, "files": {str(f.relative_to(stage)): file_record(f)
                                        for f in sorted(stage.rglob("*")) if f.is_file()},
        }, indent=2, sort_keys=True) + "\n")
        stage.rename(a.output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("registry", "gtf", "liver-counts", "output"):
        p.add_argument("--" + arg, type=Path, required=True)
    for arg in ("longitudinal", "earlier-set1", "earlier-set2"):
        p.add_argument("--" + arg, type=Path, nargs=4, required=True, metavar=("MTX", "BARCODES", "GENES", "INFO"))
    prepare(p.parse_args())


if __name__ == "__main__":
    main()
