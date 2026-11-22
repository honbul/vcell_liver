"""Fit a donor-blocked broad RNA-CNV proxy for GSE149614 hepatocytes."""
import argparse
from pathlib import Path
import tempfile
import json

import numpy as np
import pandas as pd
from scipy import sparse

from .cohorts import file_record


def normalized(matrix):
    x = sparse.csr_matrix(matrix, dtype=np.float32)
    if (x.data < 0).any() or not np.isfinite(x.data).all():
        raise ValueError("CNV proxy requires nonnegative raw count entries")
    lib_size = np.asarray(x.sum(axis=1)).ravel()
    x = (sparse.diags(1e4 / np.maximum(lib_size, 1.0)) @ x).tocsr()
    x.data = np.log1p(x.data)
    return x


def genome_projection(gene_ids, coordinates, window_size=100):
    coord = coordinates.copy()
    coord["gene_id"] = coord.gene_id.str.split(".").str[0]
    coord["chromosome"] = coord.chromosome.str.removeprefix("chr")
    chromosomes = set(map(str, range(1, 23)))
    coord = coord.loc[coord.chromosome.isin(chromosomes)].drop_duplicates("gene_id", keep=False)
    genes = pd.Series(gene_ids).fillna("").astype(str).str.split(".").str[0]
    pos = pd.DataFrame({"gene_id": genes, "column": np.arange(len(gene_ids))})
    pos = pos.loc[~pos.gene_id.duplicated(keep=False)]
    eligible = pos.merge(coord, on="gene_id", validate="one_to_one")
    eligible["chromosome"] = eligible.chromosome.astype(int)
    eligible = eligible.sort_values(["chromosome", "start", "gene_id"])
    original_cols, new_cols, weights, labels = [], [], [], []
    for chromosome, group in eligible.groupby("chromosome"):
        for offset in range(0, len(group), window_size):
            block = group.iloc[offset:offset + window_size]
            if len(block) < window_size // 2:
                continue
            original_cols.extend(block.column.tolist())
            new_cols.extend([len(labels)] * len(block))
            weights.extend([1.0 / len(block)] * len(block))
            labels.append(f"chr{chromosome}:{block.start.iloc[0]}-{block.end.iloc[-1]}")
    if len(labels) < 20:
        raise ValueError("Too few autosomal expression blocks")
    projection = sparse.csc_matrix(
        (np.asarray(weights, np.float32), (original_cols, new_cols)),
        shape=(len(gene_ids), len(labels)),
    )
    return projection, labels


def by_donor_reference(blocks, donors):
    return np.stack([blocks[donors == d].mean(axis=0) for d in sorted(set(donors))]).mean(axis=0)


def broad_cnv(blocks, reference):
    relative = blocks - reference[None, :]
    relative -= np.median(relative, axis=1, keepdims=True)
    return np.mean(np.abs(relative), axis=1)


def main():
    import anndata

    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("counts", "registry", "coordinates", "output"):
        p.add_argument(f"--{arg}", type=Path, required=True)
    p.add_argument("--held-out-fold", type=int, required=True)
    p.add_argument("--counts-layer", default="counts")
    p.add_argument("--ensembl-column", default="ensembl_id")
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)

    reg = pd.read_parquet(a.registry)
    reg = reg.loc[reg.cohort.eq("GSE149614")].set_index("barcode", verify_integrity=True)
    data = anndata.read_h5ad(a.counts, backed="r")
    if not data.obs_names.is_unique or set(data.obs_names) != set(reg.index):
        raise ValueError("Count profile barcodes do not match the frozen donor registry")
    reg = reg.loc[data.obs_names].copy()
    if a.held_out_fold not in set(reg.fold):
        raise ValueError("Held-out donor fold is unavailable")
    if a.counts_layer == "X":
        source = data.X
    elif a.counts_layer in data.layers:
        source = data.layers[a.counts_layer]
    else:
        raise ValueError("Explicit raw count layer is missing")
    coord = pd.read_csv(a.coordinates, sep="\t", dtype={"chromosome": str})
    projection, windows = genome_projection(data.var[a.ensembl_column].to_numpy(), coord)
    blocks = []
    for start in range(0, data.n_obs, 512):
        counts = normalized(source[start:start + 512])
        blocks.append((counts @ projection).toarray())
    blocks = np.concatenate(blocks)
    data.file.close()

    reference_role = reg.malignancy.eq("normal_hepatocyte") & reg.fold.ne(a.held_out_fold)
    ref_donors = reg.loc[reference_role, "donor"].to_numpy()
    held = set(reg.loc[reg.fold.eq(a.held_out_fold), "donor"])
    if set(ref_donors) & held or len(set(ref_donors)) < 3:
        raise ValueError("No independent training-donor normal-hepatocyte reference")
    ref_profiles = blocks[reference_role.to_numpy()]
    reference = by_donor_reference(ref_profiles, ref_donors)
    # Calibrate the normal tail with leave-one-training-donor-out scores.
    normal_tails = []
    for donor in sorted(set(ref_donors)):
        use = ref_donors != donor
        baseline = by_donor_reference(ref_profiles[use], ref_donors[use])
        scores = broad_cnv(ref_profiles[~use], baseline)
        normal_tails.append(np.quantile(scores, 0.99))
    cutoff = float(max(normal_tails))
    scores = broad_cnv(blocks, reference)
    result = reg.reset_index()
    result["cnv_proxy"] = scores
    result["reference_eligible"] = reference_role.to_numpy()
    result["hepatocyte_cnv_high"] = result.malignancy.eq("needs_cnv") & (scores > cutoff)
    result["analysis_role"] = "development"
    result.loc[result.fold.eq(a.held_out_fold), "analysis_role"] = "held_out"

    with tempfile.TemporaryDirectory(prefix=f".{a.output.name}.", dir=a.output.parent) as tmp:
        stage = Path(tmp)
        result.to_parquet(stage / "cnv_proxy.parquet", index=False)
        np.savez(stage / "cnv_fit.npz", windows=np.asarray(windows, dtype=str),
                 reference=reference, cutoff=np.array(cutoff))
        receipt = {
            "held_out_fold": a.held_out_fold, "fit_normal_donors": len(set(ref_donors)),
            "held_out_donors": len(held), "blocks": len(windows), "normal_score_cutoff": cutoff,
            "n_held_out_cnv_high": int(result.loc[result.analysis_role.eq("held_out"), "hepatocyte_cnv_high"].sum()),
            "meaning": "Broad RNA expression deviation proxy. Independent DNA CNV validation is unavailable.",
            "inputs": {f.name: file_record(f) for f in (a.counts, a.registry, a.coordinates)},
            "outputs": {f.name: file_record(f) for f in sorted(stage.iterdir())},
        }
        (stage / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        stage.rename(a.output)


if __name__ == "__main__":
    main()
