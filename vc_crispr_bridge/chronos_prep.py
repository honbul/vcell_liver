"""Register DepMap gene-cold and HCC-lineage-cold Chronos panels."""
import argparse
from hashlib import sha256
from pathlib import Path
import re
import tempfile

import numpy as np
import pandas as pd

from .gene_features import unambiguous_symbol_map
from .perturbseq import record
from .predict import write_json


FILES = {
    "chronos": "CRISPRGeneEffect.csv", "models": "Model.csv", "hgnc": "Gene.csv",
    "rna": "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv", "cnv": "OmicsCNGeneWGS.csv",
    "mut": "OmicsSomaticMutationsMatrixHotspot.csv",
}


def stable_group(key, seed):
    return int.from_bytes(sha256(f"chronos:{seed}:{key}".encode()).digest()[:8], "big") % 5


def entrez_axis(columns):
    out = {}
    for name in columns:
        match = re.fullmatch(r".+ \(([0-9]+)\)", name)
        if not match:
            continue
        if match[1] in out:
            raise ValueError(f"Ambiguous Entrez assay annotation for {match[1]}")
        out[match[1]] = name
    return out


def omics(path, models, entrez):
    header = pd.read_csv(path, nrows=0).columns
    axis = entrez_axis(header)
    source = pd.read_csv(path, usecols=["ModelID", "IsDefaultEntryForModel", *axis.values()],
                         dtype={name: np.float32 for name in axis.values()}, low_memory=False)
    source = source[source.IsDefaultEntryForModel.astype(str).str.lower().isin(["true", "yes", "1"])]
    source = source.loc[source.ModelID.isin(models)].drop(columns="IsDefaultEntryForModel").set_index("ModelID")
    if not source.index.is_unique:
        raise ValueError(f"Multiple default molecular profiles for a DepMap model: {path}")
    source = source.rename(columns={v: k for k, v in axis.items()})
    return source.reindex(index=models, columns=entrez).to_numpy(np.float32), set(source.index), source


def annotated_models(path):
    cols = ["ModelID", "PatientID", "StrippedCellLineName", "OncotreeLineage", "OncotreeCode",
            "AgeCategory", "ModelType"]
    data = pd.read_csv(path, usecols=cols, dtype=str).dropna(subset=["ModelID", "PatientID", "OncotreeLineage"])
    if not data.ModelID.is_unique:
        raise ValueError("Repeated DepMap model")
    data = data[data.ModelType.eq("Cell Line")].set_index("ModelID")
    liver = data.OncotreeLineage.eq("Liver")
    hepatic_donors = set(data.loc[liver, "PatientID"])
    # SK-HEP-1 has independent endothelial provenance (PMID 1371504).
    adult_hcc = liver & data.OncotreeCode.eq("HCC") & data.AgeCategory.eq("Adult") & data.index.to_series().ne("ACH-000361")
    data["partition"] = np.where(adult_hcc, "hcc", np.where(data.PatientID.isin(hepatic_donors), "exclude", "other"))
    return data[data.partition.ne("exclude")]


def pairs(m, genes, cap=None, seed=41026, outcomes=None):
    rng = np.random.default_rng(seed)
    model_indices = np.tile(m, (len(genes), 1))
    if cap is not None and len(m) > cap:
        # Draw uniformly by gene, giving each gene equal assay opportunity.
        sampled = rng.random((len(genes), len(m)), dtype=np.float32)
        if outcomes is not None:
            missing = ~np.isfinite(outcomes[np.ix_(m, genes)]).T
            sampled[missing] = np.inf
        choose = np.argpartition(sampled, cap - 1, axis=1)[:, :cap]
        model_indices = model_indices[np.arange(len(genes))[:, None], choose]
    target_indices = np.broadcast_to(genes[:, None], model_indices.shape)
    m, g = model_indices.ravel(), target_indices.ravel()
    if outcomes is not None:
        resolved = np.isfinite(outcomes[m, g])
        return m[resolved], g[resolved]
    return m, g


def role_frame(model_index, gene_index, models, gene_axis, features):
    result = pd.DataFrame({
        "model_id": models.index.to_numpy()[model_index], "donor": models.PatientID.to_numpy()[model_index],
        "gene": gene_axis[gene_index], "cell_index": model_index,
    })
    for name, array in features.items():
        result[name] = array[model_index, gene_index]
    return result


def prepare(directory, out, seed):
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = {name: directory / filename for name, filename in FILES.items()}
    genes = pd.read_csv(paths["hgnc"], usecols=["entrez_id", "ensembl_gene_id", "symbol"], dtype=str).dropna()
    symbols = unambiguous_symbol_map(paths["hgnc"])
    genes = genes[genes.ensembl_gene_id.isin(symbols.index) & ~genes.entrez_id.duplicated(keep=False)]
    header = pd.read_csv(paths["chronos"], nrows=0).columns
    source_axis = entrez_axis(header)
    genes = genes[genes.entrez_id.isin(source_axis)].set_index("entrez_id").sort_index()
    columns = [source_axis[i] for i in genes.index]
    effect = pd.read_csv(paths["chronos"], usecols=[header[0], *columns], index_col=0,
                         dtype={field: np.float32 for field in columns})
    if not effect.index.is_unique or len(genes) < 500:
        raise ValueError("Ambiguous or sparse Chronos assay")
    models = annotated_models(paths["models"])
    ready = models.index.intersection(effect.index)
    measures = {}
    for name in ("rna", "cnv", "mut"):
        measures[name] = omics(paths[name], ready, genes.index)
    complete = ready.intersection(measures["rna"][1])
    # Keep one culture per donor independent of effect magnitudes, preferring
    # complete basal omics before lexicographic tie breaking.
    models = models.loc[complete].copy()
    models["profile_count"] = models.index.isin(measures["cnv"][1]).astype(int) + models.index.isin(measures["mut"][1])
    models = models.reset_index().sort_values(["profile_count", "ModelID"], ascending=[False, True])
    models = models.drop_duplicates("PatientID").set_index("ModelID").sort_index()
    target_axis = genes.ensembl_gene_id.to_numpy(dtype=str)
    gene_bin = np.array([stable_group(g, seed + 2) for g in target_axis])
    donor_bin = np.array([stable_group(d, seed + 3) for d in models.PatientID])
    hcc = models.partition.to_numpy() == "hcc"
    held_genes = np.flatnonzero(gene_bin == 0)
    validation_genes = np.flatnonzero(gene_bin == 1)
    train_genes = np.flatnonzero(gene_bin >= 2)
    train_models = np.flatnonzero(~hcc & (donor_bin > 0))
    validation_models = np.flatnonzero(~hcc & (donor_bin == 0))
    hcc_models = np.flatnonzero(hcc)
    if min(len(hcc_models), len(validation_models), len(train_models)) < 5:
        raise ValueError("Insufficient independent HCC and source donors")
    raw_rna = measures["rna"][2].loc[models.index]
    rna_frame = raw_rna.reindex(sorted(raw_rna.columns), axis=1)
    indices = ready.get_indexer(models.index)
    assays = {name: result[0][indices] for name, result in measures.items()}
    outcome = effect.loc[models.index, columns].to_numpy(np.float32)
    train_genes = train_genes[np.isfinite(outcome[np.ix_(train_models, train_genes)]).any(axis=0)]
    with tempfile.TemporaryDirectory(prefix=f".{out.name}.", dir=out.parent) as directory:
        stage = Path(directory)
        for name, m, g, cap, observed in (
            ("fit", train_models, train_genes, 48, True), ("validation", validation_models, validation_genes, 40, True),
            ("query_gene_cold", hcc_models, held_genes, None, False),
            ("query_same_gene", hcc_models, train_genes, None, False),
        ):
            row, col = pairs(m, g, cap, seed, outcome if observed else None)
            features = role_frame(row, col, models, target_axis, assays)
            if observed:
                features["chronos"] = outcome[row, col]
            else:
                measured = np.isfinite(outcome[row, col])
                labels = features.loc[measured, ["model_id", "gene"]].assign(chronos=outcome[row[measured], col[measured]])
                labels.to_parquet(stage / f"{name.replace('query', 'labels')}.parquet", index=False)
            features.to_parquet(stage / f"{name}.parquet", index=False)
        np.save(stage / "rna.npy", rna_frame.to_numpy(np.float32), allow_pickle=False)
        models.to_csv(stage / "models.tsv", sep="\t", index_label="model_id")
        genes.to_csv(stage / "gene_map.tsv", sep="\t", index_label="entrez")
        write_json(stage / "receipt.json", {
            "version": "DepMap-25Q3", "seed": seed,
            "inputs": {name: record(path) for name, path in paths.items()},
            "training_donors": len(train_models), "validation_donors": len(validation_models), "hcc_donors": len(hcc_models),
            "training_genes": len(train_genes), "validation_genes": len(validation_genes), "cold_genes": len(held_genes),
            "hcc_model_ids": models.index[hcc_models].tolist(),
            "files": {path.name: record(path) for path in sorted(stage.iterdir())},
        })
        stage.rename(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--depmap", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--seed", type=int, default=41026)
    args = p.parse_args()
    prepare(args.depmap, args.output, args.seed)


if __name__ == "__main__":
    main()
