"""Independent TCGA-LIHC genomic, subtype and pathway annotations."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests

from .cohort import digest, write_receipt


MUTATION_GENES = ("TP53", "CTNNB1", "AXIN1", "ARID1A", "TERT")
CNA_GENES = ("FGF19", "CCND1", "MET", "VEGFA", "CDKN2A")
CODING_CLASSES = {"Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins",
                  "Splice_Site", "In_Frame_Del", "In_Frame_Ins", "Translation_Start_Site",
                  "Nonstop_Mutation"}


def read_gmt(path):
    result = {}
    for line in path.read_text().splitlines():
        name, _, *genes = line.split("\t")
        if len(set(genes)) < 20 or name in result:
            raise ValueError(f"Invalid gene set in {path}")
        result[name] = sorted(set(genes))
    return result


def matrix(cohort):
    participants = pd.read_csv(cohort / "participants.tsv", sep="\t", index_col="participant")
    genes = pd.read_csv(cohort / "genes.tsv", sep="\t")
    raw = np.load(cohort / "fpkm_unstranded.npy", allow_pickle=False)
    if raw.shape != (len(participants), len(genes)):
        raise ValueError("RNA matrix dimensions disagree with frozen axes")
    symbols = (genes.gene_type.eq("protein_coding") & genes.gene_name.notna()
               & ~genes.gene_name.duplicated(keep=False))
    expression = pd.DataFrame(np.log2(1.0 + raw[:, symbols]), index=participants.index,
                              columns=genes.loc[symbols, "gene_name"])
    return participants, expression


def mutation_table(maf_root, patients):
    statuses, lineage = set(), {}
    changes = {gene: set() for gene in MUTATION_GENES}
    columns = ["Tumor_Sample_Barcode", "Hugo_Symbol", "Variant_Classification"]
    for path in sorted(maf_root.glob("*/*.maf.gz")):
        variants = pd.read_csv(path, sep="\t", comment="#", usecols=columns, dtype=str)
        lineage[str(path.relative_to(maf_root))] = digest(path)
        variants = variants.loc[variants.Tumor_Sample_Barcode.str.slice(13, 15).eq("01")].copy()
        variants["patient"] = variants.Tumor_Sample_Barcode.str.slice(0, 12)
        statuses.update(variants.patient)
        for gene in MUTATION_GENES:
            selected = variants.Hugo_Symbol.eq(gene) & variants.Variant_Classification.isin(CODING_CLASSES)
            changes[gene].update(variants.loc[selected, "patient"])
    if not lineage:
        raise ValueError("No open TCGA MAF files in the registered directory")
    output = pd.DataFrame({f"{gene}_coding_mutation": pd.array(
        [p in changes[gene] if p in statuses else pd.NA for p in patients], dtype="boolean")
        for gene in MUTATION_GENES}, index=patients)
    return output, lineage


def cna_table(path, participants):
    cna = pd.read_csv(path, sep="\t").set_index("Hugo_Symbol")
    cna = cna.loc[list(CNA_GENES)].drop(columns=["Entrez_Gene_Id"]).T
    cna.index = cna.index.str.slice(0, 12)
    if not cna.index.is_unique or not cna.columns.is_unique:
        raise ValueError("The GISTIC cohort has ambiguous identifiers")
    return cna.reindex(participants.index).add_suffix("_gistic")


def absolute_table(path, participants):
    absolute = pd.read_csv(path, sep="\t").loc[lambda x: x["call status"].eq("called")].copy()
    absolute["sample_vial"] = absolute["sample"].str.slice(0, 16)
    exact = absolute.drop_duplicates("sample_vial", keep=False).set_index("sample_vial").purity
    barcode = participants["Sample ID"].map(exact)
    return pd.to_numeric(barcode, errors="coerce").rename("absolute_purity")


def subtype_and_hallmark(expression, raw):
    # Gene-wise normalization across the unlabelled cohort is confined to subtype annotation.
    # This removes constitutive expression differences between the three fixed templates.
    standardized = expression.sub(expression.median()).div(expression.std().replace(0, np.nan)).fillna(0)
    ranked = pd.DataFrame(rankdata(standardized, axis=1) / expression.shape[1] - .5,
                          columns=expression.columns, index=expression.index)
    stats, sets = {}, {}
    rng = np.random.default_rng(290123)
    null = []
    # Size-matched random signatures provide per-sample one-sided enrichment p-values.
    for name in ("S1", "S2", "S3"):
        signature, = read_gmt(raw / f"hoshida-{name}.gmt").values()
        present = sorted(set(signature) & set(ranked))
        if len(present) < .8 * len(signature):
            raise ValueError(f"Low fixed Hoshida signature coverage for {name}")
        sets[name] = {"present": len(present), "total": len(signature)}
        point = ranked[present].mean(axis=1)
        draws = [ranked.iloc[:, rng.choice(ranked.shape[1], len(present), replace=False)].mean(axis=1)
                 for _ in range(499)]
        distribution = np.asarray(draws)
        stats[f"hoshida_{name}"] = (point - distribution.mean(axis=0)) / distribution.std(axis=0, ddof=1)
        null.append((1 + (distribution >= point.to_numpy()).sum(axis=0)) / (1 + len(distribution)))
    output = pd.DataFrame(stats)
    chosen = output.idxmax(axis=1).str.removeprefix("hoshida_")
    best_col = chosen.map({"S1": 0, "S2": 1, "S3": 2}).to_numpy()
    p = np.asarray(null).T[np.arange(len(output)), best_col]
    q = multipletests(np.minimum(1., p * 3), method="fdr_bh")[1]
    output["hoshida_subtype"] = chosen.where(q <= .05, "unassigned")
    output["hoshida_fdr"] = q
    output["proliferative_class"] = output.hoshida_subtype.map(
        {"S1": "proliferative", "S2": "proliferative", "S3": "non_proliferative"})
    within_sample = pd.DataFrame(rankdata(expression, axis=1) / expression.shape[1] - .5,
                                 columns=expression.columns, index=expression.index)
    for name in ("e2f", "g2m"):
        signature, = read_gmt(raw / f"hallmark-{name}.gmt").values()
        present = sorted(set(signature) & set(ranked))
        sets[name] = {"present": len(present), "total": len(signature)}
        output[f"hallmark_{name}"] = within_sample[present].mean(axis=1)
    return output, sets


def pathway_scores(expression, path, top=100):
    model = pd.read_csv(path, sep="\t").sort_values(["pathway", "p.value", "gene"])
    model = model.groupby("pathway", sort=True).head(top)
    results, coverage = {}, {}
    for pathway, rows in model.groupby("pathway"):
        weight = rows.drop_duplicates("gene").set_index("gene").weight
        genes = weight.index.intersection(expression.columns)
        coverage[pathway] = {"present": len(genes), "total": len(weight)}
        if len(genes) < .6 * len(weight):
            raise ValueError(f"Low PROGENy footprint overlap for {pathway}")
        results[pathway] = expression[genes].to_numpy() @ weight.loc[genes].to_numpy()
    return pd.DataFrame(results, index=expression.index).add_prefix("progeny_"), coverage


def prepare(cohort, raw, maf, output):
    participants, expression = matrix(cohort)
    annotation, signatures = subtype_and_hallmark(expression, raw)
    progeny, footprints = pathway_scores(expression, raw / "progeny.tsv")
    variants, maf_hashes = mutation_table(maf, participants.index)
    cna = cna_table(raw / "cna.tsv", participants)
    purity = absolute_table(raw / "absolute.tsv", participants)
    combined = pd.concat([participants, annotation, progeny, variants, cna, purity], axis=1)
    output.mkdir(parents=True, exist_ok=False)
    combined.to_parquet(output / "annotations.parquet")
    # Tab-delimited expression for the official ESTIMATE package.
    expression.T.to_csv(output / "estimate_input.tsv", sep="\t", index_label="GeneSymbol")
    gmt = sorted(raw.glob("*.gmt"))
    sources = [*gmt, raw / "progeny.rda", raw / "progeny.tsv", raw / "absolute.tsv", raw / "cna.tsv"]
    write_receipt(output / "receipt.json", {
        "cohort_receipt_sha256": digest(cohort / "receipt.json"),
        "public_sources_sha256": {p.name: digest(p) for p in sources},
        "maf_sha256": maf_hashes,
        "signature_coverage": signatures,
        "progeny_footprint_coverage": footprints,
        "participants": len(combined),
        "outputs": {p.name: digest(p) for p in sorted(output.iterdir())},
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("cohort", "raw", "maf", "output"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    prepare(**vars(parser.parse_args()))
