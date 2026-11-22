"""Independent C-track comparison of target, pathway and driver-annotation features."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .function_data import alias_token, raw_sheet
from .functional import (
    cell_coordinates, chemistry_matrix, compound_anchor, feature_matrix, fit_base,
)
from .io import freeze_table, publish_json, read_contract, sha256
from .splits import assignments
from .train import IDENTIFIERS, PREDICT_INPUTS, macro_mae, prepared_data, role_rows

# Map publisher assay annotations, which include family classes as well as symbols.
# These are aggregate expression features and do not assert individual target engagement.
TARGETS = {
    "BET": "BRD2 BRD3 BRD4", "HSP90": "HSP90AA1 HSP90AB1", "BCL2": "BCL2",
    "EGFR": "EGFR", "HER2": "ERBB2", "AURORAA": "AURKA", "AURORAB": "AURKB",
    "VEGFR": "FLT1 KDR FLT4", "IAP": "BIRC2 BIRC3 XIAP", "CDK1234569": "CDK1 CDK2 CDK3 CDK4 CDK5 CDK6 CDK9",
    "ACAT1": "SOAT1", "PIM": "PIM1 PIM2 PIM3", "MEK": "MAP2K1 MAP2K2", "CHK1": "CHEK1",
    "HDAC": "HDAC1 HDAC2 HDAC3 HDAC4 HDAC5 HDAC6 HDAC7 HDAC8 HDAC9 HDAC10 HDAC11",
    "PI3K": "PIK3CA PIK3CB PIK3CD PIK3CG", "FGFR": "FGFR1 FGFR2 FGFR3 FGFR4", "PLK1": "PLK1",
    "PROTEASOME": "PSMB5 PSMB6 PSMB7", "ABL": "ABL1 ABL2", "SRC": "SRC", "PDK1": "PDPK1",
    "MET": "MET", "ALK": "ALK", "CKIT": "KIT", "VEFGR": "FLT1 KDR FLT4", "RET": "RET",
    "TOPOISOMERASEI": "TOP1", "GLUTAMINASE": "GLS", "IGF1R": "IGF1R", "ATM": "ATM",
    "CK2": "CSNK2A1 CSNK2A2", "NAMPT": "NAMPT", "DNAMETHYLATION": "DNMT1 DNMT3A DNMT3B",
    "FLT3": "FLT3", "TOPOISOMERASEII": "TOP2A TOP2B", "EZH2": "EZH2", "MTOR": "MTOR",
    "KDR": "KDR", "BTK": "BTK", "BLK": "BLK", "KSP": "KIF11", "MDM2": "MDM2",
    "PDGFR": "PDGFRA PDGFRB", "DHFR": "DHFR", "DNAPK": "PRKDC", "PARP": "PARP1 PARP2",
    "CDK46": "CDK4 CDK6", "FAK": "PTK2", "RAF": "ARAF BRAF RAF1", "CDK25": "CDK2 CDK5",
    "JAK": "JAK1 JAK2 JAK3 TYK2", "DOT1L": "DOT1L", "HMGCR": "HMGCR",
    "FARNESYLTRANSFERASE": "FNTA FNTB", "IDH12": "IDH1 IDH2", "SURVIVIN": "BIRC5",
}
# Ambiguous publisher labels are left unmatched. Enzyme synonyms above distinguish
# sterol O-acyltransferase from the acetyl-CoA acetyltransferase HGNC symbol.
HALLMARK = (
    "APOPTOSIS", "DNA_REPAIR", "E2F_TARGETS", "G2M_CHECKPOINT",
    "KRAS_SIGNALING_UP", "PI3K_AKT_MTOR_SIGNALING", "MTORC1_SIGNALING", "WNT_BETA_CATENIN_SIGNALING",
)
MUTATION = ("TP53", "CTNNB1", "AXIN1", "ARID1A", "ARID2", "TERT", "KEAP1", "NFE2L2", "PTEN", "RB1", "APC")
COPY = ("FGF19", "FGFR1", "FGFR3", "FGFR4", "CDKN2A", "AXIN1", "PTEN", "RB1")
METHODS = ("direct_ridge", "target_expression", "hallmark_interaction", "driver_mutation_cna", "all_context")
PENALTIES = (10, 100, 1000)
DRIVER_SHA256 = "931fcd83e9ce372d4f1de2c52324a9d3eea721e5dff5c160e0dc9c76b8958452"
HALLMARK_SHA256 = "eecaf6dad908334ae885406ec72bdc0646d8917588ed7c219fac92fc5363f596"


def targets_from_labels(rows: pd.DataFrame) -> dict:
    mapping = rows[["compound_id", "drug_target_source_labels"]].drop_duplicates()
    if mapping["compound_id"].duplicated().any():
        raise ValueError("A registered compound has distinct target annotations.")
    return {
        drug: sorted({gene for label in str(labels).split(",") for gene in TARGETS.get(alias_token(label), "").split()})
        for drug, labels in mapping.itertuples(index=False, name=None)
    }


def pathway_genes(path):
    reference = {}
    for line in path.read_text().splitlines():
        name, _, *genes = line.split("\t")
        reference[name] = genes
    return {name: reference["HALLMARK_" + name] for name in HALLMARK}


def load_drivers(path, rows):
    original = raw_sheet(path, "CFG alterations", 13)
    original = original.dropna(subset=["gene"]).set_index("gene")
    aliases = {alias_token(cell): cell for cell in original.columns}
    if len(aliases) != len(original.columns):
        raise ValueError("Driver table aliases are ambiguous.")
    names = rows[["cell_id", "source_cell_label"]].drop_duplicates().set_index("cell_id")["source_cell_label"]
    cells = names.map(alias_token).map(aliases)
    measured = cells.dropna()
    matrix = original[measured.tolist()].T
    matrix.index = measured.index
    known = {0, 1, 2, 3, 4, 5, 7, 8, 9, 10}
    if set(matrix.stack().astype(int).unique()) - known:
        raise ValueError("New driver annotation encoding.")
    event = matrix.reindex(columns=MUTATION).isin({1, 2, 3, 4, 7, 8, 9, 10}).astype(int)
    event.columns = ["seq_" + name for name in event.columns]
    gain = matrix.reindex(columns=COPY).eq(5).astype(int)
    gain.columns = ["gain_" + name for name in gain.columns]
    loss = matrix.reindex(columns=COPY).eq(0).astype(int)
    loss.columns = ["loss_" + name for name in loss.columns]
    return pd.concat([event, gain, loss], axis=1), sorted(cells[cells.isna()].index)


@dataclass
class ContextFeatures:
    rna: pd.DataFrame
    genes: dict
    targets: dict
    drivers: pd.DataFrame

    def fit(self, base, train):
        cells = self.rna.loc[train["cell_id"].unique()]
        relevant = sorted(set().union(*self.genes.values(), *self.targets.values()) & set(cells.columns))
        profile = cells[relevant].dropna(axis=1, how="all")
        center = profile.mean()
        spread = profile.std().clip(lower=0.10)
        molecular = train.drop_duplicates("compound_id")
        bits, descriptors = chemistry_matrix(molecular)
        chemistry = np.c_[bits, base["descriptor_scale"].transform(descriptors)]
        drug_pca = PCA(n_components=min(8, len(molecular) - 1), svd_solver="full").fit(chemistry)
        scale = StandardScaler().fit(drug_pca.transform(chemistry))
        return center, spread, drug_pca, scale

    def design(self, base, rows, transform):
        center, spread, drug_pca, drug_scale = transform
        state, _ = cell_coordinates(base["rna"], rows, self.rna)
        bits, descriptors = chemistry_matrix(rows)
        functional = feature_matrix(base, bits, descriptors, state)
        anchor = compound_anchor(base, rows["compound_id"], bits)
        drug_state = drug_scale.transform(drug_pca.transform(functional[:, :functional.shape[1] - state.shape[1]]))
        expr = self.rna.loc[rows["cell_id"].unique(), center.index]
        standardized = ((expr - center) / spread).fillna(0).clip(-5, 5)
        pathway = np.column_stack([
            standardized.loc[:, standardized.columns.intersection(gene_set)].mean(axis=1).to_numpy()
            for gene_set in self.genes.values()
        ])
        pathway = pathway[standardized.index.get_indexer(rows["cell_id"])]
        target = np.zeros((len(rows), 3))
        for compound, indices in rows.groupby("compound_id").indices.items():
            genes = self.targets[compound]
            mapped = standardized.columns.intersection(genes)
            if not len(mapped):
                continue
            values = standardized.loc[rows.iloc[indices]["cell_id"], mapped].to_numpy()
            target[indices] = np.c_[values.mean(axis=1), values.max(axis=1), np.repeat(len(mapped) / len(genes), len(indices))]
        genotype = self.drivers.loc[rows["cell_id"]].to_numpy(dtype=float)

        def interaction(view):
            product = (view[:, :, None] * drug_state[:, None, :]).reshape(len(rows), -1)
            return np.c_[view, product]

        path_view, driver_view = interaction(pathway), interaction(genotype)
        return anchor, {
            "direct_ridge": functional, "target_expression": np.c_[functional, target],
            "hallmark_interaction": np.c_[functional, path_view],
            "driver_mutation_cna": np.c_[functional, driver_view],
            "all_context": np.c_[functional, target, path_view, driver_view],
        }


def fit_compare(fit, score, context):
    base = fit_base(fit, context.rna)
    definitions = context.fit(base, fit)
    fit_anchor, design = context.design(base, fit[PREDICT_INPUTS], definitions)
    score_anchor, query = context.design(base, score[PREDICT_INPUTS], definitions)
    residual = fit["endpoint_value"].to_numpy() - fit_anchor
    return {
        (method, penalty): score_anchor + Ridge(alpha=penalty).fit(design[method], residual).predict(query[method])
        for method in METHODS for penalty in PENALTIES
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--driver-workbook", type=Path, required=True)
    parser.add_argument("--hallmark-gmt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--map-only", action="store_true")
    options = parser.parse_args()
    contract = read_contract(options.config)
    rows, rna = prepared_data(options.prepared)
    targets = targets_from_labels(rows)
    options.out.mkdir(parents=True, exist_ok=False)
    publish_json(options.out / "drug_targets.json", targets)
    if options.map_only:
        return
    if sha256(options.driver_workbook) != DRIVER_SHA256:
        raise ValueError("LIMORE Table S3 hash differs from the publisher snapshot.")
    if sha256(options.hallmark_gmt) != HALLMARK_SHA256:
        raise ValueError("Hallmark GMT hash differs from registered release.")
    drivers, omitted = load_drivers(options.driver_workbook, rows)
    pathways = pathway_genes(options.hallmark_gmt)
    context = ContextFeatures(rna, pathways, targets, drivers)
    provenance = {
        "config_sha256": sha256(options.config),
        "code_sha256": sha256(Path(__file__)),
        "rows_sha256": sha256(options.prepared / "rows.parquet"),
        "rna_sha256": sha256(options.prepared / "rna.parquet"),
        "driver_workbook_sha256": sha256(options.driver_workbook),
        "hallmark_gmt_sha256": sha256(options.hallmark_gmt),
        "rows_with_unmatched_cell_drivers": int(rows.cell_id.isin(omitted).sum()),
        "unmatched_cell_driver_ids": omitted,
        "typed_compounds_with_mapped_targets": sum(bool(symbols) for symbols in targets.values()),
        "typed_compounds_with_measured_targets": sum(bool(set(symbols) & set(rna.columns)) for symbols in targets.values()),
        "pathway_gene_counts": {path: len(set(genes) & set(rna.columns)) for path, genes in pathways.items()},
        "driver_features": {name: int(count) for name, count in drivers.sum().items()},
        "driver_mapped_models": int(len(drivers)),
    }
    seed = contract["split"]["seed"]
    manifest = assignments(rows, "C", contract["split"]["outer_folds"], seed)
    manifest.to_parquet(options.out / "splits.parquet", index=False)
    kept = rows[rows.cell_id.isin(drivers.index)]
    indexed = kept.set_index("row_id", drop=False)
    manifest = manifest[manifest.cell_id.isin(drivers.index)]
    result, selections = [], []
    with threadpool_limits(limits=1):
        for outer, roles in manifest.groupby("fold"):
            fit = role_rows(indexed, roles, "fit")
            evaluation = role_rows(indexed, roles, "withheld")
            inner = assignments(fit, "C", contract["split"]["inner_folds"], seed + outer + 100)
            scores = {pair: [] for pair in ((method, penalty) for method in METHODS for penalty in PENALTIES)}
            for _, split in inner.groupby("fold"):
                training = role_rows(indexed, split, "fit")
                validation = role_rows(indexed, split, "withheld")
                predictions = fit_compare(training, validation, context)
                for pair, forecast in predictions.items():
                    scores[pair].append(macro_mae(validation, forecast, "C"))
            predictions = fit_compare(fit, evaluation, context)
            for method in METHODS:
                penalty = min(PENALTIES, key=lambda value: (np.mean(scores[method, value]), value))
                selections.append({"outer_fold": int(outer), "method": method, "penalty": penalty,
                                   "inner_mean_macro_mae": float(np.mean(scores[method, penalty]))})
                trace = evaluation[IDENTIFIERS].copy()
                trace["track"], trace["fold"], trace["method"] = "C", outer, method
                trace["predicted_value"] = predictions[method, penalty]
                result.append(trace)
            print(f"Context comparisons C fold {outer}, fit {len(fit)}, held {len(evaluation)}", flush=True)
    freeze_table(pd.concat(result, ignore_index=True), options.out / "predictions.parquet", provenance=provenance)
    publish_json(options.out / "support.json", provenance | {
        "scored_rows": len(kept), "scored_models": int(kept.cell_id.nunique()),
        "scored_lineages": int(kept.lineage_group.nunique()), "selections": selections,
    })
    kept[IDENTIFIERS + ["endpoint_value"]].to_parquet(options.out / "labels.parquet", index=False)


if __name__ == "__main__":
    main()
