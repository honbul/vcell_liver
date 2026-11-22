"""Offline deconvolution, graph features and published ICI RNA biomarkers."""
from dataclasses import dataclass
from importlib import resources
import json

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import rankdata

from .reference import REGULATORS, TYPES, checked


IFNG6 = ("CXCL10", "CXCL9", "HLA-DRA", "IDO1", "IFNG", "STAT1")
# Ayers et al., JCI 91190 and the authors' disclosed Table 3B (AU2020204557A1).
GEP18 = {
    "CCL5": .008346, "CD27": .072293, "CD274": .042853, "CD276": -.023900,
    "CD8A": .031021, "CMKLR1": .151253, "CXCL9": .074135, "CXCR6": .004313,
    "HLA-DQA1": .020091, "HLA-DRB1": .058806, "HLA-E": .071750, "IDO1": .060679,
    "LAG3": .123895, "NKG7": .075524, "PDCD1LG2": .003734, "PSMB10": .032999,
    "STAT1": .250229, "TIGIT": .084767,
}
HOUSEKEEPING = (
    "ABCF1", "C14ORF102", "G6PD", "OAZ1", "POLR2A", "SDHA",
    "STK11IP", "TBC1D10B", "TBP", "UBB", "ZBTB34",
)
EDGE_CHANNELS = ("liana_background", *REGULATORS)
WNT_PATH = "HALLMARK_WNT_BETA_CATENIN_SIGNALING"


@dataclass
class Profiles:
    bulk: np.ndarray
    nodes: np.ndarray
    edges: np.ndarray
    fractions: np.ndarray
    unweighted_edges: np.ndarray
    unweighted_nodes: np.ndarray
    wnt_path_node: int | None
    audit: dict

    def counterfactual(self, axis, attenuation):
        channels = self.edges.copy()
        channels[..., EDGE_CHANNELS.index(axis)] *= 1 - attenuation
        nodes = self.nodes.copy()
        if axis == "WNT" and self.wnt_path_node is not None:
            # CTNNB1 is intracellular. Also attenuate the malignant-node
            # WNT activity relative to its fixed reference.
            nodes[:, TYPES.index("malignant"), self.wnt_path_node] *= 1 - attenuation
        return nodes, channels


def hallmark_sets(path):
    profiles = {}
    with path.open() as stream:
        for line in stream:
            name, _, *genes = line.rstrip().split("\t")
            if not name.startswith("HALLMARK_") or not genes or name in profiles:
                raise ValueError(f"Malformed Hallmark record: {name}")
            profiles[name] = set(genes)
    if len(profiles) != 50:
        raise ValueError("Expected the complete 50 Hallmark pathways")
    return profiles


def ranks(values):
    return ((rankdata(values, method="average", axis=1) - .5) / values.shape[1]).astype(np.float32)


def markers_and_prior(prior_dir, genes, restricted):
    receipt = json.loads((prior_dir / "receipt.json").read_text())
    ref = pd.read_parquet(checked(prior_dir, "donor_equal_reference.parquet", receipt))
    ref = ref.reindex(genes, fill_value=0).loc[:, TYPES].astype(np.float32)
    edges = pd.read_csv(checked(prior_dir, "ligand_receptor.tsv", receipt), sep="\t")
    markers = json.loads(checked(prior_dir, "markers.json", receipt).read_text())
    if restricted:
        # The assay probe list is public metadata. Choose top landmark genes
        # again within that list using only the independent reference.
        for name in TYPES:
            others = ref.drop(columns=name).mean(axis=1)
            contrast = np.log2((ref[name] + 1) / (others + 1))
            markers[name] = contrast[ref[name].ge(15)].nlargest(24).index.to_list()
    positions = np.unique(np.concatenate([genes.get_indexer(markers[kind]) for kind in TYPES]))
    positions = positions[positions >= 0]
    marker_support = {kind: len(genes.intersection(markers[kind])) for kind in TYPES}
    return ref.to_numpy(), edges, positions, marker_support


def fractions_and_expression(expr_ranks, reference, markers):
    # Transform each expression profile independently to the abundances of
    # the HCC reference. Weighted NNLS gives each landmark gene comparable
    # influence without amplifying low expression or fitting a query batch.
    ranked_abundance = np.sort(reference.mean(axis=1))
    mapped = np.interp(expr_ranks, (np.arange(len(reference)) + .5) / len(reference), ranked_abundance)
    weights = np.sqrt(reference[markers].mean(axis=1) + 10)
    design = reference[markers] / weights[:, None]
    penalty = .035 * np.linalg.norm(design, ord="fro") / np.sqrt(len(TYPES))
    design = np.vstack([design, penalty * np.eye(len(TYPES))])
    fits = np.array([
        nnls(design, np.r_[row[markers] / weights, np.zeros(len(TYPES))])[0]
        for row in mapped
    ])
    if np.any(fits.sum(axis=1) <= 0):
        raise ValueError("A sample has no deconvolvable expression")
    fractions = fits / fits.sum(axis=1, keepdims=True)
    # Borrow a fixed cell-type-specific prior and allocate the measured
    # bulk residual. Clip extreme residuals caused by highly sparse markers.
    return fractions.astype(np.float32), mapped


def expression_by_type(mapped, reference, fractions):
    expected = fractions @ reference.T
    ratio = np.clip((mapped + 2) / (expected + 2), .1, 10)
    allocated = (reference.T[None, ...] * ratio[:, None, :]).astype(np.float32)
    return allocated


def node_programs(allocated, reference, weights):
    # Relative to the HCC cell-type reference, down-weight genes for which
    # that cell type contributes little. The result is a type-weighted
    # pathway score, not a directly measured within-patient transcriptome.
    residual = np.log2(1 + allocated) - np.log2(1 + reference.T)[None, ...]
    support = np.sqrt(reference.T / (reference.T + 10))
    weighted = support[..., None] * weights[None, ...]
    denominators = weighted.sum(axis=1).clip(min=1e-6)
    return np.einsum("ntg,tgp->ntp", residual, weighted, optimize=True) / denominators


def axis_for(ligand, receptor):
    if "VEGFA" in ligand and "KDR" in receptor:
        return "VEGFA_KDR"
    if "SPP1" in ligand and "CD44" in receptor:
        return "SPP1_CD44"
    if "TGFB1" in ligand:
        return "TGFB1"
    for axis, receptor_name in (("TIGIT", "TIGIT"), ("TIM3", "HAVCR2"), ("LAG3", "LAG3")):
        if receptor_name in receptor and REGULATORS[axis].intersection(ligand):
            return axis
    if REGULATORS["WNT"].intersection(ligand) and REGULATORS["WNT"].intersection(receptor):
        return "WNT"
    return "liana_background"


def interaction_features(allocated, fractions, pairs, genes):
    output = np.zeros((len(allocated), len(TYPES), len(TYPES), len(EDGE_CHANNELS)), dtype=np.float32)
    counts = dict.fromkeys(EDGE_CHANNELS, 0)
    for row in pairs.itertuples(index=False):
        ligand, receptor = row.ligand.split("_"), row.receptor.split("_")
        left, right = genes.get_indexer(ligand), genes.get_indexer(receptor)
        if left.min() < 0 or right.min() < 0:
            continue
        axis = axis_for(ligand, receptor)
        channel = EDGE_CHANNELS.index(axis)
        sending = fractions * np.log2(1 + allocated[..., left].min(axis=-1))
        receiving = fractions * np.log2(1 + allocated[..., right].min(axis=-1))
        output[..., channel] += sending[..., None] * receiving[:, None, :]
        counts[axis] += 1
    for i, channel in enumerate(EDGE_CHANNELS):
        output[..., i] = np.log1p(output[..., i] / max(counts[channel], 1))
    if counts["liana_background"] < 20:
        raise ValueError("Gene support is too narrow for the declared TME graph")
    return output, counts


def published_markers(expression):
    result, audit = {}, {}
    housekeeping = [g for g in HOUSEKEEPING if g in expression]
    if len(housekeeping) < 5:
        raise ValueError("Required housekeeping genes are absent from the common assay")
    # JCI uses log10 counts. All registered abundance matrices are log2
    # units or author rlog, so the fixed basis conversion preserves ranks.
    z = expression.sub(expression[housekeeping].mean(axis=1), axis=0) * np.log10(2)
    weights = {
        "cd274": {"CD274": 1.0}, "ifng": {"IFNG": 1.0},
        "ifng6": dict.fromkeys(IFNG6, 1 / len(IFNG6)), "ayers_gep18": GEP18,
    }
    for label, genes in weights.items():
        present = [g for g in genes if g in z]
        if len(present) < len(genes):
            continue
        result[label] = z[present].to_numpy() @ np.array([genes[g] for g in present])
        audit[label] = len(present)
    return pd.DataFrame(result, index=expression.index), {"genes": audit, "housekeeping": housekeeping}


def tide_fit_center(fit, query):
    from tidepy.model import tide_pred
    from tidepy.pred import MODEL_DB
    from tidepy.utils import toEntrez

    transformed = toEntrez(query.sub(fit.mean(axis=0), axis=1).T.copy())
    bundled = MODEL_DB["tide"]
    names = ("Dysfunction", "Exclusion")
    genes = transformed.index.union(bundled[names[0]].index).union(bundled[names[1]].index)
    # pandas 1.5 requires an explicit common index for corrwith with a
    # Series. Use missing values rather than imputing unassayed genes.
    weights = {**bundled, **{name: bundled[name].reindex(genes) for name in names}}
    result = tide_pred(transformed.reindex(genes), cancer="Other", tide_model=weights)
    out = -result["TIDE"].reindex(query.index)
    if not np.isfinite(out).all():
        raise ValueError("TIDE produced nonfinite external scores")
    return out.to_numpy()


def tide_assets():
    data = resources.files("tidepy").joinpath("data")
    return [data.joinpath(name) for name in ("model.pkl", "Gene_Ref.pkl")]


def construct(expression, prior_dir, hallmark, restricted=False):
    genes = expression.columns
    values = expression.to_numpy(dtype=np.float32)
    if not np.isfinite(values).all() or not genes.is_unique:
        raise ValueError("Invalid bulk gene expression matrix")
    ref, pairs, markers, marker_support = markers_and_prior(prior_dir, genes, restricted)
    minimum = 3 if restricted else 8
    if min(marker_support.values()) < minimum:
        raise ValueError(f"Incomplete cellular signature: {marker_support}")
    rna = ranks(values)
    fractions, mapped = fractions_and_expression(rna, ref, markers)
    estimated = expression_by_type(mapped, ref, fractions)
    pathways = hallmark_sets(hallmark)
    names, columns = [], []
    for name, members in sorted(pathways.items()):
        legal = genes.isin(members)
        if legal.sum() >= 6:
            names.append(name)
            columns.append(legal / legal.sum())
    weights = np.array(columns, dtype=np.float32).T
    scores = node_programs(estimated, ref, weights)
    edges, pair_counts = interaction_features(estimated, fractions, pairs, genes)
    unit_fractions = np.full_like(fractions, 1 / len(TYPES))
    uniform_estimate = expression_by_type(mapped, ref, unit_fractions)
    unweighted_edges, _ = interaction_features(uniform_estimate, unit_fractions, pairs, genes)
    uniform_scores = node_programs(uniform_estimate, ref, weights)
    node = np.concatenate([scores, fractions[..., None]], axis=2)
    return Profiles(rna, node, edges, fractions, unweighted_edges,
                    np.concatenate([uniform_scores, unit_fractions[..., None]], axis=2),
                    names.index(WNT_PATH) if WNT_PATH in names else None, {
                        "genes": len(genes), "rank_axis": "assay-common intersected gene symbols",
                        "hallmark_pathways": names, "nnls_marker_support": marker_support,
                        "ligand_receptor_pairs": pair_counts,
                    })
