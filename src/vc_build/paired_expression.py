"""Learn a transferable RNA cell kernel from paired array and sequencing profiles."""

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .cross_source_data import alias_token
from .expanded_panel_cv import PENALTY, RECEIPTS, registered_inputs, same_split
from .external import KEY, _validate
from .function_data import new_output
from .io import freeze_table, publish_json, sha256
from .limore_corrected import shared_support
from .pdo_transfer_cv import device_cuda
from .response_component_benchmark import (
    BASELINES, EXTERNAL_TRACKS, FORECAST_SCHEMA, PUBLIC, external_inputs, external_reference,
)
from .rna_state_projection import score_external, score_source
from .structured_cv import ASSAYS, paired_intervals, with_assay
from .structured_kernel import KernelGrid, dot_kernel, drug_kernels, fit_cell_kernels, radial, ranked_expression, training_grid
from .v0plus_cv import _rna, molecular_inputs, panel_targets, validation_roster
from .v0plus_neural import load_auxiliary
from .v0plus_score import axis_metrics, crossed_patient_drug_ci
from .yang_hcc_external import require_hash

ARRAY = "3191100a3055bd255330f9561109e730e96c8f8bfa4c6a8c3750364a359f0b31"
HGNC = "6f43d6ff43aa9fdfa5fb2f20a20a7cace66e6e02e2a0dcf19d9b726e2e248d20"
MODELS = "9dbb9de8805696c1345816ab07edd23fb4fd95e117739f3c5c3b1cf062c1233b"
PRIMARY = "paired_array"
ARMS = ("common_axis", "rna_only_basis", "shuffled_array", "weighted_genes", PRIMARY)
CONTROLS = ("common_axis", "rna_only_basis", "shuffled_array")
DIMS = 64
REGULARIZATION = .30
WEIGHT = .35


def read_array(path):
    with gzip.open(path, "rt") as stream:
        header = {}
        for line in stream:
            if line.startswith("!series_matrix_table_begin"):
                break
            if line.startswith(("!Sample_title\t", "!Sample_geo_accession\t")):
                key, *fields = next(csv.reader((line,), delimiter="\t"))
                header[key] = fields
        array = pd.read_csv(stream, sep="\t", index_col=0, comment="!")
    if list(array.columns) != header["!Sample_geo_accession"] or len(array.columns) != 917:
        raise ValueError("Unexpected array accession/column order")
    labels = pd.Series(header["!Sample_title"], index=array.columns)
    return array, labels


def id_map(models, labels):
    # Exclude ambiguous aliases and models having multiple historically deposited arrays.
    aliases = models[["CellLineName", "StrippedCellLineName"]].stack().rename("alias").reset_index()
    aliases["alias"] = aliases.alias.map(alias_token)
    counts = aliases.groupby("alias").ModelID.nunique()
    unique = aliases[aliases.alias.isin(counts[counts.eq(1)].index)].drop_duplicates("alias")
    linked = labels.map(alias_token).map(unique.set_index("alias").ModelID).dropna()
    return linked[~linked.duplicated(keep=False)]


def register(args):
    for path, expected in ((args.array, ARRAY), (args.hgnc, HGNC), (args.models, MODELS),
                           (args.core / "receipt.json", RECEIPTS["core"])):
        require_hash(path, expected)
    receipt = json.loads((args.core / "receipt.json").read_text())
    rna_path = args.core / "train/rna.parquet"
    require_hash(rna_path, receipt["artifacts"]["train/rna.parquet"]["sha256"])
    baseline = _rna(rna_path)
    annotation = pd.read_csv(args.hgnc, sep="\t", dtype=str, usecols=["symbol", "entrez_id", "status", "locus_group"])
    annotation = annotation[annotation.status.eq("Approved") & annotation.locus_group.eq("protein-coding gene")
                            & annotation.entrez_id.notna()]
    annotation = annotation[~annotation.entrez_id.duplicated(keep=False)
                            & ~annotation.symbol.duplicated(keep=False)]
    gene_names = annotation.set_index("entrez_id").symbol
    array, labels = read_array(args.array)
    native = array.index.to_series().str.extract(r"^(\d+)_at$", expand=False)
    mapped = native.map(gene_names)
    keep = mapped.notna() & mapped.isin(baseline.columns) & ~mapped.duplicated(keep=False)
    array = array.loc[keep].set_axis(mapped[keep].to_numpy())
    models = pd.read_csv(args.models, usecols=["ModelID", "ModelType", "CellLineName", "StrippedCellLineName",
                                              "OncotreeLineage"]).set_index("ModelID", verify_integrity=True)
    models = models[models.ModelType.eq("Cell Line")]
    linkage = id_map(models, labels)
    source_link = linkage[linkage.isin(baseline.index)]
    value = array.loc[:, source_link.index].T.set_axis(source_link.to_numpy()).sort_index().astype(np.float32)
    lineage = models.OncotreeLineage.reindex(value.index).fillna("Unknown").rename("tissue").to_frame()
    if len(value) < 500 or len(value.columns) < 12_000 or not np.isfinite(value.to_numpy()).all():
        raise ValueError("Insufficient paired RNA coverage")
    with new_output(args.out) as out:
        value.to_parquet(out / "array.parquet")
        lineage.to_parquet(out / "tissue.parquet")
        publish_json(out / "receipt.json", {
            "array_sha256": ARRAY, "gene_mapping_sha256": HGNC, "model_catalogue_sha256": MODELS,
            "core_sha256": RECEIPTS["core"], "array_samples": len(labels), "unique_mapped_models": len(linkage),
            "paired_core_models": len(value), "common_protein_coding_genes": len(value.columns),
            "by_tissue": lineage.tissue.value_counts().to_dict(),
            "files": {f.name: sha256(f) for f in sorted(out.glob("*.parquet"))},
        })


def paired_data(path):
    reg = path / "receipt.json"
    card = json.loads(reg.read_text())
    if (card["array_sha256"], card["gene_mapping_sha256"], card["core_sha256"]) != (ARRAY, HGNC, RECEIPTS["core"]):
        raise ValueError("Unexpected paired expression release")
    frames = []
    for name in ("array.parquet", "tissue.parquet"):
        require_hash(path / name, card["files"][name])
        frames.append(pd.read_parquet(path / name))
    return (*frames, sha256(reg))


def row_center(data, tissues):
    adjusted = pd.DataFrame(data).groupby(tissues.to_numpy()).transform("mean").to_numpy()
    return data - adjusted


def standardized(values, reference, device):
    center = reference.mean(axis=0)
    spread = np.maximum(reference.std(axis=0), .05)
    return torch.as_tensor(np.clip((values - center) / spread, -8, 8), device=device)


def principal_scores(matrix):
    eigen, basis = torch.linalg.eigh(dot_kernel(matrix, matrix))
    return basis[:, -DIMS:] * eigen[-DIMS:].clamp_min(1e-8).sqrt()


def auxiliary_cell_grams(train, held):
    linear = dot_kernel(train, train)
    scaling = linear.diagonal().mean()
    if not torch.isfinite(scaling) or scaling <= 1e-10:
        raise ValueError("Degenerate agreement geometry")
    linear = linear / scaling
    cross = dot_kernel(held, train) / scaling
    reference = linear.diagonal()
    q = held.square().mean(dim=1) / scaling
    return .5 * (linear + radial(linear, reference, reference, 2)), .5 * (cross + radial(cross, q, reference, 2))


def paired_latent(fit, held, matrix, tissues, device):
    """Predict leading within-tissue array scores from RNA using only paired fit models."""
    pair = fit.index.intersection(matrix.index)
    labels = tissues.tissue.loc[pair]
    enough = labels.value_counts()
    labels = labels[labels.isin(enough[enough.ge(5)].index)]
    if len(labels) < 250:
        raise ValueError("Too few matched fit pairs for the registered basis")
    pair = labels.index
    raw, query, paired_array = [ranked_expression(x) for x in (fit, held, matrix.loc[pair])]
    pos = fit.index.get_indexer(pair)
    all_x = standardized(raw, raw[pos], device)
    all_held = standardized(query, raw[pos], device)
    pair_x = all_x[pos]
    centered_x = torch.as_tensor(row_center(pair_x.cpu().numpy(), labels), device=device)
    z = standardized(paired_array, paired_array, torch.device("cpu")).numpy()
    centered_y = torch.as_tensor(row_center(z, labels), device=device)
    xgram = dot_kernel(pair_x, pair_x)
    pair_scale = xgram.diagonal().mean()
    factor = torch.linalg.cholesky(xgram / pair_scale + REGULARIZATION * torch.eye(len(pair), device=device))
    source_cross = dot_kernel(all_x, pair_x) / pair_scale
    held_cross = dot_kernel(all_held, pair_x) / pair_scale
    xpca, ypca = principal_scores(centered_x), principal_scores(centered_y)
    blocks = {"rna_only_basis": xpca, PRIMARY: ypca}
    rng = np.random.default_rng(51957)
    permutation = np.arange(len(pair))
    for _, idx in labels.groupby(labels).groups.items():
        indices = rng.permutation(pair.get_indexer(idx))
        permutation[indices] = np.roll(indices, 1)
    if (permutation == np.arange(len(pair))).any():
        raise ValueError("A shuffled array row retained its true RNA pair")
    blocks["shuffled_array"] = ypca[permutation]
    paired_dot = (centered_x * centered_y).sum(dim=0)
    reliability = paired_dot / (centered_x.norm(dim=0) * centered_y.norm(dim=0)).clamp_min(1e-8)
    weights = (.25 + .75 * reliability.clamp(0, 1).square()).sqrt()
    results = {"weighted_genes": auxiliary_cell_grams(all_x * weights, all_held * weights)}
    for name, y in blocks.items():
        coeff = torch.cholesky_solve(y.contiguous(), factor)
        fitted, predicted = source_cross @ coeff, held_cross @ coeff
        mean = fitted.mean(dim=0)
        results[name] = auxiliary_cell_grams(fitted - mean, predicted - mean)
    return results, {"fit_pairs": len(pair), "paired_by_tissue": labels.value_counts().to_dict(),
                     "positive_gene_fraction": (reliability > 0).float().mean().item(),
                     "median_within_tissue_gene_pearson": reliability.median().item(), "latent_dims": DIMS}


def candidates(fit, query, rna, edges, targets, auxiliary, array, tissues):
    _validate(fit, query)
    device = device_cuda()
    genes = rna.columns.intersection(array.columns).sort_values()
    train = rna.loc[sorted(fit.cell_id.unique()), genes]
    held = rna.loc[sorted(query.cell_id.unique()), genes]
    if len(genes) < 10_000:
        raise ValueError("Too few shared genes in the external axis")
    basis, base = fit_cell_kernels(train, edges, auxiliary, device=device)
    base = (base["full"], basis.cross(held)["full"])
    extensions, audit = paired_latent(train, held, array[genes], tissues, device)
    _, anchors, aligned, _ = panel_targets(fit)
    drugs, kernels = drug_kernels(aligned, targets, device=device)
    weights, responses = training_grid(aligned, train.index, drugs, device=device)
    anchor = query.compound_id.map(anchors).to_numpy()
    result = [query[KEY].assign(method="drug_mean", predicted_value=anchor)]
    support = {}
    for name in ARMS:
        grams = base if name == "common_axis" else tuple((1 - WEIGHT) * a + WEIGHT * b
                                                         for a, b in zip(base, extensions[name]))
        grid = KernelGrid(held.index, drugs, {"full": grams}, kernels, weights, responses,
                          basis.reference[-1].shape[1])
        values, diagnostics = grid.forecast(query, "structured", PENALTY)
        result.append(query[KEY].assign(method=name, predicted_value=anchor + values))
        support[name] = diagnostics
    return pd.concat(result, ignore_index=True), audit | {"genes": len(genes), "arms": support}


def predict(args):
    array, tissues, array_sha = paired_data(args.array)
    edges, targets, priors = molecular_inputs(args.priors, args.hallmark, args.targets)
    auxiliary, aux = load_auxiliary(args.auxiliary)
    if args.mode in EXTERNAL_TRACKS:
        fit, query, rna, registered = external_inputs(args, args.mode)
        previous, reference = external_reference(args, args.mode, query)
        new, audit = candidates(fit, query, rna, edges, targets, auxiliary, array, tissues)
        joined = pd.concat([previous, new[new.method.ne("drug_mean")].assign(track=args.mode, fold=0)])
        shared_support(joined, query, (*BASELINES, *PUBLIC, *ARMS))
    else:
        original, clean, rna, _ = registered_inputs(args)
        _, fit, check = same_split(original, clean, args.fold, args.mode)
        query = validation_roster(check)
        new, audit = candidates(fit, query, rna, edges, targets, auxiliary, array, tissues)
        shared_support(new, query, ("drug_mean", *ARMS))
        joined = with_assay(new, check, args.fold)
        registered, reference = RECEIPTS["core"], None
    freeze_table(joined[FORECAST_SCHEMA], args.out, provenance={
        "mode": args.mode, "fold": args.fold, "inputs_sha256": registered, "core_sha256": RECEIPTS["core"],
        "reference_sha256": reference, "array_receipt_sha256": array_sha, "priors": priors,
        "auxiliary": aux, "diagnostics": audit,
    })


def assessor(frame, *, external):
    comparisons = (*CONTROLS, "drug_mean", "weighted_genes")
    if external:
        comparisons += ("structured", "drug_ridge", *PUBLIC)
    overall, _, _ = axis_metrics(frame, [(PRIMARY, name) for name in comparisons])
    if external:
        across = {name: crossed_patient_drug_ci(frame, PRIMARY, name, repeats=1000) for name in comparisons}
    else:
        across = {name: paired_intervals(frame, PRIMARY, name) for name in comparisons}
    within = {(row["baseline"], row["metric"]): row["ci95"] for row in overall["paired_contrasts"]}
    gates = {name: all(across[name][metric][0] > 0 for metric in ("spearman", "pearson"))
             and all(within[name, metric][0] > -.005 for metric in ("spearman", "ndcg_at_5", "top5_overlap"))
             for name in comparisons}
    return {"metrics": overall, "across_cell_ci95": across, "gates": gates,
            "groups": frame.lineage_group.nunique(), "observations": frame.row_id.nunique()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    for action, args in {
        "register": ("array", "hgnc", "models"),
        "predict": ("array", "licob", "archive", "priors", "hallmark", "targets", "auxiliary"),
        "source": ("forecasts",), "external": ("licob", "forecasts", "limore-labels", "licob-labels"),
    }.items():
        stage = stages.add_parser(action)
        for name in (("core", *args, "out") if action == "register" else ("original", "core", *args, "out")):
            stage.add_argument("--" + name, required=True, type=Path)
        if action == "predict":
            stage.add_argument("--mode", required=True, choices=(*ASSAYS, *EXTERNAL_TRACKS))
            stage.add_argument("--fold", type=int, choices=range(3), default=0)
    args = parser.parse_args()
    if args.stage == "register":
        register(args)
    elif args.stage == "predict":
        predict(args)
    elif args.stage == "source":
        score_source(args, assessor=assessor)
    else:
        score_external(args, assessor=assessor, arms=ARMS)


if __name__ == "__main__":
    main()
