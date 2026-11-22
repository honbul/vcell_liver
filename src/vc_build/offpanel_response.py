"""Learn RNA features from drug screens outside the evaluation panel."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import ndtri
from scipy.stats import rankdata

from .expanded_panel_cv import PENALTY, RECEIPTS, registered_inputs, same_split
from .external import KEY, SOURCE, _validate, native_sensitivity
from .io import freeze_table, sha256
from .limore_corrected import shared_support
from .paired_expression import auxiliary_cell_grams
from .pdo_transfer_cv import device_cuda
from .response_component_benchmark import (
    BASELINES, EXTERNAL_TRACKS, FORECAST_SCHEMA, PUBLIC, external_inputs, external_reference, source_reference,
)
from .rna_state_projection import score_external, score_source
from .structured_cv import ASSAYS, paired_intervals, with_assay
from .structured_kernel import KernelGrid, drug_kernels, fit_cell_kernels, training_grid
from .v0plus_cv import molecular_inputs, panel_targets, validation_roster
from .v0plus_neural import load_auxiliary
from .v0plus_score import axis_metrics, crossed_patient_drug_ci
from .yang_hcc_external import require_hash

GDSC_EXTENSION = "7fc3f92b576273f71a8ef41df83353471d65d98d6d53d4b9f8f6a3ddb9acf20b"
PRIMARY = "offpanel_within_tissue"
ARMS = ("rna_spectral", "coverage_only", "within_tissue_shuffle", "offpanel_global", PRIMARY)
COMPARATORS = ("structured", "drug_ridge", "drug_mean", *ARMS[:-1])
RNA_DIMENSIONS = 128
LATENT_DIMENSIONS = 64
RIDGE = 8.
WEIGHT = .25


def load_extensions(args):
    parts, hashes = [], {}
    for name, root, expected in (("ctrp_prism", args.extension, RECEIPTS["extension"]),
                                 ("gdsc2", args.gdsc_extension, GDSC_EXTENSION)):
        require_hash(root / "receipt.json", expected)
        receipt = json.loads((root / "receipt.json").read_text())
        path = root / "additional_rows.parquet"
        require_hash(path, receipt["artifacts"][path.name])
        parts.append(pd.read_parquet(path))
        hashes[name] = {"registry": expected, "table": receipt["artifacts"][path.name]}
    return pd.concat(parts, ignore_index=True), hashes


def offpanel_targets(extra, fit, query, cells, protected):
    allowed_sources = fit.source_id.unique()
    groups = fit[["cell_id", "lineage_group", "tissue_origin"]].drop_duplicates().set_index("cell_id")
    if not groups.index.is_unique:
        raise ValueError("The protected response fit requires one lineage and tissue per cell")
    all_protected = pd.Index([*protected, *query.compound_id.unique()]).str.slice(stop=14)
    selected = extra[extra.cell_id.isin(cells) & extra.source_id.isin(allowed_sources)
                     & ~extra.compound_id.str.slice(stop=14).isin(all_protected)].copy()
    reference = selected.cell_id.map(groups.lineage_group)
    if not selected.lineage_group.eq(reference).all() or selected.lineage_group.isin(query.lineage_group).any():
        raise ValueError("An off-panel row differs from the registered fit lineage or overlaps a test lineage")
    selected["oriented"] = native_sensitivity(selected)
    keys = [*SOURCE, "compound_id"]
    measured = selected.groupby(["cell_id", *keys]).oriented.median().unstack(keys).reindex(cells)
    count = measured.count().to_numpy()
    eligible = (count >= 50) & measured.std().gt(1e-8).to_numpy()
    measured = measured.loc[:, eligible]
    count = count[eligible]
    if len(measured.columns) < 100:
        raise ValueError("The auxiliary panel needs at least 100 supported tasks")
    normal = ndtri((rankdata(measured.to_numpy(), axis=0, nan_policy="omit") - .5) / count)
    normal = np.clip(normal, -3, 3)
    labels = groups.tissue_origin.reindex(cells).fillna("Unknown").to_numpy()
    within = np.full_like(normal, np.nan)
    rng = np.random.default_rng(40721)
    perm = np.arange(len(cells))
    for label in sorted(set(labels)):
        rows = np.flatnonzero(labels == label)
        matrix = normal[rows]
        counts = np.isfinite(matrix).sum(axis=0)
        supported = counts >= 8
        if supported.any():
            ix = np.ix_(rows, supported)
            values = normal[ix]
            within[ix] = values - np.nanmean(values, axis=0)
            ordered = rng.permutation(rows)
            perm[ordered] = np.roll(ordered, 1)
    support = {}
    for source in sorted(set(measured.columns.get_level_values("source_id"))):
        columns = measured.columns.get_level_values("source_id") == source
        support[source] = {
            "tasks": int(columns.sum()),
            "observations": int(np.isfinite(normal[:, columns]).sum()),
            "within_tissue_observations": int(np.isfinite(within[:, columns]).sum()),
        }
    mask = np.isfinite(within).astype(np.float32)
    blocks = {PRIMARY: within, "offpanel_global": normal, "within_tissue_shuffle": within[perm],
              "coverage_only": mask - mask.mean(axis=0)}
    sources = measured.columns.get_level_values("source_id").to_numpy()
    return blocks, sources, {
        "tasks": support, "fit_cells": len(cells),
        "cells_with_auxiliary": int(np.isfinite(within).any(axis=1).sum()),
        "connected_parents": int(measured.columns.get_level_values("compound_id").nunique()),
        "protected_connectivity_groups": int(all_protected.nunique()),
    }


def spectral_coordinates(training, cross):
    values, vectors = torch.linalg.eigh((training + training.T) * .5)
    values = values[-RNA_DIMENSIONS:].clamp_min(1e-6).sqrt()
    vectors = vectors[:, -RNA_DIMENSIONS:]
    return vectors * values, (cross @ vectors) / values


def masked_coefficients(features, targets):
    """Solve a penalized regression per task over observed response rows."""
    truth = torch.as_tensor(targets, dtype=features.dtype, device=features.device)
    supported = truth.isfinite().sum(dim=0).ge(20)
    truth = truth[:, supported]
    eye = torch.eye(features.shape[1], device=features.device, dtype=features.dtype)
    coefficients = []
    for task in truth.split(64, dim=1):
        masked = features.T[None] * task.isfinite().T[:, None, :]
        gram = masked @ features + RIDGE * eye
        rhs = (features.T @ task.nan_to_num()).T.unsqueeze(2)
        coefficients.append(torch.cholesky_solve(rhs, torch.linalg.cholesky(gram)).squeeze(2).T)
    return torch.cat(coefficients, dim=1), supported


def latent_geometry(train, query, targets, task_sources):
    coefficients, supported = masked_coefficients(train, targets)
    source = task_sources[supported.cpu().numpy()]
    _, assay_index, counts = np.unique(source, return_inverse=True, return_counts=True)
    weights = np.sqrt(counts.mean() / counts[assay_index])
    coefficients = coefficients * coefficients.new_tensor(weights)
    centered = train - train.mean(dim=0)
    fitted = centered @ coefficients
    _, singular, axes = torch.linalg.svd(fitted, full_matrices=False)
    if singular[LATENT_DIMENSIONS - 1] < 1e-7:
        raise ValueError("Degenerate rank in the fitted off-panel response features")
    mapping = coefficients @ axes[:LATENT_DIMENSIONS].T
    return auxiliary_cell_grams(centered @ mapping, (query - train.mean(dim=0)) @ mapping)


def candidate_forecasts(fit, query, rna, edges, targets, auxiliary, extra, protected):
    _validate(fit, query)
    device = device_cuda()
    training = rna.loc[sorted(fit.cell_id.unique())]
    held = rna.loc[sorted(query.cell_id.unique())]
    basis, training_grams = fit_cell_kernels(training, edges, auxiliary, device=device)
    cross = basis.cross(held)
    train_rna, held_rna = spectral_coordinates(training_grams["rna"], cross["rna"])
    blocks, task_sources, audit = offpanel_targets(extra, fit, query, training.index, protected)
    extensions = {"rna_spectral": auxiliary_cell_grams(train_rna[:, -LATENT_DIMENSIONS:],
                                                       held_rna[:, -LATENT_DIMENSIONS:])}
    extensions.update({name: latent_geometry(train_rna, held_rna, values, task_sources)
                       for name, values in blocks.items()})
    _, anchors, rows, _ = panel_targets(fit)
    drugs, kernels = drug_kernels(rows, targets, device=device)
    weights, response = training_grid(rows, training.index, drugs, device=device)
    anchor = query.compound_id.map(anchors).to_numpy()
    parts, traces = [], {}
    for name in ARMS:
        train_extra, held_extra = extensions[name]
        cell_kernel = ((1 - WEIGHT) * training_grams["full"] + WEIGHT * train_extra,
                       (1 - WEIGHT) * cross["full"] + WEIGHT * held_extra)
        grid = KernelGrid(held.index, drugs, {"full": cell_kernel},
                          kernels, weights, response, basis.reference[-1].shape[1])
        residual, traces[name] = grid.forecast(query, "structured", PENALTY)
        parts.append(query[KEY].assign(method=name, predicted_value=anchor + residual))
    return pd.concat(parts, ignore_index=True), audit | {"solver": traces, "genes": training.shape[1]}


def predict(args):
    extra, receipts = load_extensions(args)
    old, clean, rna, _ = registered_inputs(args)
    edges, targets, priors = molecular_inputs(args.priors, args.hallmark, args.targets)
    auxiliary, auxiliary_receipt = load_auxiliary(args.auxiliary)
    external = args.mode in EXTERNAL_TRACKS
    if external:
        fit, query, rna, registered = external_inputs(args, args.mode)
        previous, reference = external_reference(args, args.mode, query)
    else:
        _, fit, check = same_split(old, clean, args.fold, args.mode)
        query = validation_roster(check)
        previous, reference = source_reference(args.cv_archive, args.mode, args.fold)
        registered = RECEIPTS["core"]
    new, diagnostics = candidate_forecasts(fit, query, rna, edges, targets, auxiliary, extra,
                                            clean.compound_id.unique())
    compared = (*BASELINES, *ARMS)
    if external:
        new = new.assign(track=args.mode, fold=0)
        compared += PUBLIC
    else:
        new = with_assay(new, check, args.fold)
    joined = pd.concat([previous, new], ignore_index=True)
    matrix = joined.pivot(index=[*KEY, "track", "fold"], columns="method", values="predicted_value")
    expected_rows = len(query) if external else len(check)
    if len(matrix) != expected_rows or set(matrix) != set(compared) or not np.isfinite(matrix.to_numpy()).all():
        raise ValueError("A method does not have matched, finite forecasts on all registered rows")
    if external:
        shared_support(joined, query, compared)
    freeze_table(joined[FORECAST_SCHEMA], args.out, provenance={
        "mode": args.mode, "fold": args.fold, "inputs_sha256": registered, "core_sha256": RECEIPTS["core"],
        "reference_sha256": reference, "priors": priors, "auxiliary": auxiliary_receipt,
        "extension_inputs": receipts, "diagnostics": diagnostics,
    })


def assessor(frame, *, external):
    comparisons = (*COMPARATORS, *PUBLIC) if external else COMPARATORS
    overall, _, _ = axis_metrics(frame, [(PRIMARY, name) for name in comparisons])
    compare = crossed_patient_drug_ci if external else paired_intervals
    across = {name: compare(frame, PRIMARY, name, repeats=1000 if external else 800) for name in comparisons}
    within = {(row["baseline"], row["metric"]): row["ci95"] for row in overall["paired_contrasts"]}
    gates = {name: all(across[name][metric][0] > 0 for metric in ("spearman", "pearson"))
             and all(within[name, metric][0] > -.005 for metric in ("spearman", "ndcg_at_5", "top5_overlap"))
             for name in comparisons}
    return {"metrics": overall, "across_cell_ci95": across, "gates": gates,
            "groups": frame.lineage_group.nunique(), "observations": frame.row_id.nunique()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    inputs = {
        "predict": ("extension", "gdsc-extension", "licob", "archive", "cv-archive",
                    "priors", "hallmark", "targets", "auxiliary"),
        "source": ("forecasts",), "external": ("licob", "forecasts", "limore-labels", "licob-labels"),
    }
    for action, fields in inputs.items():
        stage = stages.add_parser(action)
        for name in ("original", "core", *fields, "out"):
            stage.add_argument("--" + name, required=True, type=Path)
        if action == "predict":
            stage.add_argument("--mode", required=True, choices=(*ASSAYS, *EXTERNAL_TRACKS))
            stage.add_argument("--fold", type=int, choices=range(3), default=0)
    args = parser.parse_args()
    if args.stage == "predict":
        predict(args)
    elif args.stage == "source":
        score_source(args, assessor=assessor)
    else:
        score_external(args, assessor=assessor, arms=ARMS)


if __name__ == "__main__":
    main()
