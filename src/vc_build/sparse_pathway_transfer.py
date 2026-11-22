"""Learn assay-balanced general sensitivity and sparse drug-specific RNA effects."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .expanded_panel_cv import RECEIPTS, registered_inputs, same_split
from .external import KEY, _validate
from .io import freeze_table
from .limore_corrected import shared_support
from .pdo_transfer_cv import device_cuda
from .response_component_benchmark import (
    BASELINES, EXTERNAL_TRACKS, FORECAST_SCHEMA, PUBLIC, external_inputs, external_reference, source_reference,
)
from .response_components import component_scores, interaction_intervals
from .rna_state_projection import assess, score_external, score_source
from .structured_cv import ASSAYS, with_assay
from .structured_kernel import drug_kernels, fit_cell_kernels, training_grid
from .v0plus_cv import molecular_inputs, panel_targets, validation_roster
from .v0plus_neural import load_auxiliary

ADDITIVE = "rna_additive"
INDEPENDENT = "sparse_independent"
PRIMARY = "sparse_drug_graph"
ARMS = (ADDITIVE, INDEPENDENT, PRIMARY)
CONTROLS = ("structured", ADDITIVE, INDEPENDENT)
L2 = .04
L1_FRACTION = .015
GRAPH = .20


def drug_laplacian(kernel):
    adjacency = (kernel - .10).clamp_min(0)
    adjacency.fill_diagonal_(0)
    # Stable sorting fixes the choice of edges for identical target/chemical weights.
    neighbors = adjacency.argsort(dim=1, descending=True, stable=True)[:, :4]
    edges = torch.zeros_like(adjacency).scatter(1, neighbors, adjacency.gather(1, neighbors))
    symmetric = torch.maximum(edges, edges.T)
    degree = symmetric.sum(dim=1)
    return (torch.diag(degree) - symmetric) / degree.max().clamp_min(1e-8)


def centered(coefficients):
    return coefficients - coefficients.mean(1, keepdim=True)


def fit_sparse(x, weights, response, *, interaction=False, laplacian=None):
    def project(values):
        return centered(values) if interaction else values

    def quadratic(coef):
        delta = x @ project(coef) - response
        penalty_grad = L2 * coef
        if laplacian is not None:
            penalty_grad = penalty_grad + GRAPH * (coef @ laplacian)
        loss = ((weights * delta.square()).sum() + (coef * penalty_grad).sum()) / 2
        return loss, project(x.T @ (weights * delta)) + penalty_grad

    coef = torch.zeros((x.shape[1], response.shape[1]), device=x.device, dtype=x.dtype)
    extrapolate = coef
    _, initial_gradient = quadratic(coef)
    scale = initial_gradient.abs().amax().clamp_min(1e-9)
    penalty = L1_FRACTION * initial_gradient.abs().amax(dim=0)

    def shrink(at, gradient, step):
        value = at - step * gradient
        return value.sign() * (value.abs() - step * penalty).clamp_min(0)

    step, momentum = 1., 1.
    for iteration in range(1, 1601):
        loss, gradient = quadratic(extrapolate)
        for _ in range(40):
            updated = shrink(extrapolate, gradient, step)
            difference = updated - extrapolate
            new_loss, _ = quadratic(updated)
            bound = loss + (gradient * difference).sum() + difference.square().sum() / (2 * step)
            if new_loss <= bound + 1e-7 * (1 + loss.abs()):
                break
            step /= 2
        else:
            raise RuntimeError("Sparse response line search failed")
        following = (1 + (1 + 4 * momentum ** 2) ** .5) / 2
        extrapolate = updated + ((momentum - 1) / following) * (updated - coef)
        coef, momentum = updated, following
        if iteration % 40 == 0:
            _, gradient = quadratic(coef)
            residual = float((shrink(coef, gradient, step) - coef).abs().amax() / (step * scale))
            if residual < .002:
                return project(coef), {"iterations": iteration, "optimality_residual": residual,
                                       "active_coefficients": int(coef.count_nonzero())}
    raise RuntimeError(f"Sparse response solver did not converge (stationarity {residual:.5g})")


def forecasts(fit, query, rna, edges, targets, auxiliary):
    _validate(fit, query)
    device = device_cuda()
    source = rna.loc[sorted(fit.cell_id.unique())]
    held = rna.loc[sorted(query.cell_id.unique())]
    representation, _ = fit_cell_kernels(source, edges, auxiliary, device=device)
    x = torch.cat(representation.reference[1:], dim=1)
    v = torch.cat(representation.vectors(held)[1:], dim=1)
    _, anchors, aligned, _ = panel_targets(fit)
    drugs, drug = drug_kernels(aligned, targets, device=device)
    weights, response = training_grid(aligned, source.index, drugs, device=device)
    weights = weights / weights.sum(dim=0)
    cell_weight = weights.mean(1, keepdim=True)
    marginal = (weights * response).mean(1, keepdim=True) / cell_weight.clamp_min(1e-9)
    additive, diagnostics = fit_sparse(x, cell_weight, marginal)
    residual = response - x @ additive
    laplacian = drug_laplacian(drug["structured"])
    indices = held.index.get_indexer(query.cell_id), drugs.get_indexer(query.compound_id)
    base = query.compound_id.map(anchors).to_numpy() + (v @ additive).cpu().numpy().ravel()[indices[0]]
    output = [query[KEY].assign(method=ADDITIVE, predicted_value=base)]
    details = {ADDITIVE: diagnostics}
    for arm, graph in ((INDEPENDENT, None), (PRIMARY, laplacian)):
        effects, details[arm] = fit_sparse(x, weights, residual, interaction=True, laplacian=graph)
        values = (v @ effects).cpu().numpy()[indices]
        output.append(query[KEY].assign(method=arm, predicted_value=base + values))
    return pd.concat(output, ignore_index=True), {
        "fit_rows": len(fit), "genes": len(rna.columns), "fit_cells": len(source), "drugs": len(drugs),
        "pathways_and_tf": representation.reference[1].shape[1], "auxiliary_features": representation.reference[2].shape[1],
        "graph_edges": int(torch.tril(laplacian, diagonal=-1).count_nonzero()), "arms": details,
    }


def predict(args):
    edges, targets, priors = molecular_inputs(args.priors, args.hallmark, args.targets)
    auxiliary, aux_stamp = load_auxiliary(args.auxiliary)
    external = args.mode in EXTERNAL_TRACKS
    if external:
        fit, query, rna, registered = external_inputs(args, args.mode)
        comparison, reference = external_reference(args, args.mode, query)
    else:
        original, clean, rna, _ = registered_inputs(args)
        _, fit, check = same_split(original, clean, args.fold, args.mode)
        query = validation_roster(check)
        comparison, reference = source_reference(args.cv_archive, args.mode, args.fold)
        registered = RECEIPTS["core"]
    new, diagnostics = forecasts(fit, query, rna, edges, targets, auxiliary)
    if external:
        rows = pd.concat([comparison, new.assign(track=args.mode, fold=0)], ignore_index=True)
        shared_support(rows, query, (*BASELINES, *PUBLIC, *ARMS))
    else:
        rows = pd.concat([comparison, with_assay(new, check, args.fold)], ignore_index=True)
        matrix = rows.pivot(index=[*KEY, "track", "fold"], columns="method", values="predicted_value")
        if set(matrix) != {*BASELINES, *ARMS} or not np.isfinite(matrix.to_numpy()).all():
            raise ValueError("Sparse and reference forecasts must have the same source support")
    freeze_table(rows[FORECAST_SCHEMA], args.out, provenance={
        "mode": args.mode, "fold": args.fold, "inputs_sha256": registered, "core_sha256": RECEIPTS["core"],
        "reference_sha256": reference, "priors": priors, "auxiliary": aux_stamp, "diagnostics": diagnostics,
    })


def assess_sparse(frame, *, external):
    report = assess(frame, external=external, primary=PRIMARY, controls=CONTROLS, require_precision=True)
    report["components"] = component_scores(frame, cell_axis="lineage_group")
    if external:
        intervals = {arm: interaction_intervals(frame, PRIMARY, arm) for arm in ("structured", INDEPENDENT)}
        report["interaction_ci95"] = intervals
        for arm, bounds in intervals.items():
            report["gates"][arm] = bool(report["gates"][arm] and bounds[0] > 0)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    fields = {
        "predict": ("licob", "archive", "cv-archive", "priors", "hallmark", "targets", "auxiliary"),
        "source": ("forecasts",), "external": ("licob", "forecasts", "limore-labels", "licob-labels"),
    }
    for action, names in fields.items():
        stage = stages.add_parser(action)
        for name in ("original", "core", *names, "out"):
            stage.add_argument(f"--{name}", required=True, type=Path)
        if action == "predict":
            stage.add_argument("--mode", required=True, choices=(*ASSAYS, *EXTERNAL_TRACKS))
            stage.add_argument("--fold", type=int, default=0, choices=range(3))
    args = parser.parse_args()
    if args.stage == "predict":
        predict(args)
    elif args.stage == "source":
        score_source(args, assessor=assess_sparse)
    else:
        score_external(args, assessor=assess_sparse, arms=ARMS)


if __name__ == "__main__":
    main()
