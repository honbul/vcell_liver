"""Fit a response model on source diffusion kernels and test inductive extrapolation."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import normalize

from .expanded_panel_cv import RECEIPTS, registered_inputs, same_split
from .external import KEY, _validate
from .io import freeze_table
from .limore_corrected import shared_support
from .pdo_transfer_cv import device_cuda
from .response_component_benchmark import (
    BASELINES, EXTERNAL_TRACKS, FORECAST_SCHEMA, PUBLIC, external_inputs, external_reference, source_reference,
)
from .response_components import component_scores, interaction_intervals
from .rna_neighbor_support import similarity
from .rna_state_projection import assess, score_external, score_source
from .structured_cv import ASSAYS, with_assay
from .structured_kernel import drug_kernels, fit_cell_kernels, solve_residuals, training_grid
from .v0plus_cv import molecular_inputs, panel_targets, validation_roster
from .v0plus_neural import load_auxiliary

ONE_STEP = "rna_diffusion_one"
PRIMARY = "rna_diffusion_two"
TRANSDUCTIVE = "rna_diffusion_query"
ARMS = (ONE_STEP, PRIMARY, TRANSDUCTIVE)
CONTROLS = ("structured", ONE_STEP)
NEIGHBORS = 20
TEMPERATURE = .05


def topology(similarities, reference_groups, query_groups):
    """Choose one representative per eligible lineage and normalize each row."""
    reference_ids = torch.as_tensor(reference_groups, device=similarities.device)
    query_ids = torch.as_tensor(query_groups, device=similarities.device)
    ordered = similarities.masked_fill(query_ids[:, None].eq(reference_ids), -torch.inf)
    order = ordered.argsort(dim=1, descending=True, stable=True)
    size = int(reference_ids.max()) + 1
    unique_at = order.new_full((len(order), size), order.shape[1])
    positions = torch.arange(order.shape[1], device=order.device).expand_as(order)
    unique_at.scatter_reduce_(1, reference_ids[order], positions, reduce="amin", include_self=True)
    first = unique_at.gather(1, reference_ids[order]).eq(positions)
    ordered_scores = ordered.gather(1, order).masked_fill(~first, -torch.inf)
    keep = ordered_scores.argsort(dim=1, descending=True, stable=True)[:, :NEIGHBORS]
    targets = order.gather(1, keep)
    scores = ordered_scores.gather(1, keep)
    available = scores.isfinite().any(dim=1)
    # A single query lineage can have no other query neighbor.
    scores = torch.where(available[:, None], scores, 0.)
    weights = torch.softmax((scores - scores.max(dim=1, keepdim=True).values) / TEMPERATURE, dim=1)
    weights = weights * available[:, None]
    return torch.zeros_like(similarities).scatter_(1, targets, weights)


def group_codes(fit, query, source, held):
    # LICOB's original roster calls each PDO a lineage, but query graph edges
    # must exclude multiple PDOs from the same patients.
    fit_groups = fit[["cell_id", "lineage_group"]]
    column = "donor_id" if "donor_id" in query else "lineage_group"
    held_groups = query[["cell_id", column]].rename(columns={column: "lineage_group"})
    if held_groups.lineage_group.isna().any():
        raise ValueError("Missing independent query graph group")
    mapping = pd.concat([fit_groups, held_groups], ignore_index=True).drop_duplicates()
    mapping = mapping.set_index("cell_id", verify_integrity=True).lineage_group
    codes = pd.Index(mapping.unique())
    return (codes.get_indexer(mapping.loc[ids]) for ids in (source, held))


def diffuse(train, cross, p, b, order):
    a = .5 * (torch.eye(len(p), dtype=p.dtype, device=p.device) + p)
    operator = a if order == 1 else a @ a
    linked = train @ operator.T
    fit = operator @ linked
    direct, source = (.5, .5 * b) if order == 1 else (.25, .5 * b + .25 * b @ p)
    query = direct * (cross @ operator.T) + source @ linked
    scale = train.diagonal().mean() / fit.diagonal().mean()
    return fit * scale, query * scale, float(scale)


def graph_forecasts(fit, query, rna, edges, targets, auxiliary, *, replay):
    _validate(fit, query)
    device = device_cuda()
    source = rna.loc[sorted(fit.cell_id.unique())]
    held = rna.loc[sorted(query.cell_id.unique())]
    fit_groups, query_groups = group_codes(fit, query, source.index, held.index)
    basis, cell = fit_cell_kernels(source, edges, auxiliary, device=device)
    raw_train, raw_cross = cell["full"], basis.cross(held)["full"]
    reference = tuple(normalize(values) for values in basis.reference)
    assessment = tuple(normalize(values) for values in basis.vectors(held))
    p = topology(similarity(reference, reference), fit_groups, fit_groups)
    b = topology(similarity(assessment, reference), fit_groups, query_groups)
    c = topology(similarity(assessment, assessment), query_groups, query_groups)
    _, anchors, labels, _ = panel_targets(fit)
    drugs, kernels = drug_kernels(labels, targets, device=device)
    chemistry = kernels["structured"]
    w, y = training_grid(labels, source.index, drugs, device=device)
    indices = held.index.get_indexer(query.cell_id), drugs.get_indexer(query.compound_id)
    if any((axis < 0).any() for axis in indices):
        raise ValueError("Registered query lost its expression or drug")
    base = query.compound_id.map(anchors).to_numpy()
    frames, details = [], {}
    for name, order in ((ONE_STEP, 1), (PRIMARY, 2)):
        train, cross, scale = diffuse(raw_train, raw_cross, p, b, order)
        train = .5 * (train + raw_train)
        weights, steps, error = solve_residuals(train, chemistry, w, y, 4.)
        variants = {name: cross}
        if order == 2:
            degree = c.sum(dim=1, keepdim=True)
            variants[TRANSDUCTIVE] = (1 - .25 * degree) * cross + .25 * c @ cross
        for variant, projected in variants.items():
            residuals = (.5 * (raw_cross + projected) @ weights @ chemistry).cpu().numpy()
            frames.append(query[KEY].assign(method=variant, predicted_value=base + residuals[indices]))
        details[name] = {"smooth_scale": scale, "iterations": steps, "residual": error}
    if replay:
        one = held.iloc[:1]
        one_features = tuple(normalize(values) for values in basis.vectors(one))
        one_b = topology(similarity(one_features, reference), fit_groups, query_groups[:1])
        independently = basis.cross(one)["full"]
        _, single, _ = diffuse(raw_train, independently, p, one_b, 2)
        discrepancy = float((single - cross[:1]).abs().max())
        if discrepancy > 2e-5:
            raise ValueError(f"Inductive kernel changes with query batch: {discrepancy}")
        details["single_query_max_error"] = discrepancy
    details["fit_edges"] = int(p.gt(0).sum())
    details["query_to_fit_edges"] = int(b.gt(0).sum())
    details["query_to_query_edges"] = int(c.gt(0).sum())
    return pd.concat(frames, ignore_index=True), details


def predict(args):
    external = args.mode in EXTERNAL_TRACKS
    if external:
        fit, query, rna, registered = external_inputs(args, args.mode)
        previous, reference = external_reference(args, args.mode, query)
    else:
        original, clean, rna, _ = registered_inputs(args)
        _, fit, check = same_split(original, clean, args.fold, args.mode)
        query = validation_roster(check)
        previous, reference = source_reference(args.cv_archive, args.mode, args.fold)
        registered = RECEIPTS["core"]
    edges, targets, priors = molecular_inputs(args.priors, args.hallmark, args.targets)
    auxiliary, auxiliary_receipt = load_auxiliary(args.auxiliary)
    new, diagnostics = graph_forecasts(fit, query, rna, edges, targets, auxiliary, replay=external)
    new = new.assign(track=args.mode, fold=0) if external else with_assay(new, check, args.fold)
    joint = pd.concat([previous, new], ignore_index=True)[FORECAST_SCHEMA]
    if external:
        shared_support(joint, query, (*BASELINES, *PUBLIC, *ARMS))
    else:
        support = joint.pivot(index=[*KEY, "track", "fold"], columns="method", values="predicted_value")
        if set(support) != {*BASELINES, *ARMS} or not np.isfinite(support).all(axis=None):
            raise ValueError("Source graph forecasts lack their matched reference")
    freeze_table(joint, args.out, provenance={
        "mode": args.mode, "fold": args.fold, "inputs_sha256": registered, "core_sha256": RECEIPTS["core"],
        "reference_sha256": reference, "priors": priors, "auxiliary": auxiliary_receipt, "graph": diagnostics,
    })


def assess_diffusion(frame, *, external):
    result = assess(frame, external=external, primary=PRIMARY, controls=CONTROLS, require_precision=True)
    if external:
        result["interaction_ci95"] = interaction_intervals(frame, PRIMARY, "structured")
        result["gates"]["structured"] &= result["interaction_ci95"][0] > 0
        result["components"] = component_scores(frame, cell_axis="lineage_group")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    fields = {"predict": ("licob", "archive", "cv-archive", "priors", "hallmark", "targets", "auxiliary"),
              "source": ("forecasts",), "external": ("licob", "forecasts", "limore-labels", "licob-labels")}
    for action, names in fields.items():
        stage = stages.add_parser(action)
        for name in ("original", "core", *names, "out"):
            stage.add_argument(f"--{name}", type=Path, required=True)
        if action == "predict":
            stage.add_argument("--mode", choices=(*ASSAYS, *EXTERNAL_TRACKS), required=True)
            stage.add_argument("--fold", choices=range(3), type=int, default=0)
    args = parser.parse_args()
    if args.stage == "predict":
        predict(args)
    elif args.stage == "source":
        score_source(args, assessor=assess_diffusion)
    else:
        score_external(args, assessor=assess_diffusion, arms=ARMS)


if __name__ == "__main__":
    main()
