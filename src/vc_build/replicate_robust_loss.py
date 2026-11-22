"""Optimize cross-assay response agreement on exact repeated training units."""

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .expanded_panel_cv import PENALTY, RECEIPTS, registered_inputs, same_split
from .external import KEY, SOURCE, _validate
from .io import freeze_table, verify_forecast
from .limore_corrected import shared_support
from .pdo_transfer_cv import device_cuda
from .response_component_benchmark import (
    BASELINES, EXTERNAL_TRACKS, FORECAST_SCHEMA, PUBLIC, external_inputs, external_reference,
)
from .response_components import component_scores
from .rna_state_projection import score_external, score_source
from .structured_cv import ASSAYS, paired_intervals, with_assay
from .structured_kernel import cell_kernels, drug_kernels, solve_residuals, training_grid
from .v0plus_cv import molecular_inputs, panel_targets, validation_roster
from .v0plus_neural import load_auxiliary
from .v0plus_score import axis_metrics, crossed_patient_drug_ci
from .yang_hcc_external import require_hash

REFERENCE_SCORE = "c2f6fa6b7beb226a762f32a8092e5399c414613e336029e1e564679518d1bc46"
PRIMARY = "replicate_robust"
BALANCED = "replicate_balanced"
SHUFFLED = "replicate_shuffled_assays"
ARMS = (PRIMARY, BALANCED, SHUFFLED)
FRACTION = .30
TEMPERATURE = .10
MIN_PAIRS = 35
SCALE_FLOOR = 1e-4


def training_repeats(rows, cells, drugs):
    """Select the widest supported set of assays on the same cell models."""
    table = (rows.groupby(["compound_id", "cell_id", *SOURCE], sort=True).residual.median()
             .unstack(SOURCE))
    rng = np.random.default_rng(37193)
    matched, details = [], {}
    for drug, matrix in table.groupby(level="compound_id", sort=True):
        matrix = matrix.droplevel("compound_id").dropna(axis=1, how="all")
        for size in range(len(matrix.columns), 1, -1):
            options = [matrix.loc[:, list(tags)].dropna() for tags in combinations(matrix.columns, size)]
            common = max(options, key=len)
            if len(common) < MIN_PAIRS:
                continue
            values = common.to_numpy()
            energy = np.square(values).mean(axis=0)
            if np.any(energy < SCALE_FLOOR) or np.any(values.std(axis=0) < .005):
                continue
            row = cells.get_indexer(common.index)
            if (row < 0).any():
                raise ValueError("Repeat support includes an unregistered fit cell")
            # Permute assay labels independently per cell, preserving that cell's measurements.
            shuffled = np.take_along_axis(values, np.argsort(rng.random(values.shape), axis=1), axis=1)
            matched.append((drugs.get_loc(drug), row, values, shuffled))
            details[drug] = {"cells": len(common), "assays": ["/".join(tag) for tag in common.columns],
                             "baseline_ms": energy.tolist()}
            break
    if not matched:
        raise ValueError("No registered exact-cell repeat set meets the protocol")
    return matched, details


class RepeatLoss:
    def __init__(self, repeats, weights, *, shuffled):
        device, dtype = weights.device, weights.dtype
        max_models = max(len(row) for _, row, _, _ in repeats)
        max_assays = max(y.shape[1] for _, _, y, _ in repeats)
        shape = (len(repeats), max_models, max_assays)
        y, mask = np.zeros(shape), np.zeros(shape, dtype=bool)
        indices, drugs = np.zeros(shape[:2], dtype=np.int64), []
        for pos, (drug, index, values, permuted) in enumerate(repeats):
            values = permuted if shuffled else values
            n, k = values.shape
            y[pos, :n, :k] = values
            mask[pos, :n, :k] = True
            indices[pos, :n] = index
            drugs.append(drug)
        self.rows = torch.as_tensor(indices, device=device)
        self.drugs = torch.as_tensor(drugs, device=device)
        self.y = torch.as_tensor(y, device=device, dtype=dtype)
        self.mask = torch.as_tensor(mask, device=device)
        count = self.mask.sum(dim=1)
        self.assays = count > 0
        self.count = count.clamp_min(1)
        power = self.y.square().sum(dim=1) / self.count
        self.scale = power.clamp_min(SCALE_FLOOR)
        self.group_count = self.assays.sum(dim=1)
        self.mass = (power * self.assays).sum(dim=1) / self.group_count * weights.sum(dim=0)[self.drugs]

    def normalized_errors(self, values):
        fitted = values[self.rows, self.drugs[:, None]][:, :, None]
        delta = (fitted - self.y) * self.mask
        return delta.square().sum(dim=1) / self.count / self.scale

    def __call__(self, values, *, robust):
        errors = self.normalized_errors(values)
        if robust:
            logits = (errors / TEMPERATURE).masked_fill(~self.assays, -torch.inf)
            loss = TEMPERATURE * (torch.logsumexp(logits, dim=1) - self.group_count.log())
        else:
            loss = (errors * self.assays).sum(dim=1) / self.group_count
        return self.mass @ loss


def factors(train, cross):
    values, vectors = torch.linalg.eigh((train + train.T) * .5)
    keep = values > values[-1] * 1e-7
    fitted = vectors[:, keep] * values[keep].sqrt()
    if cross is None:
        return fitted
    return fitted, cross @ vectors[:, keep] / values[keep].sqrt()


def optimize(cell, drug, weights, response, repeats, *, held, init, robust, shuffled=False):
    """Solve the convex smoothed worst-assay objective in whitened kernel coordinates."""
    law = RepeatLoss(repeats, weights, shuffled=shuffled)
    ordinary = weights.clone()
    ordinary[:, law.drugs] *= 1 - FRACTION
    normalizer = weights.sum()
    coefficient = torch.nn.Parameter(init.clone())
    solver = torch.optim.LBFGS([coefficient], max_iter=180, history_size=15,
                              tolerance_grad=5e-8, tolerance_change=1e-12, line_search_fn="strong_wolfe")

    def objective():
        estimate = cell @ coefficient @ drug.T
        return ((ordinary * (estimate - response).square()).sum()
                + PENALTY * coefficient.square().sum()
                + FRACTION * law(estimate, robust=robust)) / normalizer

    def closure():
        solver.zero_grad()
        loss = objective()
        loss.backward()
        return loss

    start = float(objective().detach())
    solver.step(closure)
    final = float(closure().detach())
    stop_gradient = float(coefficient.grad.abs().max())
    if not np.isfinite(final) or final > start + 1e-8 or stop_gradient > 5e-5:
        raise RuntimeError(f"Unconverged repeat loss: {start}, {final}, gradient {stop_gradient}")
    with torch.no_grad():
        errors = law.normalized_errors(cell @ coefficient @ drug.T)
        q = (errors / TEMPERATURE).masked_fill(~law.assays, -torch.inf).softmax(dim=1)
        audit = {"loss_start": start, "loss_final": final, "gradient_max": stop_gradient,
                 "iterations": int(solver.state[coefficient]["n_iter"]),
                 "mean_max_assay_weight": float(q.max(dim=1).values.mean()) if robust else None}
        return held @ coefficient @ drug.T, audit


def forecast(fit, query, rna, edges, targets, auxiliary):
    _validate(fit, query)
    device = device_cuda()
    train = rna.loc[sorted(fit.cell_id.unique())]
    held = rna.loc[sorted(query.cell_id.unique())]
    _, anchors, rows, _ = panel_targets(fit)
    drugs, kernels = drug_kernels(rows, targets, device=device)
    drug_kernel = kernels["structured"]
    grams, _ = cell_kernels(train, held, edges, auxiliary, device=device)
    cell_kernel, cross = grams["full"]
    weights, response = training_grid(rows, train.index, drugs, device=device)
    repeated, support = training_repeats(rows, train.index, drugs)
    cell, query_basis = factors(cell_kernel, cross)
    drug = factors(drug_kernel, None)
    coef, _, _ = solve_residuals(cell_kernel, drug_kernel, weights, response, PENALTY)
    init = cell.T @ coef @ drug
    replay = (query_basis @ init @ drug.T - cross @ coef @ drug_kernel).abs().max().item()
    if replay > 4e-5:
        raise RuntimeError(f"The whitened feature map did not reproduce the original kernel: {replay}")
    selector = (held.index.get_indexer(query.cell_id), drugs.get_indexer(query.compound_id))
    if any((values < 0).any() for values in selector):
        raise ValueError("Missing query identity")
    anchor = query.compound_id.map(anchors).to_numpy()
    parts, opt = [], {}
    for name in ARMS:
        raw, opt[name] = optimize(cell, drug, weights, response, repeated, held=query_basis, init=init,
                                  robust=name != BALANCED, shuffled=name == SHUFFLED)
        value = anchor + raw.cpu().numpy()[selector]
        parts.append(query[KEY].assign(method=name, predicted_value=value))
    return pd.concat(parts, ignore_index=True), {"repeats": support, "fit_cells": len(train), "optimize": opt,
                                               "kernel_reproduction_error": replay}


def source_reference(path, mode, fold):
    path_score = path / "source_score.json"
    require_hash(path_score, REFERENCE_SCORE)
    hashes = json.loads(path_score.read_text())["forecast_sha256"]
    task = f"{mode}-{fold}"
    rows, seal = verify_forecast(path / "forecasts" / f"{task}.parquet")
    if seal["sha256"] != hashes[task]:
        raise ValueError("The registered comparator prediction has changed")
    return rows[rows.method.isin(BASELINES)][FORECAST_SCHEMA], seal["sha256"]


def predict(args):
    external = args.mode in EXTERNAL_TRACKS
    if external:
        fit, query, rna, registered = external_inputs(args, args.mode)
        previous, reference = external_reference(args, args.mode, query)
    else:
        old, clean, rna, _ = registered_inputs(args)
        _, fit, check = same_split(old, clean, args.fold, args.mode)
        query = validation_roster(check)
        previous, reference = source_reference(args.cv_archive, args.mode, args.fold)
        registered = RECEIPTS["core"]
    edges, targets, priors = molecular_inputs(args.priors, args.hallmark, args.targets)
    auxiliary, aux = load_auxiliary(args.auxiliary)
    new, detail = forecast(fit, query, rna, edges, targets, auxiliary)
    new = new.assign(track=args.mode, fold=0) if external else with_assay(new, check, args.fold)
    joined = pd.concat([previous, new], ignore_index=True)
    compared = (*BASELINES, *ARMS)
    if external:
        compared += PUBLIC
        shared_support(joined, query, compared)
    wide = joined.pivot(index=[*KEY, "track", "fold"], columns="method", values="predicted_value")
    expected = len(query) if external else len(check)
    if len(wide) != expected or set(wide) != set(compared) or not np.isfinite(wide.to_numpy()).all():
        raise ValueError("Missing registered support for a repeat-aware comparison")
    freeze_table(joined[FORECAST_SCHEMA], args.out, provenance={
        "mode": args.mode, "fold": args.fold, "inputs_sha256": registered, "core_sha256": RECEIPTS["core"],
        "reference_sha256": reference, "priors": priors, "auxiliary": aux, "diagnostics": detail,
    })


def assessor(frame, *, external):
    compares = (*BASELINES, BALANCED, SHUFFLED)
    if external:
        compares += PUBLIC
    metrics, _, _ = axis_metrics(frame, [(PRIMARY, name) for name in compares])
    interval = crossed_patient_drug_ci if external else paired_intervals
    across = {name: interval(frame, PRIMARY, name, repeats=1000 if external else 800) for name in compares}
    ranks = {(row["baseline"], row["metric"]): row["ci95"] for row in metrics["paired_contrasts"]}
    gates = {name: all(across[name][m][0] > 0 for m in ("spearman", "pearson"))
             and all(ranks[name, m][0] > -.005 for m in ("spearman", "ndcg_at_5", "top5_overlap"))
             for name in compares}
    result = {"metrics": metrics, "across_cell_ci95": across, "gates": gates,
              "groups": frame.lineage_group.nunique(), "observations": frame.row_id.nunique()}
    if external:
        result["components"] = component_scores(frame, cell_axis="lineage_group")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    for stage, fields in {
        "predict": ("licob", "archive", "cv-archive", "priors", "hallmark", "targets", "auxiliary"),
        "source": ("forecasts",), "external": ("licob", "forecasts", "limore-labels", "licob-labels"),
    }.items():
        sub = subparsers.add_parser(stage)
        for field in ("original", "core", *fields, "out"):
            sub.add_argument("--" + field, type=Path, required=True)
        if stage == "predict":
            sub.add_argument("--mode", choices=(*ASSAYS, *EXTERNAL_TRACKS), required=True)
            sub.add_argument("--fold", choices=range(3), type=int, default=0)
    args = parser.parse_args()
    if args.stage == "predict":
        predict(args)
    elif args.stage == "source":
        score_source(args, assessor=assessor)
    else:
        score_external(args, assessor=assessor, arms=ARMS)


if __name__ == "__main__":
    main()
