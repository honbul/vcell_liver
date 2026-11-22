"""Separate gene membership and ranking context in the HCC response kernel."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyarrow.parquet import read_schema
import torch

from .expanded_panel_cv import PENALTY, RECEIPTS, registered_inputs, same_split
from .external import KEY, _validate
from .io import freeze_table, sha256
from .licob_full_axis import validated as validate_licob
from .limore_corrected import shared_support
from .paired_expression import paired_data, row_center
from .pdo_transfer_cv import device_cuda
from .replicate_robust_loss import source_reference
from .response_component_benchmark import (
    BASELINES, EXTERNAL_TRACKS, FORECAST_SCHEMA, PUBLIC, external_inputs, external_reference,
)
from .rna_state_projection import score_external, score_source
from .structured_cv import ASSAYS, paired_intervals, with_assay
from .structured_kernel import KernelGrid, drug_kernels, fit_cell_kernels, ranked_expression, training_grid
from .v0plus_cv import molecular_inputs, panel_targets, validation_roster
from .v0plus_neural import load_auxiliary
from .v0plus_score import axis_metrics, crossed_patient_drug_ci
from .yang_hcc_external import require_hash

ARRAY_INPUT = "0a3c330a336d7fc0b70443c4066b37578ea21a0a0c6287838e4f2a42ce3c0250"
PDO_INPUT = "63049d5a2f0d08da415685a6e41f9afe8cb042e43fc7682256f2b7395b7ee110"
PDO_AXIS = "pancreas_rerank"
PDO_CONTEXT = "pancreas_native_rank"
PDO_RANDOM = "pancreas_width_control"
STABLE = "stable_panel"
RANDOM = "stable_width_control"
PRIMARY = "stable_blend"
BLEND_CONTROL = "stable_blend_control"
ARMS = (PDO_AXIS, PDO_CONTEXT, PDO_RANDOM, STABLE, RANDOM, PRIMARY, BLEND_CONTROL)
PANEL_SIZE = 9000


def schema_columns(path, digest):
    require_hash(path, digest)
    return pd.Index(read_schema(path).names)


def public_panels(args):
    require_hash(args.pdo / "receipt.json", PDO_INPUT)
    pancreas = json.loads((args.pdo / "receipt.json").read_text())
    pancreas_genes = schema_columns(args.pdo / "rna.parquet", pancreas["files"]["rna.parquet"])
    require_hash(args.original / "receipt.json", RECEIPTS["original"])
    limore = json.loads((args.original / "receipt.json").read_text())
    limore_rna = "test_inputs/rna.parquet"
    limore_genes = schema_columns(args.original / limore_rna, limore["artifacts"][limore_rna]["sha256"])
    validate_licob(args.licob)
    licob_genes = pd.Index(read_schema(args.licob / "rna.parquet").names)
    array, tissues, seal = paired_data(args.array)
    if seal != ARRAY_INPUT:
        raise ValueError("Unexpected paired reference expression")
    common = limore_genes.intersection(licob_genes).intersection(array.columns)
    return pancreas_genes, array, tissues, common


def gene_agreement(training, array, tissues, genes):
    paired = training.index.intersection(array.index)
    groups = tissues.tissue.loc[paired]
    count = groups.value_counts()
    groups = groups[groups.isin(count[count.ge(5)].index)]
    if len(groups) < 250:
        raise ValueError("Too few paired RNA fit models with tissue support")
    seq = row_center(ranked_expression(training.loc[groups.index, genes]), groups)
    affy = row_center(ranked_expression(array.loc[groups.index, genes]), groups)
    den = np.linalg.norm(seq, axis=0) * np.linalg.norm(affy, axis=0)
    rho = np.divide((seq * affy).sum(axis=0), den, out=np.zeros_like(den), where=den > 1e-8)
    return pd.Series(rho, index=genes), len(groups)


def matched_genes(ranks, axis, eligible, seeds):
    """Sample exactly the same number of genes from each fit-only mean/variance bin."""
    bins = np.minimum(ranks.loc[eligible].rank(method="first", pct=True).mul(5).astype(int), 4)
    result = [list() for _ in seeds]
    generators = [np.random.default_rng(seed) for seed in seeds]
    for _, group in bins.groupby(["mean", "variance"], sort=True):
        count = int(group.index.isin(axis).sum())
        if count:
            for chosen, rng in zip(result, generators):
                chosen.extend(rng.choice(group.index.to_numpy(), size=count, replace=False))
    pools = [pd.Index(sorted(values)) for values in result]
    if not all(len(pool) == len(axis) and pool.is_unique for pool in pools):
        raise ValueError("Matched random gene support changed width")
    return pools


def digest(genes):
    return hashlib.sha256(("\n".join(genes) + "\n").encode()).hexdigest()


def describe(axis, comparison, ranks):
    return {
        "genes": len(axis), "sha256": digest(axis),
        "overlap": len(axis.intersection(comparison)),
        "mean_rank": float(ranks.loc[axis, "mean"].mean()),
        "mean_rank_variance": float(ranks.loc[axis, "variance"].mean()),
    }


def candidate_forecasts(fit, query, rna, edges, targets, auxiliary, panels):
    _validate(fit, query)
    pancreas, array, tissues, common = panels
    x = rna.loc[sorted(fit.cell_id.unique())].sort_index(axis=1)
    q = rna.loc[sorted(query.cell_id.unique()), x.columns]
    pdo_axis = x.columns.intersection(pancreas).sort_values()
    stable_eligible = x.columns.intersection(common).sort_values()
    if len(stable_eligible) < PANEL_SIZE or len(pdo_axis) < 10000:
        raise ValueError("The registered RNA axes no longer provide expected support")
    reliability, pairs = gene_agreement(x, array, tissues, stable_eligible)
    stable = reliability.sort_values(ascending=False, kind="stable").index[:PANEL_SIZE].sort_values()
    rank = ranked_expression(x)
    features = pd.DataFrame({"mean": rank.mean(axis=0), "variance": rank.var(axis=0)}, index=x.columns)
    pdo_random = matched_genes(features, pdo_axis, x.columns, (40103, 40109, 40111))
    ctl, = matched_genes(features, stable, stable_eligible, (40639,))
    _, anchor, rows, _ = panel_targets(fit)
    device = device_cuda()
    drugs, kernels = drug_kernels(rows, targets, device=device)
    w, y = training_grid(rows, x.index, drugs, device=device)

    def estimate(genes, *, rerank):
        if rerank:
            fitted, gram = fit_cell_kernels(x[genes], edges, auxiliary, device=device)
        else:
            fitted, gram = fit_cell_kernels(x, edges, auxiliary, device=device, genes=genes)
        cross = fitted.cross(q)["full"]
        grid = KernelGrid(q.index, drugs, {"full": (gram["full"], cross)}, kernels, w, y, fitted.reference[-1].shape[1])
        return grid.forecast(query, "structured", PENALTY)[0]

    estimates = {
        PDO_AXIS: estimate(pdo_axis, rerank=True),
        PDO_CONTEXT: estimate(pdo_axis, rerank=False),
        PDO_RANDOM: np.mean([estimate(axis, rerank=True) for axis in pdo_random], axis=0),
        STABLE: estimate(stable, rerank=True),
        RANDOM: estimate(ctl, rerank=True),
    }
    base = query.compound_id.map(anchor).to_numpy()
    records = [query[KEY].assign(method=k, predicted_value=base+values) for k, values in estimates.items()]
    audit = {
        "paired_fit_cells": pairs, "full_genes": x.shape[1], "stable_candidate_genes": len(stable_eligible),
        "pancreas_axis": describe(pdo_axis, pdo_axis, features),
        "stable_panel": describe(stable, stable, features),
        "pancreas_random": [describe(a, pdo_axis, features) for a in pdo_random],
        "stable_random": describe(ctl, stable, features),
        "median_fit_agreement": float(reliability.median()),
        "stable_median_fit_agreement": float(reliability.loc[stable].median()),
        "stable_control_median_fit_agreement": float(reliability.loc[ctl].median()),
    }
    return pd.concat(records, ignore_index=True), audit


def predict(args):
    external = args.mode in EXTERNAL_TRACKS
    panels = public_panels(args)
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
    new, audit = candidate_forecasts(fit, query, rna, edges, targets, auxiliary, panels)
    new = new.assign(track=args.mode, fold=0) if external else with_assay(new, check, args.fold)
    panel = pd.concat([previous, new], ignore_index=True)
    cols = [*KEY, "track", "fold"]
    reference_values = previous[previous.method.eq("structured")].set_index(cols).predicted_value
    blends = []
    for name, arm in ((PRIMARY, STABLE), (BLEND_CONTROL, RANDOM)):
        values = panel.loc[panel.method.eq(arm)].set_index(cols).predicted_value
        blended = (.5 * (values + reference_values)).rename("predicted_value").reset_index()
        blends.append(blended.assign(method=name))
    joint = pd.concat([panel, *blends], ignore_index=True)
    methods = (*BASELINES, *ARMS, *PUBLIC) if external else (*BASELINES, *ARMS)
    wide = joint.pivot(index=cols, columns="method", values="predicted_value")
    if set(wide) != set(methods) or len(wide) != (len(query) if external else len(check)) or not np.isfinite(wide.to_numpy()).all():
        raise ValueError("The RNA-axis comparisons do not share the registered forecast rows")
    if external:
        shared_support(joint, query, methods)
    freeze_table(joint[FORECAST_SCHEMA], args.out, provenance={
        "mode": args.mode, "fold": args.fold, "inputs_sha256": registered, "core_sha256": RECEIPTS["core"],
        "reference_sha256": reference, "array_receipt_sha256": ARRAY_INPUT, "pdo_receipt_sha256": PDO_INPUT,
        "priors": priors, "auxiliary": aux, "diagnostics": audit,
    })


def assessor(frame, *, external):
    comparisons = (*BASELINES, BLEND_CONTROL, *PUBLIC) if external else (*BASELINES, BLEND_CONTROL)
    diagnostics = ((PDO_AXIS, PDO_CONTEXT), (PDO_AXIS, PDO_RANDOM), (PDO_AXIS, "structured"), (STABLE, RANDOM))
    pairs = [(PRIMARY, name) for name in comparisons]
    if external:
        pairs += list(diagnostics)
    overall, _, _ = axis_metrics(frame, pairs)
    interval = crossed_patient_drug_ci if external else paired_intervals
    across = {name: interval(frame, PRIMARY, name, repeats=1000 if external else 800) for name in comparisons}
    ranks = {(r["baseline"], r["metric"]): r["ci95"]
             for r in overall["paired_contrasts"] if r["candidate"] == PRIMARY}
    gates = {name: all(across[name][m][0] > 0 for m in ("spearman", "pearson"))
             and all(ranks[name, m][0] > -.005 for m in ("spearman", "ndcg_at_5", "top5_overlap"))
             for name in comparisons}
    result = {"metrics": overall, "across_cell_ci95": across, "gates": gates,
              "groups": frame.lineage_group.nunique(), "observations": frame.row_id.nunique()}
    if external:
        result["rna_axis_diagnostics"] = [
            {"candidate": a, "baseline": b, "ci95": interval(frame, a, b, repeats=1000)} for a,b in diagnostics
        ]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    for stage, fields in {
        "predict": ("licob", "pdo", "array", "archive", "cv-archive", "priors", "hallmark", "targets", "auxiliary"),
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
