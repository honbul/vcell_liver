"""Evaluate drug-specific target and pathway interactions on the registered LIMORE panel."""

import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

from .evaluate import join_observations
from .external import KEY
from .io import freeze_table, publish_json, verify_forecast
from .limore_corrected import OBSERVATIONS, shared_support
from .measured_driver_transfer import MEASURED_WEIGHT, PENALTY, TARGET_WEIGHT, data, normalized_block, target_feature
from .measured_drivers import read as read_genomics
from .structured_kernel import Standardization, dot_kernel, project_drug_features
from .structured_kernel import ranked_expression, solve_residuals
from .v0plus_cv import molecular_inputs, panel_targets
from .v0plus_neural import load_auxiliary
from .v0plus_score import axis_metrics, crossed_patient_drug_ci
from .yang_hcc_external import require_hash

ARCHIVE = "12dea79f245ecebdaf48b2044665f0584788bb1d059411f4d6a9541fb4a4d1a4"
KERNEL_SOURCE = "ba5f49b462cb88c09dff1154cde0b5ed7426b3170934751207a01e375543e509"
BASE = "measured_genome"
PRIMARY = "target_pathway"
CONTROLS = (BASE, "mapped_pathway", "shuffled_pathway_link")
COMPARISONS = [(PRIMARY, name) for name in (*CONTROLS, "drug_mean", "drug_ridge", "cellhit")]


def pathway_features(fit, held, edges, drugs, targets, device):
    hallmark = edges[edges.signature.str.startswith("HALLMARK_")]
    measured_genes = set(fit.columns)
    target_genes = set().union(*(targets.get(name, ()) for name in drugs)) & measured_genes
    memberships = {name: set(part.gene) & measured_genes for name, part in hallmark.groupby("signature")}
    activity_genes = {name: sorted(genes - target_genes) for name, genes in memberships.items()
                     if len(genes - target_genes) >= 10}
    names = list(activity_genes)
    ranked = [ranked_expression(frame) for frame in (fit, held)]
    activities = []
    for values in ranked:
        columns = [values[:, fit.columns.get_indexer(activity_genes[name])].mean(axis=1) for name in names]
        activities.append(np.column_stack(columns))
    scaler = Standardization.fit(activities[0], 0.05)
    scores = [(values - scaler.center) / scaler.spread for values in activities]
    links = {
        j: tuple(k for k, name in enumerate(names) if set(targets.get(drug, ())) & memberships[name] & measured_genes)
        for j, drug in enumerate(drugs)
    }
    supported = sorted(j for j, indices in links.items() if indices)
    if not supported:
        raise ValueError("No registered drug has a measurable Hallmark link")
    rng = np.random.default_rng(54189)
    order = sorted(rng.permutation(supported).tolist(), key=lambda j: len(links[j]))
    permuted = dict(zip(order, np.roll(order, 1)))
    changed = sum(links[j] != links[other] for j, other in permuted.items())
    output = []
    for values in scores:
        correct, shuffled = (np.zeros((len(values), len(drugs)), dtype=np.float32) for _ in range(2))
        for j in supported:
            correct[:, j] = values[:, links[j]].mean(axis=1)
            shuffled[:, j] = values[:, links[permuted[j]]].mean(axis=1)
        output.append(tuple(torch.as_tensor(array, device=device) for array in (correct, shuffled)))
    return tuple(output), {"mapped_drugs": [drugs[j] for j in supported], "changed_links": changed,
                           "eligible_pathways": len(names), "excluded_target_genes": len(target_genes)}


def scaled_feature(fit, held):
    center = fit.mean(dim=0)
    scale = fit.std(dim=0, correction=0).clamp_min(0.05)
    return tuple((values - center) * (np.sqrt(TARGET_WEIGHT) / scale) for values in (fit, held))


def registered_kernel(path):
    require_hash(path, KERNEL_SOURCE)
    spec = importlib.util.spec_from_file_location("vc_build.registered_driver_kernel", path)
    kernel = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = kernel
    spec.loader.exec_module(kernel)
    return kernel


def predict(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Target-pathway transfer requires CUDA")
    device = torch.device("cuda")
    train, query, fit, held = data(args, "LIMORE")
    archived, seal = verify_forecast(args.archive)
    if seal["sha256"] != ARCHIVE or seal["provenance"]["label_sha256"] != OBSERVATIONS:
        raise ValueError("Unexpected measured-genome archive")
    edges, targets, priors = molecular_inputs(args.priors, args.hallmark, args.targets)
    auxiliary, aux_receipt = load_auxiliary(args.auxiliary)
    measurements, genomics_hash = read_genomics(args.genomics)
    kernel = registered_kernel(args.kernel_source)
    _, anchors, rows, _ = panel_targets(train)
    grid = kernel.KernelGrid.prepare(rows, fit, held, edges, targets, auxiliary, device=device)
    cell, cross = grid.cell["full"]
    drug = grid.drug["structured"]
    fit_dna, held_dna = [], []
    for name in ("coding", "copy"):
        source = measurements[f"source_{name}"].reindex(fit.index)
        query_dna = measurements[f"limore_{name}"].reindex(held.index)
        if not query_dna.notna().all(axis=None):
            raise ValueError("Registered LIMORE DNA is incomplete")
        a, b, _ = normalized_block(source, query_dna)
        a, b = (torch.as_tensor(x, device=device) for x in (a, b))
        fit_dna.append(MEASURED_WEIGHT * dot_kernel(a, a))
        held_dna.append(MEASURED_WEIGHT * dot_kernel(b, a))
    cell = cell + sum(fit_dna)
    cross = cross + sum(held_dna)

    target_fit, target_held, _ = target_feature(fit, held, grid.drugs, targets, device)
    ((path_fit, wrong_fit), (path_held, wrong_held)), support = pathway_features(
        fit, held, edges, grid.drugs, targets, device,
    )
    path = scaled_feature(path_fit, path_held)
    correct = scaled_feature(target_fit * path_fit / np.sqrt(TARGET_WEIGHT),
                             target_held * path_held / np.sqrt(TARGET_WEIGHT))
    wrong = scaled_feature(target_fit * wrong_fit / np.sqrt(TARGET_WEIGHT),
                           target_held * wrong_held / np.sqrt(TARGET_WEIGHT))
    target = (target_fit, target_held)
    arms = {BASE: (target,), "mapped_pathway": (target, path),
            PRIMARY: (target, path, correct), "shuffled_pathway_link": (target, path, wrong)}
    index = grid.cells.get_indexer(query.cell_id), grid.drugs.get_indexer(query.compound_id)
    anchor = query.compound_id.map(anchors).to_numpy()
    forecasts, diagnostics = [archived], {}
    old = archived[archived.method.eq(BASE)].set_index("row_id").predicted_value.reindex(query.row_id)
    for arm, features in arms.items():
        source, held_features = (torch.stack(parts, dim=-1) for parts in zip(*features))
        if arm == BASE:
            coefficients, steps, error = kernel.solve_residuals(
                cell, drug, grid.weights, grid.response, PENALTY, per_drug_prior=target_fit,
            )
            extra = target_held * (target_fit * coefficients).sum(dim=0)
        else:
            coefficients, steps, error = solve_residuals(
                cell, drug, grid.weights, grid.response, PENALTY, per_drug_prior=source,
            )
            extra = project_drug_features(source, held_features, coefficients)
        residual = cross @ coefficients @ drug + extra
        values = anchor + residual.cpu().numpy()[index]
        diagnostics[arm] = {"iterations": steps, "solver_residual": error}
        if arm == BASE:
            delta = float(np.max(np.abs(values - old.to_numpy())))
            if not np.isfinite(delta) or delta > 1e-5:
                raise ValueError(f"Measured-genome replay differs by {delta:.7g} ({steps} CG steps, residual {error:.7g})")
            diagnostics[arm]["max_absolute_replay_delta"] = delta
        else:
            forecasts.append(query[KEY].assign(track="LIMORE", fold=0, method=arm, predicted_value=values))
    combined = pd.concat(forecasts, ignore_index=True)
    shared_support(combined, query, {*archived.method, *arms})
    freeze_table(combined, args.out, provenance={
        "archive_sha256": ARCHIVE, "observation_sha256": OBSERVATIONS, "genomics_sha256": genomics_hash,
        "registered_kernel_sha256": KERNEL_SOURCE,
        "public_priors": priors, "auxiliary": aux_receipt, "support": support, "diagnostics": diagnostics,
    })


def score(args):
    forecast, seal = verify_forecast(args.forecast)
    provenance = seal["provenance"]
    if (provenance["archive_sha256"], provenance["observation_sha256"]) != (ARCHIVE, OBSERVATIONS):
        raise ValueError("Unexpected LIMORE forecast lineage")
    require_hash(args.labels, OBSERVATIONS)
    joined = join_observations(forecast, pd.read_parquet(args.labels))
    overall, _, _ = axis_metrics(joined, COMPARISONS)
    intervals = {b: crossed_patient_drug_ci(joined, a, b) for a, b in COMPARISONS}
    within_cell = {(row["baseline"], row["metric"]): row["ci95"] for row in overall["paired_contrasts"]
                   if row["candidate"] == PRIMARY}
    supported = provenance["support"]["mapped_drugs"]
    subset = joined[joined.compound_id.isin(supported)]
    gain = {name: all(intervals[name][metric][0] > 0 for metric in ("spearman", "pearson"))
            for _, name in COMPARISONS}
    joint_gain = {name: gain[name] and all(within_cell[name, metric][0] > 0 for metric in ("spearman", "ndcg_at_5"))
                  for name in ("drug_mean", "drug_ridge", "cellhit")}
    publish_json(args.out, {
        "scope": "previously evaluated source-independent LIMORE 50 technical comparison",
        "forecast_sha256": seal["sha256"], "observations_sha256": OBSERVATIONS,
        "lineages": joined.lineage_group.nunique(), "rows": joined.row_id.nunique(),
        "overall": overall, "crossed_ci95": intervals,
        "connected_support": len(supported), "connected_rows": subset.row_id.nunique(),
        "connected_metrics": axis_metrics(subset, ())[0] if len(supported) >= 20 else None,
        "pathway_interaction_pass": all(gain[name] for name in CONTROLS),
        "gain_on_within_drug_metrics": gain, "joint_gain_by_reference": joint_gain,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    arguments = {"predict": ("core", "limore", "archive", "priors", "hallmark", "targets", "auxiliary", "genomics",
                             "kernel-source"),
                 "score": ("forecast", "labels")}
    for stage, names in arguments.items():
        sub = stages.add_parser(stage)
        for name in (*names, "out"):
            sub.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    {"predict": predict, "score": score}[args.stage](args)


if __name__ == "__main__":
    main()
