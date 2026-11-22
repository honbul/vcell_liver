"""Use screened, reproducible RNA gene orders as drug-specific kernel features."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .expanded_panel_cv import PENALTY, RECEIPTS, registered_inputs, same_split
from .external import KEY, SOURCE, _validate
from .io import freeze_table
from .limore_corrected import shared_support
from .paired_expression import paired_data
from .paired_ordinal_kernel import (
    ARMS as ORDINAL_ARMS, COMMON, PRIMARY as PORTABLE, REF_CONTEXTS,
    lineages, pair_candidates, paired_lineages, select_pairs, sign_orders,
)
from .pdo_transfer_cv import device_cuda
from .rank_context_calibration import DRUG_METRICS, SAMPLE_METRICS
from .response_component_benchmark import (
    BASELINES, EXTERNAL_TRACKS, FORECAST_SCHEMA, PUBLIC, check_component, external_inputs,
)
from .response_components import orthogonal_parts
from .rna_gene_panel import ARRAY_INPUT, digest
from .rna_state_projection import score_external, score_source
from .structured_cv import ASSAYS, paired_intervals, with_assay
from .structured_kernel import (
    Standardization, drug_kernels, fit_cell_kernels, project_drug_features, solve_residuals, training_grid,
)
from .v0plus_cv import molecular_inputs, panel_targets, validation_roster
from .v0plus_neural import load_auxiliary
from .v0plus_score import axis_metrics, crossed_patient_drug_ci
from .yang_hcc_external import require_hash

ARCHIVE_SHA = {
    "source": "3e438151d6a7ae6c96c53bb8e88e089ee87aa6fa80c20b7db008c61571c97604",
    "external": "3e8451de6c613cd8685ba9d5f6b6662be03accf5f5898719e95183fc3a669f06",
}
PRIMARY = "drug_stable_order"
POOL = "drug_pooled_order"
SHUFFLE = "drug_order_shuffle"
ARMS = (PRIMARY, POOL, SHUFFLE)
REFERENCES = (*REF_CONTEXTS, *ORDINAL_ARMS)
WIDTH = 32
MIN_LINEAGES = 60
MIN_TISSUE = 5
MIN_ASSOCIATION = .015
WEIGHT = .35


def archive(args, seal):
    external = args.mode in EXTERNAL_TRACKS
    key = args.mode if external else f"{args.mode}-{args.fold}"
    stage = "external" if external else "source"
    score = args.archive / f"{stage}_score.json"
    require_hash(score, ARCHIVE_SHA[stage])
    hashes = json.loads(score.read_text())["forecast_sha256"]
    path = args.archive / "forecasts" / f"{key}.parquet"
    frame, sha = check_component(path, args.mode, args.fold, seal)
    if sha != hashes[key]:
        raise ValueError("The previous ordinal forecast differs from its frozen report")
    receipt = json.loads(path.with_suffix(".parquet.freeze.json").read_text())
    audit = receipt["provenance"]["ordinal_audit"]
    return frame, sha, audit


def conditional_correlation(x, y, mask, tissues):
    """Pearson of order and response after centering both within each supported tissue."""
    cov, xvar, yvar = (x.new_zeros((x.shape[1], y.shape[1])) for _ in range(3))
    count = mask.sum(dim=0).new_zeros(y.shape[1])
    for keys in tissues.groupby(tissues, sort=True).indices.values():
        m = mask[keys]
        n = m.sum(dim=0)
        valid = n >= MIN_TISSUE
        safe = n.clamp_min(1)
        values, effect = x[keys], y[keys] * m
        xsum = values.T @ m
        ysum = effect.sum(dim=0)
        cov += (values.T @ effect - xsum * (ysum / safe)) * valid
        xvar += (values.square().T @ m - xsum.square() / safe) * valid
        yvar += (effect.square().sum(dim=0) - ysum.square() / safe) * valid
        count += n * valid
    denominator = (xvar.clamp_min(0) * yvar.clamp_min(0)).sqrt()
    good = (count >= MIN_LINEAGES) & (yvar.max(dim=0).values > 1e-10)
    rho = torch.where(denominator > 1e-10, cov / denominator.clamp_min(1e-10), 0.)
    return rho.clamp(-1., 1.).cpu().numpy(), good.cpu().numpy(), count.to(torch.int64).cpu().numpy()


def assay_associations(aligned, fit_orders, drugs, device):
    labels = aligned[["lineage_group", "tissue_origin"]].drop_duplicates()
    ambiguous = labels.lineage_group.duplicated(keep=False)
    projections = labels.loc[~ambiguous].set_index("lineage_group").tissue_origin.fillna("Unknown")
    atlas, metadata = [], {"ambiguous_tissue_lineages": int(labels.loc[ambiguous, "lineage_group"].nunique())}
    for assay, group in aligned.groupby(SOURCE, sort=True):
        group = group.loc[group.lineage_group.isin(projections.index)]
        values = group.groupby(["lineage_group", "compound_id"]).residual.mean().unstack().reindex(columns=drugs)
        observed = values.notna().to_numpy()
        y = orthogonal_parts(values.fillna(0.).to_numpy(), observed.astype(float))[2]
        x = fit_orders.loc[values.index].to_numpy()
        statistic, good, counts = conditional_correlation(
            torch.as_tensor(x, device=device), torch.as_tensor(y, dtype=torch.float32, device=device),
            torch.as_tensor(observed, dtype=torch.float32, device=device), projections.loc[values.index],
        )
        atlas.append((statistic, good))
        label = "/".join(assay)
        metadata[label] = {drug: int(n) for drug, n, okay in zip(drugs, counts, good) if okay}
    return atlas, metadata


def selection(atlas, drugs):
    paired, pooled, evidence = {}, {}, {}
    for index, drug in enumerate(drugs):
        columns = np.array([rho[:, index] for rho, support in atlas if support[index]])
        if len(columns) < 2:
            continue
        retained = np.maximum(columns.min(axis=0), -columns.max(axis=0))
        order = np.argsort(-retained, kind="stable")[:WIDTH]
        if len(order) < WIDTH or retained[order[-1]] < MIN_ASSOCIATION:
            continue
        agreed = columns.mean(axis=0)
        paired[index] = order
        pooled[index] = np.argsort(-abs(agreed), kind="stable")[:WIDTH]
        evidence[drug] = {"assays": len(columns), "min_stable_association": float(retained[order[-1]]),
                          "mean_stable_association": float(retained[order].mean())}
    shuffled = {}
    if len(paired) >= 2:
        order = np.random.default_rng(37181).permutation(sorted(paired))
        shuffled = {a: paired[b] for a, b in zip(order, np.roll(order, 1))}
    elif paired:
        raise ValueError("Need two supported drugs for the linkage control")
    return {PRIMARY: paired, POOL: pooled, SHUFFLE: shuffled}, evidence


def prior_features(reference, query, choices, n_drugs, device):
    shape = (n_drugs, WIDTH)
    left = np.zeros((len(reference), *shape), dtype=np.float32)
    right = np.zeros((len(query), *shape), dtype=np.float32)
    for idx, cols in choices.items():
        train = reference[:, cols]
        scale = Standardization.fit(train, .25)
        a = np.clip((train - scale.center) / scale.spread, -8, 8)
        b = np.clip((query[:, cols] - scale.center) / scale.spread, -8, 8)
        factor = np.sqrt(WEIGHT / max(1e-8, (a * a).sum(axis=1).mean()))
        left[:, idx], right[:, idx] = a * factor, b * factor
    return torch.as_tensor(left, device=device), torch.as_tensor(right, device=device)


def candidate(fit, query, rna, array, tissues, edges, targets, auxiliary, prior, old_audit):
    _validate(fit, query)
    device = device_cuda()
    genes = rna.columns.intersection(array.columns).sort_values()
    train = rna.loc[sorted(fit.cell_id.unique()), genes]
    held = rna.loc[sorted(query.cell_id.unique()), genes]
    if digest(genes) != old_audit["gene_sha256"]:
        raise ValueError("The ordinal expression input genes differ from the reference")
    groups = lineages(fit)
    fit_pair = paired_lineages(train.index, array, groups, tissues, 5)
    select, _, _ = select_pairs(train, array[genes], fit_pair, pair_candidates(edges, genes))
    pair = select[PORTABLE]
    lines = [f"{genes[a]}\t{genes[b]}" for a, b in pair]
    pair_digest = hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()
    if pair_digest != old_audit["arms"][PORTABLE]["pairs_sha256"]:
        raise ValueError("The portable RNA pairs differ from their original freeze")
    x, z = sign_orders(train, pair), sign_orders(held, pair)
    cell_orders = pd.DataFrame(x, index=train.index).groupby(groups).mean()
    _, anchor, aligned, _ = panel_targets(fit)
    drugs, kernels = drug_kernels(aligned, targets, device=device)
    w, y = training_grid(aligned, train.index, drugs, device=device)
    associations, by_assay = assay_associations(aligned, cell_orders, drugs, device)
    methods, details = selection(associations, drugs)
    basis, k = fit_cell_kernels(train, edges, auxiliary, device=device)
    cell, cross = k["full"], basis.cross(held)["full"]
    drug = kernels["structured"]
    roster = held.index.get_indexer(query.cell_id), drugs.get_indexer(query.compound_id)
    bias = query.compound_id.map(anchor).to_numpy()

    coefficient, _, _ = solve_residuals(cell, drug, w, y, PENALTY)
    replay = bias + (cross @ coefficient @ drug).cpu().numpy()[roster]
    reference = prior.loc[prior.method.eq(COMMON)].drop_duplicates("row_id").set_index("row_id").predicted_value
    maximum = float(np.max(np.abs(replay - reference.reindex(query.row_id).to_numpy())))
    if not np.isfinite(maximum) or maximum > 1e-5:
        raise ValueError(f"Original ordinal base replay differs by {maximum:.7g}")
    predictions = []
    for arm, feature in methods.items():
        source, inferred = prior_features(x, z, feature, len(drugs), device)
        if feature:
            coef, iterations, residual = solve_residuals(cell, drug, w, y, PENALTY, per_drug_prior=source)
            term = cross @ coef @ drug + project_drug_features(source, inferred, coef)
            values = bias + term.cpu().numpy()[roster]
        else:
            values, iterations, residual = replay, 0, 0.
        predictions.append(query[KEY].assign(method=arm, predicted_value=values))
        for idx, columns in feature.items():
            validation = "\n".join(lines[j] for j in columns) + "\n"
            details[drugs[idx]][arm] = hashlib.sha256(validation.encode()).hexdigest()
        by_assay[arm] = {"iterations": iterations, "solver_residual": residual}
    return pd.concat(predictions, ignore_index=True), {
        "pair_sha256": pair_digest, "fit_drugs": len(drugs), "supported_drugs": details,
        "assays_and_solver": by_assay, "replay_max_abs": maximum,
        "query_supported_drugs": int(query.compound_id.isin(details).groupby(query.compound_id).any().sum()),
    }


def predict(args):
    external = args.mode in EXTERNAL_TRACKS
    if external:
        fit, query, rna, seal = external_inputs(args, args.mode)
    else:
        old, core, rna, _ = registered_inputs(args)
        _, fit, check = same_split(old, core, args.fold, args.mode)
        query, seal = validation_roster(check), RECEIPTS["core"]
    prior, reference, old = archive(args, seal)
    array, tissues, registration = paired_data(args.array)
    if registration != ARRAY_INPUT:
        raise ValueError("Unexpected registered paired RNA")
    edges, targets, priors = molecular_inputs(args.priors, args.hallmark, args.targets)
    auxiliary, aux = load_auxiliary(args.auxiliary)
    new, support = candidate(fit, query, rna, array, tissues, edges, targets, auxiliary, prior, old)
    joint = pd.concat([prior, new.assign(track=args.mode, fold=0) if external else with_assay(new, check, args.fold)])
    methods = (*BASELINES, *REFERENCES, *ARMS, *PUBLIC) if external else (*BASELINES, *REFERENCES, *ARMS)
    view = joint.pivot(index=[*KEY, "track", "fold"], columns="method", values="predicted_value")
    if set(view) != set(methods) or len(view) != (len(query) if external else len(check)) or not np.isfinite(view.to_numpy()).all():
        raise ValueError("The drug-specific ordinal forecasts changed the registered exact rows")
    if external:
        shared_support(joint, query, methods)
    freeze_table(joint[FORECAST_SCHEMA], args.out, provenance={
        "mode": args.mode, "fold": args.fold, "inputs_sha256": seal, "core_sha256": RECEIPTS["core"],
        "reference_sha256": reference, "paired_rna_receipt_sha256": registration,
        "priors": priors, "auxiliary": aux, "drug_order_audit": support,
    })


def assess(frame, *, external):
    controls = [*BASELINES, COMMON, PORTABLE, POOL, SHUFFLE]
    if external:
        controls.extend(PUBLIC)
    values, _, _ = axis_metrics(frame, [(PRIMARY, name) for name in controls])
    interval = crossed_patient_drug_ci if external else paired_intervals
    local = {name: interval(frame, PRIMARY, name, repeats=1000 if external else 800) for name in controls}
    ranks = {(row["baseline"], row["metric"]): row["ci95"] for row in values["paired_contrasts"]}
    gates = {name: all(local[name][metric][0] > 0 for metric in DRUG_METRICS)
             and all(ranks[name, metric][0] > -.005 for metric in SAMPLE_METRICS) for name in controls}
    return {"metrics": values, "across_cell_ci95": local, "gates": gates,
            "groups": frame.lineage_group.nunique(), "observations": frame.row_id.nunique()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    for stage, extra in {
        "predict": ("archive", "array", "licob", "priors", "hallmark", "targets", "auxiliary"),
        "source": ("forecasts",),
        "external": ("forecasts", "licob", "limore-labels", "licob-labels"),
    }.items():
        command = stages.add_parser(stage)
        for name in ("original", "core", "out", *extra):
            command.add_argument(f"--{name}", type=Path, required=True)
        if stage == "predict":
            command.add_argument("--mode", choices=(*ASSAYS, *EXTERNAL_TRACKS), required=True)
            command.add_argument("--fold", type=int, choices=range(3), default=0)
    args = parser.parse_args()
    if args.stage == "predict":
        predict(args)
    elif args.stage == "source":
        score_source(args, assessor=assess)
    else:
        score_external(args, assessor=assess, arms=(*REFERENCES, *ARMS))


if __name__ == "__main__":
    main()
