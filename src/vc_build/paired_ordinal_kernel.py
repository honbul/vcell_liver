"""Learn portable within-signature gene orders from paired public RNA assays."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .expanded_panel_cv import PENALTY, RECEIPTS, registered_inputs, same_split
from .external import KEY, _validate
from .io import freeze_table
from .limore_corrected import shared_support
from .paired_expression import auxiliary_cell_grams, paired_data, row_center
from .pdo_transfer_cv import device_cuda
from .rank_context_calibration import DRUG_METRICS, SAMPLE_METRICS, archive_component
from .response_component_benchmark import BASELINES, EXTERNAL_TRACKS, FORECAST_SCHEMA, PUBLIC, external_inputs
from .rna_gene_panel import ARRAY_INPUT, PDO_AXIS, PDO_CONTEXT, digest
from .rna_state_projection import score_external, score_source
from .structured_cv import ASSAYS, paired_intervals, with_assay
from .structured_kernel import KernelGrid, Standardization, drug_kernels, fit_cell_kernels, training_grid
from .v0plus_cv import molecular_inputs, panel_targets, validation_roster
from .v0plus_neural import load_auxiliary
from .v0plus_score import axis_metrics, crossed_patient_drug_ci

PRIMARY = "paired_ordinal"
COMMON = "ordinal_matched_rank"
SHUFFLE = "ordinal_shuffled"
SEQUENCE = "ordinal_seq_only"
ARMS = (COMMON, PRIMARY, SHUFFLE, SEQUENCE)
REF_CONTEXTS = (PDO_CONTEXT, PDO_AXIS)
PAIR_LIMIT = 96
PAIR_QUOTA = 200
PAIR_WEIGHT = .35
PAIR_SEED = 40121
SHUFFLE_SEED = 40139


def pair_candidates(edges, genes):
    registered = edges[edges.gene.isin(genes)]
    rng = np.random.default_rng(PAIR_SEED)
    pairs = []
    for _, group in registered.groupby("signature", sort=True):
        axis = genes.get_indexer(sorted(group.gene.unique()))
        if len(axis) < 10:
            continue
        lo, hi = np.triu_indices(len(axis), k=1)
        take = rng.choice(len(lo), min(PAIR_LIMIT, len(lo)), replace=False)
        pairs.append(np.column_stack((axis[lo[take]], axis[hi[take]])))
    return np.unique(np.concatenate(pairs), axis=0)


def sign_orders(frame, pairs):
    values = frame.to_numpy()
    return np.sign(values[:, pairs[:, 0]] - values[:, pairs[:, 1]]).astype(np.float32)


def lineages(rows):
    available = rows[["cell_id", "lineage_group"]].drop_duplicates().sort_values("cell_id")
    return available.set_index("cell_id", verify_integrity=True).lineage_group


def paired_lineages(ids, array, map_groups, tissues, minimum):
    ids = ids.intersection(array.index).sort_values()
    group = map_groups.loc[ids]
    keep = group.index[~group.duplicated()]
    labels = tissues.tissue.loc[keep]
    counts = labels.value_counts()
    return labels[labels.isin(counts[counts.ge(minimum)].index)]


def same_tissue_shuffle(labels):
    rng = np.random.default_rng(SHUFFLE_SEED)
    shifted = np.arange(len(labels))
    for indexes in labels.groupby(labels, sort=True).indices.values():
        current = rng.permutation(indexes)
        shifted[current] = np.roll(current, 1)
    if (shifted == np.arange(len(shifted))).any():
        raise ValueError("Invalid shuffled pairing")
    return shifted


def column_correlations(left, right):
    denominator = np.sqrt((left ** 2).sum(axis=0) * (right ** 2).sum(axis=0))
    return np.divide((left * right).sum(axis=0), denominator,
                     out=np.zeros_like(denominator), where=denominator > 0)


def select_pairs(train, array, labels, candidates):
    left = sign_orders(train.loc[labels.index], candidates)
    right = sign_orders(array.loc[labels.index], candidates)
    balanced = np.ones(len(candidates), dtype=bool)
    for values in (left, right):
        balanced &= ((values > 0).mean(axis=0) >= .1) & ((values < 0).mean(axis=0) >= .1)
    pool = candidates[balanced]
    x = row_center(left[:, balanced], labels)
    y = row_center(right[:, balanced], labels)
    variance = (x ** 2).mean(axis=0)
    corr = column_correlations(x, y)
    shuffled_corr = column_correlations(x, y[same_tissue_shuffle(labels)])
    scores = {PRIMARY: corr, SHUFFLE: shuffled_corr, SEQUENCE: variance}
    bins = np.array_split(np.argsort(variance, kind="stable"), 5)
    if min(map(len, bins)) < PAIR_QUOTA:
        raise ValueError("Too few balanced ordinal pairs for matched variance coverage")
    result = {}
    audit = {}
    for name, score in scores.items():
        chosen = np.concatenate([part[np.lexsort((part, -score[part]))[:PAIR_QUOTA]] for part in bins])
        result[name] = pool[chosen]
        audit[name] = {"fit_correlation_mean": float(corr[chosen].mean()),
                       "seq_variance_mean": float(variance[chosen].mean()),
                       "selected_pairs": len(chosen),
                       "unique_genes": int(np.unique(pool[chosen]).size)}
    return result, audit, int(balanced.sum())


def held_rna_audit(pair, held, array, labels):
    if labels.empty:
        return {}
    first = sign_orders(held.loc[labels.index], pair)
    second = sign_orders(array.loc[labels.index], pair)
    return {"paired_lineages": len(labels),
            "sign_agreement": float((first == second).mean()),
            "fit_independent_within_tissue_correlation": float(
                column_correlations(row_center(first, labels), row_center(second, labels)).mean())}


def candidate_forecasts(fit, query, rna, edges, targets, auxiliary, array, tissues):
    _validate(fit, query)
    shared = rna.columns.intersection(array.columns).sort_values()
    if len(shared) < 10_000:
        raise ValueError("Insufficient RNA common support")
    train = rna.loc[sorted(fit.cell_id.unique()), shared]
    held = rna.loc[sorted(query.cell_id.unique()), shared]
    array = array[shared]
    fit_labels = paired_lineages(train.index, array, lineages(fit), tissues, 5)
    held_labels = paired_lineages(held.index, array, lineages(query), tissues, 5)
    if len(fit_labels) < 250 or set(fit_labels.index) & set(held_labels.index):
        raise ValueError("Invalid fit-only paired lineages")
    candidates = pair_candidates(edges, shared)
    pair_sets, evidence, balanced = select_pairs(train, array, fit_labels, candidates)
    device = device_cuda()
    basis, fit_grams = fit_cell_kernels(train, edges, auxiliary, device=device)
    base = (fit_grams["full"], basis.cross(held)["full"])
    _, anchor, aligned, _ = panel_targets(fit)
    drugs, kernels = drug_kernels(aligned, targets, device=device)
    weight, y = training_grid(aligned, train.index, drugs, device=device)
    offset = query.compound_id.map(anchor).to_numpy()
    estimates = []
    for name in ARMS:
        if name == COMMON:
            grams = base
        else:
            selected = pair_sets[name]
            train_values, held_values = sign_orders(train, selected), sign_orders(held, selected)
            scale = Standardization.fit(train_values, .25)
            extra = auxiliary_cell_grams(scale.apply(train_values, device), scale.apply(held_values, device))
            grams = tuple((1 - PAIR_WEIGHT) * a + PAIR_WEIGHT * b for a, b in zip(base, extra))
            blob = "\n".join(f"{shared[a]}\t{shared[b]}" for a, b in selected) + "\n"
            evidence[name] |= {
                "pairs_sha256": hashlib.sha256(blob.encode()).hexdigest(),
                "held_assay": held_rna_audit(selected, held, array, held_labels),
                "fit_tie_fraction": float((train_values == 0).mean()),
                "held_tie_fraction": float((held_values == 0).mean()),
            }
        grid = KernelGrid(held.index, drugs, {"full": grams}, kernels, weight, y, basis.reference[-1].shape[1])
        effect, diag = grid.forecast(query, "structured", PENALTY)
        estimates.append(query[KEY].assign(method=name, predicted_value=offset + effect))
        evidence[name] = evidence.get(name, {}) | diag
    return pd.concat(estimates, ignore_index=True), {
        "genes": len(shared), "gene_sha256": digest(shared), "candidate_pairs": len(candidates),
        "balanced_pairs": balanced, "fit_paired_lineages": len(fit_labels), "fit_tissues": fit_labels.value_counts().to_dict(),
        "arms": evidence,
    }


def predict(args):
    external = args.mode in EXTERNAL_TRACKS
    array, tissues, registered = paired_data(args.array)
    if registered != ARRAY_INPUT:
        raise ValueError("Mismatched paired RNA registration")
    if external:
        fit, query, rna, seal = external_inputs(args, args.mode)
    else:
        old, core, rna, _ = registered_inputs(args)
        _, fit, check = same_split(old, core, args.fold, args.mode)
        query, seal = validation_roster(check), RECEIPTS["core"]
    prior, reference = archive_component(args, seal)
    edges, targets, priors = molecular_inputs(args.priors, args.hallmark, args.targets)
    auxiliary, aux = load_auxiliary(args.auxiliary)
    new, audit = candidate_forecasts(fit, query, rna, edges, targets, auxiliary, array, tissues)
    panel = pd.concat([prior, new.assign(track=args.mode, fold=0) if external else with_assay(new, check, args.fold)])
    methods = (*BASELINES, *REF_CONTEXTS, *ARMS, *PUBLIC) if external else (*BASELINES, *REF_CONTEXTS, *ARMS)
    wide = panel.pivot(index=[*KEY, "track", "fold"], columns="method", values="predicted_value")
    if set(wide) != set(methods) or len(wide) != (len(query) if external else len(check)) or not np.isfinite(wide.to_numpy()).all():
        raise ValueError("Ordinal conditions changed the common evaluation support")
    if external:
        shared_support(panel, query, methods)
    freeze_table(panel[FORECAST_SCHEMA], args.out, provenance={
        "mode": args.mode, "fold": args.fold, "inputs_sha256": seal, "core_sha256": RECEIPTS["core"],
        "reference_sha256": reference, "paired_rna_receipt_sha256": registered,
        "priors": priors, "auxiliary": aux, "ordinal_audit": audit,
    })


def assess(frame, *, external):
    peers = (*BASELINES, *PUBLIC) if external else BASELINES
    checks = (*peers, *REF_CONTEXTS, COMMON, SHUFFLE, SEQUENCE)
    metrics, _, _ = axis_metrics(frame, [(PRIMARY, name) for name in checks])
    interval = crossed_patient_drug_ci if external else paired_intervals
    across = {name: interval(frame, PRIMARY, name, repeats=1000 if external else 800) for name in checks}
    ranks = {(r["baseline"], r["metric"]): r["ci95"] for r in metrics["paired_contrasts"]}
    gates = {name: all(across[name][m][0] > 0 for m in DRUG_METRICS)
             and all(ranks[name, m][0] > -.005 for m in SAMPLE_METRICS) for name in checks}
    return {"metrics": metrics, "across_cell_ci95": across, "gates": gates, "groups": frame.lineage_group.nunique(),
            "observations": frame.row_id.nunique()}


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
        score_external(args, assessor=assess, arms=(*REF_CONTEXTS, *ARMS))


if __name__ == "__main__":
    main()
