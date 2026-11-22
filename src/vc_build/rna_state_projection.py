"""Assess response transfer after projecting out registered RNA state directions."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import join_observations
from .expanded_panel_cv import PENALTY, RECEIPTS, registered_inputs, same_split
from .external import KEY, _validate
from .io import freeze_table, publish_json
from .licob_node_recovery import joined_outcomes
from .limore_corrected import OBSERVATIONS, shared_support
from .pdo_transfer_cv import device_cuda
from .response_component_benchmark import (
    BASELINES, EXTERNAL_TRACKS, FORECAST_SCHEMA, PUBLIC, check_component, external_inputs,
    external_reference, source_reference,
)
from .structured_cv import ASSAYS, attach, paired_intervals, with_assay
from .structured_kernel import KernelGrid, drug_kernels, fit_cell_kernels, ranked_expression, training_grid
from .v0plus_cv import molecular_inputs, panel_targets, validation_roster
from .v0plus_licob import ASSET_HASHES, adult_donor_outcomes
from .v0plus_neural import load_auxiliary
from .v0plus_score import axis_metrics, compound_strata, crossed_patient_drug_ci
from .v0plus_wong import WORKBOOK_HASH
from .wong_paired_rna import REGISTRATIONS, registration
from .yang_hcc_external import require_hash

STATES = tuple("HALLMARK_" + x for x in (
    "E2F_TARGETS", "G2M_CHECKPOINT", "MITOTIC_SPINDLE", "EPITHELIAL_MESENCHYMAL_TRANSITION", "HYPOXIA",
))
PRIMARY = "paired_and_state"
ARMS = ("states_only", "shuffled_pairs", PRIMARY)
CONTROLS = ("structured", *ARMS[:2])


def paired_rna(organoid, tissue):
    profiles, rosters = [], []
    for name, directory in (("organoid", organoid), ("tissue", tissue)):
        path = registration(directory, name, ("query.parquet", "rna.parquet"))
        receipt = json.loads((path / "receipt.json").read_text())
        if receipt["workbook_sha256"] != WORKBOOK_HASH:
            raise ValueError("Wrong paired RNA workbook")
        roster = pd.Index(sorted(pd.read_parquet(path / "query.parquet", columns=["cell_id"]).cell_id.unique()))
        profiles.append(pd.read_parquet(path / "rna.parquet").loc[roster])
        rosters.append(roster)
    if len(rosters[0]) != 10 or not rosters[0].equals(rosters[1]):
        raise ValueError("Paired RNA must include the ten registered donor pairs")
    return profiles


def coordinates(genes, edges, pdo, tissue):
    state = (edges[edges.signature.isin(STATES)]
             .pivot(index="signature", columns="gene", values="weight")
             .reindex(index=STATES, columns=genes).fillna(0).to_numpy())
    shared = genes.intersection(pdo.columns).intersection(tissue.columns)
    if (np.count_nonzero(state, axis=1) < 10).any() or not len(shared):
        raise ValueError("RNA axis fails Hallmark or Wong support")
    _, values, axes = np.linalg.svd(state, full_matrices=False)
    if values[-1] < 1e-8:
        raise ValueError("Degenerate Hallmark directions")
    left, right = [ranked_expression(frame[shared]) for frame in (pdo, tissue)]
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Wong profiles must give finite ranks")
    order = np.random.default_rng(69403).permutation(len(pdo))
    assigned = np.empty_like(order)
    assigned[order] = np.roll(order, 1)
    paired, permuted = [np.zeros((len(pdo), len(genes))) for _ in range(2)]
    where = genes.get_indexer(shared)
    paired[:, where] = left - right
    permuted[:, where] = left[assigned] - right
    blocks = {"states_only": axes}
    for name, observed in (("shuffled_pairs", permuted), (PRIMARY, paired)):
        residual = observed - (observed @ axes.T) @ axes
        mean = residual.mean(axis=0)
        norm = np.linalg.norm(mean)
        if norm < 1e-8:
            raise ValueError("Degenerate Wong mean direction")
        shift = mean / norm
        variation = residual - np.outer(residual @ shift, shift)
        _, values, variation_axes = np.linalg.svd(variation, full_matrices=False)
        if values[2] < 1e-8:
            raise ValueError("Insufficient Wong variation")
        blocks[name] = np.vstack([axes, shift, variation_axes[:3]])
    for block in blocks.values():
        if not np.allclose(block @ block.T, np.eye(len(block)), atol=1e-8):
            raise ValueError("The projection axes must be orthonormal")
    return blocks, paired, len(shared)


def candidate_forecasts(fit, query, rna, edges, targets, auxiliary, bridge):
    _validate(fit, query)
    device = device_cuda()
    source = rna.loc[sorted(fit.cell_id.unique())]
    held = rna.loc[sorted(query.cell_id.unique())]
    axes, donor_delta, shared = coordinates(rna.columns, edges, *bridge)
    _, anchors, aligned, _ = panel_targets(fit)
    drugs, kernels = drug_kernels(aligned, targets, device=device)
    weights, response = training_grid(aligned, source.index, drugs, device=device)
    base = query.compound_id.map(anchors).to_numpy()
    shift = ranked_expression(held).mean(axis=0) - ranked_expression(source).mean(axis=0)
    rows, audit = [], {}
    for name, block in axes.items():
        basis, gram = fit_cell_kernels(source, edges, auxiliary, device=device,
                                      projection_axes=block.astype(np.float32))
        other = basis.cross(held)
        grid = KernelGrid(held.index, drugs, {"full": (gram["full"], other["full"])},
                          kernels, weights, response, basis.reference[-1].shape[1])
        residual, trace = grid.forecast(query, "structured", PENALTY)
        rows.append(query[KEY].assign(method=name, predicted_value=base + residual))
        audit[name] = {
            **trace, "dimensions": len(block),
            "rna_mean_gap_squared_fraction_removed": float(np.square(shift @ block.T).sum() /
                                                           np.square(shift).sum()),
            "paired_wong_energy_fraction_removed": float(np.square(donor_delta @ block.T).sum() /
                                                         np.square(donor_delta).sum()),
        }
    return pd.concat(rows, ignore_index=True), {"genes": len(rna.columns), "paired_genes": shared, "arms": audit}


def predict(args):
    edges, targets, priors = molecular_inputs(args.priors, args.hallmark, args.targets)
    auxiliary, auxiliary_receipt = load_auxiliary(args.auxiliary)
    bridge = paired_rna(args.organoid, args.tissue)
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
    new, diagnostics = candidate_forecasts(fit, query, rna, edges, targets, auxiliary, bridge)
    if external:
        joint = pd.concat([previous, new.assign(track=args.mode, fold=0)], ignore_index=True)
        shared_support(joint, query, (*BASELINES, *PUBLIC, *ARMS))
    else:
        joint = pd.concat([previous, with_assay(new, check, args.fold)], ignore_index=True)
        matched = joint.pivot(index=[*KEY, "track", "fold"], columns="method", values="predicted_value")
        if set(matched) != {*BASELINES, *ARMS} or not np.isfinite(matched.to_numpy()).all():
            raise ValueError("The corrected source forecasts must have shared support")
    freeze_table(joint[FORECAST_SCHEMA], args.out, provenance={
        "mode": args.mode, "fold": args.fold, "inputs_sha256": registered, "core_sha256": RECEIPTS["core"],
        "reference_sha256": reference, "priors": priors, "auxiliary": auxiliary_receipt,
        "wong_workbook_sha256": WORKBOOK_HASH, "wong_receipts": REGISTRATIONS, "diagnostics": diagnostics,
    })


def assess(frame, *, external, primary=PRIMARY, controls=CONTROLS, require_precision=False):
    comparisons = (*controls, "drug_ridge", *PUBLIC) if external else (*controls, "drug_ridge")
    overall, _, _ = axis_metrics(frame, [(primary, arm) for arm in comparisons])
    interval = crossed_patient_drug_ci if external else paired_intervals
    across = {arm: interval(frame, primary, arm) for arm in comparisons}
    within = {(row["baseline"], row["metric"]): row["ci95"] for row in overall["paired_contrasts"]}
    ranks = ("spearman", "ndcg_at_5", "top5_overlap") if require_precision else ("spearman", "ndcg_at_5")
    gates = {arm: all(across[arm][metric][0] > 0 for metric in ("spearman", "pearson"))
             and all(within[arm, metric][0] > -.005 for metric in ranks) for arm in controls}
    return {"metrics": overall, "across_cell_ci95": across, "gates": gates,
            "groups": frame.lineage_group.nunique(), "observations": frame.row_id.nunique()}


def score_source(args, *, assessor=assess):
    original, clean, _, _ = registered_inputs(args)
    result, hashes = {}, {}
    for mode in ASSAYS:
        parts = []
        for fold in range(3):
            name = f"{mode}-{fold}"
            frame, hashes[name] = check_component(args.forecasts / f"{name}.parquet", mode, fold, RECEIPTS["core"])
            _, _, labels = same_split(original, clean, fold, mode)
            parts.append(attach(frame, labels))
        result[mode] = {track: assessor(rows, external=False)
                        for track, rows in pd.concat(parts, ignore_index=True).groupby("track")}
    passed = sum(all(next(iter(result[mode].values()))["gates"].values()) for mode in ASSAYS if mode != "pooled")
    publish_json(args.out, {"modes": result, "forecast_sha256": hashes, "screen_gates_passed": passed,
                            "pass": passed >= 2})


def score_external(args, *, assessor=assess, arms=ARMS):
    result, hashes = {}, {}
    for mode in EXTERNAL_TRACKS:
        _, query, _, registered = external_inputs(args, mode)
        frame, hashes[mode] = check_component(args.forecasts / f"{mode}.parquet", mode, 0, registered)
        shared_support(frame, query, (*BASELINES, *PUBLIC, *arms))
        if mode == "LIMORE":
            require_hash(args.limore_labels, OBSERVATIONS)
            observed = join_observations(frame, pd.read_parquet(args.limore_labels))
        else:
            require_hash(args.licob_labels, ASSET_HASHES["supplement"])
            observed = adult_donor_outcomes(joined_outcomes(query, frame, args.licob_labels))
        report = assessor(observed, external=True)
        strata = compound_strata(query)
        report["strata"] = {}
        for name in ("hcc_tki", "targeted_or_kinase", "nonselective_cytotoxics"):
            block = observed[observed.compound_id.isin(strata[name])]
            if len(block):
                metrics, _, _ = axis_metrics(block, ())
                report["strata"][name] = {"drugs": block.compound_id.nunique(), "metrics": metrics}
        result[mode] = report
    publish_json(args.out, {"cohorts": result, "forecast_sha256": hashes,
                            "pass": all(all(report["gates"].values()) for report in result.values())})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    fields = {
        "predict": ("licob", "archive", "cv-archive", "priors", "hallmark", "targets", "auxiliary", "organoid", "tissue"),
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
    {"predict": predict, "source": score_source, "external": score_external}[args.stage](args)


if __name__ == "__main__":
    main()
