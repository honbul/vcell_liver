"""Learn cell sensitivity contrasts that repeat in independent source screens."""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .evaluate import join_observations
from .expanded_panel_cv import RECEIPTS, registered_inputs, same_split
from .external import KEY, SOURCE, _validate
from .hcc_3d_ordinal import gram_features
from .io import freeze_table, publish_json, verify_forecast
from .licob_node_recovery import joined_outcomes
from .limore_corrected import OBSERVATIONS, shared_support
from .pdo_transfer_cv import device_cuda
from .response_component_benchmark import (
    BASELINES, EXTERNAL_TRACKS, FORECAST_SCHEMA, PUBLIC, external_inputs, external_reference, source_reference,
)
from .structured_cv import ASSAYS, attach, paired_intervals, with_assay
from .structured_kernel import KernelGrid
from .v0plus_cv import molecular_inputs, panel_targets, validation_roster
from .v0plus_licob import ASSET_HASHES, adult_donor_outcomes
from .v0plus_neural import load_auxiliary
from .v0plus_score import axis_metrics, compound_strata, crossed_patient_drug_ci
from .yang_hcc_external import require_hash

PRIMARY = "replicated_pairs"
ARMS = ("all_sampled_pairs", PRIMARY)
CONTROLS = ("structured", ARMS[0])
CONFIRM = ("pairwise_rank", "pairwise_top5", "ecfp4_path_elasticnet")
CONFIRM_HASH = "c00b68388f256761953fff1d523baea55b2f6b2ef99e4751d27d880283154f28"
PENALTY = 4.


def registered_pairs(source):
    samples = source[["cell_id", "lineage_group"]].drop_duplicates().set_index("cell_id")
    if not samples.index.is_unique:
        raise ValueError("Ambiguous fit-lineage registration")
    pieces = []
    rng = np.random.default_rng(90721)
    for drug, table in source.groupby("compound_id", sort=True):
        assay = table.groupby(["cell_id", *SOURCE]).residual.mean().unstack(SOURCE)
        for left, right in combinations(assay.columns, 2):
            common = assay[[left, right]].dropna().sort_index()
            if len(common) < 30:
                continue
            identities = samples.loc[common.index].lineage_group.to_numpy()
            cells = common.index.to_numpy()
            a, b = np.triu_indices(len(cells), k=1)
            eligible = np.flatnonzero(identities[a] != identities[b])
            chosen = rng.choice(eligible, size=min(4000, len(eligible)), replace=False)
            if not len(chosen):
                continue
            values = common.to_numpy()
            spread = values.std(axis=0)
            if (spread < 1e-8).any():
                continue
            delta = (values[a[chosen]] - values[b[chosen]]) / spread
            same = (np.sign(delta[:, 0]) == np.sign(delta[:, 1])) & (np.abs(delta).min(axis=1) >= .15)
            pieces.append(pd.DataFrame({
                "compound_id": drug, "assay_pair": " + ".join("/".join(x) for x in (left, right)),
                "first": cells[a[chosen]], "second": cells[b[chosen]], "shared_models": len(cells),
                "target": np.clip(delta, -2., 2.).mean(axis=1) * spread.mean(), "replicated": same,
            }))
    if not pieces:
        raise ValueError("No repeated drug–cell pairs on the registered fit")
    return pd.concat(pieces, ignore_index=True)


def pair_tensors(pairs, total, cells, drugs, device):
    group_size = pairs.groupby(["compound_id", "assay_pair"]).target.transform("size")
    n_assays = pairs.groupby("compound_id").assay_pair.transform("nunique")
    weights = .35 * pairs.compound_id.map(total).to_numpy() / (group_size * n_assays).to_numpy()
    col = drugs.get_indexer(pairs.compound_id)
    first, second = (cells.get_indexer(pairs[name]) for name in ("first", "second"))
    if any((idx < 0).any() for idx in (col, first, second)) or not np.isfinite(weights).all():
        raise ValueError("Pair constraints do not map to the source grid")
    indices = [torch.as_tensor(idx * len(drugs) + col, device=device) for idx in (first, second)]
    tensors = [torch.as_tensor(x, device=device, dtype=torch.float32) for x in (weights, pairs.target.to_numpy())]
    return (*indices, *tensors)


def fit_constraints(cell, drug, w, y, pairs, *, tolerance=5e-5, max_steps=400):
    """Preconditioned conjugate gradients on the whitened product-kernel normal equations."""
    left, right, weight, target = pairs

    def divergence(values):
        flat = w.new_zeros(w.numel())
        flat.index_add_(0, left, values)
        flat.index_add_(0, right, -values)
        return flat.view_as(w)

    degree = w.new_zeros(w.numel())
    degree.index_add_(0, left, weight)
    degree.index_add_(0, right, weight)
    diagonal = cell.square().T @ (w + degree.view_as(w)) @ drug.square() + PENALTY
    rhs = cell.T @ (w * y + divergence(weight * target)) @ drug
    fit = torch.zeros_like(rhs)
    residual = rhs.clone()
    direction = residual / diagonal
    energy = (direction * residual).sum()
    threshold = max(1e-8, tolerance * float(rhs.norm()))
    if float(residual.norm()) <= threshold:
        return fit, 0, float(residual.norm())
    for step in range(max_steps):
        signal = cell @ direction @ drug.T
        flat = signal.flatten()
        contrasts = divergence(weight * (flat[left] - flat[right]))
        product = cell.T @ (w * signal + contrasts) @ drug + PENALTY * direction
        curvature = (direction * product).sum()
        update = energy / curvature.clamp_min(1e-20)
        fit += update * direction
        residual -= update * product
        error = float(residual.norm())
        if error <= threshold:
            return fit, step + 1, error
        scaled = residual / diagonal
        next_energy = (scaled * residual).sum()
        direction = scaled + (next_energy / energy) * direction
        energy = next_energy
    raise RuntimeError(f"Pair-constraint solver residual {error:.5g} exceeds {threshold:.5g}")


def candidates(fit, query, rna, edges, targets, auxiliary):
    _validate(fit, query)
    device = device_cuda()
    fit_rna = rna.loc[sorted(fit.cell_id.unique())]
    query_rna = rna.loc[sorted(query.cell_id.unique())]
    _, anchors, source, _ = panel_targets(fit)
    grid = KernelGrid.prepare(source, fit_rna, query_rna, edges, targets, auxiliary, device=device)
    pairs = registered_pairs(source)
    cell, held = (x.float() for x in gram_features(*grid.cell["full"]))
    drug = gram_features(grid.drug["structured"], None)[0].float()
    total = pd.Series(grid.weights.sum(dim=0).cpu().numpy(), index=grid.drugs)
    base = query.compound_id.map(anchors).to_numpy()
    indices = grid.cells.get_indexer(query.cell_id), grid.drugs.get_indexer(query.compound_id)
    result, info = [], {}
    for arm, selected in zip(ARMS, (pairs, pairs[pairs.replicated])):
        constraints = pair_tensors(selected, total, fit_rna.index, grid.drugs, device)
        coeff, steps, residual = fit_constraints(cell, drug, grid.weights, grid.response, constraints)
        estimates = (held @ coeff @ drug.T).cpu().numpy()[indices]
        result.append(query[KEY].assign(method=arm, predicted_value=base + estimates))
        info[arm] = {"sampled_pairs": len(selected), "drugs": sorted(selected.compound_id.unique()),
                     "support_by_assay_pair": selected.groupby("assay_pair").size().to_dict(),
                     "iterations": steps, "solver_residual": residual}
    return pd.concat(result, ignore_index=True), info


def methods(mode):
    additions = CONFIRM if mode == "LIMORE" else ()
    public = PUBLIC if mode in EXTERNAL_TRACKS else ()
    return (*BASELINES, *public, *ARMS, *additions)


def predict(args):
    edges, targets, priors = molecular_inputs(args.priors, args.hallmark, args.targets)
    aux, receipt = load_auxiliary(args.auxiliary)
    extra_hash = None
    if args.mode in EXTERNAL_TRACKS:
        fit, query, rna, inputs = external_inputs(args, args.mode)
        previous, prior_hash = external_reference(args, args.mode, query)
        fresh, diagnostics = candidates(fit, query, rna, edges, targets, aux)
        parts = [previous, fresh.assign(track=args.mode, fold=0)]
        if args.mode == "LIMORE":
            confirmed, stamp = verify_forecast(args.confirm)
            if stamp["sha256"] != CONFIRM_HASH or stamp["provenance"]["observation_sha256"] != OBSERVATIONS:
                raise ValueError("Matched external modern comparators changed")
            use = confirmed[confirmed.method.isin(CONFIRM)]
            shared_support(use, query, CONFIRM)
            parts.append(use.assign(track=args.mode, fold=0))
        joint = pd.concat(parts, ignore_index=True)
        shared_support(joint, query, methods(args.mode))
        extra_hash = CONFIRM_HASH if args.mode == "LIMORE" else None
    else:
        original, clean, rna, _ = registered_inputs(args)
        _, fit, check = same_split(original, clean, args.fold, args.mode)
        query = validation_roster(check)
        previous, prior_hash = source_reference(args.cv_archive, args.mode, args.fold)
        fresh, diagnostics = candidates(fit, query, rna, edges, targets, aux)
        joint = pd.concat([previous, with_assay(fresh, check, args.fold)], ignore_index=True)
        matrix = joint.pivot(index=[*KEY, "track", "fold"], columns="method", values="predicted_value")
        if set(matrix) != set(methods(args.mode)) or not np.isfinite(matrix.to_numpy()).all():
            raise ValueError("Unmatched corrected assay support")
        inputs = RECEIPTS["core"]
    freeze_table(joint[FORECAST_SCHEMA], args.out, provenance={
        "mode": args.mode, "fold": args.fold, "inputs_sha256": inputs, "core_sha256": RECEIPTS["core"],
        "reference_sha256": prior_hash, "confirm_sha256": extra_hash, "priors": priors, "auxiliary": receipt,
        "fit": diagnostics,
    })


def frozen(path, mode, fold, inputs):
    prediction, stamp = verify_forecast(path)
    p = stamp["provenance"]
    if (p["mode"], p["fold"], p["inputs_sha256"], p["core_sha256"]) != (mode, fold, inputs, RECEIPTS["core"]):
        raise ValueError("Wrong held screen, fold, or cohort")
    return prediction, stamp["sha256"], p["fit"][PRIMARY]["drugs"]


def score_family(joined, *, external, added=()):
    all_pairs = [(PRIMARY, name) for name in (*CONTROLS, *added)]
    overall, _, _ = axis_metrics(joined, all_pairs)
    interval = crossed_patient_drug_ci if external else paired_intervals
    across = {b: interval(joined, a, b) for a, b in all_pairs}
    within = {(row["baseline"], row["metric"]): row["ci95"] for row in overall["paired_contrasts"]}
    gates = {b: all(across[b][name][0] > 0 for name in ("spearman", "pearson"))
             and all(within[b, m][0] > -.005 for m in ("spearman", "ndcg_at_5")) for b in CONTROLS}
    support = {}
    for name, part in joined.groupby("pair_support"):
        if part.compound_id.nunique() >= 2:
            support[str(name)] = axis_metrics(part, ())[0]["across_cell_scores"]
    return {"overall": overall, "pairs_ci95": across, "gates": gates, "drug_support": support,
            "groups": joined.lineage_group.nunique(), "observed_pairs": joined.row_id.nunique()}


def score_source(args):
    old, clean, _, _ = registered_inputs(args)
    report, hashes = {}, {}
    for mode in ASSAYS:
        parts = []
        for fold in range(3):
            name = f"{mode}-{fold}"
            forecast, hashes[name], supported = frozen(
                args.forecasts / f"{name}.parquet", mode, fold, RECEIPTS["core"],
            )
            _, _, labels = same_split(old, clean, fold, mode)
            parts.append(attach(forecast, labels).assign(pair_support=lambda d: d.compound_id.isin(supported)))
        report[mode] = {track: score_family(rows, external=False)
                        for track, rows in pd.concat(parts, ignore_index=True).groupby("track")}
    passed = sum(all(next(iter(report[name].values()))["gates"].values()) for name in ASSAYS if name != "pooled")
    publish_json(args.out, {"screen_gates_passed": passed, "pass": passed >= 2, "modes": report, "forecast_sha256": hashes})


def score_external(args):
    result, hashes = {}, {}
    for mode in EXTERNAL_TRACKS:
        _, query, _, inputs = external_inputs(args, mode)
        frame, hashes[mode], supported = frozen(args.forecasts / f"{mode}.parquet", mode, 0, inputs)
        shared_support(frame, query, methods(mode))
        if mode == "LIMORE":
            require_hash(args.limore_labels, OBSERVATIONS)
            observed = join_observations(frame, pd.read_parquet(args.limore_labels))
        else:
            require_hash(args.licob_labels, ASSET_HASHES["supplement"])
            observed = adult_donor_outcomes(joined_outcomes(query, frame, args.licob_labels))
        observed = observed.assign(pair_support=observed.compound_id.isin(supported))
        added = ("drug_mean", "drug_ridge", "cellhit", "codeae_inductive") + (CONFIRM if mode == "LIMORE" else ())
        report = score_family(observed, external=True, added=added)
        grouping = compound_strata(query)
        report["subgroups"] = {}
        for name in ("hcc_tki", "targeted_or_kinase", "nonselective_cytotoxics"):
            if grouping[name]:
                section = observed[observed.compound_id.isin(grouping[name])]
                report["subgroups"][name] = {"drugs": len(grouping[name]),
                    "scores": axis_metrics(section, ())[0]["across_cell_scores"]}
        result[mode] = report
    publish_json(args.out, {"cohorts": result, "forecast_sha256": hashes,
                            "pass": all(all(report["gates"].values()) for report in result.values())})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    fields = {
        "predict": ("licob", "archive", "cv-archive", "confirm", "priors", "hallmark", "targets", "auxiliary"),
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
