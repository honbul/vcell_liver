"""Test fit-only assay consistency weighting against the corrected native kernel."""

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtri

from .evaluate import score, score_across_cells
from .expanded_panel_cv import PENALTY, RECEIPTS, registered_inputs, same_split
from .external import KEY, SOURCE, _validate
from .hcc_conditional_cv import fold_centered_correlations
from .io import freeze_table, publish_json, sha256, verify_forecast
from .structured_cv import ASSAYS, attach, observations, paired_intervals, with_assay
from .v0plus_cv import molecular_inputs, panel_targets, validation_roster

PREVIOUS_SCORE = "3a294d50c9e693e7f64a5db6a30c5ea659c214a8afc37f01492f6a91002ed4a1"
PRIMARY = "agreement_rank"
BLEND = "agreement_ridge_average"
CENTERED = "centered_corrected_core"
CORE = ("corrected_core", "corrected_ridge", "corrected_drug_mean")
CORE_BLEND = "corrected_ridge_average"
METHODS = {*CORE, CORE_BLEND, CENTERED, "normal_rank", PRIMARY, BLEND}
CONTRASTS = ((PRIMARY, "corrected_core"), (PRIMARY, CENTERED), (PRIMARY, "normal_rank"),
             (PRIMARY, "corrected_ridge"), (BLEND, CORE_BLEND), (BLEND, "corrected_ridge"))


def normal_score(ranks, counts):
    return ndtri((ranks - .5) / counts)


def rank_residuals(rows):
    assay_drug = [*SOURCE, "compound_id"]
    groups = rows.groupby(assay_drug).residual
    rms = np.sqrt(groups.var(ddof=0).groupby("compound_id").mean())
    z = normal_score(groups.rank(), groups.transform("size"))
    ranks = z.groupby([rows[name] for name in assay_drug])
    spread = ranks.transform("std", ddof=0)
    standardized = (z - ranks.transform("mean")) / spread.where(spread.gt(0))
    return rows.assign(residual=standardized.fillna(0) * rows.compound_id.map(rms))


def confidence_weights(rows):
    # Compute concordance only for identical fitting lineages within each parent.
    keys = ["compound_id", "lineage_group", *SOURCE]
    lineage_values = rows.groupby(keys).residual.mean().unstack(SOURCE)
    records, agreement = [], []
    for compound, table in lineage_values.groupby(level="compound_id"):
        for left, right in combinations(table.columns, 2):
            common = table[[left, right]].dropna()
            count = len(common)
            if count < 20 or not common.nunique().gt(1).all():
                continue
            ranks = common.rank()
            rho = float(ranks.corr().iloc[0, 1])
            records.append({"compound_id": compound, "lineages": count, "left": list(left), "right": list(right),
                            "rho": rho, "shrunk_rho": rho * count / (count + 40)})
            values = normal_score(ranks, count)
            similarity = 1 / (1 + .5 * (values[left] - values[right]).pow(2))
            for source in (left, right):
                agreement.append(similarity.rename("similarity").reset_index().assign(
                    **dict(zip(SOURCE, source)),
                ))
    if not records:
        return np.ones(len(rows)), {"pairs": [], "supported_drugs": [], "matched_training_rows": 0}
    known = pd.DataFrame(records).groupby("compound_id").shrunk_rho.mean()
    drug_weight = .65 + .35 * known.clip(lower=0)
    by_lineage = pd.concat(agreement).groupby(keys).similarity.mean()
    observation_weight = by_lineage.reindex(pd.MultiIndex.from_frame(rows[keys]))
    weights = rows.compound_id.map(drug_weight).fillna(1).to_numpy() * observation_weight.fillna(1).to_numpy()
    return weights, {"pairs": records, "supported_drugs": sorted(known.index),
                     "matched_training_rows": int(observation_weight.notna().sum()),
                     "weight_quantiles": np.quantile(weights, [0, .25, .5, .75, 1]).tolist()}


def predicted_values(fit, roster, rna, edges, targets, auxiliary, device):
    from .structured_kernel import cell_kernels, drug_kernels, solve_residuals, training_grid

    _validate(fit, roster)
    _, anchor, native, _ = panel_targets(fit)
    fit_rna = rna.loc[sorted(fit.cell_id.unique())]
    query_rna = rna.loc[sorted(roster.cell_id.unique())]
    cells, _ = cell_kernels(fit_rna, query_rna, edges, auxiliary, device=device)
    compounds, drugs = drug_kernels(native, targets, device=device)
    cell, cross = cells["full"]
    drug = drugs["structured"]
    anchors = roster.compound_id.map(anchor).to_numpy()
    q = query_rna.index.get_indexer(roster.cell_id), compounds.get_indexer(roster.compound_id)
    if any((index < 0).any() for index in q) or not np.isfinite(anchors).all():
        raise ValueError("Unmatched query for the consensus kernel")
    rank = rank_residuals(native)
    weighting, diagnostics = confidence_weights(native)
    estimates = {}
    for arm, rows, row_weight in ((CENTERED, native, None), ("normal_rank", rank, None), (PRIMARY, rank, weighting)):
        weight, response = training_grid(rows, fit_rna.index, compounds, device=device, row_weight=row_weight)
        alpha, iterations, error = solve_residuals(cell, drug, weight, response, PENALTY)
        fitted_coef = alpha @ drug
        fitted = cell @ fitted_coef
        center = (weight * fitted).sum(dim=0) / weight.sum(dim=0).clamp_min(1e-8)
        estimates[arm] = anchors + (cross @ fitted_coef - center).cpu().numpy()[q]
        diagnostics[arm] = {"solver_steps": iterations, "solver_residual": error}
    return estimates, diagnostics


def predict(args):
    import torch

    from .v0plus_neural import load_auxiliary

    if not torch.cuda.is_available():
        raise RuntimeError("Registered assay concordance models require CUDA")
    earlier = args.baseline / "score.json"
    if sha256(earlier) != PREVIOUS_SCORE:
        raise ValueError("Unregistered baseline score")
    old, clean, rna, _ = registered_inputs(args)
    _, fit, check = same_split(old, clean, args.fold, args.assay)
    edges, targets, priors = molecular_inputs(args.priors, args.hallmark, args.targets)
    auxiliary, aux_receipt = load_auxiliary(args.auxiliary)
    roster = validation_roster(check)
    estimates, fit_diagnostics = predicted_values(fit, roster, rna, edges, targets, auxiliary, torch.device("cuda"))
    fresh = pd.concat([roster[KEY].assign(method=name, predicted_value=value) for name, value in estimates.items()])
    fresh = with_assay(fresh, check, args.fold)
    path = args.baseline / "forecasts" / f"{args.assay}-{args.fold}.parquet"
    stored, receipt = verify_forecast(path)
    if receipt["sha256"] != json.loads(earlier.read_text())["forecast_sha256"][path.stem]:
        raise ValueError("Unregistered corrected baseline predictions")
    index = [*KEY, "track", "fold"]
    prior = stored[stored.method.isin(CORE)]
    joint = pd.concat([prior, fresh], ignore_index=True)
    matrix = joint.pivot(index=index, columns="method", values="predicted_value")
    if not np.isfinite(matrix.to_numpy()).all():
        raise ValueError("Expected all comparators on the same corrected outcome rows")
    matrix[BLEND] = (matrix[PRIMARY] + matrix.corrected_ridge) / 2
    matrix[CORE_BLEND] = (matrix.corrected_core + matrix.corrected_ridge) / 2
    all_forecasts = matrix.reset_index().melt(id_vars=index, var_name="method", value_name="predicted_value")
    freeze_table(all_forecasts, args.out, provenance={
        "assay": args.assay, "fold": args.fold, "core_receipts": RECEIPTS, "fit_sources": sorted(fit.source_id.unique()),
        "baseline_sha256": receipt["sha256"], "fit_rows": len(fit), "query_rows": len(check),
        "priors": priors, "auxiliary": aux_receipt, "diagnostics": fit_diagnostics,
    })


def contrasts(frame, by_fold):
    ranking, _ = score(frame, endpoint_direction="higher", rank_only=True,
                       comparisons=(*CONTRASTS, (PRIMARY, "corrected_drug_mean")))
    drug_scores, _ = score_across_cells(frame)
    by_drug = drug_scores.groupby(["track", "method"], as_index=False)[["spearman", "pearson"]].mean()
    labelled = drug_scores.assign(agreement_support=[
        drug in by_fold[fold] for drug, fold in zip(drug_scores.compound_id, drug_scores.fold)
    ])
    grouped = labelled.groupby(["track", "agreement_support", "method"], as_index=False)
    strata = grouped.agg(spearman=("spearman", "mean"), pearson=("pearson", "mean"),
                         eligible_drug_folds=("spearman", "count"))
    intervals = [{"candidate": a, "baseline": b, "ci95": paired_intervals(frame, a, b)}
                 for a, b in CONTRASTS]
    gate = {}
    for track in frame.track.unique():
        rank = [row for row in ranking["paired_contrasts"] if row["track"] == track and row["candidate"] == PRIMARY
                and row["metric"] in ("spearman", "ndcg_at_5")]
        vs_core = [row["ci95"][0] for row in rank if row["baseline"] == "corrected_core"]
        means = pd.DataFrame(ranking["scores"]).query("track == @track").set_index("method")
        endpoints = ["macro_spearman", "macro_ndcg_at_5"]
        over_mean = means.loc[PRIMARY, endpoints] >= means.loc["corrected_drug_mean", endpoints]
        gate[track] = {
            "within_drug_positive": all(value[0] > 0 for value in intervals[0]["ci95"].values()),
            "within_cell_noninferior": len(vs_core) == 2 and min(vs_core) > -.005,
            "within_cell_above_mean": bool(over_mean.all()),
        }
    return {"within_drug": by_drug.to_dict("records"), "within_cell": ranking["scores"],
            "within_drug_ci95": intervals, "within_cell_contrasts": ranking["paired_contrasts"],
            "agreement_support_strata": strata.astype(object).where(strata.notna(), None).to_dict("records"),
            "gate": gate}


def evaluate(args):
    old, clean, _, _ = registered_inputs(args)
    adult = set(clean.loc[clean.lineage_role.eq("adult_hcc"), "cell_id"])
    modes, seals, provenance = {}, {}, {}
    for assay in ASSAYS:
        parts, support = [], {}
        for fold in range(3):
            path = args.forecasts / f"{assay}-{fold}.parquet"
            frame, seal = verify_forecast(path)
            record = seal["provenance"]
            if (record["assay"], record["fold"], record["core_receipts"]) != (assay, fold, RECEIPTS):
                raise ValueError("Unexpected frozen development split")
            _, _, check = same_split(old, clean, fold, assay)
            keys = [*KEY, "track"]
            expected = set(observations(check).set_index(keys).index)
            if set(frame.method) != METHODS or any(
                    len(part) != len(expected) or set(part.set_index(keys).index) != expected
                    for _, part in frame.groupby("method")):
                raise ValueError("Methods must cover the same registered corrected outcomes")
            seals[path.stem], provenance[path.stem] = seal["sha256"], record
            support[fold] = set(record["diagnostics"]["supported_drugs"])
            parts.append(attach(frame, check))
        scored = pd.concat(parts, ignore_index=True)
        modes[assay] = contrasts(scored, support)
        modes[assay]["adult_hcc_fold_centered"] = fold_centered_correlations(scored[scored.cell_id.isin(adult)])
    gate = all(all(checks.values()) for assay in ASSAYS[1:] for checks in modes[assay]["gate"].values())
    publish_json(args.out, {"modes": modes, "forecast_sha256": seals, "provenance": provenance, "promoted": gate})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action, options in (
            ("predict", ("baseline", "priors", "hallmark", "targets", "auxiliary")), ("score", ("forecasts",))):
        command = commands.add_parser(action)
        for name in ("original", "core", "out", *options):
            command.add_argument("--" + name, type=Path, required=True)
        if action == "predict":
            command.add_argument("--assay", choices=ASSAYS, required=True)
            command.add_argument("--fold", type=int, choices=range(3), required=True)
    arguments = parser.parse_args()
    {"predict": predict, "score": evaluate}[arguments.action](arguments)


if __name__ == "__main__":
    main()
