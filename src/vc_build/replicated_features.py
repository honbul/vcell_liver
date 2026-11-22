"""Test feature-level replication across matched assay lineages."""

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .assay_consensus_cv import CORE, PREVIOUS_SCORE
from .evaluate import score_across_cells
from .expanded_panel_cv import RECEIPTS, registered_inputs, same_split
from .external import KEY, SOURCE, _validate
from .hcc_conditional_cv import fold_centered_correlations
from .io import freeze_table, publish_json, sha256, verify_forecast
from .pdo_transfer_cv import device_cuda
from .structured_cv import ASSAYS, attach, observations, paired_intervals, with_assay
from .v0plus_cv import molecular_inputs, panel_targets, validation_roster
from .v0plus_features import ExpressionFeatures
from .v0plus_score import axis_metrics

CANDIDATE = "replicated"
ABLATION = "matched_pooled"
BASELINE = "corrected_core"
ANCHOR = "corrected_drug_mean"
FRESH = (CANDIDATE, ABLATION)
METHODS = (*CORE, *FRESH)
COMPARISONS = ((CANDIDATE, BASELINE), (CANDIDATE, ABLATION), (CANDIDATE, ANCHOR))
PANEL_LIMIT = 128
MIN_REPEAT = 30
ALPHA = 100.


def unit_columns(values):
    values = values - values.mean(dim=0)
    return values / values.norm(dim=0).clamp_min(1e-8)


def contrast_scores(feature, ranks, *, pair_weights):
    """Match the data and feature count across consensus and replication."""
    import torch

    features = unit_columns(feature).T
    left, right = (features @ ranks).T
    consistent = torch.where(
        left * right > 0,
        left.sign() * torch.minimum(left.abs(), right.abs()),
        torch.zeros_like(left),
    )
    pooled = (features @ unit_columns(ranks.mean(dim=1, keepdim=True))).flatten()
    return pair_weights * torch.stack((consistent, pooled))


def choose_features(outcomes, traits, lineage, device):
    import torch

    statistic = torch.zeros((2, traits.shape[1]), device=device)
    pairs = []
    for left, right in combinations(outcomes.columns, 2):
        common = outcomes[[left, right]].dropna()
        count = len(common)
        if count < MIN_REPEAT or not common.nunique().gt(1).all():
            continue
        index = lineage.get_indexer(common.index)
        if (index < 0).any():
            raise ValueError("Assay lineages are missing registered RNA")
        ranks = rankdata(common.to_numpy(), axis=0).astype(np.float32)
        # Equalize rank variance for differing tied assay values before combining ranks.
        y = unit_columns(torch.as_tensor(ranks, device=device))
        statistic += contrast_scores(traits[index], y, pair_weights=count - 3)
        pairs.append({"left": list(left), "right": list(right), "lineages": count})
    signals = statistic[0].abs()
    n = min(int(signals.gt(0).sum().item()), PANEL_LIMIT)
    columns = statistic.abs().argsort(dim=1, descending=True, stable=True)[:, :n]
    return columns, pairs


def predict_panel(fit, roster, rna, edges, targets, device):
    import torch

    _validate(fit, roster)
    _, anchor, residuals, calibration = panel_targets(fit)
    fit_cells = sorted(fit.cell_id.unique())
    held_cells = sorted(roster.cell_id.unique())
    fit_rna = rna.loc[fit_cells]
    schema = ExpressionFeatures.fit(fit_rna, edges, targets, n_variable=768, rank=True)
    mapping = fit[["cell_id", "lineage_group"]].drop_duplicates().set_index("cell_id").lineage_group
    observed_rna = schema.transform(fit_rna).groupby(mapping).mean()
    query_rna = schema.transform(rna.loc[held_cells])
    lookup, device_query = query_rna.index, torch.as_tensor(query_rna.to_numpy(), device=device)
    traits = torch.as_tensor(observed_rna.to_numpy(), device=device)
    by_lineage = residuals.groupby(["compound_id", "lineage_group", *SOURCE], sort=True).residual.mean()
    rows = roster.assign(_anchor=roster.compound_id.map(anchor)).reset_index(drop=True)
    if not np.isfinite(rows["_anchor"]).all():
        raise ValueError("Query is missing a registered drug anchor")
    output = np.full((len(FRESH), len(rows)), np.nan)
    support = {}
    fit_sizes = []
    for compound, q in rows.groupby("compound_id"):
        measured = by_lineage.loc[compound]
        columns, pairs = choose_features(measured.unstack(SOURCE), traits, observed_rna.index, device)
        if not columns.shape[1]:
            continue
        identity = measured.index.get_level_values("lineage_group")
        f = observed_rna.index.get_indexer(identity)
        if (f < 0).any():
            raise ValueError("Measured assay is missing fit-only expression")
        n_assay = measured.groupby(level=SOURCE).transform("size")
        weight = len(measured) / (n_assay * measured.index.droplevel("lineage_group").nunique())
        w = torch.as_tensor(weight.to_numpy(dtype=np.float32), device=device)
        x = traits[f][:, columns].permute(1, 0, 2).contiguous()
        y = torch.as_tensor(measured.to_numpy(dtype=np.float32), device=device)
        x_bar = (x * w[None, :, None]).sum(dim=1) / w.sum()
        y_bar = (y * w).sum() / w.sum()
        z = x - x_bar[:, None, :]
        g = z.transpose(1, 2) @ (w[None, :, None] * z)
        rhs = z.transpose(1, 2) @ (w * (y - y_bar))[:, None]
        coef = torch.linalg.solve(g + ALPHA * torch.eye(columns.shape[1], device=device), rhs).squeeze(-1)
        query_index = lookup.get_indexer(q.cell_id)
        if (query_index < 0).any():
            raise ValueError("Missing registered query RNA")
        held = device_query[query_index][:, columns].permute(1, 0, 2)
        forecast = ((held - x_bar[:, None, :]) * coef[:, None, :]).sum(dim=2) + y_bar
        output[:, q.index] = q["_anchor"].to_numpy() + forecast.cpu().numpy()
        chosen, control = columns.cpu().numpy()
        support[compound] = {"pairs": pairs, "selected_features": len(chosen),
                             "features_shared_with_control": len(np.intersect1d(chosen, control))}
        fit_sizes.append(len(measured))
    diagnostics = {"support": support, "fit_rna_features": len(schema.columns),
                   "median_fit_lineage_assay_rows": float(np.median(fit_sizes)) if fit_sizes else None,
                   "calibration": calibration}
    return output, diagnostics


def frozen_baseline(path, assay, fold):
    summary = path / "score.json"
    if sha256(summary) != PREVIOUS_SCORE:
        raise ValueError("Unexpected corrected development score")
    fname = path / "forecasts" / f"{assay}-{fold}.parquet"
    rows, receipt = verify_forecast(fname)
    if receipt["sha256"] != json.loads(summary.read_text())["forecast_sha256"][fname.stem]:
        raise ValueError("Unexpected corrected development freeze")
    return rows[rows.method.isin(CORE)], receipt["sha256"]


def predict(args):
    old, corrected, rna, _ = registered_inputs(args)
    _, fit, check = same_split(old, corrected, args.fold, args.assay)
    check = check[[*KEY, *SOURCE, "isomeric_smiles"]]
    roster = validation_roster(check)
    edges, targets, priors = molecular_inputs(args.priors, args.hallmark, args.targets)
    fitted, diagnostics = predict_panel(fit, roster, rna, edges, targets, device_cuda())
    baseline, baseline_sha = frozen_baseline(args.baseline, args.assay, args.fold)
    index = [*KEY, "track", "fold"]
    fresh = with_assay(pd.concat([roster[KEY].assign(method=arm, predicted_value=fitted[i])
                                  for i, arm in enumerate(FRESH)], ignore_index=True), check, args.fold)
    wide = pd.concat([fresh, baseline], ignore_index=True).pivot(
        index=index, columns="method", values="predicted_value")
    if set(wide.columns) != set(METHODS) or not np.isfinite(wide[list(CORE)].to_numpy()).all():
        raise ValueError("Methods must cover the same registered validation rows")
    eligible = wide.index.get_level_values("compound_id").isin(diagnostics["support"])
    if (not np.isfinite(wide.loc[eligible, list(FRESH)].to_numpy()).all()
            or wide.loc[~eligible, list(FRESH)].notna().any(axis=None)):
        raise ValueError("Shared replication support changed across both feature selections")
    for arm in FRESH:
        wide[arm] = .5 * (wide[arm].fillna(wide[BASELINE]) + wide[BASELINE])
    freeze_table(wide.reset_index().melt(id_vars=index, var_name="method", value_name="predicted_value"),
                 args.out, provenance={"assay": args.assay, "fold": args.fold, "core_receipts": RECEIPTS,
                     "fit_sources": sorted(fit.source_id.unique()), "baseline_sha256": baseline_sha,
                     "fit_rows": len(fit), "query_rows": len(check), "priors": priors,
                     "diagnostics": diagnostics})


def assess(whole, supported):
    metrics, _, _ = axis_metrics(whole, COMPARISONS)
    exact, _ = score_across_cells(supported)
    summary = (exact.groupby(["track", "method"], as_index=False)
               .agg(spearman=("spearman", "mean"), pearson=("pearson", "mean"), drug_folds=("pearson", "count")))
    eligible = exact[exact.method.eq(CANDIDATE)].dropna(subset=["spearman", "pearson"])
    counts = {track: {str(fold): int(n) for fold, n in part.groupby("fold").compound_id.nunique().items()}
              for track, part in eligible.groupby("track")}
    enough = (set(counts) == set(whole.track.unique())
              and all(len(item) == 3 and min(item.values()) >= 5 for item in counts.values()))
    ci = {
        name: paired_intervals(supported, CANDIDATE, name) for name in (BASELINE, ABLATION)
    } if enough else {}
    ranks = [c["ci95"][0] for c in metrics["paired_contrasts"] if c["baseline"] == BASELINE
             and c["metric"] in ("spearman", "ndcg_at_5")]
    means = pd.DataFrame(metrics["scores"]).set_index(["method", "track"])
    axes = ["macro_spearman", "macro_ndcg_at_5"]
    above_mean = bool((means.loc[CANDIDATE, axes] >= means.loc[ANCHOR, axes]).all(axis=None))
    lowers = [limits[0] for cmp in ci.values() for limits in cmp.values()]
    return {
        "full": metrics, "supported_within_drug": summary.astype(object).where(summary.notna(), None).to_dict("records"),
        "supported_ci95": ci, "eligible_drugs_per_fold": counts,
        "whole_within_drug_ci95": {base: paired_intervals(whole, CANDIDATE, base) for base in (BASELINE, ABLATION)},
        "gate": {"support": enough, "positive": len(lowers) == 4 and min(lowers) > 0,
                 "drug_noninferior": len(lowers) == 4 and min(lowers) > -.01,
                 "ranking_noninferior": len(ranks) == 2 * whole.track.nunique() and min(ranks) > -.005,
                 "rank_point_at_least_mean": above_mean},
    }


def evaluate(args):
    old, corrected, _, _ = registered_inputs(args)
    adult = set(corrected.loc[corrected.lineage_role.eq("adult_hcc"), "cell_id"])
    reports, freeze, fit_notes = {}, {}, {}
    for assay in ASSAYS:
        full, supported = [], []
        for fold in range(3):
            path = args.forecast / f"{assay}-{fold}.parquet"
            part, seal = verify_forecast(path)
            info = seal["provenance"]
            if (info["assay"], info["fold"], info["core_receipts"]) != (assay, fold, RECEIPTS):
                raise ValueError("Forecast and fixed split disagree")
            _, _, truth = same_split(old, corrected, fold, assay)
            keys = [*KEY, "track"]
            expect = observations(truth)[keys].set_index(keys).sort_index().index
            if set(part.method) != set(METHODS) or any(
                    not part_method.set_index(keys).sort_index().index.equals(expect)
                    for _, part_method in part.groupby("method")):
                raise ValueError("Missing common row or method in verified prediction")
            joined = attach(part, truth)
            support = info["diagnostics"]["support"]
            supported.append(joined[joined.compound_id.isin(support)])
            full.append(joined)
            freeze[path.stem], fit_notes[path.stem] = seal["sha256"], info
        full = pd.concat(full, ignore_index=True)
        reports[assay] = assess(full, pd.concat(supported, ignore_index=True))
        reports[assay]["adult_hcc_fold_centered"] = fold_centered_correlations(full[full.cell_id.isin(adult)])
    gates = [reports[assay]["gate"] for assay in ASSAYS[1:]]
    promoted = sum(gate["positive"] for gate in gates) >= 2 and all(
        all(gate[k] for k in ("support", "drug_noninferior", "ranking_noninferior", "rank_point_at_least_mean"))
        for gate in gates)
    publish_json(args.out, {"reports": reports, "prediction_sha256": freeze, "provenance": fit_notes,
                           "promoted": promoted})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    for stage, fields in {"predict": ("baseline", "priors", "targets", "hallmark"), "score": ("forecast",)}.items():
        sub = stages.add_parser(stage)
        for name in ("original", "core", "out", *fields):
            sub.add_argument("--" + name, type=Path, required=True)
        if stage == "predict":
            sub.add_argument("--assay", choices=ASSAYS, required=True)
            sub.add_argument("--fold", type=int, choices=range(3), required=True)
    args = parser.parse_args()
    {"predict": predict, "score": evaluate}[args.stage](args)


if __name__ == "__main__":
    main()
