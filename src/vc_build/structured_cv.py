"""Seal grouped source-only forecasts and score held lineages and assays separately."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import GroupKFold

from .evaluate import score, score_across_cells
from .external import KEY, SOURCE, _validate, native_sensitivity
from .io import freeze_table, publish_json, sha256, verify_forecast
from .v0plus_cv import (
    GAMMAS, candidate_forecasts, choose, molecular_inputs, panel_targets, settings_for,
    summarize, training_inputs, validation_roster,
)
from .v0plus_score import correlations

ASSAYS = ("pooled", "CTRPv2", "GDSC2", "PRISM20Q2")
PENALTIES = (4.0, 16.0, 64.0)
STRUCTURAL_ARMS = ("structured", "no_drug_sharing", "rna_chem_only")
ARCHIVED_ARMS = {
    "linear": ("cv-778c6fa", ("drug_mean", "drug_ridge")),
    "cellhit": ("cv-2c90b62", ("cellhit",)),
    "v0plus_inductive": ("cv-7924f1a", ("v0plus_inductive",)),
}


def split(train, fold, assay):
    fit_index, check_index = list(GroupKFold(n_splits=3).split(train, groups=train.lineage_group))[fold]
    fit, check = train.iloc[fit_index], train.iloc[check_index]
    if assay != "pooled":
        fit, check = fit[fit.source_id.ne(assay)], check[check.source_id.eq(assay)]
    return fit, check[check.compound_id.isin(fit.compound_id)]


def with_assay(prediction, check, fold):
    identity = ["cell_id", "lineage_group", "compound_id"]
    expected = check[[*identity, *SOURCE]].drop_duplicates()
    return prediction.merge(expected, on=identity, validate="many_to_many").assign(
        track=lambda frame: frame.source_id + "/" + frame.endpoint_id, fold=fold,
    ).drop(columns=SOURCE)


def observations(check):
    return check.assign(
        row_id=check.cell_id + "|" + check.compound_id,
        track=check.source_id + "/" + check.endpoint_id,
        endpoint_value=native_sensitivity(check),
    )[[*KEY, "track", "endpoint_value"]]


def attach(prediction, check):
    truth = observations(check)
    joined = prediction.merge(truth, on=[*KEY, "track"], validate="many_to_one", how="left")
    if joined.endpoint_value.isna().any():
        raise ValueError("Forecast identities differ from source observations")
    return joined


def forecasts(fit, roster, rna, edges, targets, auxiliary, device, settings=None):
    from .structured_kernel import KernelGrid

    _validate(fit, roster)
    _, anchor, rows, calibration = panel_targets(fit)
    fit_rna = rna.loc[sorted(fit.cell_id.unique())]
    query_rna = rna.loc[sorted(roster.cell_id.unique())]
    grid = KernelGrid.prepare(rows, fit_rna, query_rna, edges, targets, auxiliary, device=device)
    base = roster.compound_id.map(anchor).to_numpy()
    parts, solve = [], {}
    for arm, penalty, scales in (
        [(arm, penalty, GAMMAS) for arm in STRUCTURAL_ARMS for penalty in PENALTIES] if settings is None
        else [(arm, penalty, (scale,)) for arm, penalty, _, scale in settings]
    ):
        residual, diagnostic = grid.forecast(roster, arm, penalty)
        solve[f"{arm}/{penalty}"] = diagnostic
        for scale in scales:
            parts.append(roster[KEY].assign(method=f"{arm}/{penalty}/0/{scale}",
                                            predicted_value=base + scale * residual))
    return pd.concat(parts, ignore_index=True), {
        "solver": solve, "auxiliary_features": grid.auxiliary_features, "calibration": calibration,
    }


def ridge_candidates(fit, roster, rna, edges, targets, selected=None):
    if selected is None:
        selected = [(*item, scale) for item in settings_for("linear")
                    if item[0] == "drug_ridge" for scale in GAMMAS]
        selected.append(("drug_mean", 0.0, 256, 1.0))
    return candidate_forecasts("linear", fit, roster, rna, edges, targets, selected=selected)[0]


def archived(base, fold, part):
    parts, hashes = [], {}
    for arm, (folder, methods) in ARCHIVED_ARMS.items():
        root = base / folder / arm / f"fold{fold}"
        prediction, receipt = verify_forecast(root / "forecast.parquet")
        selected = prediction[prediction.method.isin(methods)]
        key = ["row_id", "track", "fold"]
        indexed = part.set_index(key).index
        for name in methods:
            observed = selected[selected.method.eq(name)].set_index(key).index
            if set(indexed) != set(observed):
                raise ValueError(f"Archived {name} used a different pooled source split")
        ids = part[[*KEY, "track", "fold"]]
        parts.append(selected.merge(ids, on=key, validate="many_to_one"))
        hashes[arm] = receipt["sha256"]
    return parts, hashes


def predict(args):
    import torch

    from .v0plus_neural import load_auxiliary

    if not torch.cuda.is_available():
        raise RuntimeError("Structured kernel CV requires the requested G1 CUDA device")
    device = torch.device("cuda")
    train, rna, _ = training_inputs(args.bundle)
    edges, targets, prior = molecular_inputs(args.priors, args.hallmark, args.targets)
    auxiliary, aux_receipt = load_auxiliary(args.auxiliary)
    fit, check = split(train, args.fold, args.assay)
    inner = []
    for index, (a, b) in enumerate(GroupKFold(n_splits=2).split(fit, groups=fit.lineage_group)):
        inner_fit, inner_check = fit.iloc[a], fit.iloc[b]
        inner_check = inner_check[inner_check.compound_id.isin(inner_fit.compound_id)]
        roster = validation_roster(inner_check)
        forecast, _ = forecasts(inner_fit, roster, rna, edges, targets, auxiliary, device)
        if args.assay != "pooled":
            forecast = pd.concat([forecast, ridge_candidates(inner_fit, roster, rna, edges, targets)])
        inner.append(attach(with_assay(forecast, inner_check, index), inner_check))
    fitted = choose(summarize(pd.concat(inner, ignore_index=True)))
    roster = validation_roster(check)
    qualified = [item for item in fitted if item[0] in STRUCTURAL_ARMS]
    prediction, receipt = forecasts(fit, roster, rna, edges, targets, auxiliary, device, settings=qualified)
    prediction.method = prediction.method.str.split("/").str[0]
    parts = [with_assay(prediction, check, args.fold)]
    prior_forecasts = {}
    if args.assay == "pooled":
        prior_parts, prior_forecasts = archived(args.archive, args.fold, parts[0].query("method == 'structured'"))
        parts.extend(prior_parts)
    else:
        baseline = ridge_candidates(fit, roster, rna, edges, targets,
                                    selected=[item for item in fitted if item[0] in {"drug_ridge", "drug_mean"}])
        baseline.method = baseline.method.str.split("/").str[0]
        parts.append(with_assay(baseline, check, args.fold))
    predictions = pd.concat(parts, ignore_index=True)
    average = (predictions[predictions.method.isin(["structured", "drug_ridge"])]
               .groupby([*KEY, "track", "fold"], as_index=False).predicted_value.mean()
               .assign(method="ridge_structured_average"))
    predictions = pd.concat([predictions, average], ignore_index=True)
    if not np.isfinite(predictions.predicted_value).all():
        raise ValueError("Methods require identical supported source observations")
    receipt |= {
        "assay": args.assay, "fold": args.fold, "training_rows": len(fit),
        "training_sources": sorted(fit.source_id.unique()), "scoring_rows": len(check),
        "compounds": int(check.compound_id.nunique()), "lineages": int(check.lineage_group.nunique()),
        "inner_settings": fitted, "priors": prior, "auxiliary_receipt": aux_receipt,
        "archive_forecasts_sha256": prior_forecasts, "train_receipt_sha256": sha256(args.bundle / "receipt.json"),
    }
    freeze_table(predictions[[*KEY, "track", "fold", "method", "predicted_value"]],
                 args.out, provenance=receipt)


def paired_intervals(frame, candidate, baseline, *, repeats=800):
    """Pair lineage draws within the fold and sample eligible drugs within each assay."""
    rng = np.random.default_rng(629137)
    draws = []
    selected = frame[frame.method.isin([candidate, baseline])]
    for fold, block in selected.groupby("fold", sort=True):
        names = sorted(block.lineage_group.unique())
        sample = rng.integers(len(names), size=(repeats, len(names)))
        for _, assay in block.groupby("track", sort=True):
            actual = assay.groupby(["lineage_group", "compound_id"]).endpoint_value.mean().unstack().reindex(names)
            by_method = [assay[assay.method.eq(name)].groupby(["lineage_group", "compound_id"])
                         .predicted_value.mean().unstack().reindex(index=names, columns=actual.columns)
                         for name in (candidate, baseline)]
            present = actual.notna().sum().ge(5) & actual.std().gt(1e-8)
            eligible = np.flatnonzero(present.to_numpy())
            if not len(eligible):
                raise ValueError(f"No drugs support paired resampling in fold {fold}")
            drug_draw = rng.choice(eligible, size=(repeats, len(eligible)))
            matrix = []
            for start in range(0, repeats, 32):
                ix = sample[start:start + 32]
                drug_ix = drug_draw[start:start + 32]
                y, p, q = [table.to_numpy()[ix] for table in [actual, *by_method]]
                ranking = [rankdata(array, axis=1, nan_policy="omit") for array in (y, p, q)]
                pairs = [(y, p, q), ranking]
                changes = [
                    np.nanmean(np.take_along_axis(correlations(a, b) - correlations(a, c), drug_ix, axis=1), axis=1)
                    for a, b, c in pairs
                ]
                matrix.append(np.stack(changes, axis=1))
            draws.append(np.concatenate(matrix))
    intervals = np.nanquantile(np.stack(draws).mean(axis=0), [0.025, 0.975], axis=0).T
    return dict(zip(("pearson", "spearman"), intervals.tolist()))


def summarize_scores(frame):
    all_methods = set(frame.method)
    pairs = [(candidate, baseline)
             for candidate in ("structured", "ridge_structured_average")
             for baseline in ("drug_mean", "drug_ridge", "cellhit") if baseline in all_methods]
    scores, _ = score(frame, endpoint_direction="higher", rank_only=True, comparisons=pairs)
    drugs, _ = score_across_cells(frame)
    within_drug = (drugs.groupby(["track", "method"])[["spearman", "pearson"]].mean()
                   .rename(columns=lambda col: "within_drug_" + col).reset_index())
    results = pd.DataFrame(scores["scores"]).merge(within_drug, on=["track", "method"], validate="1:1")
    metrics = [key for key in results if key.startswith(("within_drug_", "macro_"))]
    summary = results.groupby("method")[metrics].mean().reset_index()
    result = {
        "table": summary.to_dict("records"), "by_source": results.to_dict("records"),
        "crossed_drug_intervals": [
            {"candidate": a, "baseline": b, "ci95": paired_intervals(frame, a, b)} for a, b in pairs
        ],
        "within_cell_intervals": scores["paired_contrasts"],
    }
    comparison = next(item["ci95"] for item in result["crossed_drug_intervals"]
                      if item["candidate"] == "structured" and item["baseline"] == "drug_ridge")
    cell_gate = [item["ci95"][0] for item in scores["paired_contrasts"]
                 if item["candidate"] == "structured" and item["baseline"] == "drug_mean"
                 and item["metric"] in {"ndcg_at_5", "spearman"}]
    result["gate"] = {
        "within_drug_spearman_vs_ridge_positive_interval": comparison["spearman"][0] > 0,
        "within_cell_both_positive_vs_mean_in_every_assay": (
            len(cell_gate) == 2 * frame.track.nunique() and min(cell_gate) > 0
        ),
    }
    return result


def score_frozen(args):
    train, _, _ = training_inputs(args.bundle)
    tables, hashes, support = {}, {}, {}
    for assay in ASSAYS:
        parts = []
        support[assay] = []
        for fold in range(3):
            path = args.forecasts / f"{assay}-{fold}.parquet"
            part, receipt = verify_forecast(path)
            locked = receipt["provenance"]
            if locked["assay"] != assay or locked["fold"] != fold or (
                locked["train_receipt_sha256"] != sha256(args.bundle / "receipt.json")
            ):
                raise ValueError("Wrong registered split or source bundle")
            _, check = split(train, fold, assay)
            expected = observations(check).drop(columns="endpoint_value")
            for _, method in part.groupby("method", sort=False):
                counts = method.merge(expected, on=[*KEY, "track"], validate="one_to_one")
                if not len(counts) == len(expected) == len(method):
                    raise ValueError("Forecasts did not retain the exact source assay roster")
            parts.append(attach(part, check))
            support[assay].append({key: locked[key] for key in (
                "fold", "training_rows", "scoring_rows", "compounds", "lineages", "inner_settings",
                "solver", "auxiliary_features",
            )})
            hashes[path.stem] = receipt["sha256"]
        tables[assay] = summarize_scores(pd.concat(parts, ignore_index=True))
    publish_json(args.out, {"tables": tables, "forecast_sha256": hashes, "support": support})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    forecast = sub.add_parser("predict")
    for key in ("bundle", "priors", "hallmark", "targets", "auxiliary", "archive", "out"):
        forecast.add_argument("--" + key, required=True, type=Path)
    forecast.add_argument("--fold", choices=range(3), type=int, required=True)
    forecast.add_argument("--assay", choices=ASSAYS, required=True)
    scoring = sub.add_parser("score")
    for key in ("bundle", "forecasts", "out"):
        scoring.add_argument("--" + key, required=True, type=Path)
    args = parser.parse_args()
    {"predict": predict, "score": score_frozen}[args.action](args)


if __name__ == "__main__":
    main()
