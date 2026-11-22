"""Calibrate an HCC drug anchor from other studies without changing patient effects."""

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata

from .evaluate import join_observations, paired_gain, score_across_cells
from .external import KEY
from .hcc_study_pool import STUDIES, data as pool_data, registered as registered_pool
from .io import freeze_table, publish_json, sha256, verify_forecast
from .limore_corrected import shared_support
from .tki_gr50 import PANEL, registration as registered_tki
from .tki_gr50_score import labels as gr50_labels
from .v0plus_score import crossed_patient_drug_ci
from .wong_expanded import checked as checked_wong, observations as wong_labels
from .yang_hcc_external import require_hash

POOL = "bd27132354dcf5537b717227099147514851ec0218d0c96920ec4c8a00a96a4c"
BASE_FORECAST = "d3cfcde3c4d4e71d29edacf1200176b2abbe5828b5ad0bddc7d266fc466f0616"
T4 = "082a0c426c7a0b40bcc1a306f75669098c7e213eda2f1be437f728832a3e07ec"
T4_FORECAST = "b375a08b3d974a802171f8ce92c2c4a3c2f2a12343735938825d73d785cdf3bb"
T4_TRUTH = "06d72dc855aa5570adb6cfe0a046334c8bb31cbcc0fad87e294180027d5a1d82"
WONG_EXPANDED = "f7ad39cd33ca3f945cd38a9a31793193a4a5a9f23ecb6f9b939ff0cdb4f70b90"
HOLDOUTS = (*STUDIES, "GR50")
ANCHOR = "2d_mean"
TEACHER = "2d_teacher"
MEAN = "3d_order_mean"
PRIMARY = "3d_order_teacher"
CONTINUOUS = "3d_continuous_teacher"
SHARED = {
    "3d_order_ridge": "2d_ridge",
    "3d_order_cellhit": "2d_cellhit",
    "3d_order_residual": "balanced_residual",
}
TEMPERATURE = .7
PRIOR = .05


def registered(args):
    if registered_pool(args.inputs) != POOL or registered_tki(args.tki / "inputs")[1] != T4:
        raise ValueError("Unexpected study registration")


def roster(args, study):
    path = args.tki / "inputs" / "query.parquet" if study == "GR50" else args.inputs / f"{study}.parquet"
    return pd.read_parquet(path)


def training_data(args, study, compounds):
    sources = []
    for name in STUDIES:
        if name == study:
            continue
        if study == "GR50" and name == "Wong":
            if checked_wong(args.expanded_wong, ("query.parquet",)) != WONG_EXPANDED:
                raise ValueError("Unexpected four-TKI Wong registration")
            query = pd.read_parquet(args.expanded_wong / "query.parquet")
            query = query[query.compound_id.isin(compounds)]
            frame = wong_labels(query, args.wong_book).assign(study=name)
            if len(frame) != 40:
                raise ValueError("Expected four independently measured Wong TKI")
        else:
            frame = pool_data(args, name, full=False)
            frame = frame[frame.compound_id.isin(compounds)]
        sources.append(frame[["study", "lineage_group", "compound_id", "endpoint_value"]])
    return pd.concat(sources, ignore_index=True)


def calibration_pairs(train, compounds):
    records = []
    sizes = {}
    for study, section in train.groupby("study"):
        outcomes = (section.groupby(["lineage_group", "compound_id"]).endpoint_value.mean()
                    .unstack("compound_id").reindex(columns=compounds))
        finite = outcomes.to_numpy()
        panel = finite[np.isfinite(finite)]
        scale = np.percentile(panel, 75) - np.percentile(panel, 25)
        if scale <= 1e-8:
            raise ValueError(f"Flat donor assay scale for {study}")
        sizes[study] = {"donors": len(outcomes), "drugs": int(outcomes.notna().any().sum()), "scale": float(scale)}
        for left, right in combinations(range(len(compounds)), 2):
            difference = (outcomes.iloc[:, left] - outcomes.iloc[:, right]).dropna().to_numpy()
            if len(difference) >= 5:
                records.append((left, right, study, len(difference),
                                float((.5 + .5 * np.sign(difference)).mean()),
                                float(np.median(difference) / scale)))
    evidence = pd.DataFrame(records, columns=["left", "right", "study", "donors", "win", "delta"])
    if evidence.empty:
        raise ValueError("No tenable pairwise anchor calibration")
    return evidence, sizes


def pairwise_fit(train, anchor):
    compounds = anchor.index
    evidence, supports = calibration_pairs(train, compounds)
    pairs = evidence.groupby(["left", "right"], as_index=False)[["win", "delta"]].mean()
    identity = np.eye(len(anchor))
    design = identity[pairs.left] - identity[pairs.right]
    center = anchor.median()
    scale = anchor.quantile(.75) - anchor.quantile(.25)
    if scale <= 1e-8:
        raise ValueError("The source drug anchor has no spread")
    prior = (anchor.to_numpy() - center) / scale
    wins = pairs.win.to_numpy()

    def objective(current):
        estimate = design @ current / TEMPERATURE
        penalty = current - prior
        value = (np.logaddexp(0., estimate) - wins * estimate).mean() + .5 * PRIOR * np.mean(penalty ** 2)
        grad = design.T @ (expit(estimate) - wins) / (len(pairs) * TEMPERATURE) + PRIOR * penalty / len(anchor)
        return value, grad

    out = minimize(objective, prior, jac=True, method="L-BFGS-B",
                   options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 300})
    if not out.success or not np.isfinite(out.x).all():
        raise ValueError(f"Ordinal anchor failed to converge: {out.message}")
    regularizer = PRIOR * len(pairs) / len(anchor)
    continuous = np.linalg.solve(design.T @ design + regularizer * identity,
                                 design.T @ pairs.delta.to_numpy() + regularizer * prior)
    return pd.DataFrame({"ordinal": center + scale * out.x, "continuous": center + scale * continuous},
                        index=compounds), {
        "training_support": supports, "pair_count": len(pairs), "iterations": int(out.nit),
        "pair_training_studies": {str(key): int(value) for key, value in evidence.groupby(
            ["left", "right"]).study.nunique().value_counts().items()},
        "paired_records_by_study": evidence.groupby("study").size().to_dict(),
        "anchor_scale": float(scale),
    }


def reference_columns(args, query, study):
    if study == "GR50":
        historical, seal = verify_forecast(args.tki / "final.parquet")
        expected = T4_FORECAST
        names = {"drug_mean": ANCHOR, "structured": TEACHER, "drug_ridge": "2d_ridge", "cellhit": "2d_cellhit",
                 "growth_adjusted": "growth_adjusted"}
    else:
        historical, seal = verify_forecast(args.old)
        historical = historical[historical.track.eq(study)]
        expected = BASE_FORECAST
        names = {name: name for name in (ANCHOR, TEACHER, "2d_ridge", "2d_cellhit", "balanced_residual")}
    if seal["sha256"] != expected:
        raise ValueError("Unexpected archived forecasts")
    historical = historical[historical.method.isin(names)].copy()
    shared_support(historical, query, names)
    matrix = historical.pivot(index=KEY, columns="method", values="predicted_value").rename(columns=names)
    return matrix.reset_index(), expected


def reanchor(table, anchors):
    fit = table.join(anchors, on="compound_id", validate="many_to_one")
    fit[MEAN] = fit.ordinal
    methods = {PRIMARY: TEACHER, **SHARED}
    for target, reference in methods.items():
        if reference in fit.columns:
            fit[target] = fit[reference] + (fit.ordinal - fit[ANCHOR])
    fit[CONTINUOUS] = fit[TEACHER] + (fit.continuous - fit[ANCHOR])
    for new, native in ((CONTINUOUS, TEACHER), *methods.items()):
        if new in fit:
            difference = (fit[new] - fit[native]).groupby(fit.compound_id)
            if (difference.max() - difference.min()).max() > 1e-10:
                raise ValueError("Changing an anchor affected within-drug patient residuals")
    return fit.drop(columns=["ordinal", "continuous"])


def predict(args):
    registered(args)
    if ((args.holdout == "LICOB" and args.licob_book is not None)
            or (args.holdout == "Wong" and args.wong_book is not None)):
        raise ValueError("Held labels were supplied to prediction")
    query = roster(args, args.holdout)
    baseline, hashref = reference_columns(args, query, args.holdout)
    train = training_data(args, args.holdout, set(query.compound_id))
    if args.holdout in set(train.study):
        raise ValueError("Held study entered anchor training")
    means = baseline.groupby("compound_id")[ANCHOR]
    if not means.nunique().eq(1).all():
        raise ValueError("Non-invariant 2D drug anchor")
    old = means.first().sort_index()
    anchors, info = pairwise_fit(train, old)
    estimates = reanchor(baseline, anchors)
    predicted = estimates.melt(id_vars=KEY, var_name="method", value_name="predicted_value")
    if not np.isfinite(predicted.predicted_value).all():
        raise ValueError("New anchor depends on an unmatched drug")
    freeze_table(predicted.assign(track=args.holdout, fold=0), args.out, provenance={
        "holdout": args.holdout, "pool_sha256": POOL, "gr50_sha256": T4, "prior_forecast_sha256": hashref,
        "diagnostics": info, "anchor_values": anchors.assign(old=old).to_dict("index"),
    })


def merge(args):
    registered(args)
    parts, hashes = [], {}
    for study in HOLDOUTS:
        rows, seal = verify_forecast(args.parts / f"{study}.parquet")
        p = seal["provenance"]
        if (p["holdout"], p["pool_sha256"], p["gr50_sha256"], p["prior_forecast_sha256"]) != (
                study, POOL, T4, T4_FORECAST if study == "GR50" else BASE_FORECAST):
            raise ValueError("Incompatible held study forecast")
        if set(p["diagnostics"]["training_support"]) != set(STUDIES) - {study}:
            raise ValueError("External study was not held out")
        shared_support(rows, roster(args, study), set(rows.method))
        parts.append(rows)
        hashes[study] = {"sha256": seal["sha256"], "diagnostics": p["diagnostics"], "anchor_values": p["anchor_values"]}
    freeze_table(pd.concat(parts, ignore_index=True), args.out, provenance={
        "pool_sha256": POOL, "gr50_sha256": T4, "components": hashes,
    })


def donor_scores(block):
    group = block.groupby(["lineage_group", "compound_id", "track", "fold", "method"], as_index=False)
    reduced = group[["predicted_value", "endpoint_value"]].mean()
    for method in (ANCHOR, MEAN):
        original = block[block.method.eq(method)].groupby("compound_id").predicted_value
        if not original.nunique().eq(1).all():
            raise ValueError("A constant anchor changed across patients")
        mask = reduced.method.eq(method)
        reduced.loc[mask, "predicted_value"] = reduced.loc[mask, "compound_id"].map(original.first())
    return reduced.assign(row_id=reduced.lineage_group + "|" + reduced.compound_id, cell_id=reduced.lineage_group)


def rank_rows(donors):
    from .evaluate import rank_metrics

    cells = []
    for tag, group in donors.groupby(["track", "fold", "lineage_group", "method"]):
        values, forecast = group.endpoint_value.to_numpy(), group.predicted_value.to_numpy()
        if len(values) <= 5:
            y, p = rankdata(values), rankdata(forecast)
            maximum = (y == y.max()), (p == p.max())
            rho = float(np.corrcoef(y, p)[0, 1]) if np.ptp(y) and np.ptp(p) else 0.0
            metrics = {"spearman": rho, "top_1": float((maximum[0] & maximum[1]).sum() /
                                                         (maximum[0].sum() * maximum[1].sum()))}
        else:
            metrics = rank_metrics(values, forecast, k=5)
        cells.append(dict(zip(("track", "fold", "lineage_group", "method"), tag)) | metrics)
    return pd.DataFrame(cells).assign(cell_id=lambda table: table.lineage_group)


def diagnostic_loo(donors):
    # Diagnostic only. It uses outcomes from other members of the evaluated study.
    cases = donors[donors.method.eq(TEACHER)].copy()
    source = donors[donors.method.eq(ANCHOR)].groupby("compound_id").predicted_value.first()
    source_iqr = source.quantile(.75) - source.quantile(.25)
    parts = []
    for held, query in cases.groupby("lineage_group"):
        avg = cases[cases.lineage_group.ne(held)].groupby("compound_id").endpoint_value.mean()
        iqr = avg.quantile(.75) - avg.quantile(.25)
        if iqr <= 1e-8:
            raise ValueError("Unable to scale a constant within-study diagnostic anchor")
        anchor = query.compound_id.map(source.median() + source_iqr * (avg - avg.median()) / iqr)
        residual = query.predicted_value - query.compound_id.map(source)
        parts.extend([query.assign(method="loo_observed_anchor", predicted_value=anchor),
                      query.assign(method="loo_observed_anchor_teacher", predicted_value=anchor + residual)])
    return rank_summary(rank_rows(pd.concat(parts, ignore_index=True)))


def rank_summary(cells):
    axes = [col for col in ("spearman", "ndcg_at_5", "top5_overlap", "top_1") if col in cells]
    return cells.groupby("method")[axes].mean().reset_index().to_dict("records")


def summarize(donors):
    methods = set(donors.method)
    references = (TEACHER, MEAN, "3d_order_ridge", "3d_order_cellhit", CONTINUOUS, "3d_order_residual")
    cells = rank_rows(donors)
    axes = [a for a in ("spearman", "ndcg_at_5", "top5_overlap", "top_1") if a in cells]
    paired = {ref: {a: paired_gain(cells, PRIMARY, ref, a) for a in axes} for ref in references if ref in methods}
    within, _ = score_across_cells(donors)
    macro = within.groupby("method")[["spearman", "pearson"]].mean().reset_index()
    crossings = {ref: crossed_patient_drug_ci(donors, PRIMARY, ref) for ref in references if ref in methods}
    rank_gate = ("ndcg_at_5" in axes and all(paired[TEACHER][a] is not None and paired[TEACHER][a]["ci95"][0] > 0
                                             for a in ("spearman", "ndcg_at_5")))
    individualized = ("ndcg_at_5" in axes and all(paired[MEAN][a] is not None and paired[MEAN][a]["ci95"][0] > 0
                                                  for a in ("spearman", "ndcg_at_5"))
                      and all(crossings[MEAN].get(a, [0])[0] > 0 for a in ("spearman", "pearson")))
    difference = macro.set_index("method").loc[[PRIMARY, TEACHER]].diff().iloc[-1].abs()
    if difference.max() > 1e-10:
        raise ValueError("Drug anchor changed a within-drug statistic")
    return {"n_donors": donors.lineage_group.nunique(), "n_drugs": donors.compound_id.nunique(),
            "within_drug": macro.to_dict("records"), "rank": rank_summary(cells),
            "rank_contrasts": paired, "drug_ci95": crossings,
            "rank_gate": rank_gate, "personalization_gate": individualized,
            "loo_observed_diagnostic": diagnostic_loo(donors)}


def score(args):
    registered(args)
    rows, seal = verify_forecast(args.forecast)
    p = seal["provenance"]
    if (p["pool_sha256"], p["gr50_sha256"], set(p["components"])) != (POOL, T4, set(HOLDOUTS)):
        raise ValueError("Unexpected merged model forecast")
    results = {}
    for study in HOLDOUTS:
        if study == "GR50":
            path = args.tki / "table2.tsv"
            require_hash(path, T4_TRUTH)
            truth = gr50_labels(roster(args, study), path)[0]["GR50"]
        else:
            truth = pool_data(args, study, full=True)
        block = join_observations(rows[rows.track.eq(study)], truth)
        results[study] = summarize(donor_scores(block))
    gates = [results[s] for s in ("LICOB", "Wong")]
    publish_json(args.out, {"forecast_sha256": seal["sha256"], "scopes": results,
                           "rank_gate": all(q["rank_gate"] for q in gates),
                           "joint_gate": all(q["rank_gate"] and q["personalization_gate"] for q in gates),
                           "fit_anchors": p["components"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    tasks = parser.add_subparsers(dest="stage", required=True)
    fields = {"predict": ("old", "transfer", "expanded-wong"), "merge": ("parts",), "score": ("forecast", "transfer")}
    for stage, args in fields.items():
        sub = tasks.add_parser(stage)
        for path in ("inputs", "tki", "out", *args):
            sub.add_argument("--" + path, type=Path, required=True)
        if stage != "merge":
            sub.add_argument("--licob-book", type=Path)
            sub.add_argument("--wong-book", type=Path)
        if stage == "predict":
            sub.add_argument("--holdout", choices=HOLDOUTS, required=True)
    args = parser.parse_args()
    {"predict": predict, "merge": merge, "score": score}[args.stage](args)


if __name__ == "__main__":
    main()
