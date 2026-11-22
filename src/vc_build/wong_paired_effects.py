"""Test a fixed additive decomposition for prospective held-donor response ordering."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import join_observations
from .external import KEY
from .hcc_3d_fewshot import clustered_sensitivity
from .io import freeze_table, publish_json, sha256, verify_forecast
from .v0plus_score import axis_metrics
from .wong_assay_bridge import SCOPE, FORECASTS, interaction
from .yang_hcc_external import require_hash

REGISTERED = "bb8066fbd52e0120f0ea8286b67f54a82cab3f33b86ca5635afdaefae3d7906f"
PRIMARY = "interaction_only"
ARMS = ("fit_3d_mean", "undivided", PRIMARY, "interaction_and_broad")
COMPARISONS = ((PRIMARY, "fit_3d_mean"), (PRIMARY, "undivided"), ("interaction_and_broad", PRIMARY))


def slope(x, y):
    energy = np.square(x).sum()
    return float((x * y).sum() / energy) if energy > 1e-12 else 0.


def effects(x, y, query):
    if not all(np.isfinite(a).all() for a in (x, y, query)) or (
        x.shape != y.shape or x.shape[1] != query.shape[1]
    ):
        raise ValueError("Need finite, complete and matched fit and query panels")
    xmean, ymean = x.mean(axis=0), y.mean(axis=0)
    x0, y0 = x - xmean, y - ymean
    held = query - xmean
    held_broad = held.mean(axis=1, keepdims=True)
    gains = {
        "shared": slope(x0, y0), "interaction": max(0., slope(interaction(x), interaction(y))),
        "broad": slope(x0.mean(axis=1), y0.mean(axis=1)),
    }
    selectivity = (held - held_broad) * gains["interaction"]
    values = (np.broadcast_to(ymean, held.shape), ymean + gains["shared"] * held,
              ymean + selectivity, ymean + selectivity + gains["broad"] * held_broad)
    return values, gains


def predict(args):
    receipt_path = args.inputs / "receipt.json"
    require_hash(receipt_path, REGISTERED)
    receipt = json.loads(receipt_path.read_text())
    if receipt["scope"] != SCOPE:
        raise ValueError("Wrong fit cohort")
    for filename, digest in receipt["files"].items():
        require_hash(args.inputs / filename, digest)
    roster = pd.read_parquet(args.inputs / "roster.parquet").sort_values(["cell_id", "compound_id"])
    tissue, seal = verify_forecast(args.tissue)
    if seal["sha256"] != FORECASTS["tissue"]:
        raise ValueError("The parental-tissue source prediction changed")
    tissue = tissue[tissue.method.eq(PRIMARY) & tissue.cell_id.isin(roster.cell_id)]
    roster = roster.merge(tissue[["cell_id", "compound_id", "predicted_value"]], on=["cell_id", "compound_id"],
                          how="left", validate="one_to_one")
    panels = {
        "measured_100": (roster, "measured_2d"),
        "measured_62": (roster.dropna(subset=["predicted_value"]), "measured_2d"),
        "tissue_rna_62": (roster.dropna(subset=["predicted_value"]), "predicted_value"),
    }
    records, gains = [], {}
    for track, (panel, field) in panels.items():
        matrix = panel.pivot(index="lineage_group", columns="compound_id", values=field)
        if matrix.shape != (8, 100 if track == "measured_100" else 62):
            raise ValueError("Paired assay comparison support changed")
        gains[track] = {}
        for number, fit in receipt["fit_groups"].items():
            labels = pd.read_parquet(args.inputs / f"fit-{number}.parquet")
            if set(labels.lineage_group) != set(fit) or len(fit) != 3:
                raise ValueError("Holdout 3D labels cannot enter prediction")
            target = labels.pivot(index="lineage_group", columns="compound_id", values="endpoint_value")
            held = matrix.index.difference(fit)
            x, y, query = [table.to_numpy() for table in (
                matrix.loc[fit], target.loc[fit, matrix.columns], matrix.loc[held])]
            values, gains[track][number] = effects(x, y, query)
            block = panel[panel.lineage_group.isin(held)][KEY]
            records.extend(block.assign(method=m, predicted_value=v.ravel(), track=track, fold=int(number))
                           for m, v in zip(ARMS, values))
    freeze_table(pd.concat(records, ignore_index=True), args.out, provenance={
        "fit_receipt_sha256": REGISTERED, "tissue_forecast_sha256": FORECASTS["tissue"], "gains": gains,
    })


def summarize(frame, gains):
    report, _, _ = axis_metrics(frame, COMPARISONS)
    intervals = clustered_sensitivity(frame, COMPARISONS)
    across = {m["baseline"]: m["ci95"] for m in intervals if m["candidate"] == PRIMARY}
    ranks = {(m["baseline"], m["metric"]): m["ci95"] for m in report["paired_contrasts"] if m["candidate"] == PRIMARY}
    gate = {c: all(across[c][metric][0] > 0 for metric in ("spearman", "pearson"))
            and all(ranks[c, metric][0] > -.005 for metric in ("spearman", "ndcg_at_5"))
            for c in ("fit_3d_mean", "undivided")}
    return {"report": report, "across_cell_ci95": intervals, "gate": gate,
        "patients": int(frame.lineage_group.nunique()), "drugs": int(frame.compound_id.nunique()),
        "folds": int(frame.fold.nunique()), "gain_min_median_max": {
            k: np.quantile([item[k] for item in gains.values()], [0, .5, 1]).tolist()
            for k in ("shared", "interaction", "broad")}}


def score(args):
    fit = args.inputs / "predict/receipt.json"
    require_hash(fit, REGISTERED)
    root = json.loads((args.inputs / "receipt.json").read_text())
    outcomes = args.inputs / "score_only.parquet"
    require_hash(outcomes, root["files"][outcomes.name])
    rows, seal = verify_forecast(args.forecast)
    p = seal["provenance"]
    if (p["fit_receipt_sha256"], p["tissue_forecast_sha256"]) != (REGISTERED, FORECASTS["tissue"]):
        raise ValueError("Incorrect frozen input for selectivity scoring")
    results = {}
    for name, frame in join_observations(rows, pd.read_parquet(outcomes)).groupby("track"):
        results[name] = summarize(frame, p["gains"][name])
    publish_json(args.out, {"input_sha256": sha256(args.inputs / "receipt.json"),
        "forecast_sha256": seal["sha256"], "results": results,
        "main_measured_gate": all(results["measured_100"]["gate"].values()),
        "tissue_rna_gate": all(results["tissue_rna_62"]["gate"].values())})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    for stage, fields in {"predict": ("inputs", "tissue"), "score": ("inputs", "forecast")}.items():
        p = sub.add_parser(stage)
        for field in (*fields, "out"):
            p.add_argument("--" + field, type=Path, required=True)
    args = parser.parse_args()
    {"predict": predict, "score": score}[args.stage](args)


if __name__ == "__main__":
    main()
