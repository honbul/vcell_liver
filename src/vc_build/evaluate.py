"""Score one verified forecast against separately supplied observations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .io import publish_json, read_table, sha256, verify_forecast

IDENTITIES = ["cell_id", "lineage_group", "compound_id"]


def join_observations(predictions, outcomes):
    required = {"row_id", "endpoint_value"}
    observations = outcomes.rename(columns={"record_id": "row_id", "cell_group": "lineage_group"}).copy()
    if not required <= set(observations):
        raise ValueError(f"Observations lack {sorted(required - set(observations))}")
    if observations.row_id.isna().any() or observations.row_id.duplicated().any():
        raise ValueError("Observed row identifiers must be unique and present")
    forecasts = predictions.copy()
    forecasts["row_id"] = forecasts.row_id.astype(str)
    observations["row_id"] = observations.row_id.astype(str)
    observed_columns = list(required) + [key for key in IDENTITIES if key in observations]
    joined = forecasts.merge(
        observations[observed_columns], on="row_id", how="left", validate="many_to_one",
        indicator=True, suffixes=("", "_observed"),
    )
    if (joined["_merge"] != "both").any():
        raise ValueError("Every frozen identity must have a registered observation")
    for column in IDENTITIES:
        observed_column = column + "_observed"
        if observed_column in joined and not joined[column].astype(str).equals(joined[observed_column].astype(str)):
            raise ValueError(f"Forecast and observation {column} disagree")
    joined["endpoint_value"] = pd.to_numeric(joined.endpoint_value, errors="raise")
    if not np.isfinite(joined.endpoint_value).all():
        raise ValueError("Registered observations must be finite")
    joined["predicted_value"] = pd.to_numeric(joined.predicted_value, errors="raise")
    if np.isinf(joined.predicted_value).any():
        raise ValueError("Infinite forecasts are invalid")
    if not set(IDENTITIES) <= set(joined) or joined[IDENTITIES].isna().any().any():
        raise ValueError("Scoring requires cell, lineage and compound identity")
    return joined.drop(columns="_merge")


def rank_metrics(actual, forecast, k: int = 5) -> dict:
    """Rank correlation, rank-gain NDCG, and expected top-k identity overlap under ties."""
    y = rankdata(actual)
    p = rankdata(forecast)
    if len(y) < 2 or np.ptp(y) == 0:
        return {"spearman": None, "ndcg_at_5": None, "top5_overlap": None}
    if np.ptp(p) == 0:
        spearman = 0.0
    else:
        spearman = float(np.corrcoef(y, p)[0, 1])

    def top_probability(scores):
        cutoff = np.sort(scores)[-min(k, len(scores))]
        above = scores > cutoff
        equal = scores == cutoff
        return above + equal * ((min(len(scores), k) - above.sum()) / equal.sum())

    gain = (y - 1) / (len(y) - 1)
    discounts = 1 / np.log2(np.arange(2, min(len(y), k) + 2))
    ordered = np.argsort(-p, kind="stable")
    mean_gain = pd.Series(gain).groupby(p).transform("mean").to_numpy()
    dcg = np.dot(mean_gain[ordered[:k]], discounts)
    ideal = np.dot(np.sort(gain)[::-1][:k], discounts)
    overlap = np.dot(top_probability(y), top_probability(p)) / min(k, len(y))
    return {"spearman": spearman, "ndcg_at_5": float(dcg / ideal), "top5_overlap": float(overlap)}


def _cell_metrics(group, endpoint_direction, rank_only):
    actual = group.endpoint_value.to_numpy()
    forecast = group.predicted_value.to_numpy()
    direction = 1 if endpoint_direction == "higher" else -1
    metrics = rank_metrics(direction * actual, direction * forecast)
    if not rank_only:
        error = actual - forecast
        metrics["mae"] = float(np.mean(np.abs(error)))
        metrics["mse"] = float(np.mean(error ** 2))
    if {"interval_lower", "interval_upper"} <= set(group):
        lo = pd.to_numeric(group.interval_lower, errors="raise")
        hi = pd.to_numeric(group.interval_upper, errors="raise")
        valid = lo.notna() & hi.notna()
        if (lo[valid] > hi[valid]).any() or not np.isfinite(lo[valid]).all() or not np.isfinite(hi[valid]).all():
            raise ValueError("Malformed prediction interval")
        metrics["interval_fraction"] = float(np.mean(valid))
        if valid.any():
            metrics["interval_coverage"] = float(np.mean((lo[valid] <= actual[valid]) & (hi[valid] >= actual[valid])))
            metrics["interval_width"] = float(np.mean(hi[valid] - lo[valid]))
    return {"rows": len(group), **metrics}


def paired_gain(cells, candidate, baseline, metric, *, repeats=4000, seed=14822):
    """Resample paired lineage blocks after averaging cells within each lineage."""
    key = ["track", "fold", "cell_id", "lineage_group"]
    if metric not in cells:
        return None
    a = cells[cells.method == candidate].set_index(key)[metric].rename("candidate")
    b = cells[cells.method == baseline].set_index(key)[metric].rename("baseline")
    paired = pd.concat([a, b], axis=1, join="inner").dropna()
    groups = paired.groupby(level="lineage_group").mean()
    if len(groups) < 2:
        return None
    values = groups.to_numpy()
    rng = np.random.default_rng(seed)
    estimate = values.mean(axis=0)
    samples = values[rng.integers(len(values), size=(repeats, len(values)))].mean(axis=1)
    if metric == "mae":
        if estimate[1] <= 0 or (samples[:, 1] <= 0).any():
            return None
        gain = (estimate[1] - estimate[0]) / estimate[1]
        draws = (samples[:, 1] - samples[:, 0]) / samples[:, 1]
    else:
        gain = estimate[0] - estimate[1]
        draws = samples[:, 0] - samples[:, 1]
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"gain": float(gain), "ci95": [float(low), float(high)], "lineages": len(groups)}


def score(joined, *, endpoint_direction, rank_only=False, comparisons=()):
    keys = ["track", "fold", "cell_id", "lineage_group"]
    rows = []
    method_summary = []
    for track, section in joined.groupby("track", sort=True):
        methods = sorted(section.method.unique())
        available = section[np.isfinite(section.predicted_value)]
        support = available.groupby(["fold", "row_id"]).method.nunique().rename("methods").reset_index()
        common = support[support.methods == len(methods)][["fold", "row_id"]]
        comparable = available.merge(common, on=["fold", "row_id"], validate="many_to_one")
        for tag, group in comparable.groupby([*keys, "method"], sort=True):
            rows.append(dict(zip([*keys, "method"], tag)) | _cell_metrics(group, endpoint_direction, rank_only or tag[-1] == "random_priority_order"))
        for method, group in section.groupby("method", sort=True):
            covered = int(np.isfinite(group.predicted_value).sum())
            method_summary.append({"track": track, "method": method, "registered_rows": len(group),
                                   "covered_rows": covered, "common_rows": len(common)})
    cells = pd.DataFrame(rows)
    metrics = ["mae", "mse", "spearman", "ndcg_at_5", "top5_overlap",
               "interval_fraction", "interval_coverage", "interval_width"]
    for item in method_summary:
        if cells.empty:
            continue
        group = cells[(cells.track == item["track"]) & (cells.method == item["method"])]
        item["cells"] = int(group.cell_id.nunique())
        item["lineages"] = int(group.lineage_group.nunique())
        for name in metrics:
            if name in group and group[name].notna().any():
                item["macro_" + name] = float(group[name].mean())
    contrasts = []
    if not cells.empty:
        for track, group in cells.groupby("track", sort=True):
            for candidate, baseline in comparisons:
                for metric in ("mae", "ndcg_at_5", "spearman"):
                    result = paired_gain(group, candidate, baseline, metric)
                    if result is not None:
                        contrasts.append(dict(track=track, candidate=candidate, baseline=baseline, metric=metric, **result))
    return {"scores": method_summary, "paired_contrasts": contrasts}, cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--split", required=True)
    parser.add_argument("--endpoint-direction", choices=("higher", "lower"), required=True)
    parser.add_argument("--rank-only", action="store_true")
    parser.add_argument("--compare", nargs=2, action="append", default=[], metavar=("CANDIDATE", "BASELINE"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    forecasts, receipt = verify_forecast(args.prediction)
    if args.split not in set(forecasts.track):
        parser.error("split was not registered in forecast")
    which = forecasts[forecasts.track == args.split]
    observations = read_table(args.observations)
    joined = join_observations(which, observations)
    scores, cells = score(joined, endpoint_direction=args.endpoint_direction,
                          rank_only=args.rank_only, comparisons=args.compare)
    output = {
        "split": args.split,
        "freeze_sha256": receipt["sha256"],
        "observations_sha256": sha256(args.observations),
        "endpoint_direction": args.endpoint_direction,
        "rank_only": args.rank_only,
        **scores,
        "cell_scores": cells.astype(object).where(pd.notna(cells), None).to_dict(orient="records"),
    }
    publish_json(args.out, output)


if __name__ == "__main__":
    main()
