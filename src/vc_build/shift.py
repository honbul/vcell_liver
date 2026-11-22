"""Build immutable public source-shift forecasts with disjoint sentinel calibration."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .calibration import finite_sample_quantile
from .evaluate import join_observations
from .io import freeze_table, read_table, sha256, verify_forecast


def _affine(predictor, actual, penalty):
    if len(actual) < 2 or np.var(predictor) < 1e-10:
        return 0., float(np.mean(actual))
    fit = Ridge(alpha=penalty, positive=True).fit(np.asarray(predictor)[:, None], actual)
    return float(fit.coef_[0]), float(fit.intercept_)


def calibrate_public(forecast, fit, interval, sentinel, *, coverage=0.90, weight=8.):
    query = forecast[forecast.role == "score"].copy()
    fit_ids = set(fit.row_id)
    interval_ids = set(interval.row_id)
    sentinel_ids = set(sentinel.row_id)
    score_ids = set(query.row_id)
    partitions = (fit_ids, interval_ids, sentinel_ids, score_ids)
    if any(a & b for i, a in enumerate(partitions) for b in partitions[i + 1:]):
        raise ValueError("Calibration, sentinel and scoring rows must be disjoint")
    role_frames = {}
    for name, outcomes in (("shift_fit", fit), ("interval", interval), ("sentinel", sentinel)):
        part = forecast[forecast.role == name]
        if set(part.row_id) != set(outcomes.row_id):
            raise ValueError(f"The locked {name} roster differs from its observed subset")
        role_frames[name] = join_observations(part, outcomes)
    model_keys = ["track", "fold", "method"]
    outputs = []
    for names, query_rows in query.groupby(model_keys, sort=True):
        keep = {name: value for name, value in zip(model_keys, names)}
        parts = {key: value.loc[(value[model_keys] == pd.Series(keep)).all(axis=1)] for key, value in role_frames.items()}
        fit_rows, interval_rows, sentinel_rows = (parts[role] for role in ("shift_fit", "interval", "sentinel"))
        for part in (fit_rows, interval_rows, sentinel_rows):
            if not np.isfinite(part.predicted_value).all():
                raise ValueError("Calibration requires observed and predicted values on all registered support")
        if fit_rows.empty or interval_rows.empty:
            raise ValueError("Both independent public calibration roles are required")
        fit_groups, interval_groups, score_groups = (
            set(frame.lineage_group) for frame in (fit_rows, interval_rows, query_rows)
        )
        if fit_groups & interval_groups or fit_groups & score_groups or interval_groups & score_groups:
            raise ValueError("Source-scale calibration must be disjoint by lineage")
        if set(sentinel_rows.compound_id) & set(query_rows.compound_id):
            raise ValueError("Sentinel compounds must be outside the scoring candidate pool")
        slope, offset = _affine(fit_rows.predicted_value, fit_rows.endpoint_value, penalty=1.)
        scale_hat = slope * interval_rows.predicted_value + offset
        scores = interval_rows.assign(error=np.abs(scale_hat - interval_rows.endpoint_value)).groupby("lineage_group").error.max()
        try:
            radius = finite_sample_quantile(scores.to_numpy(), coverage)
        except ValueError:
            radius = np.nan
        result = query_rows.assign(predicted_value=slope * query_rows.predicted_value + offset)
        result["method"] = names[-1] + "_source_scale"
        result["interval_lower"] = result.predicted_value - radius
        result["interval_upper"] = result.predicted_value + radius
        outputs.append(result)
        if not sentinel_rows.empty:
            shifted = sentinel_rows.assign(
                residual=sentinel_rows.endpoint_value - (slope * sentinel_rows.predicted_value + offset)
            ).groupby("cell_id").residual.agg(["sum", "count"])
            local = (shifted["sum"] / (shifted["count"] + weight)).rename("adjustment")
            adjusted = result.join(local, on="cell_id")
            adjusted = adjusted.assign(
                predicted_value=(adjusted.predicted_value + adjusted.adjustment).where(adjusted.adjustment.notna()),
                method=names[-1] + "_sentinel_scale",
                interval_lower=np.nan,
                interval_upper=np.nan,
            )
            outputs.append(adjusted.drop(columns="adjustment"))
    if not outputs:
        raise ValueError("No registered source-shift queries")
    return pd.concat(outputs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("forecast", "fit", "interval", "sentinel", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    forecasts, receipt = verify_forecast(args.forecast)
    parts = [read_table(getattr(args, name)) for name in ("fit", "interval", "sentinel")]
    calibrated = calibrate_public(forecasts, *parts)
    freeze_table(calibrated, args.out, provenance={
        "prior_forecast_sha256": receipt["sha256"],
        "calibration_sha256": {name: sha256(getattr(args, name)) for name in ("fit", "interval", "sentinel")},
        "interval": "group-maximum absolute error on separate calibration lineages, 90% nominal",
        "local_site": "data_wait",
    })


if __name__ == "__main__":
    main()
