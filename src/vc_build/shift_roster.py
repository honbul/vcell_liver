"""Register public shift calibration roles using the frozen external query roster only."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from .io import freeze_table, publish_json, sha256, verify_forecast


def ordered(keys, scope):
    return sorted(keys, key=lambda value: hashlib.sha256(f"14822:{scope}:{value}".encode()).digest())


def roster(forecasts):
    eligible = forecasts[forecasts.method.isin(["drug_mean", "direct_structure_rna"])].copy()
    groups = ordered(eligible.lineage_group.unique(), "lineage")
    chemistry = ordered(eligible.compound_id.unique(), "sentinel")
    if len(groups) < 35 or len(chemistry) < 15:
        raise ValueError("Public shift calibration needs at least 35 independent lineages and 15 shared drugs")
    fit, interval = set(groups[:15]), set(groups[15:25])
    sentinel = set(chemistry[:5])
    eligible["role"] = "score"
    eligible.loc[eligible.compound_id.isin(sentinel), "role"] = "sentinel"
    eligible.loc[eligible.lineage_group.isin(fit), "role"] = "shift_fit"
    eligible.loc[eligible.lineage_group.isin(interval), "role"] = "interval"
    eligible["track"] = "X_public_shift"
    report = {
        "fit_lineages": sorted(fit), "interval_lineages": sorted(interval), "score_lineages": sorted(set(groups[25:])),
        "sentinel_compounds": sorted(sentinel), "roles_per_method": eligible.groupby(["role", "method"]).size()
        .unstack(fill_value=0).to_dict(),
    }
    return eligible, report


def register(source, directory):
    predictions, stage = verify_forecast(source)
    records, report = roster(predictions)
    directory.mkdir(parents=True, exist_ok=False)
    source_hash = stage["sha256"]
    observation_hash = stage["provenance"].get("registered_observations_sha256")
    freeze_table(records, directory / "registered.parquet", provenance={
        "external_freeze_sha256": source_hash, "registered_observations_sha256": observation_hash, **report,
    })
    publish_json(directory / "roles.json", report)


def selective_observations(directory, observed):
    records, receipt = verify_forecast(directory / "registered.parquet")
    if sha256(observed) != receipt["provenance"]["registered_observations_sha256"]:
        raise ValueError("Source-shift observation file changed since registration")
    actual = pd.read_parquet(observed)
    actual = actual.rename(columns={"record_id": "row_id", "cell_group": "lineage_group"})
    if actual["row_id"].duplicated().any():
        raise ValueError("Duplicate observations in public source-shift bundle")
    for role in ("shift_fit", "interval", "sentinel", "score"):
        subset = records[records.role == role].row_id.unique()
        table = actual[actual.row_id.isin(subset)]
        if set(table.row_id) != set(subset):
            raise ValueError("Role and observation support differ")
        with (directory / f"observations_{role}.parquet").open("xb") as stream:
            table.to_parquet(stream, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--register-from", type=Path)
    action.add_argument("--select-observations-from", type=Path)
    args = parser.parse_args()
    if args.register_from:
        register(args.register_from, args.dir)
    else:
        selective_observations(args.dir, args.select_observations_from)


if __name__ == "__main__":
    main()
