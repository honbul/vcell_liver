"""Fit the VCB functional prototype and freeze outcome-free forecasts."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import platform
import subprocess
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from rdkit import rdBase
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from .functional import (
    CONTEXT_METHODS, NONINTERACTION_METHODS, SEEDS, SETTINGS,
    available_rna, chemistry_matrix, fit_arm, fit_base, predict_arm, structure_similarity,
)
from .io import freeze_table, read_contract, sha256
from .splits import assignments, chemical_groups

IDENTIFIERS = ["row_id", "cell_id", "lineage_group", "compound_id"]
REQUIRED = {*IDENTIFIERS[1:], "connectivity_group", "scaffold_group", "isomeric_smiles", "endpoint_value"}
PREDICT_INPUTS = IDENTIFIERS + ["isomeric_smiles"]


def prepared_data(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = pd.read_parquet(directory / "rows.parquet")
    rna = pd.read_parquet(directory / "rna.parquet")
    if "cell_id" in rna.columns:
        rna = rna.set_index("cell_id")
    rows = rows.rename(columns={"cell_group": "lineage_group"})
    missing = REQUIRED - set(rows.columns)
    if missing:
        raise ValueError(f"Functional outcome table is missing {sorted(missing)}.")
    if rows[sorted(REQUIRED - {"scaffold_group"})].isna().any().any():
        raise ValueError("Required functional values must be mapped before training.")
    rows = rows.sort_values(["cell_id", "compound_id"]).reset_index(drop=True)
    if rows.duplicated(subset=["cell_id", "compound_id"]).any():
        raise ValueError("Repeated responses require an explicit endpoint aggregation contract.")
    if "row_id" not in rows:
        rows["row_id"] = rows["cell_id"].astype(str) + "::" + rows["compound_id"].astype(str)
    if rows["row_id"].duplicated().any() or rows["row_id"].isna().any():
        raise ValueError("Prepared response rows must have unique identities.")
    rows["chemical_group"] = chemical_groups(rows)
    rows["endpoint_value"] = pd.to_numeric(rows["endpoint_value"])
    if not np.isfinite(rows["endpoint_value"]).all():
        raise ValueError("Non-finite endpoint in prepared outcome table.")
    rna = available_rna(rna)
    if not rows["cell_id"].isin(rna.index).all():
        raise ValueError("Prepared outcomes contain a cell without basal RNA.")
    return rows, rna


def held_in_valid_rows(fit: pd.DataFrame, query: pd.DataFrame, track: str) -> pd.DataFrame:
    if track == "C":
        return query.loc[query["compound_id"].isin(fit["compound_id"])]
    if track == "D":
        return query.loc[query["cell_id"].isin(fit["cell_id"])]
    return query


def macro_mae(validation: pd.DataFrame, predictions: np.ndarray, track: str) -> float:
    residual = (validation["endpoint_value"] - predictions).abs()
    groups = "chemical_group" if track == "D" else "lineage_group"
    return float(residual.groupby(validation[groups]).mean().mean())


def role_rows(indexed_rows: pd.DataFrame, roles: pd.DataFrame, role: str) -> pd.DataFrame:
    return indexed_rows.loc[roles.loc[roles["role"].eq(role), "row_id"]].reset_index(drop=True)


def tune(rows: pd.DataFrame, rna: pd.DataFrame, track: str, folds: int, seed: int,
         methods: tuple[str, ...] = tuple(SETTINGS)) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    manifest = assignments(rows, track, folds, seed)
    indexed = rows.set_index("row_id", drop=False)
    summary = []
    validation_rows = []
    for fold, roles in manifest.groupby("fold", sort=True):
        fit = role_rows(indexed, roles, "fit")
        validation = held_in_valid_rows(fit, role_rows(indexed, roles, "withheld"), track)
        if validation.empty:
            raise ValueError(f"No matched development support in {track}, fold {fold}.")
        base = fit_base(fit, rna)
        query = validation[PREDICT_INPUTS]
        for method in methods:
            for setting in SETTINGS[method]:
                bundle = fit_arm(base, fit, rna, method, setting)
                prediction = predict_arm(bundle, query, rna)
                summary.append({
                    "inner_fold": int(fold), "method": method, "setting": setting,
                    "validation_rows": len(validation),
                    "macro_mae": macro_mae(validation, prediction, track),
                })
                result = validation[IDENTIFIERS + ["endpoint_value"]].copy()
                result["inner_fold"] = fold
                result["method"], result["setting"] = method, setting
                result["predicted_value"] = prediction
                validation_rows.append(result)
    losses = pd.DataFrame(summary)
    by_setting = losses.groupby(["method", "setting"], as_index=False)["macro_mae"].mean()
    ordered = by_setting.sort_values(["macro_mae", "setting", "method"], kind="stable")
    winners = ordered.drop_duplicates(subset="method").set_index("method")
    selection = {method: int(winners.loc[method, "setting"]) for method in methods}
    for arm, candidates in (
        ("selected_context", CONTEXT_METHODS), ("selected_noninteraction", NONINTERACTION_METHODS)
    ):
        eligible = winners.loc[winners.index.isin(candidates)]
        if not eligible.empty:
            selection[arm] = str(eligible.index[0])
    return selection, losses, pd.concat(validation_rows, ignore_index=True)


def method_predictions(train: pd.DataFrame, test: pd.DataFrame, rna: pd.DataFrame,
                       selection: dict, track: str, fold: int,
                       development: pd.DataFrame) -> pd.DataFrame:
    base = fit_base(train, rna)
    query = test[PREDICT_INPUTS]
    metadata = query[IDENTIFIERS].copy()
    metadata["track"], metadata["fold"], metadata["role"] = track, fold, "withheld"
    fingerprints, _ = chemistry_matrix(query)
    metadata["nearest_fit_tanimoto"] = structure_similarity(
        fingerprints, base["fit_fingerprints"],
    ).max(axis=1)
    output = []

    def add(method, values, calibration_method=None):
        frame = metadata.copy()
        frame["method"], frame["predicted_value"] = method, values
        half_width = interval_half_width(
            development, calibration_method or method, selection.get(calibration_method or method),
        )
        frame["interval_lower"] = values - half_width if half_width is not None else np.nan
        frame["interval_upper"] = values + half_width if half_width is not None else np.nan
        output.append(frame)

    fits = {}
    for method in (name for name in SETTINGS if name in selection):
        seeds = SEEDS if method == selection.get("selected_context") else SEEDS[:1]
        bundle = fit_arm(base, train, rna, method, selection[method], seeds)
        fits[method] = bundle
        add(method, predict_arm(bundle, query, rna))
    for label in ("selected_context", "selected_noninteraction"):
        if label in selection:
            bundle = fits[selection[label]]
            add(label, predict_arm(bundle, query, rna), selection[label])
    if track == "C" and "selected_context" in selection:
        bundle = fits[selection["selected_context"]]
        add("selected_context_rna_drop", predict_arm(bundle, query, rna, drop_rna=True))
        query_groups = query[["cell_id", "lineage_group"]].drop_duplicates().sort_values("cell_id")
        swap = query.copy()
        donors = {}
        for cell, lineage in query_groups.itertuples(index=False, name=None):
            other_test = query_groups.loc[query_groups["lineage_group"].ne(lineage), "cell_id"]
            other_fit = train.loc[train["lineage_group"].ne(lineage), "cell_id"].sort_values()
            other = other_test if not other_test.empty else other_fit
            donors[cell] = other.iloc[0]
        swap["cell_id"] = swap["cell_id"].map(donors)
        add("selected_context_rna_swap", predict_arm(bundle, swap, rna))
        rng = np.random.default_rng(24000 + fold)
        numbers = rng.uniform(size=len(metadata))
        add("random_priority_order", numbers)
    return pd.concat(output, ignore_index=True)


def generate_forecasts(rows, rna, manifest, inner_folds: int, seed: int, smoke: bool):
    indexed = rows.set_index("row_id", drop=False)
    forecasts, shortlist, calibration = [], [], []
    for (track, fold), roles in manifest.groupby(["track", "fold"], sort=True):
        fit = role_rows(indexed, roles, "fit")
        test = held_in_valid_rows(fit, role_rows(indexed, roles, "withheld"), track)
        if smoke:
            selection = {"drug_mean": 0, "additive": 10, "selected_noninteraction": "drug_mean"}
            dev_predictions = pd.DataFrame()
        else:
            selection, losses, dev_predictions = tune(fit, rna, track, inner_folds, seed + fold + 100)
            shortlist.append(losses.assign(track=track, fold=fold))
            dev_predictions["track"], dev_predictions["fold"] = track, fold
            calibration.append(dev_predictions)
        forecasts.append(method_predictions(fit, test, rna, selection, track, fold, dev_predictions))
        print(f"Frozen-forecast inputs {track} fold {fold} fit={len(fit)} holdout={len(test)}", flush=True)
    return (pd.concat(forecasts, ignore_index=True),
            pd.concat(shortlist, ignore_index=True) if shortlist else pd.DataFrame(),
            pd.concat(calibration, ignore_index=True) if calibration else pd.DataFrame())


def endpoint_metadata(rows: pd.DataFrame, config: dict) -> dict:
    def constant(column):
        values = rows[column].dropna().unique() if column in rows else []
        if len(values) > 1:
            raise ValueError(f"Mixed {column} cannot be used in a single endpoint head.")
        if len(values):
            value = values[0]
            return value.item() if isinstance(value, np.generic) else value
        return config.get(column)

    direction = constant("direction") or config.get("endpoint_direction", "higher")
    return {
        "endpoint_id": constant("endpoint_id"),
        "predicted_unit": constant("endpoint_unit"),
        "source_protocol": constant("source_protocol"),
        "source_id": constant("source_id"),
        "assay_id": constant("assay_id"),
        "time_h": constant("time_h") or constant("exposure_hours"),
        "expression_scale": constant("expression_scale") or "log2(FPKM+1)",
        "baseline_is_proxy": constant("baseline_is_proxy"),
        "endpoint_contract": "curve_summary",
        "supported_dose_response": False,
        "program_prediction_available": False,
        "program_definition_version": None,
        "dose_uM": None,
        "endpoint_direction": direction.split("_", 1)[0],
        "culture_format": constant("culture_format") or config.get("culture_format", "2d_monolayer"),
        "supported_primary_input": "baseline RNA plus registered compound identity",
        "input_mode": "baseline_rna_and_structure",
        "cell_rna_ood": "nearest fit-cell distance in fit-only standardized RNA PCA",
        "structure_ood": "exact isomeric SMILES registration and nearest fit ECFP Tanimoto",
    }


def interval_half_width(development: pd.DataFrame, method: str, setting: int | None) -> float | None:
    if development.empty or setting is None:
        return None
    data = development.loc[
        development["method"].eq(method) & development["setting"].eq(setting)
    ]
    if data.empty:
        return None
    error = (data["endpoint_value"] - data["predicted_value"]).abs().to_numpy()
    sizes = data.groupby("lineage_group")["lineage_group"].transform("size").to_numpy()
    order = np.argsort(error, kind="stable")
    weights = np.cumsum(1 / sizes[order])
    return float(error[order[np.searchsorted(weights, 0.90 * weights[-1])]])


def save_package(path, rows, rna, method, setting, endpoint, development, seed, smoke):
    path.mkdir(exist_ok=False)
    base = fit_base(rows, rna)
    seeds = SEEDS if method == "interaction_mlp" and not smoke else (seed,)
    bundle = fit_arm(base, rows, rna, method, setting, seeds)
    joblib.dump(bundle, path / "checkpoint.joblib", compress=3)
    contract = dict(
        endpoint, model_version=path.name, method=method, setting=setting, seeds=list(seeds),
        min_gene_fraction=bundle["rna"]["min_gene_fraction"],
        rna_distance_threshold=bundle["rna"]["distance_threshold"],
        registered_panel_size=len(bundle["structures"]), known_panel_only=True,
        interval=None,
        versions={"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                  "rdkit": rdBase.rdkitVersion},
    )
    if not development.empty:
        development = development.loc[
            development["method"].eq(method) & development["setting"].eq(setting)
        ].copy()
        development["absolute_error"] = (development["endpoint_value"] - development["predicted_value"]).abs()
        development.to_csv(path / "calibration.tsv", sep="\t", index=False)
        contract["interval"] = {
            "method": "development_lineage_weighted_C_cross_fit_absolute_error_quantile",
            "nominal_coverage": 0.90,
            "half_width": interval_half_width(development, method, setting),
        }
    (path / "contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    files = {entry.name: sha256(entry) for entry in sorted(path.iterdir()) if entry.is_file()}
    (path / "receipt.json").write_text(json.dumps({"files": files}, indent=2, sort_keys=True) + "\n")
    return contract


def run(config_path: Path, data_dir: Path, run_dir: Path, tracks: list[str],
        *, smoke: bool = False) -> None:
    config = read_contract(config_path)
    functional = config.get("functional", {})
    split = config["split"]
    seed = int(split["seed"])
    outer_folds = int(split["outer_folds"])
    inner_folds = int(split["inner_folds"])
    start = time.perf_counter()
    rows, rna = prepared_data(data_dir)
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    endpoint = endpoint_metadata(rows, functional)
    run_dir.mkdir(parents=True, exist_ok=False)
    if smoke:
        # Declared minimal slice remains real endpoint development data.
        rows = rows.loc[rows["compound_id"].isin(sorted(rows["compound_id"].unique())[:12])]
        outer_folds, tracks = 3, ["C"]
    manifest = pd.concat([
        assignments(rows, track, outer_folds, seed) for track in tracks
    ], ignore_index=True)
    manifest.to_parquet(run_dir / "splits.parquet", index=False)
    context = threadpool_limits(limits=1) if config.get("execution", {}).get("cpu_threads", 1) == 1 else nullcontext()
    with context, warnings.catch_warnings(record=True) as training_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        forecasts, shortlist, calibration = generate_forecasts(
            rows, rna, manifest, inner_folds, seed, smoke,
        )
        forecasts = forecasts.sort_values(["track", "fold", "method", "row_id"])
        provenance = {
            "scope": "real endpoint pipeline smoke; no comparative claim" if smoke else
            "existing public development data; outer targets excluded under this contract",
            "code_commit": revision, "config_sha256": sha256(config_path),
            "rows_sha256": sha256(data_dir / "rows.parquet"),
            "rna_sha256": sha256(data_dir / "rna.parquet"),
            "splits_sha256": sha256(run_dir / "splits.parquet"),
        }
        freeze_table(forecasts, run_dir / "predictions.parquet", provenance=provenance)
        if smoke:
            deployment = "additive"
            settings, _, dev = tune(rows, rna, "C", inner_folds, seed + 900, ("additive",))
        else:
            settings, deployment_losses, dev = tune(rows, rna, "C", inner_folds, seed + 900)
            deployment = settings["selected_context"]
            deployment_losses.to_csv(run_dir / "release_selection.tsv", sep="\t", index=False)
        direct = save_package(
            run_dir / ("v0-smoke" if smoke else "v0"), rows, rna,
            deployment, settings[deployment], endpoint, dev, SEEDS[0], smoke,
        )
        if not smoke:
            reference = settings["selected_noninteraction"]
            save_package(run_dir / "v0_baseline", rows, rna, reference, settings[reference],
                         endpoint, dev, SEEDS[0], smoke)
    if not shortlist.empty:
        shortlist.to_csv(run_dir / "inner_selection.tsv", sep="\t", index=False)
        calibration.to_parquet(run_dir / "inner_calibration.parquet", index=False)
    labels = rows[IDENTIFIERS + ["endpoint_value"]].copy()
    labels.to_parquet(run_dir / "labels.parquet", index=False)
    first = rows.iloc[0]
    ref_genes = joblib.load(run_dir / ("v0-smoke" if smoke else "v0") / "checkpoint.joblib")["rna"]["genes"]
    sample = {
        "request_id": "limore_offline_example",
        "candidate_pool_id": "limore_observed_panel_excerpt",
        "input_mode": direct["input_mode"],
        "assay_id": direct["assay_id"],
        "time_h": direct["time_h"],
        "cell": {
            "cell_id": str(first["cell_id"]), "lineage_role": "adult_hcc",
            "expression_scale": direct["expression_scale"], "culture_format": direct["culture_format"],
            "baseline_rna": {
                gene: float(value) for gene, value in rna.loc[first["cell_id"], ref_genes].dropna().items()
            },
        },
        "candidates": rows[["compound_id", "isomeric_smiles"]].drop_duplicates().head(8)
        .rename(columns={"isomeric_smiles": "smiles"}).to_dict(orient="records"),
    }
    (run_dir / "sample_request.json").write_text(json.dumps(sample, indent=2, sort_keys=True) + "\n")
    receipt = dict(
        provenance, smoke=smoke, track_names=tracks, forecast_rows=len(forecasts),
        outcome_rows=len(rows), cells=int(rows["cell_id"].nunique()),
        lineage_groups=int(rows["lineage_group"].nunique()), compounds=int(rows["compound_id"].nunique()),
        deployment_method=deployment, elapsed_s=time.perf_counter() - start,
        convergence_warnings=sum(issubclass(w.category, ConvergenceWarning) for w in training_warnings),
        artifact_sha256={
            str(path.relative_to(run_dir)): sha256(path)
            for path in sorted(run_dir.rglob("*")) if path.is_file()
        },
    )
    (run_dir / "training_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tracks", nargs="+", choices=["C", "D", "CD"], default=["C", "D", "CD"])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = read_contract(args.config)
    data_dir = args.data_dir or Path(config["functional_dataset"])
    run(args.config, data_dir, args.run_dir, args.tracks, smoke=args.smoke)


if __name__ == "__main__":
    main()
