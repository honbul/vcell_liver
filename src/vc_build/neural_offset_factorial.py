"""Separate compound, sample and interaction contributions in frozen forecasts."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .expanded_panel_cv import RECEIPTS
from .external import KEY
from .io import freeze_table, verify_forecast
from .response_component_benchmark import BASELINES, EXTERNAL_TRACKS, FORECAST_SCHEMA, PUBLIC
from .response_components import component_scores, interaction_intervals, orthogonal_parts
from .rna_state_projection import score_external, score_source
from .structured_cv import ASSAYS, paired_intervals
from .v0plus_score import axis_metrics, crossed_patient_drug_ci
from .yang_hcc_external import require_hash

ARCHIVE_RECEIPT = "5c8ad37177fd60919c8b9aeccfc6336cc3a53e373daeebc6211a757b24837ab4"
BASE = "structured"
FULL = "interaction_neural_blend"
PRIMARY = "offset_interaction"
COALITIONS = {
    0: BASE,
    1: "offset_drug",
    2: "offset_sample",
    3: "offset_drug_sample",
    4: PRIMARY,
    5: "offset_drug_interaction",
    6: "offset_sample_interaction",
    7: FULL,
}
OFFSETS = tuple(COALITIONS[i] for i in range(1, 7))
ARMS = (*OFFSETS, FULL, "native_neural_blend")
CONTROLS = (BASE, FULL)
PREDICTION_KEYS = [*KEY, "track", "fold"]
RANK_METRICS = ("spearman", "ndcg_at_5", "top5_overlap")


def separated_predictions(prediction, base, full):
    raw = prediction.assign(delta=full - base)
    grid = raw.pivot(index="cell_id", columns="compound_id", values="delta")
    support = grid.notna().to_numpy(dtype=float)
    centered = grid.to_numpy() - raw.delta.mean()
    sample, drug, interaction = orthogonal_parts(centered, support)
    row = grid.index.get_indexer(raw.cell_id)
    column = grid.columns.get_indexer(raw.compound_id)
    pieces = np.stack([drug[column], sample[row], interaction[row, column]])
    if not np.allclose(pieces.sum(axis=0) + raw.delta.mean(), raw.delta, atol=1e-12):
        raise ValueError("Separated forecast must recover the original prediction")
    results = []
    for coalition, name in COALITIONS.items():
        if name in OFFSETS:
            active = [i for i in range(3) if coalition & (1 << i)]
            results.append(prediction.assign(method=name, predicted_value=base + pieces[active].sum(axis=0)))
    return results, {
        "samples": len(grid), "drugs": len(grid.columns), "observed_pairs": len(raw),
        "grand_offset": float(raw.delta.mean()),
        "part_rms": dict(zip(("drug", "sample", "interaction"), np.sqrt(np.square(pieces).mean(axis=1)).tolist())),
    }


def prediction_stage(args):
    require_hash(args.receipt, ARCHIVE_RECEIPT)
    recorded = json.loads(args.receipt.read_text())
    name = args.mode if args.mode in EXTERNAL_TRACKS else f"{args.mode}-{args.fold}"
    rows, stamp = verify_forecast(args.archive / f"{name}.parquet")
    prior = recorded["forecasts"][name]
    info = stamp["provenance"]
    if (stamp["sha256"], info["inputs_sha256"], info["mode"], info["fold"], info["core_sha256"]) != (
        prior["sha256"], prior["inputs_sha256"], args.mode, args.fold, RECEIPTS["core"],
    ):
        raise ValueError("Forecast differs from the registered neural interaction analysis")
    public = PUBLIC if args.mode in EXTERNAL_TRACKS else ()
    comparison = {*BASELINES, *public, FULL, "native_neural_blend"}
    result, audits = [rows], {}
    for track, block in rows.groupby("track", sort=True):
        matrix = block.pivot(index=PREDICTION_KEYS, columns="method", values="predicted_value").reset_index()
        if set(matrix.columns) != {*PREDICTION_KEYS, *comparison} or not np.isfinite(matrix[list(comparison)]).all(axis=None):
            raise ValueError("Expected the full archive comparator support")
        fresh, audits[track] = separated_predictions(matrix[PREDICTION_KEYS], matrix[BASE], matrix[FULL])
        result.extend(fresh)
    freeze_table(pd.concat(result, ignore_index=True)[FORECAST_SCHEMA], args.out, provenance={
        "mode": args.mode, "fold": args.fold, "core_sha256": RECEIPTS["core"], "inputs_sha256": info["inputs_sha256"],
        "previous_sha256": stamp["sha256"], "previous_registration_sha256": ARCHIVE_RECEIPT, "diagnostics": audits,
    })


def shapley_attribution(cells):
    base_keys = ["track", "fold", "cell_id", "lineage_group"]
    subset = cells[cells.method.isin(COALITIONS.values())]
    by_metric = {}
    for metric in RANK_METRICS:
        values = subset.pivot(index=base_keys, columns="method", values=metric).dropna()
        values = values.groupby(level="lineage_group").mean()
        if len(values) < 2:
            raise ValueError(f"Insufficient paired lineages for {metric} attribution")
        effects = {}
        for component, bit in enumerate((1, 2, 4)):
            gain = sum(
                (values[COALITIONS[mask | bit]] - values[COALITIONS[mask]])
                * (1 / 6 if mask.bit_count() == 1 else 1 / 3)
                for mask in range(8) if not mask & bit
            )
            effects[("drug", "sample", "interaction")[component]] = gain
        effect = pd.DataFrame(effects)
        difference = values[FULL] - values[BASE]
        if not np.allclose(effect.sum(axis=1), difference, atol=1e-12):
            raise ValueError("Factorial attributions do not sum to the measured difference")
        rng = np.random.default_rng(14822)
        draw = rng.integers(len(effect), size=(4000, len(effect)))
        bands = np.quantile(effect.to_numpy()[draw].mean(axis=1), (.025, .975), axis=0).T
        by_metric[metric] = {
            "delta": float(difference.mean()), "groups": len(effect),
            "components": {name: {"point": float(series.mean()), "ci95": ci.tolist()}
                           for (name, series), ci in zip(effect.items(), bands)},
        }
    return by_metric


def assess(frame, *, external):
    comparisons = [(PRIMARY, arm) for arm in CONTROLS] + [("offset_drug", BASE), (FULL, BASE)]
    metrics, drug_scores, cell_scores = axis_metrics(frame, comparisons)
    interval = crossed_patient_drug_ci if external else paired_intervals
    across = {arm: interval(frame, PRIMARY, arm) for arm in CONTROLS}
    ranks = {(row["baseline"], row["metric"]): row["ci95"] for row in metrics["paired_contrasts"]
             if row["candidate"] == PRIMARY}
    gates = {arm: all(across[arm][metric][0] > 0 for metric in ("spearman", "pearson"))
             and all(ranks[arm, metric][0] > -.005 for metric in RANK_METRICS) for arm in CONTROLS}
    report = {"metrics": metrics, "across_cell_ci95": across, "gates": gates,
              "groups": frame.lineage_group.nunique(), "observations": frame.row_id.nunique()}
    if external:
        components = component_scores(frame, cell_axis="lineage_group")
        interactions = {arm: interaction_intervals(frame, PRIMARY, arm) for arm in CONTROLS}
        for arm, bounds in interactions.items():
            gates[arm] = gates[arm] and bounds[0] > 0
        report.update(components=components, interaction_ci95=interactions, shapley=shapley_attribution(cell_scores))
        across_panel = drug_scores.pivot(index="compound_id", columns="method", values=["spearman", "pearson"])
        for metric in ("spearman", "pearson"):
            if not np.allclose(across_panel[metric][BASE], across_panel[metric]["offset_drug"], atol=1e-12, equal_nan=True):
                raise ValueError("A compound-constant offset changed within-drug correlations")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    fields = {
        "predict": ("archive", "receipt"),
        "source": ("original", "core", "forecasts"),
        "external": ("original", "core", "licob", "forecasts", "limore-labels", "licob-labels"),
    }
    for action, names in fields.items():
        stage = stages.add_parser(action)
        for name in (*names, "out"):
            stage.add_argument(f"--{name}", required=True, type=Path)
        if action == "predict":
            stage.add_argument("--mode", required=True, choices=(*ASSAYS, *EXTERNAL_TRACKS))
            stage.add_argument("--fold", type=int, default=0, choices=range(3))
    args = parser.parse_args()
    if args.stage == "predict":
        prediction_stage(args)
    elif args.stage == "source":
        score_source(args, assessor=assess)
    else:
        score_external(args, assessor=assess, arms=ARMS)


if __name__ == "__main__":
    main()
