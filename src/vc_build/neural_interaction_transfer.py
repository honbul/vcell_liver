"""Transfer molecular interaction residuals on top of a fixed structural predictor."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .expanded_panel_cv import RECEIPTS, registered_inputs, same_split
from .external import KEY, SOURCE, _validate
from .io import freeze_table
from .limore_corrected import shared_support
from .response_component_benchmark import (
    BASELINES, EXTERNAL_TRACKS, FORECAST_SCHEMA, PUBLIC, external_inputs, external_reference, source_reference,
)
from .response_components import component_scores, interaction_intervals, orthogonal_parts
from .rna_state_projection import assess, score_external, score_source
from .structured_cv import ASSAYS, with_assay
from .v0plus_cv import molecular_inputs, panel_targets, validation_roster
from .v0plus_neural import fit_molecular, load_auxiliary

ARM = "native_neural_blend"
PRIMARY = "interaction_neural_blend"
ARMS = (ARM, PRIMARY)
CONTROLS = ("structured", ARM)
SCALE = .5
WIDTH, EPOCHS = 192, 70


def selective_labels(rows):
    effects = np.empty(len(rows))
    traces = {}
    for assay, positions in rows.groupby(SOURCE, sort=True).indices.items():
        sample = rows.iloc[positions]
        grid = sample.pivot(index="cell_id", columns="compound_id", values="residual")
        measured = grid.notna().to_numpy(dtype=float)
        i = grid.index.get_indexer(sample.cell_id)
        j = grid.columns.get_indexer(sample.compound_id)
        _, _, interaction = orthogonal_parts(grid.to_numpy(), measured)
        effects[positions] = interaction[i, j]
        traces["/".join(assay)] = {
            "rows": len(sample), "cells": len(grid), "drugs": len(grid.columns),
            "original_rms": float(np.sqrt(np.mean(sample.residual.to_numpy() ** 2))),
            "interaction_rms": float(np.sqrt(np.mean(effects[positions] ** 2))),
        }
    return rows.assign(residual=effects), traces


def response_matrix(model, rna):
    drugs = model.molecular.drugs
    full = pd.MultiIndex.from_product((rna.index, drugs), names=["cell_id", "compound_id"]).to_frame(index=False)
    _, predictions = model.predict(full, rna)
    if not np.isfinite(predictions).all():
        raise ValueError("Incomplete predictions on the full fit compound panel")
    return predictions.reshape(len(rna), len(drugs))


def centered_prediction(fit_mean, predictions):
    residual = predictions - fit_mean
    return residual - residual.mean(axis=1, keepdims=True)


def neural_residuals(fit, query, rna, edges, targets, auxiliary):
    _validate(fit, query)
    train_rna = rna.loc[sorted(fit.cell_id.unique())]
    held_rna = rna.loc[sorted(query.cell_id.unique())]
    _, anchors, aligned, _ = panel_targets(fit)
    residuals, target_audit = selective_labels(aligned)
    parts = []
    diagnostics = {"fit_cells": len(train_rna), "genes": len(train_rna.columns),
                   "assays": target_audit, "scale": SCALE, "width": WIDTH, "epochs": EPOCHS}
    for name, rows in ((ARM, aligned), (PRIMARY, residuals)):
        fitted = fit_molecular("v0plus_inductive", WIDTH, EPOCHS, rows, anchors,
                               train_rna, None, edges, targets, auxiliary,
                               gpu=True, interaction_target=name == PRIMARY)
        fit_prediction = response_matrix(fitted, train_rna)
        held_prediction = response_matrix(fitted, held_rna)
        centered = centered_prediction(fit_prediction.mean(axis=0), held_prediction)
        samples = held_rna.index.get_indexer(query.cell_id)
        drugs = pd.Index(fitted.molecular.drugs).get_indexer(query.compound_id)
        if (drugs < 0).any():
            raise ValueError("Every scored compound needs a fit-only neural prediction")
        parts.append(query[KEY].assign(method=name, predicted_value=SCALE * centered[samples, drugs]))
        diagnostics[name] = {
            "fit_drugs": fit_prediction.shape[1], "fit_rms": float(np.sqrt(np.mean(fit_prediction**2))),
            "held_rms": float(np.sqrt(np.mean(held_prediction**2))),
            "held_centered_rms": float(np.sqrt(np.mean(centered**2))),
        }
        print(f"{name} complete", flush=True)
    return pd.concat(parts, ignore_index=True), diagnostics


def predict(args):
    edges, targets, priors = molecular_inputs(args.priors, args.hallmark, args.targets)
    auxiliary, aux_stamp = load_auxiliary(args.auxiliary)
    external = args.mode in EXTERNAL_TRACKS
    if external:
        fit, query, rna, registered = external_inputs(args, args.mode)
        comparison, reference = external_reference(args, args.mode, query)
    else:
        original, clean, rna, _ = registered_inputs(args)
        _, fit, check = same_split(original, clean, args.fold, args.mode)
        query = validation_roster(check)
        comparison, reference = source_reference(args.cv_archive, args.mode, args.fold)
        registered = RECEIPTS["core"]
    offsets, diagnostics = neural_residuals(fit, query, rna, edges, targets, auxiliary)
    offsets = offsets.assign(track=args.mode, fold=0) if external else with_assay(offsets, check, args.fold)
    keys = [*KEY, "track", "fold"]
    baseline = comparison[comparison.method.eq("structured")][[*keys, "predicted_value"]]
    offsets = offsets.merge(baseline.rename(columns={"predicted_value": "base"}), on=keys,
                            validate="many_to_one", how="left")
    offsets.predicted_value += offsets.pop("base")
    rows = pd.concat([comparison, offsets], ignore_index=True)
    if external:
        shared_support(rows, query, (*BASELINES, *PUBLIC, *ARMS))
    else:
        matrix = rows.pivot(index=keys, columns="method", values="predicted_value")
        if set(matrix) != {*BASELINES, *ARMS} or not np.isfinite(matrix.to_numpy()).all():
            raise ValueError("Neural and reference forecasts must have the same source support")
    freeze_table(rows[FORECAST_SCHEMA], args.out, provenance={
        "mode": args.mode, "fold": args.fold, "inputs_sha256": registered, "core_sha256": RECEIPTS["core"],
        "reference_sha256": reference, "priors": priors, "auxiliary": aux_stamp, "diagnostics": diagnostics,
    })


def score_neural(frame, *, external):
    report = assess(frame, external=external, primary=PRIMARY, controls=CONTROLS, require_precision=True)
    report["components"] = component_scores(frame, cell_axis="lineage_group")
    if external:
        intervals = {arm: interaction_intervals(frame, PRIMARY, arm) for arm in CONTROLS}
        report["interaction_ci95"] = intervals
        for arm, limits in intervals.items():
            report["gates"][arm] = bool(report["gates"][arm] and limits[0] > 0)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    fields = {
        "predict": ("licob", "archive", "cv-archive", "priors", "hallmark", "targets", "auxiliary"),
        "source": ("forecasts",), "external": ("licob", "forecasts", "limore-labels", "licob-labels"),
    }
    for action, names in fields.items():
        stage = stages.add_parser(action)
        for name in ("original", "core", *names, "out"):
            stage.add_argument(f"--{name}", required=True, type=Path)
        if action == "predict":
            stage.add_argument("--mode", required=True, choices=(*ASSAYS, *EXTERNAL_TRACKS))
            stage.add_argument("--fold", type=int, default=0, choices=range(3))
    args = parser.parse_args()
    if args.stage == "predict":
        predict(args)
    elif args.stage == "source":
        score_source(args, assessor=score_neural)
    else:
        score_external(args, assessor=score_neural, arms=ARMS)


if __name__ == "__main__":
    main()
