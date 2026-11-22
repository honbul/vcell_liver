"""Compare 2D residual models with exactly the same frozen 3D drug anchors."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import score, score_across_cells
from .external import KEY
from .io import freeze_table, publish_json, sha256, verify_forecast
from .pdo_transfer_cv import PDO_HASH, STUDIES, cells_for_study, inputs, source_teacher
from .v0plus_score import crossed_patient_drug_ci

SCORE_SHA256 = "42f819534ac115a2b4dbbecc473ef4dc149bf0f4797a8f2f984b31a7e4e991b0"
STRUCTURE = "pdo_anchor_transfer"
RIDGE = "pdo_anchor_ridge"
AVERAGE = "pdo_anchor_average"
COMPARISONS = ((STRUCTURE, RIDGE), (AVERAGE, RIDGE), (AVERAGE, STRUCTURE),
               (STRUCTURE, "pdo_mean"), (RIDGE, "pdo_mean"))


def predict(args):
    if sha256(args.previous / "score.json") != SCORE_SHA256:
        raise ValueError("Unexpected registered previous 3D result")
    recorded = json.loads((args.previous / "score.json").read_text())
    metadata, _ = inputs(args.pdo)
    teacher, metadata, teacher_hash = source_teacher(args.previous / "teacher.parquet", metadata)
    if teacher_hash != recorded["teacher_sha256"]:
        raise ValueError("2D forecast hash differs from the 3D receipt")
    hashes = {}
    for study in STUDIES:
        filename = study.replace(" ", "_") + ".parquet"
        old, seal = verify_forecast(args.previous / "forecasts" / filename)
        if seal["sha256"] != recorded["forecast_sha256"][study]:
            raise ValueError("Expected the frozen original 3D study forecasts")
        fit, held = cells_for_study(metadata, study)
        pairs = old.pivot(index=KEY, columns="method", values="predicted_value").reset_index()
        donor_residual = fit.join(teacher[["2d_mean", "2d_ridge"]], on=["cell_id", "compound_id"])
        fit_center = (donor_residual["2d_ridge"] - donor_residual["2d_mean"]).groupby(
            donor_residual.compound_id,
        ).mean()
        ridge = pairs["pdo_mean"] + pairs["2d_ridge"] - pairs["2d_mean"]
        ridge -= pairs.compound_id.map(fit_center).fillna(0)
        blend = .5 * (ridge + pairs[STRUCTURE])
        new = pd.concat([pairs[KEY].assign(method=RIDGE, predicted_value=ridge),
                         pairs[KEY].assign(method=AVERAGE, predicted_value=blend)])
        if not np.isfinite(new.predicted_value).all() or set(new.row_id) != set(held.row_id):
            raise ValueError("An anchored control has unsupported study values")
        selected = old[old.method.isin({"pdo_mean", STRUCTURE, "2d_ridge", "2d_structured"})]
        receipt = freeze_table(pd.concat([selected, new.assign(track=study, fold=0)]), args.out / filename, provenance={
            "original_study_sha256": seal["sha256"], "teacher_sha256": teacher_hash,
            "study": study, "previous_score_sha256": SCORE_SHA256,
        })
        hashes[study] = receipt["sha256"]
    publish_json(args.out / "receipt.json", {"forecast_sha256": hashes, "previous_score_sha256": SCORE_SHA256})


def evaluate(args):
    metadata, _ = inputs(args.pdo)
    truth = pd.read_parquet(args.pdo / "observations.parquet", columns=["row_id", "endpoint_value"])
    manifest = json.loads((args.forecasts / "receipt.json").read_text())
    if manifest["previous_score_sha256"] != SCORE_SHA256:
        raise ValueError("Expected the locked original score receipt")
    studies = {}
    for study in STUDIES:
        filename = study.replace(" ", "_") + ".parquet"
        forecast, receipt = verify_forecast(args.forecasts / filename)
        if receipt["sha256"] != manifest["forecast_sha256"][study]:
            raise ValueError("Anchored PDO predictions changed")
        if not set(forecast.row_id) <= set(metadata.loc[metadata.source.eq(study), "row_id"]):
            raise ValueError("Study identity mismatch in anchored 3D controls")
        rows = forecast.merge(truth, on="row_id", validate="many_to_one")
        rows.endpoint_value *= -1
        ranking, _ = score(rows, endpoint_direction="higher", rank_only=True, comparisons=COMPARISONS)
        sufficient = rows.lineage_group.nunique() >= 5
        drugs, _ = score_across_cells(rows, min_lineages=5 if sufficient else 4)
        associations = drugs.groupby("method")[["spearman", "pearson"]].mean()
        intervals = [{"candidate": a, "baseline": b,
                      "ci95": crossed_patient_drug_ci(rows, a, b, repeats=2400)}
                     for a, b in COMPARISONS] if sufficient else []
        for adapted, original in ((STRUCTURE, "2d_structured"), (RIDGE, "2d_ridge")):
            if not np.allclose(associations.loc[adapted], associations.loc[original], rtol=1e-7, atol=1e-8):
                raise ValueError("A constant per drug should preserve sensitivity correlations")
        studies[study] = {"sensitivity": associations.reset_index().to_dict("records"),
                          "ranking": ranking["scores"], "paired_ranking": ranking["paired_contrasts"],
                          "crossed_sensitivity_intervals": intervals}
    publish_json(args.out, {"studies": studies, **manifest})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="action", required=True)
    for action, operand in (("predict", "previous"), ("score", "forecasts")):
        command = stages.add_parser(action)
        for name in ("pdo", operand, "out"):
            command.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    {"predict": predict, "score": evaluate}[args.action](args)


if __name__ == "__main__":
    main()
