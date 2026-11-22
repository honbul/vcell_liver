"""Isolate restored exact-parent PRISM measurements on the fixed LICOB panel."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .expanded_panel_cv import expanded_targets
from .external import KEY
from .io import freeze_table, publish_json, sha256, verify_forecast
from .licob_cold import checked, models
from .licob_node_recovery import joined_outcomes
from .limore_corrected import shared_support
from .pdo_transfer_cv import device_cuda
from .v0plus_cv import molecular_inputs
from .v0plus_licob import ASSET_HASHES, adult_donor_outcomes
from .v0plus_neural import load_auxiliary
from .v0plus_score import axis_metrics, crossed_patient_drug_ci
from .vinorelbine_transfer import EXTENSION_SHA
from .yang_hcc_external import require_hash


OLD_INPUT = "67c11aba490282e30be9ef6d3d4d28ad192a2f5620a9e0b84c978019a0ccdb9e"
OLD_FORECAST = "4cc7e3afa71fee91c0ca9a12ce06002d7a1016f9d4951b1833237830e3d7a2a1"
RESTORED = {
    "OMKHWTRUYNAGFG-IEBDPFPHSA-N": 466,
    "PIQCTGMSNWUMAF-UHFFFAOYSA-N": 670,
    "YOZBGTLTNGAVFU-UHFFFAOYSA-N": 420,
    "HZTYDQRUAWIZRE-UHFFFAOYSA-N": 417,
    "QJZRFPJCWMNVAV-HHHXNRCGSA-N": 571,
    "MZDKLVOWGIOKTN-UHFFFAOYSA-N": 461,
    "PHXJVRSECIGDHY-UHFFFAOYSA-N": 671,
}
METHODS = ("old_mean", "old_interaction", "restored_residual", "restored_direct")
PRIMARY = "restored_residual"
RANK_METRICS = {"spearman", "ndcg_at_5", "top5_overlap"}


def predict(args):
    area = args.historical / "inputs"
    if checked(area, ("train", "query", "rna")) != OLD_INPUT:
        raise ValueError("Unexpected historical fit")
    historical, history = verify_forecast(args.historical / "parts/fold-3.parquet")
    if history["sha256"] != OLD_FORECAST or history["provenance"]["input_sha256"] != OLD_INPUT:
        raise ValueError("Unexpected historical predictions")
    train, query, rna = (pd.read_parquet(area / f"{key}.parquet") for key in ("train", "query", "rna"))
    metadata = query.drop_duplicates("compound_id").set_index("compound_id")
    if set(train.compound_id) & set(RESTORED) or not metadata.loc[list(RESTORED), "remaining_unspecified_stereo"].eq(0).all():
        raise ValueError("Restored axis is not an exact, previously unobserved parent set")
    receipt = args.extension / "receipt.json"
    require_hash(receipt, EXTENSION_SHA)
    artifact = args.extension / "additional_rows.parquet"
    require_hash(artifact, json.loads(receipt.read_text())["artifacts"][artifact.name])
    added = pd.read_parquet(artifact, filters=[("compound_id", "in", list(RESTORED))])
    if (not added.source_id.eq("PRISM20Q2").all() or added.groupby("compound_id").size().to_dict() != RESTORED
            or not set(added.cell_id) <= set(train.cell_id) or set(added.lineage_group) & set(query.lineage_group)):
        raise ValueError("Changed PRISM, training lineage or query support")
    combined = pd.concat([train, added.assign(core_rows=False)[train.columns]], ignore_index=True)
    edges, targets, priors = molecular_inputs(args.priors, args.hallmark, args.targets)
    reference = pd.concat([train, query], ignore_index=True)
    targets, _ = expanded_targets(reference[~reference.compound_id.isin(targets)], targets, rna.columns)
    auxiliaries, aux = load_auxiliary(args.auxiliary)
    predictions, fit = models(combined, query, rna, edges, targets, auxiliaries, device_cuda())
    if history["provenance"]["model"]["calibration"] != fit["calibration"]:
        raise ValueError("Endpoint calibration moved after adding disjoint parents")
    old = historical.pivot(index="row_id", columns="method", values="predicted_value").reindex(query.row_id)
    new = predictions.pivot(index="row_id", columns="method", values="predicted_value").reindex(query.row_id)
    restored = query.compound_id.isin(RESTORED).to_numpy()
    mean = old.chemical_mean
    parent = old.interaction_transfer
    residual = mean + (new.interaction_transfer - new.chemical_mean)
    arrays = [mean, parent, parent.where(~restored, residual), parent.where(~restored, new.interaction_transfer)]
    outputs = pd.concat([query[KEY].assign(method=method, predicted_value=array.to_numpy())
                         for method, array in zip(METHODS, arrays)], ignore_index=True)
    shared_support(outputs, query, METHODS)
    freeze_table(outputs.assign(track="LICOB_PRISM_RESTORED", fold=0), args.run / "forecast.parquet", provenance={
        "refitted": fit, "old_input_sha256": OLD_INPUT, "old_forecast_sha256": OLD_FORECAST,
        "restored_prism_receipt": EXTENSION_SHA, "priors": priors, "auxiliary": aux["endpoints"],
        "added": RESTORED, "added_hcc": added[added.lineage_role.eq("adult_hcc")].groupby("compound_id").size().to_dict(),
    })


def score(args):
    area = args.historical / "inputs"
    if checked(area, ("query",)) != OLD_INPUT:
        raise ValueError("Unexpected LICOB evaluation roster")
    query = pd.read_parquet(area / "query.parquet")
    rows, seal = verify_forecast(args.run / "forecast.parquet")
    shared_support(rows, query, METHODS)
    require_hash(args.supplement, ASSET_HASHES["supplement"])
    values = adult_donor_outcomes(joined_outcomes(query, rows, args.supplement))
    mapping = rows[rows.method.eq("old_mean")].groupby("compound_id").predicted_value.first()
    mask = values.method.eq("old_mean")
    values.loc[mask, "predicted_value"] = values.loc[mask, "compound_id"].map(mapping)
    pairs = [(candidate, baseline) for candidate in METHODS[2:] for baseline in METHODS[:2]]
    result = {}
    for scope, subset in (("restored", values[values.compound_id.isin(RESTORED)]), ("full", values)):
        report, drugs, _ = axis_metrics(subset, pairs)
        patient = {f"{a}:{b}": crossed_patient_drug_ci(subset, a, b) for a, b in pairs}
        result[scope] = {"report": report, "within_drug_ci95": patient, "donors": subset.lineage_group.nunique(),
                         "drugs": subset.compound_id.nunique(),
                         "per_drug": drugs.astype(object).where(drugs.notna(), None).to_dict("records")}
    primary = result["restored"]
    acceptable = {}
    for baseline in METHODS[:2]:
        ranked = {row["metric"]: row["ci95"][0] for row in primary["report"]["paired_contrasts"]
                  if row["candidate"] == PRIMARY and row["baseline"] == baseline}
        acceptable[baseline] = (all(ci[0] > 0 for ci in
                                    primary["within_drug_ci95"][f"{PRIMARY}:{baseline}"].values())
                                and all(ranked.get(metric, float("-inf")) > -.005 for metric in RANK_METRICS))
    publish_json(args.run / "score.json", {
        "freeze_sha256": seal["sha256"], "outcome_sha256": ASSET_HASHES["supplement"],
        "results": result, "primary_checks": acceptable, "pass": all(acceptable.values()),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    for name, task, paths in (("predict", predict, ("extension", "priors", "hallmark", "targets", "auxiliary")),
                              ("score", score, ("supplement",))):
        stage = stages.add_parser(name)
        for arg in ("run", "historical", *paths):
            stage.add_argument("--" + arg, type=Path, required=True)
        stage.set_defaults(task=task)
    args = parser.parse_args()
    args.task(args)


if __name__ == "__main__":
    main()
