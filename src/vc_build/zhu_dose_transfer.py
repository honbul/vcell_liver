"""Compare matched physical-dose supervision and paired Zhu measurements."""

import argparse
from contextlib import closing
import json
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from .evaluate import join_observations
from .expanded_panel_cv import RECEIPTS, expanded_targets, registered_inputs
from .external import KEY, SOURCE, native_sensitivity
from .function_data import new_output
from .hcc_3d_repeat_components import repeat_stats
from .hcc_panel_support import GDSC_RECEIPT, read_registered
from .io import freeze_table, publish_json, sha256, verify_forecast
from .licob_dose_sources import CTRP_QC, RAW_SHA256, SCREENS, TREATMENTS_HASH, ctrp, gdsc, prism
from .licob_endpoints import fit_matched
from .limore_corrected import shared_support
from .v0plus_cv import molecular_inputs
from .v0plus_neural import load_auxiliary
from .v0plus_score import axis_metrics, crossed_patient_drug_ci
from .yang_hcc_external import require_hash
from .zhu_hcc_confirmation import (
    ASSETS, BASE, EXPANDED, ZHU, observed, read_rosters, training_bundle,
)

ORIGINAL = "663eb5717b30c6f03ea43c04c4bb076bd233ea0c05405450e17f0c9e85f7fa86"
FORECAST = "b82b797eb121ac56e9618dec23709d03d2356f1fe0dfded9a8e842189f2369fb"
DOSE = 10.0
MIN_COMPOUNDS, MIN_FIT_CELLS = 12, 30
TABLE = "dose.parquet"
MAIN = "matched_endpoint"
REPLAY = (BASE, "drug_ridge", "cellhit", "icc_offset_drug_mean", "icc_offset_codeae_inductive")
COMPARATORS = ("matched_auc", "drug_mean", *REPLAY)
PAIRED_METHODS = (*REPLAY, "icc_anchor_only", "icc_anchor_and_residual")
IDENTITY = ["cell_id", "compound_id", *SOURCE]
NATIVE_METADATA = ["native_compound_id", "source_run", "dose_min_uM", "dose_max_uM"]


def dose_value(dose, viability, *, minimum, second, maximum):
    """Allow an administered setting or short interpolation in log concentration."""
    if len(dose) < 5:
        return None
    y = np.clip(viability, 0, 1)
    direct = np.flatnonzero(np.isclose(dose, DOSE, rtol=.005, atol=0))
    if len(direct):
        return (float(y[direct[np.argmin(np.abs(dose[direct] - DOSE))]]),)
    j = np.searchsorted(dose, DOSE)
    if j == 0 or j == len(dose) or dose[j] / dose[j - 1] > 4.001:
        return None
    return (float(np.interp(np.log(DOSE), np.log(dose), y)),)


def restore_native_metadata(fit, args):
    # The expanded model table retained the signed row ID but stripped raw assay identifiers.
    columns = ["row_id", *IDENTITY, *NATIVE_METADATA]
    _, core, _, extension = registered_inputs(args, include_extension=True)
    receipt = args.gdsc_extension / "receipt.json"
    require_hash(receipt, GDSC_RECEIPT)
    digest = json.loads(receipt.read_text())["artifacts"]["additional_rows.parquet"]
    gdsc = read_registered(args.gdsc_extension / "additional_rows.parquet", digest, columns)
    registry = pd.concat([core[columns], extension[columns], gdsc])
    source = fit.drop(columns=NATIVE_METADATA, errors="ignore").merge(
        registry, on=["row_id", *IDENTITY], how="left", validate="one_to_one",
    )
    if source.native_compound_id.isna().any() or source.native_compound_id.eq("").any():
        raise ValueError("Cannot recover the exact raw identifiers of every selected source row")
    return source


def extract(args):
    require_hash(args.original / "receipt.json", ORIGINAL)
    original = json.loads((args.original / "receipt.json").read_text())["inputs"]
    for name in ("ctrp_cells", "ctrp_experiments", "depmap_models", "prism_curves"):
        require_hash(Path(original[name]["path"]), original[name]["sha256"])
    source, _, _, _, _, _ = training_bundle(args)
    roster = read_rosters(args.zhu)["single_dose"]
    source = source[source.source_id.eq(args.assay) & source.compound_id.isin(roster.compound_id)].copy()
    source = restore_native_metadata(source, args)
    if source.duplicated(["cell_id", "compound_id"]).any():
        raise ValueError("Dose extraction requires one registered protocol per source, drug and cell")
    # Legacy extractors use maximum and second to prefilter the PRISM physical range.
    source = source.assign(minimum=DOSE, second=DOSE, maximum=DOSE)
    columns = ["dose_viability"]
    if args.assay == "CTRPv2":
        values, qc = ctrp(source, args.raw, original, dose_value, columns)
        raw_hash = CTRP_QC
    elif args.assay == "GDSC2":
        values, qc = gdsc(source, args.raw, original, dose_value, columns)
        raw_hash = RAW_SHA256
    else:
        values, qc = prism(source, args.raw, original["prism_curves"]["path"], dose_value, columns)
        raw_hash = TREATMENTS_HASH
    result = source[IDENTITY + ["endpoint_value"]].merge(values, on=["cell_id", "compound_id"], validate="one_to_one")
    with new_output(args.out) as directory:
        result.to_parquet(directory / TABLE, index=False)
        publish_json(directory / "receipt.json", {
            "cohort": ZHU, "expanded": EXPANDED, "core": RECEIPTS["core"], "source": args.assay,
            "native_metadata_registries": [RECEIPTS["extension"], GDSC_RECEIPT],
            "raw_sha256": raw_hash, "curve_sha256": sha256(directory / TABLE), "qc": qc,
            "eligible_before_curve_check": len(source), "matched_rows": len(result),
            "matched_drugs": int(result.compound_id.nunique()),
        })


def old_forecast(path):
    predictions, seal = verify_forecast(path)
    if (seal["sha256"], seal["provenance"]["cohort_sha256"]) != (FORECAST, ZHU):
        raise ValueError("Expected the first Zhu frozen predictions")
    return predictions


def predict(args):
    fit, _, rna, _, _, added = training_bundle(args)
    rosters = read_rosters(args.zhu)
    repeats, hashes, support = [], {}, {}
    for source in SCREENS:
        path = args.extracts / source
        stamp = json.loads((path / "receipt.json").read_text())
        if (stamp["cohort"], stamp["expanded"], stamp["core"], stamp["source"]) != (
                ZHU, EXPANDED, RECEIPTS["core"], source):
            raise ValueError("Unexpected source extraction scope")
        if stamp["native_metadata_registries"] != [RECEIPTS["extension"], GDSC_RECEIPT]:
            raise ValueError("Unverified raw model and assay identifiers")
        require_hash(path / TABLE, stamp["curve_sha256"])
        table = pd.read_parquet(path / TABLE)
        if len(table) != stamp["matched_rows"] or not table.source_id.eq(source).all():
            raise ValueError("Source extraction disagrees with its receipt")
        repeats.append(table)
        hashes[source] = sha256(path / "receipt.json")
        support[source] = {k: stamp[k] for k in ("matched_rows", "matched_drugs", "eligible_before_curve_check")}
    physical = pd.concat(repeats, ignore_index=True)
    counts = physical.groupby("compound_id").cell_id.nunique()
    drugs = counts[counts.ge(MIN_FIT_CELLS)].index
    roster = rosters["single_dose"]
    held = roster[roster.compound_id.isin(drugs)]
    overlap = physical[physical.compound_id.isin(drugs)]
    prior = old_forecast(args.prior)
    paired = rosters["auc"][KEY]
    candidates = prior[prior.method.isin(PAIRED_METHODS)].merge(paired[KEY[1:]], on=KEY[1:],
                                                                validate="many_to_one")
    pieces = [candidates.assign(track="paired_" + candidates.track)]
    audit = {"compounds_supported": len(drugs), "fit_rows": len(overlap),
             "fit_cells_by_drug": counts.to_dict(), "used_rows_by_source": overlap.source_id.value_counts().to_dict(),
             "sources": support, "fit_performed": len(drugs) >= MIN_COMPOUNDS}
    if audit["fit_performed"]:
        fit = fit.merge(overlap, on=[*IDENTITY, "endpoint_value"], validate="one_to_one")
        if len(fit) != len(overlap):
            raise ValueError("Dosage and native AUC must have identical training identities")
        # The shared fit routine negates viability and uses one equally weighted drug mean per source.
        fit["source_auc"] = -native_sensitivity(fit)
        edges, targets, _ = molecular_inputs(args.priors, args.hallmark, args.targets)
        targets, _ = expanded_targets(fit[fit.compound_id.isin(added)], targets, rna.columns)
        auxiliary, _ = load_auxiliary(args.auxiliary)
        estimates, features, solves = fit_matched(fit, held, rna, edges, targets, auxiliary, "dose_viability")
        native_means = (-fit.source_auc).groupby([fit[c] for c in [*SOURCE, "compound_id"]]).mean()
        native_means = native_means.groupby(level="compound_id").mean()
        physical_anchor = estimates[estimates.method.eq("drug_mean")].groupby("compound_id").predicted_value.first()
        mask = estimates.method.eq("matched_auc")
        estimates.loc[mask, "predicted_value"] += estimates.loc[mask, "compound_id"].map(physical_anchor - native_means)
        references = prior[prior.track.eq("single_dose") & prior.row_id.isin(held.row_id) & prior.method.isin(REPLAY)]
        joint = pd.concat([estimates, references], ignore_index=True).assign(track="dose_matched")
        shared_support(joint, held, (MAIN, *COMPARATORS))
        pieces.append(joint)
        audit.update(features=features, solves=solves)
    freeze_table(pd.concat(pieces, ignore_index=True), args.out, provenance={
        "cohort": ZHU, "prior_sha256": FORECAST, "extracts": hashes, "dose_support": audit,
    })


def paired_assays(predictions, truth):
    key = ["cell_id", "lineage_group", "compound_id"]
    a, b = (truth[name].pivot(index="lineage_group", columns="compound_id", values="endpoint_value")
            for name in ("auc", "single_dose"))
    b = b.loc[a.index, a.columns]
    if a.shape != (16, 18) or not a.notna().all(axis=None) or not b.notna().all(axis=None):
        raise ValueError("Expected complete same-donor, same-drug endpoint overlap")
    scores = {}
    for endpoint, observations in truth.items():
        section = predictions[predictions.track.eq("paired_" + endpoint)]
        labels = observations.merge(truth["auc"][key], on=key, validate="one_to_one")
        shared_support(section, labels, PAIRED_METHODS)
        scores[endpoint] = axis_metrics(join_observations(section, labels), ())[0]
    return {"observed_concordance": repeat_stats(a, b, reciprocal=True), "same_forecasts_by_endpoint": scores}


def score(args):
    predictions, seal = verify_forecast(args.forecast)
    if (seal["provenance"]["cohort"], seal["provenance"]["prior_sha256"]) != (ZHU, FORECAST):
        raise ValueError("Missing the matched source prediction freeze")
    rosters = read_rosters(args.zhu)
    require_hash(args.labels, ASSETS["zhu-s4.xlsx"])
    with closing(load_workbook(args.labels, read_only=True, data_only=True)) as book:
        truth = {name: observed(book, name, roster) for name, roster in rosters.items()}
    outcome = {
        "forecast_sha256": seal["sha256"], "label_sha256": ASSETS["zhu-s4.xlsx"],
        "support": seal["provenance"]["dose_support"], "paired_assay": paired_assays(predictions, truth),
    }
    candidate = predictions[predictions.track.eq("dose_matched")]
    if not candidate.empty:
        joined = join_observations(candidate, truth["single_dose"])
        pairs = [(MAIN, b) for b in COMPARATORS]
        values = axis_metrics(joined, pairs)[0]
        ranking = {(x["baseline"], x["metric"]): x["ci95"][0] for x in values["paired_contrasts"]}
        intervals = {b: crossed_patient_drug_ci(joined, MAIN, b, repeats=2400) for b in COMPARATORS}
        gates = {b: all(intervals[b][m][0] > 0 for m in ("spearman", "pearson"))
                 and all(ranking[b, m] > -.005 for m in ("spearman", "ndcg_at_5", "top5_overlap"))
                 for b in COMPARATORS}
        outcome["matched_10uM"] = {
            "metrics": values, "intervals": intervals, "gates": gates, "joint_pass": all(gates.values()),
            "rows": joined.row_id.nunique(), "donors": joined.lineage_group.nunique(),
        }
    publish_json(args.out, outcome)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    tasks = parser.add_subparsers(dest="stage", required=True)
    training = ("zhu", "core", "expanded", "icc", "licob")
    for stage, fields in {
        "extract": (*training, "original", "extension", "gdsc-extension", "raw"),
        "predict": (*training, "extracts", "prior", "priors", "hallmark", "targets", "auxiliary"),
        "score": ("zhu", "forecast", "labels"),
    }.items():
        p = tasks.add_parser(stage)
        for field in ("out", *fields):
            p.add_argument("--" + field, type=Path, required=True)
        if stage == "extract":
            p.add_argument("--assay", choices=SCREENS, required=True)
    args = parser.parse_args()
    {"extract": extract, "predict": predict, "score": score}[args.stage](args)


if __name__ == "__main__":
    main()
