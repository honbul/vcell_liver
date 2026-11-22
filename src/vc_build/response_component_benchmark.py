"""Freeze a source-grouped and external HCC component-supervised benchmark."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .assay_consensus_cv import PREVIOUS_SCORE
from .evaluate import join_observations
from .expanded_panel_cv import RECEIPTS, registered_inputs, same_split
from .external import KEY
from .io import freeze_table, publish_json, sha256, verify_forecast
from .licob_full_axis import validated as licob_registration
from .licob_node_recovery import joined_outcomes
from .limore_corrected import OBSERVATIONS, shared_support
from .pdo_transfer_cv import device_cuda
from .response_components import ARMS, PRIMARY, component_scores, forecast, interaction_intervals
from .structured_cv import ASSAYS, attach, paired_intervals, with_assay
from .v0plus_cv import molecular_inputs, training_inputs, validation_roster
from .v0plus_licob import ASSET_HASHES, adult_donor_outcomes
from .v0plus_release import load_query
from .v0plus_score import axis_metrics, compound_strata, crossed_patient_drug_ci
from .yang_hcc_external import require_hash

EXTERNAL_TRACKS = ("LIMORE", "LICOB")
EXTERNAL_HASHES = {
    "LIMORE": "fe7f9e656801d63fd1263ec3e94093cad86b049130aa038d4f2e8fdd059b02bb",
    "LICOB": "fc5ab2b2902c2ee0fdca5b4027544208f6630fe2ec7b7304615ab36389c1dde7",
}
CV_BASELINES = {"corrected_core": "structured", "corrected_drug_mean": "drug_mean", "corrected_ridge": "drug_ridge"}
BASELINES = tuple(CV_BASELINES.values())
PUBLIC = ("cellhit", "codeae_inductive", "velodrome_inductive", "v0plus_inductive")
CORE_PAIRS = tuple((PRIMARY, model) for model in ("structured", "drug_ridge", "marginal_only", "interaction_only"))
PAIR_INTERACTION = ((PRIMARY, "structured"), (PRIMARY, "drug_ridge"))
FORECAST_SCHEMA = [*KEY, "track", "fold", "method", "predicted_value"]


def external_inputs(args, study):
    require_hash(args.core / "receipt.json", RECEIPTS["core"])
    fit, rna, _ = training_inputs(args.core)
    if study == "LIMORE":
        require_hash(args.original / "receipt.json", RECEIPTS["original"])
        query, held, _ = load_query(args.original, fit)
        registered = RECEIPTS["original"]
    else:
        registered = licob_registration(args.licob)
        query = pd.read_parquet(args.licob / "query.parquet")
        held = pd.read_parquet(args.licob / "rna.parquet")
    genes = rna.columns.intersection(held.columns)
    joint = pd.concat([rna[genes], held[genes]], verify_integrity=True)
    expected = (50, 68) if study == "LIMORE" else (43, 53)
    if (query.cell_id.nunique(), query.compound_id.nunique()) != expected or len(query) != expected[0] * expected[1]:
        raise ValueError("External RNA or response panel identity changed")
    return fit, query, joint, registered


def external_reference(args, study, query):
    existing, seal = verify_forecast(args.archive / study / "final.parquet")
    if seal["sha256"] != EXTERNAL_HASHES[study]:
        raise ValueError("Unexpected registered external comparison")
    prefix = "corrected/" if study == "LIMORE" else "wide/"
    references = [*BASELINES, *PUBLIC]
    parts = existing[existing.method.isin({prefix+name for name in references})].copy()
    parts.method = parts.method.str.removeprefix(prefix)
    shared_support(parts, query, references)
    return parts[FORECAST_SCHEMA].assign(track=study, fold=0), seal["sha256"]


def source_reference(path, mode, fold):
    require_hash(path / "score.json", PREVIOUS_SCORE)
    hashes = json.loads((path / "score.json").read_text())["forecast_sha256"]
    key = f"{mode}-{fold}"
    data, seal = verify_forecast(path / "forecasts" / f"{key}.parquet")
    if seal["sha256"] != hashes[key]:
        raise ValueError("Corrected paired source forecasts changed")
    subset = data[data.method.isin(CV_BASELINES)].copy()
    subset.method = subset.method.map(CV_BASELINES)
    return subset[FORECAST_SCHEMA], seal["sha256"]


def add_blend(rows):
    subset = rows[rows.method.isin([PRIMARY, "drug_ridge"])]
    by_row = subset.groupby([*KEY, "track", "fold"], as_index=False)
    if not by_row.size()["size"].eq(2).all():
        raise ValueError("Component/Ridge blend requires paired rows")
    combined = by_row.predicted_value.mean().assign(method=ARMS[-1])
    return pd.concat([rows, combined], ignore_index=True)[FORECAST_SCHEMA]


def predict(args):
    from .v0plus_neural import load_auxiliary

    edges, targets, priors = molecular_inputs(args.priors, args.hallmark, args.targets)
    auxiliary, aux = load_auxiliary(args.auxiliary)
    if args.mode in EXTERNAL_TRACKS:
        fit, roster, rna, registered = external_inputs(args, args.mode)
        previous, reference = external_reference(args, args.mode, roster)
        new, audit = forecast(fit, roster, rna, edges, targets, auxiliary, device_cuda())
        new = new.assign(track=args.mode, fold=0)
        joint = add_blend(pd.concat([previous, new]))
        shared_support(joint, roster, [*BASELINES, *PUBLIC, *ARMS])
    else:
        old, corrected, rna, _ = registered_inputs(args)
        _, fit, check = same_split(old, corrected, args.fold, args.mode)
        roster = validation_roster(check)
        previous, reference = source_reference(args.cv_archive, args.mode, args.fold)
        computed, audit = forecast(fit, roster, rna, edges, targets, auxiliary, device_cuda())
        new = with_assay(computed, check, args.fold)[FORECAST_SCHEMA]
        matrix = pd.concat([previous, new]).pivot(index=[*KEY, "track", "fold"], columns="method", values="predicted_value")
        if set(matrix.columns) != {*BASELINES, *ARMS[:-1]} or not np.isfinite(matrix.to_numpy()).all():
            raise ValueError("Component candidates do not match the original locked source support")
        joint = add_blend(pd.concat([previous, new]))
        registered = RECEIPTS["core"]
    freeze_table(joint, args.out, provenance={"mode": args.mode, "fold": args.fold, "inputs_sha256": registered,
        "core_sha256": RECEIPTS["core"], "reference_sha256": reference, "priors": priors, "auxiliary": aux, "training": audit})


def check_component(path, name, fold, digest):
    rows, seal = verify_forecast(path)
    info = seal["provenance"]
    if (info["mode"], info["fold"], info["inputs_sha256"], info["core_sha256"]) != (
            name, fold, digest, RECEIPTS["core"]):
        raise ValueError("Invalid benchmark component provenance")
    return rows, seal["sha256"]


def score_source(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    old, corrected, _, _ = registered_inputs(args)
    by_mode, hashes = {}, {}
    for mode in ASSAYS:
        folds = []
        for fold in range(3):
            task = f"{mode}-{fold}"
            rows, hashes[task] = check_component(args.forecasts / f"{task}.parquet", mode, fold, RECEIPTS["core"])
            _, _, labels = same_split(old, corrected, fold, mode)
            folds.append(attach(rows, labels))
        matched = pd.concat(folds, ignore_index=True)
        summary = {}
        for track, section in matched.groupby("track"):
            results, _, _ = axis_metrics(section, CORE_PAIRS)
            results["crossed_ci95"] = [{"candidate": a, "baseline": b, "ci95": paired_intervals(section, a, b)}
                                       for a,b in PAIR_INTERACTION]
            results["components_by_fold"] = {str(fold): component_scores(rows, cell_axis="lineage_group")
                                             for fold, rows in section.groupby("fold")}
            for groups in results["components_by_fold"].values():
                if groups["support"] < 5:
                    raise ValueError("Unusable rate support")
            means = {x["method"]: x for x in results["scores"]}
            marginal_delta = [x for x in results["paired_contrasts"] if x["candidate"] == PRIMARY
                              and x["baseline"] == "structured" and x["metric"] in ("spearman", "ndcg_at_5")]
            gate = all(v[0] > 0 for v in results["crossed_ci95"][0]["ci95"].values())
            results["gate_vs_structured"] = bool(gate and len(marginal_delta) == 2
                and all(x["ci95"][0] > -.005 for x in marginal_delta)
                and all(means[PRIMARY][x] >= means["drug_mean"][x] for x in ("macro_spearman", "macro_ndcg_at_5")))
            summary[track] = results
        by_mode[mode] = summary
    publish_json(args.out, {"stage": "source", "corrected_core_sha256": RECEIPTS["core"],
        "development_gate": sum(next(iter(by_mode[m].values()))["gate_vs_structured"] for m in ASSAYS if m != "pooled") >= 2,
        "results": by_mode, "forecast_sha256": hashes})


def rich_metrics(frame, query):
    pairs = (*CORE_PAIRS, (PRIMARY, "cellhit"), (ARMS[-1], "drug_ridge"))
    report, _, _ = axis_metrics(frame, pairs)
    interaction = component_scores(frame, cell_axis="lineage_group")
    intervals = [{"candidate": a, "baseline": b, "ci95": crossed_patient_drug_ci(frame, a, b)} for a, b in pairs]
    interaction_ci = [{"candidate": a, "baseline": b, "ci95": interaction_intervals(frame, a, b)}
                      for a, b in (*PAIR_INTERACTION, (PRIMARY, "cellhit"))]
    ranks = {(r["baseline"], r["metric"]): r["ci95"] for r in report["paired_contrasts"] if r["candidate"] == PRIMARY}
    standard = {r["baseline"]: r["ci95"] for r in intervals if r["candidate"] == PRIMARY}
    gate = {r["baseline"]: r["ci95"][0] > 0 and all(standard[r["baseline"]][m][0] > 0 for m in ("pearson", "spearman"))
             and all(ranks[r["baseline"], m][0] > -.005 for m in ("spearman", "ndcg_at_5")) for r in interaction_ci}
    strata = {}
    for name in ("hcc_tki", "targeted_or_kinase", "nonselective_cytotoxics"):
        ids = compound_strata(query)[name]
        if ids:
            observed = frame[frame.compound_id.isin(ids)]
            sub = axis_metrics(observed, ())[0]
            if len(ids) <= 5:
                for row in sub["scores"]:
                    row.pop("macro_ndcg_at_5", None)
                    row.pop("macro_top5_overlap", None)
            strata[name] = {"drugs": len(ids), "report": sub, "components": component_scores(observed, cell_axis="lineage_group")}
    return {"models": int(frame.cell_id.nunique()), "observations": int(frame.row_id.nunique()),
            "report": report, "components": interaction, "paired_interaction_ci95": interaction_ci,
            "paired_drug_ci95": intervals, "gate": gate, "joint_gate": all(gate.values()), "strata": strata}


def score_external(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    result, seals = {}, {}
    for study in EXTERNAL_TRACKS:
        _, query, _, digest = external_inputs(args, study)
        rows, seals[study] = check_component(args.forecasts / f"{study}.parquet", study, 0, digest)
        shared_support(rows, query, [*BASELINES, *PUBLIC, *ARMS])
        if study == "LIMORE":
            require_hash(args.limore_labels, OBSERVATIONS)
            full = join_observations(rows, pd.read_parquet(args.limore_labels))
            result[study] = {"all": rich_metrics(full, query)}
        else:
            require_hash(args.licob_labels, ASSET_HASHES["supplement"])
            full = joined_outcomes(query, rows, args.licob_labels)
            result[study] = {"all": rich_metrics(adult_donor_outcomes(full), query),
                             "nine": rich_metrics(adult_donor_outcomes(full[full.new_donor]), query)}
    publish_json(args.out, {"stage": "external_technical", "results": result, "forecast_sha256": seals,
        "both_external_gate": all(result[s]["all"]["joint_gate"] for s in EXTERNAL_TRACKS)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="action", required=True)
    paths = {"predict": ("original", "core", "licob", "archive", "cv_archive", "priors", "hallmark", "targets", "auxiliary"),
             "source": ("original", "core", "forecasts"),
             "external": ("original", "core", "licob", "forecasts", "limore_labels", "licob_labels")}
    for action, fields in paths.items():
        command = stages.add_parser(action)
        for field in (*fields, "out"):
            command.add_argument("--" + field.replace("_", "-"), required=True, type=Path)
        if action == "predict":
            command.add_argument("--mode", choices=(*ASSAYS, *EXTERNAL_TRACKS), required=True)
            command.add_argument("--fold", choices=range(3), type=int, default=0)
    args = parser.parse_args()
    {"predict": predict, "source": score_source, "external": score_external}[args.action](args)


if __name__ == "__main__":
    main()
