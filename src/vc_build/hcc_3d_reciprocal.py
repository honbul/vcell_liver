"""Freeze reciprocal 3D HCC forecasts on 21 exact matched compounds."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import join_observations, score_across_cells
from .external import KEY, _validate
from .function_data import new_output
from .hcc_3d_study_transfer import (
    COHORT_HASHES, DONORS_HASH, SOURCE_ARMS, STUDIES, fit_source_forecast, parameters,
    recipient_outcomes, target_inputs,
)
from .io import freeze_table, publish_json, sha256, verify_forecast
from .pdo_transfer_cv import device_cuda
from .v0plus_score import axis_metrics, compound_strata, crossed_patient_drug_ci
from .yang_hcc_adaptation import BENCHMARK_INPUTS, aggregate, predictor
from .yang_hcc_external import SETTINGS, read_inputs, require_hash

SELECTED = {"pdo_ridge": [2.], "pdo_structured": [2.]}
PRIMARY = "2d_pdo_average"
CANDIDATES = {*SELECTED, "drug_mean", "2d_anchor_pdo_ridge", "3d_anchor_2d_shape", PRIMARY}
METHODS = SOURCE_ARMS | CANDIDATES | {"2d_cellhit"}
BASELINES = ("2d_ridge", "2d_teacher")
PAIRS = tuple((PRIMARY, name) for name in (*BASELINES, "2d_mean", "2d_cellhit", "drug_mean")) + (
    ("pdo_ridge", "drug_mean"), ("2d_anchor_pdo_ridge", "pdo_ridge"),
    ("3d_anchor_2d_shape", "2d_teacher"), ("2d_anchor_pdo_ridge", "2d_mean"),
)


def prepare(args):
    read_inputs(args.benchmark, BENCHMARK_INPUTS)
    for name, path in zip(STUDIES, (args.licob, args.wong)):
        require_hash(path / "receipt.json", COHORT_HASHES[name])
    sources = [pd.read_parquet(path / "query.parquet", columns=["compound_id"])
               for path in (args.licob, args.wong)]
    compounds = sorted(set(sources[0].compound_id) & set(sources[1].compound_id))
    if len(compounds) != 21:
        raise ValueError("Expected the 21 exact previously registered common parents")
    train = pd.read_parquet(args.benchmark / "train.parquet")
    source_rna = pd.read_parquet(args.benchmark / "rna.parquet").loc[sorted(set(train.cell_id))]
    rna, rosters = [source_rna], {}
    for study, directory in zip(STUDIES, (args.licob, args.wong)):
        meta, expression = target_inputs(study, directory, args.patient_audit, compounds=compounds)
        rosters[study] = meta
        rna.append(expression)
    genes = sorted(set.intersection(*(set(data.columns) for data in rna)))
    if len(genes) < 10000:
        raise ValueError("Insufficient common established RNA axis")
    all_rna = pd.concat([x[genes] for x in rna], verify_integrity=True)
    query = pd.concat([meta[[*KEY, "isomeric_smiles"]].assign(study=study)
                       for study, meta in rosters.items()], ignore_index=True)
    _validate(train, query)
    labels = (rosters["Wong"].drop_duplicates("compound_id").set_index("compound_id")
              .source_compound_label.to_dict())
    files = {"train2d.parquet": train, "rna.parquet": all_rna, "application.parquet": query}
    files.update({f"{study}.parquet": frame for study, frame in rosters.items()})
    with new_output(args.out) as temp:
        checksums = {}
        for name, table in files.items():
            path = temp / name
            table.to_parquet(path, index=name == "rna.parquet")
            checksums[name] = sha256(path)
        publish_json(temp / "receipt.json", {
            "benchmark_inputs_sha256": BENCHMARK_INPUTS, "recipient_cohort_sha256": COHORT_HASHES,
            "licob_patient_sha256": DONORS_HASH, "gene_count": len(genes), "compound_labels": labels,
            "support": {study: {"patients": data.lineage_group.nunique(), "organoids": data.cell_id.nunique(),
                                "expected_rows": len(data)} for study, data in rosters.items()},
            "artifacts": checksums,
        })


def checked_inputs(path):
    receipt = path / "receipt.json"
    document = json.loads(receipt.read_text())
    if (document["benchmark_inputs_sha256"], document["recipient_cohort_sha256"], len(document["compound_labels"])) != (
            BENCHMARK_INPUTS, COHORT_HASHES, 21):
        raise ValueError("Unexpected reciprocal study input manifest")
    for filename, checksum in document["artifacts"].items():
        require_hash(path / filename, checksum)
    return document, sha256(receipt)


def source(args):
    _, registered = checked_inputs(args.inputs)
    require_hash(args.settings, SETTINGS)
    train = pd.read_parquet(args.inputs / "train2d.parquet")
    query = pd.read_parquet(args.inputs / "application.parquet")
    if args.component == "cellhit":
        parents = np.array_split(sorted(query.compound_id.unique()), 3)[args.shard]
        query = query[query.compound_id.isin(parents)]
    return fit_source_forecast(args, registered, train, query)


def anchored_forecasts(fit, held, result):
    methods = result.pivot(index="row_id", columns="method", values="predicted_value")
    fit_donors = fit.groupby(["lineage_group", "compound_id"])[["endpoint_value", "2d_teacher"]].mean()
    by_drug = fit_donors.groupby("compound_id")
    scale = by_drug["2d_teacher"].std(ddof=0) / by_drug.endpoint_value.std(ddof=0)
    query = held.set_index("row_id").loc[methods.index]
    relative = query.compound_id.map(scale)
    if not np.isfinite(relative).all() or not relative.gt(0).all():
        raise ValueError("Invalid fit-only scaling between 2D and 3D residuals")
    mean_teacher = query.compound_id.map(by_drug["2d_teacher"].mean())
    shifted = methods["2d_mean"] + relative * (methods["pdo_ridge/2.0"] - methods.drug_mean)
    inverse = methods.drug_mean + (methods["2d_teacher"] - mean_teacher) / relative
    joint = (methods["2d_teacher"] + shifted) / 2
    return {"2d_anchor_pdo_ridge": shifted, "3d_anchor_2d_shape": inverse, PRIMARY: joint}


def predict(args):
    _, registered = checked_inputs(args.inputs)
    source_rows, seal = verify_forecast(args.teacher)
    if (seal["provenance"]["input_sha256"] != registered or seal["provenance"]["mode"] != "teacher"
            or set(source_rows.method) != SOURCE_ARMS):
        raise ValueError("Unexpected corrected 2D comparison in reciprocal fit")
    donor = next(name for name in STUDIES if name != args.study)
    labels = pd.read_parquet(args.inputs / f"{donor}.parquet")
    fit = recipient_outcomes(donor, labels, args.fit_book)
    held = pd.read_parquet(args.inputs / f"{args.study}.parquet")
    if len(fit) != len(labels):
        raise ValueError("Fit observations differ from the previous complete panel audit")
    prior = source_rows.pivot(index="row_id", columns="method", values="predicted_value")[["2d_mean", "2d_teacher"]]
    fit = fit.join(prior, on="row_id", validate="one_to_one")
    held = held.join(prior, on="row_id", validate="one_to_one")
    rna = pd.read_parquet(args.inputs / "rna.parquet")
    edges, targets, provenance = parameters(args)
    panel = {key: value for key, value in targets.items() if key in set(fit.compound_id)}
    result, diagnostics = predictor(fit, held, rna, edges, panel, device_cuda(), SELECTED)
    transfer = anchored_forecasts(fit, held, result)
    extra = [held[KEY].assign(method=arm, predicted_value=held.row_id.map(estimates))
             for arm, estimates in transfer.items()]
    result.method = result.method.str.split("/").str[0]
    full = pd.concat([result[result.method.isin(CANDIDATES)], *extra], ignore_index=True)
    if set(full.method) != CANDIDATES or not np.isfinite(full.predicted_value).all():
        raise ValueError("Incomplete reciprocal transfer estimates")
    freeze_table(full.assign(track=args.study, fold=0), args.out, provenance={
        "input_sha256": registered, "teacher_sha256": seal["sha256"], "fit_study": donor,
        "target_study": args.study, "fit_label_sha256": sha256(args.fit_book),
        "priors": provenance, "solver": diagnostics,
    })


def merge(args):
    _, registered = checked_inputs(args.inputs)
    names = ["teacher", "cellhit0", "cellhit1", "cellhit2", *STUDIES]
    parts, hashes = [], {}
    for name in names:
        forecasts, seal = verify_forecast(args.parts / f"{name}.parquet")
        if seal["provenance"]["input_sha256"] != registered:
            raise ValueError("Component inputs do not match the registered comparison")
        if name in STUDIES and (seal["provenance"]["teacher_sha256"] != hashes["teacher"]
                                or seal["provenance"]["target_study"] != name):
            raise ValueError("Independent 3D transfer component changed")
        if name.startswith("cellhit") and set(forecasts.method) != {"2d_cellhit"}:
            raise ValueError("Unexpected CellHit component arm")
        parts.append(forecasts)
        hashes[name] = seal["sha256"]
    joined = pd.concat(parts, ignore_index=True)
    for study in STUDIES:
        registered_roster = pd.read_parquet(args.inputs / f"{study}.parquet", columns=KEY)
        block = joined[joined.track.eq(study)]
        if (set(block.method) != METHODS or any(len(part) != len(registered_roster)
                or set(part.row_id) != set(registered_roster.row_id) for _, part in block.groupby("method"))):
            raise ValueError("Every comparison method must have the exact same held registry")
    freeze_table(joined, args.out, provenance={"input_sha256": registered, "components": hashes})


def summarize(forecasts, actual, annotations):
    aligned = join_observations(forecasts[forecasts.row_id.isin(actual.row_id)], actual)
    donors = aggregate(aligned)
    table, drugs, _ = axis_metrics(donors, PAIRS)
    sensitivity = [{"candidate": arm, "baseline": base, "ci95": crossed_patient_drug_ci(donors, arm, base)}
                   for arm, base in PAIRS]
    checks = {}
    for base in BASELINES:
        relevant = next(x["ci95"] for x in sensitivity if x["candidate"] == PRIMARY and x["baseline"] == base)
        ranking = [x["ci95"][0] for x in table["paired_contrasts"]
                   if x["candidate"] == PRIMARY and x["baseline"] == base and x["metric"] in ("spearman", "ndcg_at_5")]
        checks[base] = len(relevant) == len(ranking) == 2 and min([v[0] for v in relevant.values()] + ranking) > 0
    strata = {}
    for name, parents in compound_strata(annotations).items():
        if parents:
            eligible = drugs[drugs.compound_id.isin(parents)]
            summary = eligible.groupby("method", as_index=False)[["spearman", "pearson"]].mean()
            strata[name] = {"drugs": len(parents), "metrics": summary.to_dict("records")}
    shifts = {}
    for arm in ("drug_mean", "2d_mean"):
        anchor = donors[donors.method.eq(arm)]
        actual_mean = anchor.groupby("compound_id").endpoint_value.mean()
        forecast_mean = anchor.groupby("compound_id").predicted_value.first()
        if not anchor.groupby("compound_id").predicted_value.nunique().eq(1).all():
            raise ValueError("An exact drug anchor must be constant")
        shifts[arm] = float(forecast_mean.corr(actual_mean, method="spearman"))
    return {"patients": donors.lineage_group.nunique(), "patient_rows": donors.row_id.nunique(),
            "unaggregated_rows": aligned.row_id.nunique(), "overall": table,
            "sensitivity_ci95": sensitivity, "strata": strata, "gates": checks,
            "anchor_across_drug_spearman": shifts, "per_drug": drugs.to_dict("records")}


def score(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    inputs, fingerprint = checked_inputs(args.inputs)
    predictions, seal = verify_forecast(args.forecast)
    if seal["provenance"]["input_sha256"] != fingerprint:
        raise ValueError("Expected the registered reciprocal model freeze")
    scored = {}
    annotations = pd.read_parquet(args.inputs / "Wong.parquet").drop_duplicates("compound_id")
    for study, book in zip(STUDIES, (args.licob_book, args.wong_book)):
        registered = pd.read_parquet(args.inputs / f"{study}.parquet")
        observed = recipient_outcomes(study, registered, book)
        scored[study] = summarize(predictions[predictions.track.eq(study)], observed, annotations)
    publish_json(args.out, {"scope": "reciprocal technical HCC 3D transfer on exposed cohorts",
                            "forecast_sha256": seal["sha256"], "compounds": inputs["compound_labels"],
                            "studies": scored, "bidirectional_gain": all(
                                all(result["gates"].values()) for result in scored.values())})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="action", required=True)
    molecular = ("priors", "hallmark", "targets")
    options = {
        "prepare": ("benchmark", "licob", "wong", "patient_audit"),
        "source": ("inputs", *molecular, "settings", "auxiliary", "upstream"),
        "predict": ("inputs", *molecular, "teacher", "fit_book"),
        "merge": ("inputs", "parts"),
        "score": ("inputs", "forecast", "licob_book", "wong_book"),
    }
    for name, fields in options.items():
        item = stages.add_parser(name)
        for key in (*fields, "out"):
            item.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
        if name == "source":
            item.add_argument("--component", choices=("teacher", "cellhit"), required=True)
            item.add_argument("--shard", type=int, choices=range(3), default=0)
        if name == "predict":
            item.add_argument("--study", choices=STUDIES, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "source": source, "predict": predict, "merge": merge, "score": score}[args.action](args)


if __name__ == "__main__":
    main()
