"""Measure cross-study HCC organoid transfer from Yang on matched TKI support."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from .evaluate import join_observations, paired_gain, rank_metrics, score_across_cells
from .external import KEY, _validate
from .function_data import new_output
from .io import freeze_table, publish_json, sha256, verify_forecast
from .pdo_transfer_cv import device_cuda
from .v0plus_cv import candidate_forecasts, molecular_inputs
from .v0plus_licob import ASSET_HASHES, checked_registration
from .v0plus_score import crossed_patient_drug_ci
from .v0plus_sota import verify_upstream
from .v0plus_wong import auc_mean, read_registration, workbook
from .yang_hcc_adaptation import BENCHMARK_INPUTS, aggregate, predictor
from .yang_hcc_audit import COHORT
from .yang_hcc_external import (
    EXPOSED_DONORS, SETTINGS, SMALL_MOLECULE_TARGETS, kernel_predictions, outcomes, read_inputs, require_hash,
)

STUDIES = ("LICOB", "Wong")
COHORT_HASHES = {
    "LICOB": "8257fd833e86a7e60ac8168c802a19916b7cc17a33b833ff3e5edcda0f87a678",
    "Wong": "1b7871c110bf61a7dd22d374a95e5bb68e0cdad6831eeb8f6de54c8cd469cc0d",
}
DONORS_HASH = "8a882f8e01e5033378872260d054150df32f7906fdb68de8f47472c73f162c9c"
DRUGS = {
    "WOSKHXYHFSIKNG-UHFFFAOYSA-N": "Lenvatinib",
    "MLDQJTXFUGDVEO-UHFFFAOYSA-N": "Sorafenib",
    "FNHKPVJBJVTLMP-UHFFFAOYSA-N": "Regorafenib",
}
SOURCE_ARMS = {"2d_mean", "2d_ridge", "2d_structured", "2d_teacher"}
SELECTED = {"teacher_affine": [32.], "pdo_ridge": [2.], "pdo_structured": [2.], "pdo_signed_transfer": [2.]}
PDO_ARMS = {*SELECTED, "drug_mean", "pdo_ridge_coral"}
PRIMARY = "pdo_ridge"
PAIRS = tuple((PRIMARY, arm) for arm in ("drug_mean", "2d_ridge", "2d_teacher", "2d_cellhit",
                                         "pdo_signed_transfer", "pdo_structured")) + (("pdo_ridge_coral", PRIMARY),)


def target_inputs(study, cohort, patient_audit, *, compounds=DRUGS):
    require_hash(cohort / "receipt.json", COHORT_HASHES[study])
    frame = pd.read_parquet(cohort / "query.parquet")
    frame = frame[frame.compound_id.isin(compounds)].copy()
    if study == "LICOB":
        checked_registration(cohort)
        require_hash(patient_audit, DONORS_HASH)
        audit = json.loads(patient_audit.read_text())
        if audit["registration_sha256"] != COHORT_HASHES[study]:
            raise ValueError("LICOB donor map is unrelated to the cohort registry")
        donors = pd.DataFrame(audit["models"]).set_index("cell_id", verify_integrity=True)
        donor = frame.cell_id.map(donors.donor_id)
        adult = frame.cell_id.map(donors.adult_age_confirmed)
        if donor.isna().any() or adult.isna().any():
            raise ValueError("Unknown LICOB patient in the prepared roster")
        frame = frame.assign(lineage_group=donor)[adult]
    else:
        read_registration(cohort)
    patients = 29 if study == "LICOB" else 10
    if (frame.lineage_group.nunique() != patients
            or not frame.groupby("cell_id").compound_id.nunique().eq(len(compounds)).all()
            or frame.duplicated(["cell_id", "compound_id"]).any()):
        raise ValueError("Expected every registered parent in each recipient model")
    frame["model_native"] = frame.cell_id
    frame["cell_id"] = study + ":" + frame.cell_id
    frame["lineage_group"] = study + ":" + frame.lineage_group.astype(str)
    frame["row_id"] = frame.cell_id + "|" + frame.compound_id
    rna = pd.read_parquet(cohort / "rna.parquet").rename(index=lambda label: study + ":" + label)
    return frame, rna.loc[sorted(frame.cell_id.unique())]


def prepare(args):
    read_inputs(args.benchmark, BENCHMARK_INPUTS)
    train = pd.read_parquet(args.benchmark / "train.parquet")
    yang = pd.read_parquet(args.benchmark / "query.parquet")
    yang = yang[~yang.lineage_group.isin(EXPOSED_DONORS)]
    yang_fit = outcomes(args.yang_workbook, yang)
    if (len(yang_fit) != 1085 or yang_fit.lineage_group.nunique() != 74
            or not np.isfinite(yang_fit.endpoint_value).all()):
        raise ValueError("Unexpected Yang fit support")
    panel = yang[["compound_id", "isomeric_smiles"]].drop_duplicates()
    source_rna = pd.read_parquet(args.benchmark / "rna.parquet")
    assays, application, rna = {}, {COHORT: yang[[*KEY, "isomeric_smiles"]]}, [source_rna]
    for study, directory in zip(STUDIES, (args.licob, args.wong)):
        meta, expression = target_inputs(study, directory, args.patient_audit)
        targets = meta[["cell_id", "lineage_group"]].drop_duplicates().merge(panel, how="cross")
        application[study] = targets.assign(row_id=targets.cell_id + "|" + targets.compound_id)
        assays[study] = meta
        rna.append(expression)
    genes = sorted(set.intersection(*(set(data.columns) for data in rna)))
    if len(genes) < 10000:
        raise ValueError("Too few common query and fit genes")
    common_rna = pd.concat([data[genes] for data in rna], verify_integrity=True)
    all_query = pd.concat([x.assign(study=study) for study, x in application.items()], ignore_index=True)
    _validate(train, all_query)
    files = {"train2d.parquet": train, "fit3d.parquet": yang_fit, "application.parquet": all_query,
             "rna.parquet": common_rna}
    files.update({f"{study}.parquet": frame for study, frame in assays.items()})
    with new_output(args.out) as tmp:
        hashes = {}
        for name, data in files.items():
            path = tmp / name
            data.to_parquet(path, index=name == "rna.parquet")
            hashes[name] = sha256(path)
        publish_json(tmp / "receipt.json", {
            "benchmark_inputs_sha256": BENCHMARK_INPUTS, "recipient_cohort_sha256": COHORT_HASHES,
            "licob_patient_sha256": DONORS_HASH, "gene_count": len(genes), "yang_donors": 74,
            "recipient_support": {study: {"patients": data.lineage_group.nunique(),
                                          "organoids": data.cell_id.nunique(), "rows": len(data)}
                                  for study, data in assays.items()},
            "artifacts": hashes,
        })


def validated_inputs(path):
    receipt = path / "receipt.json"
    document = json.loads(receipt.read_text())
    if (document["benchmark_inputs_sha256"], document["recipient_cohort_sha256"]) != (BENCHMARK_INPUTS, COHORT_HASHES):
        raise ValueError("Unregistered HCC 3D transfer inputs")
    for name, value in document["artifacts"].items():
        require_hash(path / name, value)
    return sha256(receipt)


def parameters(args):
    edges, targets, provenance = molecular_inputs(args.priors, args.hallmark, args.targets)
    return edges, targets | SMALL_MOLECULE_TARGETS, provenance


def predict_source(args):
    registered = validated_inputs(args.inputs)
    require_hash(args.settings, SETTINGS)
    train = pd.read_parquet(args.inputs / "train2d.parquet")
    query = pd.read_parquet(args.inputs / "application.parquet")
    if args.component == "cellhit":
        query = query[query.study.isin(STUDIES) & query.compound_id.isin(DRUGS)]
    return fit_source_forecast(args, registered, train, query)


def fit_source_forecast(args, registered, train, query):
    _validate(train, query)
    rna = pd.read_parquet(args.inputs / "rna.parquet")
    edges, targets, provenance = parameters(args)
    config = json.loads(args.settings.read_text())["settings"]
    if args.component == "teacher":
        linear, _ = candidate_forecasts("linear", train, query, rna, edges, targets,
                                        selected=[r for r in map(tuple, config["linear"])
                                                  if r[0] in {"drug_mean", "drug_ridge"}])
        linear.method = linear.method.str.split("/").str[0].map({"drug_mean": "2d_mean", "drug_ridge": "2d_ridge"})
        from .v0plus_neural import load_auxiliary

        auxiliary, _ = load_auxiliary(args.auxiliary)
        kernel = kernel_predictions(train, query, rna, edges, targets, auxiliary)
        kernel = kernel[kernel.method.eq("structured")].assign(method="2d_structured")
        core = pd.concat([linear, kernel], ignore_index=True)
        blend = (core[core.method.isin({"2d_ridge", "2d_structured"})].groupby(KEY, as_index=False)
                 .predicted_value.mean().assign(method="2d_teacher"))
        parts = pd.concat([core, blend], ignore_index=True)
    else:
        verify_upstream(args.upstream)
        parts, _ = candidate_forecasts("cellhit", train, query, rna, edges, targets,
                                       selected=[tuple(config["cellhit"][0])], upstream=args.upstream)
        parts.method = "2d_cellhit"
    if not np.isfinite(parts.predicted_value).all():
        raise ValueError("Missing source predictions for a supported recipient")
    freeze_table(parts.assign(track=parts.row_id.map(query.set_index("row_id").study), fold=0),
                 args.out, provenance={"input_sha256": registered, "mode": args.component,
                                       "settings_sha256": SETTINGS, "priors": provenance})


def predict_recipient(args):
    registered = validated_inputs(args.inputs)
    past, seal = verify_forecast(args.teacher)
    if (seal["provenance"]["input_sha256"] != registered or seal["provenance"]["mode"] != "teacher"
            or set(past.method) != SOURCE_ARMS):
        raise ValueError("The common 2D teacher is not frozen")
    teacher = past.pivot(index="row_id", columns="method", values="predicted_value")[["2d_mean", "2d_teacher"]]
    application = pd.read_parquet(args.inputs / "application.parquet")
    query = application[application.study.eq(args.study)].join(teacher, on="row_id", validate="one_to_one")
    fit = pd.read_parquet(args.inputs / "fit3d.parquet").join(teacher, on="row_id", validate="one_to_one")
    rna = pd.read_parquet(args.inputs / "rna.parquet")
    device = device_cuda()
    edges, targets, provenance = parameters(args)
    target = {key: value for key, value in targets.items() if key in set(fit.compound_id)}
    inductive, diagnostic = predictor(fit, query, rna, edges, target, device, SELECTED)
    coral, _ = predictor(fit, query, rna, edges, target, device, {"pdo_ridge": [2.]}, query_coral=True)
    inductive.method = inductive.method.str.split("/").str[0]
    aligned = coral[coral.method.eq("pdo_ridge/2.0")].assign(method="pdo_ridge_coral")
    parts = pd.concat([inductive[inductive.method.isin(PDO_ARMS)], aligned], ignore_index=True)
    common = parts[parts.compound_id.isin(DRUGS)].assign(track=args.study)
    freeze_table(common, args.out, provenance={
        "input_sha256": registered, "2d_sha256": seal["sha256"], "study": args.study,
        "priors": provenance, "yang_fit_donors": 74, "estimator_diagnostics": diagnostic,
    })


def merge(args):
    registered = validated_inputs(args.inputs)
    sources, teacher = verify_forecast(args.teacher)
    reference, cellhit = verify_forecast(args.cellhit)
    if any(seal["provenance"]["input_sha256"] != registered for seal in (teacher, cellhit)):
        raise ValueError("Component has a different evaluation input manifest")
    predicted = [reference, sources[sources.track.isin(STUDIES) & sources.compound_id.isin(DRUGS)]]
    hashes = {"2d": teacher["sha256"], "cellhit": cellhit["sha256"]}
    for name in STUDIES:
        table, seal = verify_forecast(args.parts / f"{name}.parquet")
        status = seal["provenance"]
        if (status["input_sha256"], status["2d_sha256"], status["study"]) != (registered, hashes["2d"], name):
            raise ValueError("Unexpected provenance for the 3D model")
        predicted.append(table)
        hashes[name] = seal["sha256"]
    joint = pd.concat(predicted, ignore_index=True)
    for name in STUDIES:
        query = pd.read_parquet(args.inputs / f"{name}.parquet", columns=KEY)
        models = joint[joint.track.eq(name)]
        complete = set(models.method) == (SOURCE_ARMS | PDO_ARMS | {"2d_cellhit"})
        if not complete or any(len(part) != len(query) or set(part.row_id) != set(query.row_id)
                               for _, part in models.groupby("method")):
            raise ValueError("Frozen TKI comparison is missing a method or a cohort identity")
    freeze_table(joint, args.out, provenance={"input_sha256": registered, "components": hashes})


def recipient_outcomes(study, query, book):
    if study == "LICOB":
        require_hash(book, ASSET_HASHES["supplement"])
        rows = list(load_workbook(book, read_only=True, data_only=True)["S6. AUC"].values)
        response = []
        for model in query.itertuples(index=False):
            record = rows[model.response_row - 1]
            value = record[model.response_column]
            if record[0] != model.model_native or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError("Unmatched adult LICOB normalized AUC")
            response.append(-float(value))
    else:
        records = workbook(book)["Supplementary Table 7"].values
        header = next(records)
        panel = list(records)
        response = [auc_mean(panel[model.response_row - 2][header.index(model.model_native)])
                    for model in query.itertuples(index=False)]
    return query.assign(endpoint_value=response).dropna(subset=["endpoint_value"])


def ranking_cells(donors):
    rows = []
    for identity, group in donors.groupby(["track", "fold", "lineage_group", "cell_id", "method"]):
        if len(group) != 3:
            continue
        scores = rank_metrics(group.endpoint_value, group.predicted_value, k=3)
        first = rank_metrics(group.endpoint_value, group.predicted_value, k=1)
        rows.append(dict(zip(("track", "fold", "lineage_group", "cell_id", "method"), identity)) | {
            "spearman": scores["spearman"], "ndcg_at_3": scores["ndcg_at_5"], "precision_at_1": first["top5_overlap"],
        })
    if not rows:
        raise ValueError("No patient has all three observed TKI responses")
    return pd.DataFrame(rows)


def summarize(section):
    donors = aggregate(section)
    drugs, _ = score_across_cells(donors)
    associations = drugs.groupby("method", as_index=False)[["spearman", "pearson"]].mean()
    cells = ranking_cells(donors)
    ranks = cells.groupby("method", as_index=False)[["spearman", "ndcg_at_3", "precision_at_1"]].mean()
    ranking_intervals = []
    for arm, baseline in PAIRS:
        for metric in ("spearman", "ndcg_at_3", "precision_at_1"):
            interval = paired_gain(cells, arm, baseline, metric)
            if interval is not None:
                ranking_intervals.append({"candidate": arm, "baseline": baseline, "metric": metric, **interval})
    return {"patients": donors.lineage_group.nunique(), "models": section.cell_id.nunique(),
            "patient_response_rows": donors.row_id.nunique(), "models_response_rows": section.row_id.nunique(),
            "complete_tki_patients": cells.lineage_group.nunique(),
            "sensitivity": associations.to_dict("records"), "per_drug": drugs.to_dict("records"),
            "ranking": ranks.to_dict("records"), "ranking_ci95": ranking_intervals,
            "sensitivity_ci95": [{"candidate": arm, "baseline": base,
                                  "ci95": crossed_patient_drug_ci(donors, arm, base)} for arm, base in PAIRS]}


def score(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    registered = validated_inputs(args.inputs)
    predicted, seal = verify_forecast(args.forecast)
    if seal["provenance"]["input_sha256"] != registered:
        raise ValueError("Recipient data registration changed after predictions were frozen")
    result = {}
    for study, book in zip(STUDIES, (args.licob_book, args.wong_book)):
        query = pd.read_parquet(args.inputs / f"{study}.parquet")
        actual = recipient_outcomes(study, query, book)
        selected = predicted[predicted.track.eq(study) & predicted.row_id.isin(actual.row_id)]
        result[study] = summarize(join_observations(selected, actual))
    publish_json(args.out, {"role": "cross-study technical transfer on previously evaluated HCC cohorts",
                            "studies": result, "compounds": DRUGS, "forecast_sha256": seal["sha256"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    model_inputs = ("priors", "hallmark", "targets")
    options = {
        "prepare": ("benchmark", "yang_workbook", "licob", "wong", "patient_audit"),
        "source": ("inputs", *model_inputs, "settings", "auxiliary", "upstream"),
        "recipient": ("inputs", *model_inputs, "teacher"),
        "merge": ("inputs", "teacher", "cellhit", "parts"),
        "score": ("inputs", "forecast", "licob_book", "wong_book"),
    }
    for action, fields in options.items():
        item = actions.add_parser(action)
        for key in (*fields, "out"):
            item.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
        if action == "source":
            item.add_argument("--component", required=True, choices=("teacher", "cellhit"))
        if action == "recipient":
            item.add_argument("--study", choices=STUDIES, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "source": predict_source, "recipient": predict_recipient, "merge": merge,
     "score": score}[args.action](args)


if __name__ == "__main__":
    main()
