"""Lock source-only selections, train release artifacts and freeze public forecasts."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .external import KEY, _rna, _validate
from .io import freeze_table, publish_json, read_table, sha256, verify_forecast
from .v0plus_cv import (
    MOLECULAR_ARMS, PUBLIC_ARMS, choose, design_drugs, fit_baseline, molecular_inputs, panel_targets,
    summarize, training_inputs,
)
from .v0plus_features import ExpressionFeatures
from .v0plus_sota import public_predictions, verify_upstream

ARMS = ("linear", "catboost", "lightgbm", *PUBLIC_ARMS, *MOLECULAR_ARMS)


def lock_settings(roots, out):
    artifact_hashes, source_predictions, selections = {}, [], {}
    for arm in ARMS:
        folders = [root / arm for root in roots if (root / arm).is_dir()]
        if len(folders) != 1:
            raise ValueError(f"Expected one complete development run for {arm}")
        files = [folders[0] / f"fold{i}" / "metrics.json" for i in range(3)]
        runs = [json.loads(path.read_text()) for path in files]
        for fold, (path, run) in enumerate(zip(files, runs)):
            if run["fold"] != fold or run["arm"] != arm:
                raise ValueError(f"Incorrect validation receipt at {path}")
            verified, _ = verify_forecast(path.parent / "forecast.parquet")
            scored = pd.read_parquet(path.parent / "validation.parquet")
            comparison = [*verified.columns]
            if not scored[comparison].equals(verified):
                raise ValueError(f"Source score does not match freeze: {path}")
            source_predictions.append(scored)
            artifact_hashes[f"{arm}/fold{fold}"] = sha256(path)
        inner = [entry for run in runs for entry in run["inner_metrics"]]
        selections[arm] = choose(pd.DataFrame(inner)) if inner else runs[0]["selected"]
    joined = pd.concat(source_predictions, ignore_index=True)
    summary = summarize(joined)
    source_macro = summary.groupby("method")[["spearman", "pearson", "macro_spearman",
                                               "macro_ndcg_at_5", "macro_top5_overlap"]].mean()
    primary = source_macro.assign(selection_utility=(source_macro.spearman * 0.5
                                                       + source_macro.macro_spearman * 0.25
                                                       + source_macro.macro_ndcg_at_5 * 0.25))
    traditional = primary.loc[["drug_ridge", "drug_elasticnet", "additive", "lightgbm", "catboost"]]
    conventional = traditional.selection_utility.idxmax()
    public_inductive = primary.loc[["cellhit", "velodrome_inductive", "codeae_inductive"]]
    sota = public_inductive.selection_utility.idxmax()
    output = {
        "settings": selections, "development_receipts_sha256": artifact_hashes,
        "development_metrics": primary.reset_index().to_dict("records"),
        "source_metrics": summary.to_dict("records"), "best_traditional": conventional,
        "best_inductive_public": sota, "test_outcomes_used": False,
    }
    publish_json(out, output)


def load_query(bundle, train):
    receipt = json.loads((bundle / "receipt.json").read_text())
    for relative in ["test_inputs/rows.parquet", "test_inputs/rna.parquet"]:
        if sha256(bundle / relative) != receipt["artifacts"][relative]["sha256"]:
            raise ValueError(f"Query input differs from cohort registration: {relative}")
    frame = read_table(bundle / "test_inputs/rows.parquet").rename(
        columns={"record_id": "row_id", "cell_group": "lineage_group"}
    )
    _validate(train, frame)
    return frame, _rna(bundle / "test_inputs/rna.parquet"), receipt


def save_research_model(out, method, model, gamma):
    if method in ("v0plus_inductive", "v0plus_transductive"):
        model.network = model.network.cpu()
    path = out / (method + ".joblib")
    joblib.dump(model, path, compress=3)
    return {"path": path.name, "sha256": sha256(path), "multiplier": gamma}


def predict(args):
    settings = json.loads(args.settings.read_text())
    train, fit_rna, train_receipt = training_inputs(args.bundle)
    edges, targets, priors = molecular_inputs(args.priors, args.hallmark, args.targets)
    query, query_rna, cohort_receipt = load_query(args.query_bundle, train)
    panel, anchor, source, calibration = panel_targets(train)
    aux_receipt = None
    if args.arm in MOLECULAR_ARMS:
        from .v0plus_neural import fit_molecular, load_auxiliary

        auxiliary, aux_receipt = load_auxiliary(args.auxiliary)
    else:
        variable = 512 if args.arm in ("linear", *PUBLIC_ARMS) else 256
        features = ExpressionFeatures.fit(fit_rna, edges, targets, n_variable=variable, rank=True)
        chemistry = design_drugs(train) if args.arm in ("catboost", "lightgbm") else None
    upstream = verify_upstream(args.upstream) if args.arm in PUBLIC_ARMS else None

    args.out.mkdir(parents=True, exist_ok=False)
    frames, packages = [], {}
    for method, penalty, width, multiplier in settings["settings"][args.arm]:
        if args.arm in MOLECULAR_ARMS:
            model = fit_molecular(args.arm, int(penalty), width, source, anchor, fit_rna, query_rna,
                                  edges, targets, auxiliary, gpu=args.gpu)
            ref, residual = model.predict(query, query_rna)
        elif args.arm in PUBLIC_ARMS:
            ref = query.compound_id.map(anchor).to_numpy()
            residual = public_predictions(method, source, query, features.transform(fit_rna),
                                          features.transform(query_rna), args.upstream, penalty, gpu=args.gpu)
        else:
            model = fit_baseline(method, (penalty, width), panel, anchor, fit_rna, features, targets,
                                 chemistry=chemistry, gpu=args.gpu)
            ref, residual = model.predict(query, query_rna)
        if args.package and method in {"v0plus_inductive", settings["best_traditional"], "drug_mean"}:
            packages[method] = save_research_model(args.out, method, model, multiplier)
        frames.append(query[KEY].assign(method=method, base_value=ref, predicted_value=ref + multiplier * residual,
                                        track=args.cohort, fold=0))
    prediction = pd.concat(frames, ignore_index=True)
    input_receipt = args.query_bundle / "receipt.json"
    provenance = {
        "settings_sha256": sha256(args.settings), "training_receipt_sha256": sha256(args.bundle / "receipt.json"),
        "query_receipt_sha256": sha256(input_receipt),
        "registered_observations_sha256": cohort_receipt["artifacts"]["test_labels/outcomes.parquet"]["sha256"],
        "cohort": args.cohort, "cohort_rows": len(query), "query_lineages": int(query.lineage_group.nunique()),
        "training_rows": train_receipt["train_rows"], "software_arm": args.arm,
        "calibration": calibration, "molecular_priors": priors, "auxiliary_priors": aux_receipt,
        "upstream_sha256": upstream, "model_packages": packages,
    }
    freeze_table(prediction, args.out / "forecast.parquet", provenance=provenance)


def merge_forecasts(args):
    frames, hashes, trained = [], {}, {}
    settings_hash = sha256(args.settings)
    for arm in ARMS:
        file = args.source / arm / "forecast.parquet"
        frame, receipt = verify_forecast(file)
        provenance = receipt["provenance"]
        if provenance["settings_sha256"] != settings_hash:
            raise ValueError(f"Changing model setup across final arms: {arm}")
        if provenance["cohort"] != args.cohort or provenance["software_arm"] != arm:
            raise ValueError(f"Incorrect external prediction role: {arm}")
        cohort = provenance["query_receipt_sha256"], provenance["registered_observations_sha256"]
        if frames and cohort != cohort_hashes:
            raise ValueError("Final arms use different outcome/query registration")
        cohort_hashes = cohort
        frames.append(frame)
        hashes[arm] = receipt["sha256"]
        for name, info in provenance["model_packages"].items():
            location = file.parent / info["path"]
            if sha256(location) != info["sha256"]:
                raise ValueError(f"Changed model checkpoint {location}")
            trained[name] = info | {"path": str(location.resolve())}
    table = pd.concat(frames, ignore_index=True)
    if args.ridge_blend:
        ridge = table[table.method.eq("drug_ridge")].copy()
        molecular = table[table.method.eq("v0plus_inductive")].set_index("row_id")
        mate = molecular.loc[ridge.row_id]
        if len(ridge) != len(molecular) or not np.array_equal(ridge.base_value, mate.base_value):
            raise ValueError("The fixed blend requires the same drug anchor and registered roster")
        ridge["predicted_value"] = (ridge.predicted_value.to_numpy() + mate.predicted_value.to_numpy()) / 2
        table = pd.concat([table, ridge.assign(method="ridge_molecular_average")], ignore_index=True)
    support = table.groupby("method").row_id.agg(["nunique", "size"])
    if support["size"].nunique() != 1 or (support["nunique"] != support["size"]).any():
        raise ValueError("Missing or repeated registered forecast identifiers across methods")
    freeze_table(table, args.out, provenance={
        "settings_sha256": settings_hash, "cohort": args.cohort, "query_receipt_sha256": cohort_hashes[0],
        "registered_observations_sha256": cohort_hashes[1], "component_freezes_sha256": hashes,
        "research_model_packages": trained,
        "fixed_ridge_molecular_average": args.ridge_blend,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    lock = sub.add_parser("lock")
    lock.add_argument("--cv-root", type=Path, action="append", required=True)
    lock.add_argument("--out", type=Path, required=True)
    freeze = sub.add_parser("predict")
    for key in ("settings", "bundle", "query-bundle", "priors", "hallmark", "targets", "out"):
        freeze.add_argument("--" + key, type=Path, required=True)
    for key in ("upstream", "auxiliary"):
        freeze.add_argument("--" + key, type=Path)
    freeze.add_argument("--cohort", required=True)
    freeze.add_argument("--arm", required=True, choices=ARMS)
    freeze.add_argument("--gpu", action="store_true")
    freeze.add_argument("--package", action="store_true")
    merge = sub.add_parser("merge")
    for key in ("settings", "source", "out"):
        merge.add_argument("--" + key, type=Path, required=True)
    merge.add_argument("--cohort", required=True)
    merge.add_argument("--ridge-blend", action="store_true")
    args = parser.parse_args()
    if args.command == "lock":
        lock_settings(args.cv_root, args.out)
    elif args.command == "predict":
        predict(args)
    else:
        merge_forecasts(args)


if __name__ == "__main__":
    main()
