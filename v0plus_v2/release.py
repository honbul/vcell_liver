"""Package source-selected sensitivity ensembles and independently score frozen LIMORE forecasts."""

import argparse
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from vc_build.external import KEY, _rna
from vc_build.io import freeze_table, publish_json, sha256, verify_forecast
from vc_build.v0plus_score import axis_metrics, compound_strata, crossed_patient_drug_ci, join_observations

from .lean import fm_paths
from .release_types import Release, SourceMember
from .round4 import AXES, angle
from .source import load, load_drug_models

HISTORICAL = ("drug_ridge", "cellhit", "v0plus_inductive", "drug_mean", "ridge_molecular_average")



def load_cpu(path):
    """Open trusted study joblib files with tagged torch storages placed on the CPU."""
    import torch

    storage_load = torch.storage._load_from_bytes
    try:
        torch.storage._load_from_bytes = lambda raw: torch.load(
            io.BytesIO(raw), map_location="cpu", weights_only=False)
        return joblib.load(path)
    finally:
        torch.storage._load_from_bytes = storage_load


def read_receipt(path):
    return json.loads(path.read_text())


def choose(runs):
    receipts = [runs / "round4-01" / f"fold{i}" / "receipt.json" for i in range(3)]
    records = [r for path in receipts for r in read_receipt(path)["outer"]]
    fit_summary = pd.DataFrame(records).groupby("method")[list(AXES)].mean()
    fit_summary = fit_summary.drop(index="fold_selected")
    numbers = fit_summary.apply(angle, axis=1).sort_values(ascending=False)
    # Exactly two source-selected models may be evaluated once.
    return list(numbers.head(2).index), fit_summary, {str(path): sha256(path) for path in receipts}


def package(args):
    args.out.mkdir(exist_ok=False, parents=True)
    train, rna, _ = load(args.bundle)
    methods, development, receipts = choose(args.runs)
    registry = train[["compound_id", "isomeric_smiles", "source_compound_label", "drug_target_source_labels"]]
    registry = registry.groupby(["compound_id", "isomeric_smiles"], as_index=False).first()
    if len(registry) != 68 or registry.compound_id.duplicated().any():
        raise ValueError("Registered panel changed")
    members, selected = [], set()
    for fold in range(3):
        names = dict(round1="models.joblib", round2="trees.joblib", round3="neural.joblib", round4="meta.joblib")
        steps = {name: (load_drug_models if name=="round1" else load_cpu)(args.runs / f"{name}-{'02' if name in ('round1','round2') else '01'}"
                     / f"fold{fold}" / file) for name, file in names.items()}
        fitted = {name: steps["round4"][name] for name in methods}
        desired = set().union(*(model.columns for model in fitted.values()))
        linear, tree, neural = (steps[f"round{i}"] for i in range(1, 4))
        for base in (linear, tree):
            base.heads = {name: heads for name, heads in base.heads.items() if name in desired}
            for drugs in base.heads.values():
                for head in drugs.values():
                    selected.update(head.columns.tolist())
        neural = {name: model for name, model in neural.items() if name in desired}
        members.append(SourceMember(linear, tree, neural, fitted))
    model = Release(list(rna.columns), rna.median(axis=0).to_numpy(), members, methods, registry,
                    np.array(sorted(selected), dtype=int))
    endpoint = args.out / "checkpoint.joblib"
    joblib.dump(model, endpoint, compress=2)
    registry.to_csv(args.out / "compounds.tsv", sep="\t", index=False)
    development.to_csv(args.out / "internal_stack.tsv", sep="\t")
    publish_json(args.out / "release.json", {"checkpoint_sha256": sha256(endpoint),
        "compound_sha256": sha256(args.out / "compounds.tsv"),
        "candidates": methods, "default": methods[0], "version": "V0+ v2", "tuning_uses_limore": False,
        "model_folds": 3, "scores": "screen aligned sensitivity, higher means more sensitive",
        "source_only_development_receipts": receipts})


def load_release(root):
    manifest = read_receipt(root/"release.json")
    path = root/"checkpoint.joblib"
    if sha256(path) != manifest["checkpoint_sha256"] or manifest["version"] != "V0+ v2":
        raise ValueError("The V0+ v2 release checkpoint differs from the manifest")
    return load_cpu(path), manifest


def historical(path, names):
    table, stamp = verify_forecast(path)
    return table.loc[table.method.isin(names), KEY+["method", "predicted_value"]], stamp["sha256"]


def predict_table(query, expression, fitted, banks, track, *, min_coverage=.85, device="cpu"):
    aligned, coverage, valid = fitted.align(expression, min_coverage)
    if not valid.loc[query.cell_id].all():
        raise ValueError(f"Some registered {track} inputs did not pass RNA coverage")
    values, _ = fitted.predict(query, aligned, banks, device=device)
    result = pd.concat([query[KEY].assign(method="v2_"+name, predicted_value=pred)
                        for name, pred in values.items()], ignore_index=True)
    return result.assign(track=track, fold=0), coverage


def cohort_input(bundle):
    stamp = read_receipt(bundle/"receipt.json")
    paths = [bundle/"test_inputs"/(name + ".parquet") for name in ("rows", "rna")]
    for path in paths:
        if sha256(path) != stamp["artifacts"][str(path.relative_to(bundle))]["sha256"]:
            raise ValueError("Frozen LIMORE input manifest changed")
    query = pd.read_parquet(paths[0]).rename(columns={"record_id": "row_id", "cell_group": "lineage_group"})
    if (len(query), query.lineage_group.nunique(), query.compound_id.nunique()) != (3400, 50, 68):
        raise ValueError("The registered LIMORE evaluation scope changed")
    return query, _rna(paths[1]), stamp


def predict(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    model, _ = load_release(args.package)
    query, rna, roster = cohort_input(args.bundle)
    banks, fm_sha = fm_paths(args.fm, limore=True)
    label_key = "test_labels/outcomes.parquet"
    old, old_sha = historical(args.baselines, HISTORICAL)
    if old.method.nunique() != len(HISTORICAL) or len(old) != len(query)*len(HISTORICAL):
        raise ValueError("Missing matched historical LIMORE forecasts")
    cohort = "X_C_LIMORE"
    new, coverage = predict_table(query, rna, model, banks, cohort)
    controls = old.assign(track=cohort, fold=0)
    freeze_table(pd.concat([new, controls], ignore_index=True), args.out, provenance={
        "settings_sha256": sha256(args.package/"release.json"), "query_sha256": sha256(args.bundle/"receipt.json"),
        "registered_observations_sha256": roster["artifacts"][label_key]["sha256"],
        "historical_matched_forecast_sha256": old_sha, "verified_fm_sha256": fm_sha,
        "min_rna_coverage": float(coverage.min().min()), "cohort": cohort,
    })


def evaluate(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    forecast, f = verify_forecast(args.forecast)
    stamp = f["provenance"]
    if stamp["settings_sha256"] != sha256(args.package / "release.json"):
        raise ValueError("External model choices changed after seal")
    query, _, _ = cohort_input(args.bundle)
    labels = args.bundle/"test_labels/outcomes.parquet"
    if stamp["registered_observations_sha256"] != sha256(labels):
        raise ValueError("LIMORE observations diverged from preregistered hash")
    observed = pd.read_parquet(labels)
    merged = join_observations(forecast, observed)
    v2 = [name for name in merged.method.unique() if name.startswith("v2_")]
    versus = [(a, b) for a in v2 for b in HISTORICAL[:3]]
    overview, details, _ = axis_metrics(merged, versus)
    overview["crossed_ci95"] = [{"candidate": a, "baseline": b, **crossed_patient_drug_ci(merged, a, b)}
                                 for a, b in versus]
    strata = {}
    for name, drugs in compound_strata(query).items():
        sample = merged.loc[merged.compound_id.isin(drugs)]
        if len(sample):
            result, _, _ = axis_metrics(sample, ())
            strata[name] = {"drug_count": len(drugs), **result}
    labels_drug = query[["compound_id", "source_compound_label"]].drop_duplicates()
    drugs = details.merge(labels_drug, on="compound_id", validate="many_to_one")
    drugs = drugs.astype(object).where(drugs.notna(), None)
    publish_json(args.out, {"checkpoint_freeze_sha256": f["sha256"],
        "registered_labels_sha256": stamp["registered_observations_sha256"], "lineages": 50, "drugs": 68, "rows": 3400,
        "full": overview, "strata": strata, "drug_rows": drugs.to_dict("records"), "external_score_passes": 1})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("package", "predict", "score"))
    for key in ("runs", "bundle", "out", "package", "fm", "baselines", "forecast"):
        parser.add_argument("--" + key, type=Path)
    args = parser.parse_args()
    {"package": package, "predict": predict, "score": evaluate}[args.action](args)
