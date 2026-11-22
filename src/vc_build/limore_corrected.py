"""Isolate the single-agent training correction on the previously scored LIMORE panel."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .evaluate import join_observations
from .expanded_panel_cv import RECEIPTS
from .external import KEY
from .io import freeze_table, publish_json, sha256, verify_forecast
from .v0plus_cli import predict_offline
from .v0plus_cv import candidate_forecasts, molecular_inputs, training_inputs
from .v0plus_release import load_query
from .v0plus_score import axis_metrics, compound_strata, crossed_patient_drug_ci
from .v0plus_sota import verify_upstream
from .yang_hcc_external import SETTINGS, kernel_predictions, require_hash

TRACK = "LIMORE50"
ARCHIVE = "894d5dc7a5fb7efdca49f261ba87aa58cf7b520595993a681fdb336175c17dba"
OBSERVATIONS = "b03a6e17b5c9743b47c06e6ddea0bf49057f685e0a1146ecc066ede0ab598e17"
PACKAGE = "16bbdda46351a5ab6abb8d884e5761164e88c52b4bf06fa059e22cc592b81046"
CPU = ("linear", "lightgbm", "cellhit")
GPU = ("catboost", "velodrome_inductive", "velodrome_transductive", "codeae_inductive",
       "codeae_transductive", "v0plus_inductive", "v0plus_transductive", "structured", "structured_old")
ARMS = {name: (name,) for name in (*CPU, *GPU) if name not in ("linear", "structured", "structured_old")} | {
    "linear": ("drug_mean", "additive", "drug_ridge", "drug_elasticnet"),
    "structured": ("structured",), "structured_old": ("structured",),
}
PACKAGE_METHODS = ("drug_mean", "drug_ridge", "v0plus_inductive")
BLENDS = {"ridge_molecular_average": "v0plus_inductive", "ridge_structured_average": "structured"}


def registered(args, role):
    require_hash(args.original / "receipt.json", RECEIPTS["original"])
    require_hash(args.core / "receipt.json", RECEIPTS["core"])
    directory = args.original if role == "historical" else args.core
    train, rna, receipt = training_inputs(directory)
    if role == "corrected" and receipt["original_bundle_sha256"] != RECEIPTS["original"]:
        raise ValueError("Unexpected corrected training lineage")
    query, query_rna, _ = load_query(args.original, train)
    if len(query) != 3400 or query.lineage_group.nunique() != 50 or query.compound_id.nunique() != 68:
        raise ValueError("Unexpected LIMORE RNA and compound roster")
    if not rna.columns.equals(query_rna.columns) or set(rna.index) & set(query_rna.index):
        raise ValueError("LIMORE RNA and the reference training axis must match without model overlap")
    return train, query, pd.concat([rna, query_rna], verify_integrity=True)


def predict(args):
    require_hash(args.settings, SETTINGS)
    role = "historical" if args.component == "structured_old" else "corrected"
    train, query, rna = registered(args, role)
    edges, targets, priors = molecular_inputs(args.priors, args.hallmark, args.targets)
    auxiliary, aux_receipt = None, None
    if args.component.startswith(("v0plus", "structured")):
        from .v0plus_neural import load_auxiliary

        auxiliary, aux_receipt = load_auxiliary(args.auxiliary)
    if args.component.startswith("structured"):
        result = kernel_predictions(train, query, rna, edges, targets, auxiliary)
        result = result[result.method.eq("structured")]
    else:
        if args.component.startswith(("cellhit", "codeae", "velodrome")):
            verify_upstream(args.upstream)
        selected = [tuple(x) for x in json.loads(args.settings.read_text())["settings"][args.component]]
        result, _ = candidate_forecasts(args.component, train, query, rna, edges, targets,
                                        selected=selected, gpu=args.gpu, upstream=args.upstream, auxiliary=auxiliary)
        result.method = result.method.str.split("/").str[0]
    if set(result.method) != set(ARMS[args.component]) or not np.isfinite(result.predicted_value).all():
        raise ValueError("Model did not predict all registered LIMORE profiles")
    result = result.assign(method=role + "/" + result.method, track=TRACK, fold=0)
    freeze_table(result, args.out, provenance={
        "component": args.component, "training_role": role,
        "registration_sha256": RECEIPTS["original"], "training_sha256":
            RECEIPTS["original" if role == "historical" else "core"],
        "settings_sha256": SETTINGS, "training_rows": len(train), "genes": rna.shape[1],
        "priors": priors, "auxiliary": aux_receipt,
    })


def apply_package(args):
    require_hash(args.package / "release.json", PACKAGE)
    _, query, rna = registered(args, "corrected")
    profiles = rna.loc[sorted(query.cell_id.unique())].rename_axis("sample_id")
    with TemporaryDirectory(prefix="limore-corrected-cli-") as temporary:
        folder = Path(temporary)
        p, d = folder / "expression.tsv", folder / "compounds.tsv"
        profiles.to_csv(p, sep="\t")
        query[["compound_id"]].drop_duplicates().to_csv(d, sep="\t", index=False)
        outputs = []
        for method in PACKAGE_METHODS:
            path = folder / f"{method}.tsv"
            predict_offline(SimpleNamespace(package=args.package, model=method, rna=p, scale="bulk_log2_tpm_plus_1",
                                            compounds=d, gene_map=None, out=path))
            raw = pd.read_csv(path, sep="\t").rename(columns={
                "sample_id": "cell_id", "sensitivity_score": "predicted_value",
            })
            if len(raw) != len(query) or not raw.status.eq("predicted").all():
                raise ValueError("The deployed 70-drug model does not cover the original 68-drug panel")
            cols = ["cell_id", "compound_id", "predicted_value"]
            forecast = query[KEY].merge(raw[cols], on=["cell_id", "compound_id"], validate="one_to_one")
            outputs.append(forecast.assign(method=f"release70/{method}", track=TRACK, fold=0))
    freeze_table(pd.concat(outputs), args.out, provenance={"component": "release70", "package_sha256": PACKAGE,
                                                          "registration_sha256": RECEIPTS["original"]})


def shared_support(rows, query, methods):
    if set(rows.method) != set(methods) or not np.isfinite(rows.predicted_value).all():
        raise ValueError("Expected all locked LIMORE models")
    expected = query[KEY].set_index("row_id").sort_index()
    for _, part in rows.groupby("method"):
        if not part[KEY].set_index("row_id").sort_index().equals(expected):
            raise ValueError("Missing or extra LIMORE identities in the joint prediction")


def merge(args):
    _, query, _ = registered(args, "corrected")
    past, previous = verify_forecast(args.historical)
    prior = previous["provenance"]
    if (previous["sha256"], prior["settings_sha256"], prior["query_receipt_sha256"]) != (
            ARCHIVE, SETTINGS, RECEIPTS["original"]):
        raise ValueError("Unexpected historical LIMORE comparison")
    kept = set().union(*(set(names) for key, names in ARMS.items() if not key.startswith("structured")))
    old = past[past.method.isin(kept)].copy()
    old.method = "historical/" + old.method
    columns = [*KEY, "method", "predicted_value", "track", "fold"]
    reports, slices = {"historical": ARCHIVE}, [old[columns]]
    expected = set(old.method)
    for component in (*ARMS, "release70"):
        table, seal = verify_forecast(args.parts / f"{component}.parquet")
        info = seal["provenance"]
        if (info["component"], info["registration_sha256"]) != (component, RECEIPTS["original"]):
            raise ValueError("Unexpected component registration")
        if component == "release70":
            if info["package_sha256"] != PACKAGE:
                raise ValueError("Frozen CLI model registration changed")
            arms = {f"release70/{name}" for name in PACKAGE_METHODS}
        else:
            role = "historical" if component == "structured_old" else "corrected"
            expected_data = RECEIPTS["original" if role == "historical" else "core"]
            if (info["settings_sha256"], info["training_sha256"]) != (SETTINGS, expected_data):
                raise ValueError("Unexpected training role or settings")
            arms = {f"{role}/{name}" for name in ARMS[component]}
        shared_support(table, query, arms)
        expected |= arms
        reports[component] = seal["sha256"]
        slices.append(table[columns])
    both = pd.concat(slices, ignore_index=True)
    for split in ("historical", "corrected"):
        for blend, method in BLENDS.items():
            selected = both[both.method.isin({f"{split}/{method}", f"{split}/drug_ridge"})]
            if not selected.groupby("row_id").size().eq(2).all():
                raise ValueError("Incomplete paired blend components")
            combo = selected.groupby([*KEY, "track", "fold"], as_index=False).predicted_value.mean()
            name = f"{split}/{blend}"
            slices.append(combo.assign(method=name))
            expected.add(name)
    combined = pd.concat(slices, ignore_index=True)
    shared_support(combined, query, expected)
    freeze_table(combined, args.out, provenance={
        "component_hashes": reports, "registration_sha256": RECEIPTS["original"],
        "core_sha256": RECEIPTS["core"], "settings_sha256": SETTINGS, "label_sha256": OBSERVATIONS,
    })


def pairs(methods):
    revised = sorted(name.removeprefix("corrected/") for name in methods if name.startswith("corrected/"))
    corrections = [(f"corrected/{method}", f"historical/{method}") for method in revised]
    ratings = [(f"corrected/{name}", f"corrected/{base}") for name in ["v0plus_inductive", *BLENDS]
               for base in ("drug_mean", "drug_ridge", "cellhit")]
    deployed = [(f"release70/{model}", f"corrected/{model}") for model in PACKAGE_METHODS]
    return corrections + ratings + deployed


def digest_rows(frame):
    return frame.astype(object).where(frame.notna(), None).to_dict("records")


def evaluate(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    _, query, _ = registered(args, "corrected")
    prediction, seal = verify_forecast(args.forecast)
    provenance = seal["provenance"]
    if (provenance["core_sha256"], provenance["settings_sha256"], provenance["label_sha256"]) != (
            RECEIPTS["core"], SETTINGS, OBSERVATIONS):
        raise ValueError("The joint correction and LIMORE response registration changed")
    query_labels = args.original / "test_labels/outcomes.parquet"
    require_hash(query_labels, OBSERVATIONS)
    labels = pd.read_parquet(query_labels)
    joined = join_observations(prediction, labels)
    contrasts = pairs(set(joined.method))
    overall, drugs, _ = axis_metrics(joined, contrasts)
    sensitivity = [{"candidate": arm, "baseline": base, "ci95": crossed_patient_drug_ci(joined, arm, base)}
                   for arm, base in contrasts]
    meta = json.loads((args.core / "receipt.json").read_text())
    affected = {key["compound_id"]: key["compound_label"] for key in meta["affected_source_compounds"]}
    if len(affected) != 2 or not set(affected) <= set(query.compound_id):
        raise ValueError("Expected both documented CTRPv2 correction parents")
    panels = compound_strata(query)
    panels["affected"] = sorted(affected)
    panels["unaffected"] = sorted(set(query.compound_id) - set(affected))
    strata = {}
    for name, ids in panels.items():
        if ids:
            sub = joined[joined.compound_id.isin(ids)]
            summary, _, _ = axis_metrics(sub, ())
            if len(ids) < 6:
                for method in summary["scores"]:
                    method.pop("macro_ndcg_at_5", None)
                    method.pop("macro_top5_overlap", None)
            strata[name] = {"compounds": len(ids), "metrics": summary}
    publish_json(args.out, {
        "scope": "technical correction audit on previously evaluated LIMORE50",
        "forecast_sha256": seal["sha256"], "observations_sha256": OBSERVATIONS,
        "compounds": query.drop_duplicates("compound_id").set_index("compound_id").source_compound_label.to_dict(),
        "n_models": query.cell_id.nunique(), "n_lineages": query.lineage_group.nunique(),
        "common_observations": len(query), "affected_compounds": affected, "overall": overall,
        "across_cell_ci95": sensitivity, "strata": strata, "by_drug": digest_rows(drugs),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="action", required=True)
    options = {
        "predict": ("settings", "priors", "hallmark", "targets", "upstream", "auxiliary"),
        "package": ("package",),
        "merge": ("historical", "parts"),
        "score": ("forecast",),
    }
    for mode, items in options.items():
        command = stages.add_parser(mode)
        for name in ("original", "core", *items, "out"):
            command.add_argument("--" + name, type=Path, required=True)
        if mode == "predict":
            command.add_argument("--component", choices=ARMS, required=True)
            command.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    {"predict": predict, "package": apply_package, "merge": merge, "score": evaluate}[args.action](args)


if __name__ == "__main__":
    main()
