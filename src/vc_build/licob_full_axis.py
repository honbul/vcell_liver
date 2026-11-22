"""Extend the immutable LICOB roster with a verified wider RNA axis and two exact drugs."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from . import measured_drivers
from .expanded_panel_cv import RECEIPTS, expanded_targets, registered_inputs
from .external import KEY, _validate
from .function_data import new_output
from .io import freeze_table, publish_json, sha256, verify_forecast
from .licob_node_recovery import GENOMICS_HASH, MODELS, joined_outcomes, metrics, registration, TRACK
from .limore_corrected import shared_support
from .measured_driver_transfer import expected_components, fit_reference, structured_models
from .v0plus_cv import molecular_inputs, panel_targets, training_inputs
from .v0plus_licob import ASSET_HASHES, adult_donor_outcomes, checked_registration
from .v0plus_score import axis_metrics, crossed_patient_drug_ci
from .yang_hcc_external import SETTINGS, kernel_predictions, require_hash

RUN = "LICOB_WIDE_RNA"
RNA_HASH = "d1dcd5dcb984c9bbac8e7bef31d753586beaf5e9c4303baa688cacfaff1df77d"
RELEASE = "fc66d2991e499a03b92778347a8a251d0bd9eb7c"
PREFIX_INPUTS = "d44d8e0bf1577e80bb064c045e3d74929d8b0b39968552800644c682595fe88f"
PREFIX_FORECAST = "7fb8595294828c7953690e5b82fd3912491f1ce890f956812854419ebc9655a8"
EXTENDED = {"LSXUTRRVVSPWDZ-UHFFFAOYSA-N": "AT-406", "LIOLIMKSCNQPLV-UHFFFAOYSA-N": "Capmatinib"}
EXTENDED_MODELS = ("drug_mean", "drug_ridge", "structured", "ridge_structured_average")


def register(args):
    require_hash(args.rna, RNA_HASH)
    if registration(args.prefix) != PREFIX_INPUTS:
        raise ValueError("The RNA reconstruction registry must be identical to the previous score")
    checked_registration(args.licob)
    _, train, reference, extra = registered_inputs(args, include_extension=True)
    question = pd.read_parquet(args.prefix / "query.parquet")
    prior_rna = pd.read_parquet(args.prefix / "rna.parquet")
    coding = pd.read_parquet(args.prefix / "coding.parquet")
    author_rna = pd.read_parquet(args.licob / "rna.parquet")
    raw = pd.read_parquet(args.rna)
    if raw.shape != (65, 23121) or not raw.index.is_unique or not raw.columns.is_unique:
        raise ValueError("Unexpected DrugDis LICOB sample or gene axis")
    models = sorted(question.cell_id.unique())
    genes = sorted(set(reference.columns) & set(raw.columns))
    same = raw.loc[prior_rna.index, prior_rna.columns].to_numpy(dtype=float)
    difference = float(abs(same - prior_rna.to_numpy(dtype=float)).max())
    if len(genes) != 14421 or difference > 3e-7 or not np.isfinite(raw.to_numpy()).all():
        raise ValueError("DrugDis measurements fail the independent RNA prefix identity check")
    author_genes = sorted(set(author_rna.columns) & set(genes))
    a, b = [rankdata(table.loc[author_rna.index, author_genes].to_numpy(), axis=1)
            for table in (raw, author_rna)]
    concordance = [float(np.corrcoef(x, y)[0, 1]) for x, y in zip(a, b)]
    if min(concordance) < .90 or len(author_genes) != 14421:
        raise ValueError("DrugDis RNA ranks do not identify the same samples as author RNA")
    chemistry = pd.read_parquet(args.licob / "chemistry.parquet")
    new_drugs = chemistry[chemistry.compound_id.isin(EXTENDED)].drop(columns="isomeric_smiles")
    extension = extra[extra.compound_id.isin(EXTENDED)].copy()
    structures = extension[["compound_id", "isomeric_smiles"]].drop_duplicates()
    new_drugs = new_drugs.merge(structures, on="compound_id", validate="one_to_one")
    if len(new_drugs) != 2 or len(extension) != 920:
        raise ValueError("Unexpected exact single-agent pharmacology support")
    profiles = question[["cell_id", "donor_id", "adult_age_confirmed", "new_donor", "lineage_group",
                         "response_row"]].drop_duplicates()
    extra_query = profiles.merge(new_drugs, how="cross")
    combined_query = pd.concat([question, extra_query.assign(
        row_id=lambda q: TRACK + "|" + q.cell_id + "|" + q.compound_id,
    )], ignore_index=True)
    _validate(train, question)
    _validate(pd.concat([train, extension]), combined_query)
    with new_output(args.out) as temp:
        hashes = {}
        for name, table in {"query.parquet": question, "expanded_query.parquet": combined_query,
                            "rna.parquet": raw.loc[models, genes].rename_axis("cell_id"),
                            "coding.parquet": coding, "extension.parquet": extension}.items():
            path = temp / name
            table.to_parquet(path, index=name in ("rna.parquet", "coding.parquet"))
            hashes[name] = sha256(path)
        publish_json(temp / "receipt.json", {"stage": RUN, "drugdis_revision": RELEASE, "drugdis_rna_sha256": RNA_HASH,
            "prefix_inputs_sha256": PREFIX_INPUTS, "core_sha256": RECEIPTS["core"],
            "source_expansion_sha256": RECEIPTS["extension"], "artifacts": hashes, "input_genes": len(genes),
            "rna_rows": len(raw), "rna_genes_in_release": raw.shape[1], "verified_prefix_models": len(prior_rna),
            "verified_prefix_genes": prior_rna.shape[1], "maximum_prefix_difference": difference,
            "author_rank_spearman_median": float(np.median(concordance)),
            "author_rank_spearman_p05": float(np.quantile(concordance, .05)),
            "model_support": len(models), "donor_support": int(question.donor_id.nunique()),
            "extra_source_support": {str(source): int(rows) for source, rows in extension.source_id.value_counts().items()},
            "additional_training_rows": extension.groupby("compound_id").size().to_dict(),
            "candidate_exact_drugs": len(EXTENDED), "unavailable_exact_drugs": 76 - combined_query.compound_id.nunique(),
        })


def validated(path):
    receipt = path / "receipt.json"
    info = json.loads(receipt.read_text())
    if (info["stage"], info["drugdis_rna_sha256"], info["prefix_inputs_sha256"], info["core_sha256"]) != (
            RUN, RNA_HASH, PREFIX_INPUTS, RECEIPTS["core"]):
        raise ValueError("Wider LICOB input registration changed")
    for filename, digest in info["artifacts"].items():
        require_hash(path / filename, digest)
    return sha256(receipt)


def predict(args):
    from .v0plus_neural import load_auxiliary

    receipt = validated(args.input)
    require_hash(args.settings, SETTINGS)
    require_hash(args.core / "receipt.json", RECEIPTS["core"])
    train, source, _ = training_inputs(args.core)
    held = pd.read_parquet(args.input / "rna.parquet").sort_index()
    fit = source.loc[sorted(train.cell_id.unique()), held.columns]
    edges, targets, _ = molecular_inputs(args.priors, args.hallmark, args.targets)
    expanded = args.component == "expanded"
    query = pd.read_parquet(args.input / ("expanded_query.parquet" if expanded else "query.parquet"))
    details = {}
    if expanded:
        auxiliary, aux = load_auxiliary(args.auxiliary)
        added = pd.read_parquet(args.input / "extension.parquet")
        joint = pd.concat([train, added], ignore_index=True)
        _, old_anchor, _, old_calibration = panel_targets(train)
        _, combined_anchor, _, combined_calibration = panel_targets(joint)
        if old_calibration != combined_calibration or not np.allclose(
                old_anchor, combined_anchor.reindex(old_anchor.index), atol=1e-13, rtol=0):
            raise ValueError("Exact drug augmentation must preserve common calibration and potency")
        targets, _ = expanded_targets(added, targets, fit.columns)
        work = argparse.Namespace(**(vars(args) | {"component": "linear"}))
        linear, _ = fit_reference(work, joint, query, fit, held, edges, targets)
        structure = kernel_predictions(joint, query, pd.concat([fit, held], verify_integrity=True), edges, targets, auxiliary)
        both = pd.concat([linear, structure[structure.method.eq("structured")]])
        predictions = add_blend(both)
        details = {"auxiliary": aux, "additional_train_rows": len(added), "common_calibration_equal": True}
    elif args.component == "structured":
        auxiliary, aux = load_auxiliary(args.auxiliary)
        features, fingerprint = measured_drivers.read(args.genomics)
        if fingerprint != GENOMICS_HASH:
            raise ValueError("Unexpected measured driver support")
        coding = pd.read_parquet(args.input / "coding.parquet")
        blocks = {"coding": (features["source_coding"], coding)}
        groups = query.assign(lineage_group=query.donor_id)
        predictions, details = structured_models(train, query, fit, held, edges, targets, auxiliary, blocks, groups)
        details |= {"genomics_sha256": fingerprint, "auxiliary": aux}
    else:
        predictions, query = fit_reference(args, train, query, fit, held, edges, targets)
    shared_support(predictions, query, predictions.method.unique())
    freeze_table(predictions.assign(track=RUN, fold=0), args.out, provenance={"registration_sha256": receipt,
                 "core_sha256": RECEIPTS["core"], "settings_sha256": SETTINGS, "component": args.component, **details})


def add_blend(prediction):
    group = prediction[prediction.method.isin(["structured", "drug_ridge"])].groupby(KEY, as_index=False)
    return pd.concat([prediction, group.predicted_value.mean().assign(method="ridge_structured_average")])


def merge(args):
    report = validated(args.input)
    q53 = pd.read_parquet(args.input / "query.parquet")
    q55 = pd.read_parquet(args.input / "expanded_query.parquet")
    components, records = {}, []
    for component in (*expected_components("LICOB"), "expanded"):
        forecasts, stamp = verify_forecast(args.parts / f"{component}.parquet")
        p = stamp["provenance"]
        if (p["registration_sha256"], p["settings_sha256"], p["component"]) != (report, SETTINGS, component):
            raise ValueError("Wider-axis prediction failed component verification")
        components[component] = stamp["sha256"]
        if component == "expanded":
            expanded = forecasts
        else:
            records.append(forecasts)
    full = add_blend(pd.concat(records)).assign(track=RUN, fold=0)
    partial, old = verify_forecast(args.historical)
    if old["sha256"] != PREFIX_FORECAST:
        raise ValueError("Unexpected paired RNA-prefix forecast")
    components["prefix"] = PREFIX_FORECAST
    parts = []
    for name, table, roster, models in (("wide", full, q53, MODELS), ("prefix", partial, q53, MODELS),
                                         ("expanded", expanded, q55, EXTENDED_MODELS)):
        shared_support(table, roster, models)
        copy = table[[*KEY, "track", "fold", "method", "predicted_value"]].copy()
        parts.append(copy.assign(track=RUN, method=name+"/"+copy.method))
    freeze_table(pd.concat(parts, ignore_index=True), args.out, provenance={"registration_sha256": report,
        "components_sha256": components, "labels_sha256": ASSET_HASHES["supplement"]})


def paired(frame, models, *, comparison):
    pairs = [(f"{comparison}/{method}", f"prefix/{method}") for method in models]
    block = frame[frame.method.isin({name for pair in pairs for name in pair})]
    # Aggregate each method on its own so an exactly constant drug anchor remains constant.
    parts = []
    for method, rows in block.groupby("method"):
        if method.endswith("/drug_mean"):
            scored = adult_donor_outcomes(rows.assign(method="drug_mean")).assign(method=method)
        else:
            scored = adult_donor_outcomes(rows)
        parts.append(scored)
    donors = pd.concat(parts)
    report = axis_metrics(donors, pairs)[0]
    report["crossed_ci95"] = [{"candidate": a, "baseline": b, "ci95": crossed_patient_drug_ci(donors, a, b)}
                              for a, b in pairs]
    bounds = next(row["ci95"] for row in report["crossed_ci95"] if row["candidate"] == f"{comparison}/structured")
    gains = {row["metric"]: row["gain"] for row in report["paired_contrasts"]
             if row["candidate"] == f"{comparison}/structured"}
    report["rna_axis_gain_supported"] = (all(bounds[name][0] > 0 for name in ("spearman", "pearson"))
                                         and all(gains[name] >= 0 for name in ("spearman", "ndcg_at_5")))
    return report


def score(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    receipt = validated(args.input)
    frame, seal = verify_forecast(args.forecast)
    if seal["provenance"]["registration_sha256"] != receipt or seal["provenance"]["labels_sha256"] != ASSET_HASHES["supplement"]:
        raise ValueError("Score requires the locked paired-axis prediction set")
    require_hash(args.supplement, ASSET_HASHES["supplement"])
    query = pd.read_parquet(args.input / "expanded_query.parquet")
    full = joined_outcomes(query, frame, args.supplement)
    main = full[full.method.str.startswith("wide/")].assign(method=lambda x: x.method.str.removeprefix("wide/"))
    extra = full[full.method.str.startswith("expanded/")].assign(method=lambda x: x.method.str.removeprefix("expanded/"))
    donors = adult_donor_outcomes(extra)
    new_chemistry = donors[donors.compound_id.isin(EXTENDED)]
    added, per_added, _ = axis_metrics(new_chemistry, ())
    report = {"stage": RUN, "registration_sha256": receipt, "forecast_sha256": seal["sha256"],
        "labels_sha256": ASSET_HASHES["supplement"], "primary": "structured", "new_compounds": EXTENDED,
        "common_53": {"new_donors": metrics(adult_donor_outcomes(main[main.new_donor]), query),
                      "previous_donors": metrics(adult_donor_outcomes(main[~main.new_donor]), query),
                      "all_donors": metrics(adult_donor_outcomes(main), query)},
        "paired_wide_vs_prefix": {"new_donors": paired(full[full.new_donor], sorted(MODELS), comparison="wide"),
                                 "all_donors": paired(full, sorted(MODELS), comparison="wide")},
        "expanded_55": axis_metrics(donors, ())[0], "extra_two_38donors": added["across_cell_scores"],
        "extra_per_drug": per_added.astype(object).where(per_added.notna(), None).to_dict("records")}
    publish_json(args.out, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    fields = {"register": ("rna", "prefix", "licob", "core", "original", "extension"),
              "predict": ("input", "core", "settings", "genomics", "priors", "hallmark", "targets", "upstream", "auxiliary"),
              "merge": ("input", "parts", "historical"), "score": ("input", "forecast", "supplement")}
    for mode, path_fields in fields.items():
        sub = stages.add_parser(mode)
        for name in (*path_fields, "out"):
            sub.add_argument("--"+name, type=Path, required=True)
        if mode == "predict":
            sub.add_argument("--component", required=True, choices=(*expected_components("LICOB"), "expanded"))
            sub.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    {"register": register, "predict": predict, "merge": merge, "score": score}[args.stage](args)


if __name__ == "__main__":
    main()
