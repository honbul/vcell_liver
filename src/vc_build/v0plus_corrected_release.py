"""Build and replay V0+ with the corrected known-drug source core."""

import argparse
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .external import KEY, _validate
from .function_data import new_output
from .hcc_3d_study_transfer import parameters
from .io import freeze_table, publish_json, sha256, verify_forecast
from .v0plus_cli import predict_offline
from .v0plus_cv import fit_baseline, panel_targets
from .v0plus_features import ExpressionFeatures
from .v0plus_release import save_research_model
from .yang_hcc_adaptation import BENCHMARK_FORECAST, BENCHMARK_INPUTS
from .yang_hcc_external import SETTINGS, read_inputs, require_hash
from .yang_hcc_audit import COHORT

METHODS = ("drug_mean", "drug_ridge", "v0plus_inductive")
REVISION = "single-agent-corrected-70"
SCORE = "61acdfb63552ef0c14fcfb0a9002c26fa76caeae9357c1d651b90b127b329fc0"
TOLERANCE = {"drug_mean": 1e-9, "drug_ridge": 1e-9, "v0plus_inductive": 5e-4}


def inputs(path):
    registration = read_inputs(path, BENCHMARK_INPUTS)
    train = pd.read_parquet(path / "train.parquet")
    rna = pd.read_parquet(path / "rna.parquet")
    query = pd.read_parquet(path / "query.parquet", columns=[*KEY, "isomeric_smiles"])
    _validate(train, query)
    return train, rna, query, registration


def reference(path):
    table, seal = verify_forecast(path)
    if seal["sha256"] != BENCHMARK_FORECAST or seal["provenance"]["inputs_sha256"] != BENCHMARK_INPUTS:
        raise ValueError("Unexpected frozen Yang RNA forecast")
    return table[table.method.isin(METHODS)]


def verify_prediction(predicted, documented):
    expected = documented.set_index([*KEY, "method"]).predicted_value
    names = set(predicted.method)
    candidate = predicted.set_index([*KEY, "method"]).predicted_value
    expected = expected[expected.index.get_level_values("method").isin(names)]
    pair = expected.to_frame("reference").join(candidate.rename("refit"), how="outer", validate="one_to_one")
    if not np.isfinite(pair.to_numpy()).all():
        raise ValueError("Refit does not have the expected Yang prediction identities")
    errors = (pair.refit - pair.reference).abs().groupby(level="method").max().to_dict()
    if any(error > TOLERANCE[name] for name, error in errors.items()):
        raise ValueError(f"The corrected model refit did not reproduce the frozen predictions: {errors}")
    return errors


def fit(args):
    require_hash(args.settings, SETTINGS)
    train, rna, query, _ = inputs(args.inputs)
    expected = reference(args.reference)
    edges, targets, priors = parameters(args)
    panel, anchor, rows, calibration = panel_targets(train)
    fit_rna = rna.loc[sorted(train.cell_id.unique())]
    query_rna = rna.loc[sorted(query.cell_id.unique())]
    settings = json.loads(args.settings.read_text())["settings"]
    selected = settings["linear"] if args.component == "linear" else settings["v0plus_inductive"]
    selected = [tuple(setting) for setting in selected if setting[0] in METHODS]
    if args.component == "linear":
        features = ExpressionFeatures.fit(fit_rna, edges, targets, n_variable=512, rank=True)
    else:
        from .v0plus_neural import fit_molecular, load_auxiliary

        auxiliary, aux_receipt = load_auxiliary(args.auxiliary)
    with new_output(args.out) as temp:
        outputs, model_cards = [], {}
        for name, parameter, width, gamma in selected:
            if args.component == "linear":
                model = fit_baseline(name, (parameter, width), panel, anchor, fit_rna, features, targets)
            else:
                model = fit_molecular(name, int(parameter), width, rows, anchor, fit_rna, query_rna,
                                      edges, targets, auxiliary, gpu=True)
            base, residual = model.predict(query, query_rna)
            outputs.append(query[KEY].assign(method=name, predicted_value=base + gamma * residual))
            model_cards[name] = save_research_model(temp, name, model, gamma)
        forecast = pd.concat(outputs, ignore_index=True)
        errors = verify_prediction(forecast, expected)
        freeze_table(forecast.assign(track=COHORT, fold=0), temp / "forecast.parquet", provenance={
            "data_revision": REVISION, "input_sha256": BENCHMARK_INPUTS, "reference_sha256": BENCHMARK_FORECAST,
            "settings_sha256": SETTINGS, "model_cards": model_cards, "max_abs_errors": errors,
            "public_priors": priors, "auxiliary_priors": aux_receipt if args.component == "molecular" else None,
            "calibration": calibration,
        })


def pack(args):
    train, _, _, registered = inputs(args.inputs)
    require_hash(args.score, SCORE)
    result = json.loads(args.score.read_text())
    if result["forecast_sha256"] != BENCHMARK_FORECAST:
        raise ValueError("Unexpected Yang benchmark score binding")
    drug_table = (train[["compound_id", "isomeric_smiles", "source_compound_label"]]
                  .drop_duplicates().sort_values("compound_id"))
    if len(drug_table) != 70 or not drug_table.compound_id.is_unique:
        raise ValueError("Expected the registered 70 exact compound identities")
    cards, errors, component_hashes = {}, {}, {}
    with new_output(args.out) as temp:
        for component in ("linear", "molecular"):
            root = args.parts / component
            _, receipt = verify_forecast(root / "forecast.parquet")
            details = receipt["provenance"]
            if (details["data_revision"], details["input_sha256"], details["reference_sha256"]) != (
                    REVISION, BENCHMARK_INPUTS, BENCHMARK_FORECAST):
                raise ValueError("Corrected model export and RNA replay registration differ")
            for name, metadata in details["model_cards"].items():
                source = root / metadata["path"]
                require_hash(source, metadata["sha256"])
                shutil.copy2(source, temp / metadata["path"])
                require_hash(temp / metadata["path"], metadata["sha256"])
                cards[name] = metadata
            errors.update(details["max_abs_errors"])
            component_hashes[component] = receipt["sha256"]
        if set(cards) != set(METHODS) or set(errors) != set(METHODS):
            raise ValueError("Missing corrected release heads")
        drug_table.to_csv(temp / "compounds.tsv", sep="\t", index=False)
        publish_json(temp / "release.json", {
            "schema_version": 1, "version": "V0+", "data_revision": REVISION, "models": cards,
            "default_model": "v0plus_inductive", "cohort": COHORT, "inductive_query": True,
            "endpoint": "screen_aligned_sensitivity_score_higher_is_more_sensitive",
            "supported_axis": "exact parent and pretreatment bulk RNA-seq from research cultures",
            "clinical_response": False, "three_dimensional_validation": "Yang tested five of the 70 supported compounds",
            "allowed_formats": ["bulk_tpm", "bulk_counts", "bulk_log2_tpm_plus_1"],
            "source_rows": registered["training_rows"], "source_panels": registered["training_by_source"],
            "registered_compounds": 70, "external_compounds": 5, "external_gate_passed": result["promoted"],
            "drug_registry_sha256": sha256(temp / "compounds.tsv"),
            "prediction_freeze_sha256": BENCHMARK_FORECAST, "external_score_sha256": SCORE,
            "source_input_sha256": BENCHMARK_INPUTS, "settings_sha256": SETTINGS,
            "refit_max_abs_errors": errors, "refit_components_sha256": component_hashes,
        })


def replay(args):
    card = json.loads((args.package / "release.json").read_text())
    if card["data_revision"] != REVISION or card["prediction_freeze_sha256"] != BENCHMARK_FORECAST:
        raise ValueError("Unexpected corrected package version")
    _, rna, query, _ = inputs(args.inputs)
    expected = reference(args.reference)
    use = sorted(query.cell_id.unique())[:3]
    input_rna = rna.loc[use].rename_axis("sample_id")
    input_rna.loc["no_expression"] = 0.
    with new_output(args.out) as work:
        frame = work / "input.tsv"
        input_rna.to_csv(frame, sep="\t")
        results = {}
        for method in METHODS:
            path = work / f"{method}.tsv"
            options = SimpleNamespace(package=args.package, model=method, rna=frame, scale="bulk_log2_tpm_plus_1",
                                      compounds=None, gene_map=None, out=path)
            predict_offline(options)
            read = pd.read_csv(path, sep="\t")
            estimated = read[read.status.eq("predicted")].set_index(["sample_id", "compound_id"])
            comparators = expected[expected.method.eq(method) & expected.cell_id.isin(use)]
            observations = comparators.set_index(["cell_id", "compound_id"]).predicted_value
            delta = estimated.loc[observations.index].sensitivity_score.to_numpy() - observations.to_numpy()
            maximum = float(np.abs(delta).max())
            if (len(estimated) != 210 or len(observations) != 15 or maximum > TOLERANCE[method]
                    or not read.loc[read.sample_id.eq("no_expression"), "status"].eq("insufficient_rna_coverage").all()):
                raise ValueError(f"Incorrect corrected release CPU replay or RNA rejection for {method}: {maximum}")
            results[method] = {"valid_rows": len(estimated), "replayed_reference_rows": len(observations),
                               "max_abs_difference": maximum, "insufficient_expression_rows": len(read) - len(estimated)}
        compounds = work / "unsupported.tsv"
        pd.DataFrame({"compound_id": ["ZZZZZZZZZZZZZZ-ZZZZZZZZZZ-Z"]}).to_csv(compounds, sep="\t", index=False)
        path = work / "unsupported_predictions.tsv"
        predict_offline(SimpleNamespace(package=args.package, model=None, rna=frame, scale="bulk_log2_tpm_plus_1",
                                        compounds=compounds, gene_map=None, out=path))
        absent = pd.read_csv(path, sep="\t")
        if len(absent) != 4 or not absent.status.eq("compound_outside_training_panel").all():
            raise ValueError("Corrected CLI outside-panel diagnostic failed")
        publish_json(work / "receipt.json", {"release_sha256": sha256(args.package / "release.json"),
                                             "reference_forecast_sha256": BENCHMARK_FORECAST,
                                             "models": results, "outside_panel_rows": len(absent)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    fields = {
        "fit": ("inputs", "reference", "settings", "priors", "hallmark", "targets", "auxiliary"),
        "pack": ("inputs", "score", "parts"),
        "replay": ("inputs", "reference", "package"),
    }
    for action, names in fields.items():
        item = commands.add_parser(action)
        for field in (*names, "out"):
            item.add_argument("--" + field, required=True, type=Path)
        if action == "fit":
            item.add_argument("--component", choices=("linear", "molecular"), required=True)
    args = parser.parse_args()
    {"fit": fit, "pack": pack, "replay": replay}[args.mode](args)


if __name__ == "__main__":
    main()
