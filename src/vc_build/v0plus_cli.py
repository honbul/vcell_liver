"""Offline known-panel prioritization from bulk RNA-seq and exact compound identity."""

import argparse
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd

from .function_data import canonical_chemistry
from .io import publish_json, sha256, verify_forecast
from .v0plus_cv import training_inputs


def build_package(args):
    _, sealed = verify_forecast(args.forecast)
    results = json.loads(args.results.read_text())
    if results["forecast_sha256"] != sealed["sha256"]:
        raise ValueError("Research model card must use the selected forecast's score")
    train, _, audit = training_inputs(args.bundle)
    table = train[["compound_id", "isomeric_smiles", "source_compound_label"]].drop_duplicates()
    if not table.compound_id.is_unique:
        raise ValueError("Training panel does not contain one chemical identity per compound")
    models = sealed["provenance"]["research_model_packages"]
    for card in models.values():
        if sha256(card["path"]) != card["sha256"]:
            raise ValueError("Release model checkpoint failed hash verification")
    args.out.mkdir(parents=True, exist_ok=False)
    table.to_csv(args.out / "compounds.tsv", sep="\t", index=False)
    relative_models = {}
    for name, card in models.items():
        source = Path(card["path"])
        target = args.out / source.name
        shutil.copy2(source, target)
        if sha256(target) != card["sha256"]:
            raise ValueError("Packaged checkpoint failed hash verification")
        relative_models[name] = card | {"path": target.name}
    publish_json(args.out / "release.json", {
        "schema_version": 1, "version": "V0+", "models": relative_models, "default_model": "v0plus_inductive",
        "supported_axis": "exact parent and pretreatment bulk RNA-seq from research cultures",
        "endpoint": "screen_aligned_sensitivity_score_higher_is_more_sensitive",
        "clinical_response": False, "three_dimensional_validation": args.three_d_status,
        "allowed_formats": ["bulk_tpm", "bulk_counts", "bulk_log2_tpm_plus_1"],
        "inductive_query": True, "patient_level_permission": "research culture prioritization",
        "source_rows": audit["train_rows"], "source_panels": audit["train_by_source"],
        "drug_registry_sha256": sha256(args.out / "compounds.tsv"), "prediction_freeze_sha256": sealed["sha256"],
        "external_score_sha256": sha256(args.results), "cohort": results["cohort"],
    })


def read_expression(path, scale, gene_map):
    raw = pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")
    if {"sample_id", "gene", "value"} <= set(raw):
        raw = raw.pivot(index="sample_id", columns="gene", values="value")
    else:
        if "sample_id" not in raw:
            raise ValueError("RNA needs sample_id and one column per gene, or long sample_id/gene/value")
        raw = raw.set_index("sample_id")
    if not raw.index.is_unique or raw.index.hasnans or not raw.columns.is_unique:
        raise ValueError("RNA input has repeated or missing sample/gene identity")
    if gene_map:
        mapping = pd.read_csv(gene_map, sep="\t", dtype=str)
        if set(mapping) != {"input_id", "gene_symbol"} or not mapping.input_id.is_unique:
            raise ValueError("Gene map needs unique input_id and gene_symbol columns")
        raw = raw.rename(columns=dict(zip(mapping.input_id, mapping.gene_symbol)))
        if not raw.columns.is_unique:
            raise ValueError("Genes become ambiguous after symbol mapping")
    raw = raw.apply(pd.to_numeric, errors="raise")
    if np.isinf(raw.to_numpy()).any() or (raw < 0).any().any():
        raise ValueError("RNA measurements must be nonnegative finite numbers or missing")
    if scale == "bulk_counts":
        libsize = raw.sum(axis=1)
        raw = raw.div(libsize.replace(0., np.nan), axis=0) * 1e6
    if scale != "bulk_log2_tpm_plus_1":
        raw = np.log2(raw + 1)
    return raw


def requested_panel(table, candidates):
    if candidates is None:
        return table.assign(requested=table.compound_id, input_index=np.arange(len(table)))
    raw = pd.read_csv(candidates, sep="\t" if candidates.suffix == ".tsv" else ",", dtype=str).fillna("")
    if not {"compound_id", "smiles"} & set(raw):
        raise ValueError("Compound file must have compound_id or smiles")
    ids = raw.get("compound_id", pd.Series("", index=raw.index))
    structures = raw.get("smiles", pd.Series("", index=raw.index))
    matched = []
    for compound, smiles in zip(ids, structures):
        if smiles:
            try:
                _, exact, *_ = canonical_chemistry(smiles)
            except (ValueError, RuntimeError):
                exact = ""
            if compound and exact != compound:
                raise ValueError(f"Compound ID and exact standardized SMILES disagree for {compound}")
            compound = exact
        matched.append(compound)
    lookup = pd.DataFrame({"compound_id": matched, "requested": ids.where(ids.ne(""), structures),
                           "input_index": np.arange(len(raw))})
    return lookup.merge(table, on="compound_id", how="left", validate="m:1")


def predict_offline(args):
    card_path = args.package / "release.json"
    card = json.loads(card_path.read_text())
    if card["schema_version"] != 1 or card["version"] != "V0+":
        raise ValueError("Unsupported predictor release")
    method = args.model or card["default_model"]
    model_record = card["models"][method]
    checkpoint = Path(model_record["path"])
    if not checkpoint.is_absolute():
        checkpoint = args.package / checkpoint
    if sha256(checkpoint) != model_record["sha256"]:
        raise ValueError("Release checkpoint changed")
    registry = args.package / "compounds.tsv"
    if sha256(registry) != card["drug_registry_sha256"]:
        raise ValueError("Compound registry changed")
    # Checkpoints are generated within this study and consumed from local trusted storage.
    model = joblib.load(checkpoint)
    features = model.molecular.features if hasattr(model, "molecular") else model.features
    panel = requested_panel(pd.read_csv(registry, sep="\t", dtype=str), args.compounds)
    inputs = read_expression(args.rna, args.scale, args.gene_map)
    matched = inputs.reindex(columns=features.genes)
    complete = matched.notna()
    coverage = complete.mean(axis=1)
    selected_coverage = complete.iloc[:, features.indices].mean(axis=1)
    measurable = coverage.ge(0.85) & selected_coverage.ge(0.85) & matched.max(axis=1).gt(0)
    filled = matched.fillna(pd.Series(features.raw_median, index=features.genes))
    all_rows = pd.DataFrame({"cell_id": inputs.index}).merge(panel, how="cross")
    all_rows["rna_gene_coverage"] = all_rows.cell_id.map(coverage)
    all_rows["selected_gene_coverage"] = all_rows.cell_id.map(selected_coverage)
    all_rows["status"] = "predicted"
    all_rows.loc[~all_rows.cell_id.isin(measurable[measurable].index), "status"] = "insufficient_rna_coverage"
    all_rows.loc[all_rows.isomeric_smiles.isna(), "status"] = "compound_outside_training_panel"
    for name in ("sensitivity_score", "drug_mean_anchor", "personalization_delta"):
        all_rows[name] = np.nan
    eligible = all_rows.status.eq("predicted")
    if eligible.any():
        subset = all_rows.loc[eligible]
        ref, residual = model.predict(subset, filled.loc[subset.cell_id.unique()])
        delta = model_record["multiplier"] * residual
        if not (np.isfinite(ref) & np.isfinite(delta)).all():
            raise ValueError("Release produced a missing prediction for a supported sample–compound pair")
        all_rows.loc[eligible, ["sensitivity_score", "drug_mean_anchor", "personalization_delta"]] = np.column_stack(
            [ref + delta, ref, delta],
        )
    all_rows["model"] = method
    all_rows["version"] = card["version"]
    if "data_revision" in card:
        all_rows["data_revision"] = card["data_revision"]
    all_rows["presented_endpoint"] = card["endpoint"]
    all_rows["checkpoint_sha256"] = model_record["sha256"]
    all_rows["external_score_sha256"] = card["external_score_sha256"]
    all_rows = all_rows.rename(columns={"cell_id": "sample_id"})
    with args.out.open("x") as stream:
        all_rows.to_csv(stream, sep="\t", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    package = sub.add_parser("package")
    for name in ("forecast", "results", "bundle", "out"):
        package.add_argument("--" + name, required=True, type=Path)
    package.add_argument("--three-d-status", required=True)
    predict = sub.add_parser("predict")
    for name in ("package", "rna", "out"):
        predict.add_argument("--" + name, required=True, type=Path)
    predict.add_argument("--compounds", type=Path)
    predict.add_argument("--gene-map", type=Path)
    predict.add_argument("--model")
    predict.add_argument("--scale", required=True, choices=("bulk_tpm", "bulk_counts", "bulk_log2_tpm_plus_1"))
    args = parser.parse_args()
    if args.command == "package":
        build_package(args)
    else:
        predict_offline(args)


if __name__ == "__main__":
    main()
