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

BULK_FORMATS = ("bulk_tpm", "bulk_counts", "bulk_log2_tpm_plus_1")
RANK_FORMAT = "bulk_rank_preserving"


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


def read_expression(path, scale, gene_map, gene_lengths=None):
    if scale not in (*BULK_FORMATS, RANK_FORMAT):
        raise ValueError(f"Unsupported RNA input scale: {scale}")
    if gene_lengths is not None and scale != "bulk_counts":
        raise ValueError("--gene-lengths requires --scale bulk_counts")
    raw = pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",",
                      dtype=str if scale == RANK_FORMAT else None)
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
    if scale == RANK_FORMAT:
        raw = raw.astype(float)
        if np.isinf(raw.to_numpy()).any():
            raise ValueError("Rank-preserving RNA requires finite values or missing measurements")
        return raw
    raw = raw.apply(pd.to_numeric, errors="raise")
    if np.isinf(raw.to_numpy()).any() or (raw < 0).any().any():
        raise ValueError("RNA measurements must be nonnegative finite numbers or missing")
    if scale == "bulk_counts":
        if gene_lengths is not None:
            lengths = pd.read_csv(gene_lengths, sep="\t", dtype={"gene": str})
            if (set(lengths) != {"gene", "length_bp"} or lengths.empty or lengths.gene.isna().any()
                    or not lengths.gene.is_unique or lengths.gene.str.strip().eq("").any()):
                raise ValueError("Gene lengths need one row per gene with gene and length_bp columns")
            size = pd.to_numeric(lengths.set_index("gene").length_bp, errors="raise")
            if not np.isfinite(size).all() or not size.gt(0).all():
                raise ValueError("Gene lengths must be finite positive values in bases")
            raw = raw.div(size.reindex(raw.columns), axis=1)
        libsize = raw.sum(axis=1)
        if not np.isfinite(libsize).all():
            raise ValueError("RNA count normalization produced a nonfinite library size")
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
    if args.scale not in card.get("allowed_formats", BULK_FORMATS):
        raise ValueError(f"RNA input scale {args.scale} is not enabled for this predictor release")
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
    if "checkpoint_entry" in model_record:
        model = model[model_record["checkpoint_entry"]]
    features = model.molecular.features if hasattr(model, "molecular") else model.features
    panel = requested_panel(pd.read_csv(registry, sep="\t", dtype=str), args.compounds)
    gene_lengths = getattr(args, "gene_lengths", None)
    inputs = read_expression(args.rna, args.scale, args.gene_map, gene_lengths)
    matched = inputs.reindex(columns=features.genes)
    complete = matched.notna()
    coverage = complete.mean(axis=1)
    selected_coverage = complete.iloc[:, features.indices].mean(axis=1)
    informative = matched.max(axis=1).gt(matched.min(axis=1)) if args.scale == RANK_FORMAT else matched.max(axis=1).gt(0)
    measurable = coverage.ge(0.85) & selected_coverage.ge(0.85) & informative
    filled = matched.fillna(pd.Series(features.raw_median, index=features.genes))
    all_rows = pd.DataFrame({"cell_id": inputs.index}).merge(panel, how="cross")
    all_rows["rna_gene_coverage"] = all_rows.cell_id.map(coverage)
    all_rows["selected_gene_coverage"] = all_rows.cell_id.map(selected_coverage)
    all_rows["status"] = "predicted"
    all_rows.loc[~all_rows.cell_id.isin(measurable[measurable].index), "status"] = "insufficient_rna_coverage"
    outside_status = ("compound_outside_registered_panel" if card.get("fixed_registry")
                      else "compound_outside_training_panel")
    all_rows.loc[all_rows.isomeric_smiles.isna(), "status"] = outside_status
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
    if args.scale == RANK_FORMAT:
        all_rows["rna_normalization"] = "within_sample_rank"
    if gene_lengths is not None:
        all_rows["rna_normalization"] = "counts_per_base_to_tpm"
        all_rows["gene_lengths_sha256"] = sha256(gene_lengths)
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
    predict.add_argument("--gene-lengths", type=Path,
                         help="Optional gene and length_bp TSV for bulk_counts, keyed after --gene-map")
    predict.add_argument("--model")
    predict.add_argument("--scale", required=True, choices=(*BULK_FORMATS, RANK_FORMAT))
    args = parser.parse_args()
    if args.command == "package":
        build_package(args)
    else:
        predict_offline(args)


if __name__ == "__main__":
    main()
