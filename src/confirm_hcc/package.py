"""Assemble a hash-registered research package and replay its real CPU CLI."""

import argparse
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from vc_build.io import publish_json, sha256, verify_forecast
from vc_build.structured_release import run_cli

from .contract import RNA_SHA, TRAIN_SHA, inputs, read_json, verified
from .ledger import checkpoint

MODEL = "structured"
REVISION = "single-agent-structured-68-confirm-20261123"
TOL = 5e-6
UNKNOWN = "ZZZZZZZZZZZZZZ-ZZZZZZZZZZ-Z"
PAIR = ["sample_id", "compound_id"]


def dependency_versions():
    records = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "torch", "rdkit",
                 "pyarrow", "openpyxl", "xgboost", "lightgbm", "catboost"):
        try:
            records[name] = version(name)
        except PackageNotFoundError:
            continue
    return records


def pack(args):
    _, stamp = verify_forecast(args.forecast)
    official = read_json(args.official / "score_receipt.json")
    decision = read_json(args.task2 / "reference_decision.json")
    if (stamp["provenance"]["training_sha256"], official["prediction_sha256"]) != (TRAIN_SHA, stamp["sha256"]):
        raise ValueError("The release must use the corrected, verified external fit")
    train, _, _, _ = inputs()
    registry = train[["compound_id", "isomeric_smiles", "source_compound_label"]].drop_duplicates()
    if len(registry) != 68 or not registry.compound_id.is_unique:
        raise ValueError("Expected a unique set of 68 exact registered parents")
    args.out.mkdir(parents=True, exist_ok=False)
    drug_list = args.out / "compounds.tsv"
    registry.sort_values("compound_id").to_csv(drug_list, sep="\t", index=False)
    path = checkpoint(args.task1, MODEL)
    immutable = args.out / "structured.joblib"
    shutil.copyfile(path, immutable)
    verified(immutable, sha256(path))
    model = joblib.load(immutable)
    gene_registry = args.out / "genes.tsv"
    axis = pd.Index(model.features.genes)
    pd.DataFrame({"gene": axis, "selected": np.isin(np.arange(len(axis)), model.features.indices)}).to_csv(
        gene_registry, sep="\t", index=False)
    publish_json(args.out / "release.json", {
        "schema_version": 1, "version": "V0+", "data_revision": REVISION, "default_model": MODEL,
        "models": {MODEL: {"path": immutable.name, "sha256": sha256(immutable), "multiplier": 1.0}},
        "drug_registry_sha256": sha256(drug_list), "gene_registry_sha256": sha256(gene_registry),
        "source_rows": len(train), "source_cell_models": int(train.cell_id.nunique()),
        "source_lineages": int(train.lineage_group.nunique()), "registered_compounds": len(registry),
        "source_train_rows_sha256": TRAIN_SHA, "source_train_rna_sha256": RNA_SHA,
        "endpoint": "screen_aligned_sensitivity_score_higher_is_more_sensitive",
        "supported_axis": "research culture pretreatment bulk RNA, 68 exact parents",
        "allowed_formats": ["bulk_tpm", "bulk_counts", "bulk_log2_tpm_plus_1"],
        "preprocessing": {"complete_axis_min": .85, "selected_axis_min": .85, "imputation": "source median",
                          "expression_units": "log2(TPM+1) or input converted before sample-rank transform"},
        "clinical_response": False, "inductive_query": True, "clinical_dose_validation": False,
        "prediction_freeze_sha256": stamp["sha256"],
        "external_score_sha256": sha256(args.official / "score_receipt.json"),
        "task2_designated_reference_model": decision["designated_model"],
        "task2_designation_scope": decision["scope"],
        "python": platform.python_version(), "runtime_versions": dependency_versions(),
    })


def write_rna(rna, path):
    rna.rename_axis("sample_id").to_csv(path, sep="\t")


def supported(result):
    return result[result.status.eq("predicted")].set_index(PAIR).sensitivity_score.sort_index()


def replay(args):
    import torch

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_available():
        raise ValueError("CPU replay requires GPU masking")
    card = read_json(args.package / "release.json")
    forecast, stamp = verify_forecast(args.forecast)
    if (card["source_train_rows_sha256"], card["prediction_freeze_sha256"], card["default_model"]) != (
            TRAIN_SHA, stamp["sha256"], MODEL):
        raise ValueError("Package does not correspond to the jointly scored structured checkpoint")
    verified(args.package / "genes.tsv", card["gene_registry_sha256"])
    model = joblib.load(verified(args.package / card["models"][MODEL]["path"], card["models"][MODEL]["sha256"]))
    _, _, query, qrna = inputs()
    qa = args.out / "qa"
    qa.mkdir(parents=True, exist_ok=False)
    rna = qrna.sort_index()
    coverage_low = rna.iloc[:1].copy().set_axis(["qc_low_full"], axis=0)
    coverage_low.iloc[:, :int(rna.shape[1] * .25)] = np.nan
    selected_low = rna.iloc[1:2].copy().set_axis(["qc_low_selected"], axis=0)
    selected_low.iloc[:, model.features.indices] = np.nan
    zero = rna.iloc[2:3].copy().set_axis(["qc_zero"], axis=0) * 0
    mixed = qa / "batch.tsv"
    write_rna(pd.concat([rna, coverage_low, selected_low, zero]), mixed)
    batch, batch_info = run_cli(args.package, mixed, qa / "batch_result.tsv")
    expected = (forecast[forecast.method.eq(MODEL)].set_index(["cell_id", "compound_id"]).predicted_value
                .rename_axis(PAIR).sort_index())
    actual = supported(batch)
    if not actual.index.equals(expected.index) or len(actual) != 3400:
        raise ValueError("CLI support differs from the jointly scored external forecast")
    frozen_diff = float(np.max(np.abs(actual - expected)))
    if frozen_diff > TOL:
        raise ValueError(f"CPU replay exceeds frozen score tolerance: {frozen_diff}")
    for label in (coverage_low.index[0], selected_low.index[0], zero.index[0]):
        bad = batch[batch.sample_id.eq(label)]
        if len(bad) != 68 or not bad.status.eq("insufficient_rna_coverage").all() or not bad.sensitivity_score.isna().all():
            raise ValueError(f"Bad RNA profile produced predictions: {label}")

    cold, errors = [], []
    for i in (0, len(rna) // 2, len(rna) - 1):
        file = qa / f"single-{i}.tsv"
        write_rna(rna.iloc[i:i+1], file)
        result, info = run_cli(args.package, file, qa / f"single-{i}-result.tsv")
        solo = supported(result)
        if len(solo) != 68 or not np.isfinite(solo).all() or not solo.index.isin(actual.index).all():
            raise ValueError("Singleton supported rows differ from their batch")
        errors.append(float(np.max(np.abs(solo - actual.loc[solo.index]))))
        cold.append(info["seconds"])
    if max(errors) > TOL:
        raise ValueError("Single-input and batch CLI results differ")

    negative = rna.iloc[[0]].copy()
    negative.iloc[0, 0] = -1.
    negative_file = qa / "negative.tsv"
    write_rna(negative, negative_file)
    _, negative_info = run_cli(args.package, negative_file, qa / "negative-result.tsv",
                               expected_error="RNA measurements must be nonnegative finite numbers or missing")
    unknown_file = qa / "outside.tsv"
    pd.DataFrame({"compound_id": [UNKNOWN]}).to_csv(unknown_file, sep="\t", index=False)
    unknown, unknown_info = run_cli(args.package, qa / "single-0.tsv", qa / "outside-result.tsv",
                                    compounds=unknown_file)
    if len(unknown) != 1 or not unknown.status.eq("compound_outside_training_panel").all() or (
            not unknown.sensitivity_score.isna().all()):
        raise ValueError("Unregistered drug was accepted")

    # Check the TPM and equivalent log2(TPM+1) entry points with the same registered sample.
    tpm = qa / "tpm.tsv"
    write_rna(np.expm1(rna.iloc[[0]] * np.log(2.)), tpm)
    tpm_out = qa / "tpm-result.tsv"
    subprocess.run([sys.executable, "-m", "vc_build.v0plus_cli", "predict", "--package", str(args.package),
                    "--rna", str(tpm), "--scale", "bulk_tpm", "--out", str(tpm_out)], check=True)
    tpm_values = supported(pd.read_csv(tpm_out, sep="\t", dtype={name: str for name in PAIR}))
    tpm_diff = float(np.max(np.abs(tpm_values - actual.loc[tpm_values.index])))
    if len(tpm_values) != 68 or tpm_diff > TOL:
        raise ValueError("Equivalent registered TPM scales give different predictions")

    # Check every row against singleton kernel inference also, and time warm CPU service computation.
    runtime, warm_errors = [], []
    batch_base, batch_residual = model.predict(query, rna)
    for sample in rna.index:
        q = query[query.cell_id.eq(sample)]
        t0 = perf_counter()
        base, residual = model.predict(q, rna.loc[[sample]])
        runtime.append(perf_counter() - t0)
        mask = query.cell_id.eq(sample)
        warm_errors.append(float(np.max(np.abs(base + residual - (batch_base[mask] + batch_residual[mask])))))
    if max(warm_errors) > TOL:
        raise ValueError("Kernel singleton replay changed a valid registered sample")
    cpu_name = next((s.split(":", 1)[1].strip() for s in Path("/proc/cpuinfo").read_text().splitlines()
                     if s.startswith("model name")), platform.processor())
    publish_json(args.out / "package_replay_receipt.json", {
        "model_id": MODEL, "data_revision": REVISION, "training_rows": 65678, "train_sha256": TRAIN_SHA,
        "train_rna_sha256": RNA_SHA, "package_release_sha256": sha256(args.package / "release.json"),
        "checkpoint_sha256": card["models"][MODEL]["sha256"], "drug_manifest_sha256": card["drug_registry_sha256"],
        "gene_manifest_sha256": card["gene_registry_sha256"], "frozen_prediction_sha256": stamp["sha256"],
        "gene_count": len(model.features.genes), "selected_gene_count": len(model.features.indices),
        "drug_count": card["registered_compounds"], "external_replay_profiles": 50, "external_replay_rows": 3400,
        "frozen_cpu_replay_max_abs_diff": frozen_diff, "single_batch_cli_max_abs_diff": max(errors),
        "single_batch_kernel_all_profiles_max_abs_diff": max(warm_errors), "max_abs_diff_tolerance": TOL,
        "tpm_log2_equivalence_max_abs_diff": tpm_diff,
        "input_validation": {"negative": negative_info, "outside_training_panel": unknown_info,
                             "whole_gene_coverage_rejections": 68, "selected_gene_coverage_rejections": 68,
                             "all_zero_rejections": 68, "rna_min_fraction": .85},
        "cpu_latency_seconds": {"cold_cli_single_68_drugs_all": cold, "cold_cli_single_68_drugs_median": float(np.median(cold)),
                                "cli_batch_53_profiles_input_checks": batch_info["seconds"],
                                "warm_kernel_single_68_drugs_median": float(np.median(runtime)),
                                "warm_kernel_single_68_drugs_p95": float(np.quantile(runtime, .95))},
        "device": "cpu", "cpu": cpu_name, "cpu_threads": torch.get_num_threads(),
        "runtime_versions": dependency_versions(), "python": platform.python_version(),
        "batch_output_sha256": batch_info["file_sha256"],
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pack", "replay"))
    for name in ("task1", "task2", "official", "out", "forecast", "package"):
        parser.add_argument("--" + name, type=Path, required=name in ("forecast", "out"))
    args = parser.parse_args()
    {"pack": pack, "replay": replay}[args.stage](args)


if __name__ == "__main__":
    main()
