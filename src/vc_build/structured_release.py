"""Export the registered LIMORE structured model and verify the public CPU entry point."""

import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter

import numpy as np
import pandas as pd

from .expanded_panel_cv import RECEIPTS
from .function_data import new_output
from .io import publish_json, sha256, verify_forecast
from .licob_cold import checked as checked_cold
from .licob_dose_sources import COLD
from .limore_corrected import registered, shared_support
from .pdo_transfer_cv import device_cuda
from .v0plus_cv import fit_baseline, molecular_inputs, panel_targets, training_inputs
from .v0plus_features import ExpressionFeatures
from .v0plus_release import save_research_model
from .yang_hcc_external import PENALTY, SETTINGS, require_hash

REVISION = "single-agent-structured-68"
FORECAST = "fe7f9e656801d63fd1263ec3e94093cad86b049130aa038d4f2e8fdd059b02bb"
SCORE = "ec9e064cc2413bbb72c5adbf3146a26f86d0acc69b81d4f3eefc2ddb36ed66fd"
TOLERANCES = {"structured": 5e-4, "drug_ridge": 1e-6, "drug_mean": 1e-9}
WONG_RNA = "224f100887e9a5aa03f81c8ead855ab1e9c2636e6a96c432b0988398bab0b22c"
WONG_PANEL = "58f45948f59816f13fb6963ac0c8fe193ac7cffcd83276af125f3733ffd9d4b8"
PAIR = ["sample_id", "compound_id"]


def fit_kernel(rna, source, anchor, edges, targets, auxiliary, *, device=None):
    from .structured_kernel import (
        KernelExpressionSchema, StructuredPredictor, drug_kernels, fit_cell_kernels, solve_residuals, training_grid,
    )

    if device is None:
        device = device_cuda()
    basis, cell = fit_cell_kernels(rna, edges, auxiliary, device=device)
    drugs, matrices = drug_kernels(source, targets, device=device)
    drug = matrices["structured"]
    weights, response = training_grid(source, rna.index, drugs, device=device)
    coefficients, steps, error = solve_residuals(cell["full"], drug, weights, response, PENALTY)
    schema = KernelExpressionSchema(basis.genes, basis.indices, np.median(rna.to_numpy(), axis=0))
    info = {"iterations": steps, "residual_norm": error, "penalty": PENALTY,
            "auxiliary_features": int(basis.reference[-1].shape[1]), "gene_features": len(schema.genes)}
    return StructuredPredictor(schema, basis.cpu(), drugs, anchor, coefficients.cpu(), drug.cpu()), info


def pack(args):
    from .v0plus_neural import load_auxiliary

    require_hash(args.core / "receipt.json", RECEIPTS["core"])
    require_hash(args.settings, SETTINGS)
    require_hash(args.score, SCORE)
    if json.loads(args.score.read_text())["forecast_sha256"] != FORECAST:
        raise ValueError("Report does not correspond to the registered LIMORE comparison")
    train, rna, _ = training_inputs(args.core)
    rna = rna.loc[sorted(train.cell_id.unique())]
    edges, targets, prior_hashes = molecular_inputs(args.priors, args.hallmark, args.targets)
    auxiliary, aux_hashes = load_auxiliary(args.auxiliary)
    panel, anchor, aligned, calibration = panel_targets(train)
    fixed, fit_notes = fit_kernel(rna, aligned, anchor, edges, targets, auxiliary)
    selected = json.loads(args.settings.read_text())["settings"]["linear"]
    settings = {name: (penalty, width, gain) for name, penalty, width, gain in selected
                if name in ("drug_mean", "drug_ridge")}
    if set(settings) != {"drug_mean", "drug_ridge"}:
        raise ValueError("Expected two registered comparison settings")
    baseline_schema = ExpressionFeatures.fit(rna, edges, targets, n_variable=512, rank=True)
    compounds = (train[["compound_id", "isomeric_smiles", "source_compound_label"]]
                 .drop_duplicates().sort_values("compound_id"))
    if len(compounds) != 68 or not compounds.compound_id.is_unique:
        raise ValueError("Expected exactly 68 deployable parent identities")
    with new_output(args.out) as output:
        cards = {"structured": save_research_model(output, "structured", fixed, 1.)}
        for method, (penalty, width, gain) in settings.items():
            model = fit_baseline(method, (penalty, width), panel, anchor, rna, baseline_schema, targets)
            cards[method] = save_research_model(output, method, model, gain)
        registry = output / "compounds.tsv"
        compounds.to_csv(registry, sep="\t", index=False)
        publish_json(output / "release.json", {
            "version": "V0+", "schema_version": 1, "data_revision": REVISION,
            "default_model": "structured", "models": cards,
            "endpoint": "screen_aligned_sensitivity_score_higher_is_more_sensitive",
            "clinical_response": False, "inductive_query": True,
            "supported_axis": "research cultures, pretreatment bulk RNA, exact parent",
            "allowed_formats": ["bulk_tpm", "bulk_counts", "bulk_log2_tpm_plus_1"],
            "cohort": "LIMORE50", "registered_compounds": len(compounds),
            "source_rows": len(train), "source_models": len(rna),
            "source_panels": train.source_id.value_counts().to_dict(), "source_input_sha256": RECEIPTS["core"],
            "drug_registry_sha256": sha256(registry), "settings_sha256": SETTINGS,
            "prediction_freeze_sha256": FORECAST, "external_score_sha256": SCORE,
            "fit": fit_notes, "calibration": calibration, "public_priors": prior_hashes,
            "auxiliary_priors": aux_hashes, "comparator_settings": settings,
        })


def run_cli(package, expression, output, *, method=None, compounds=None, expected_error=None):
    command = [sys.executable, "-m", "vc_build.v0plus_cli", "predict", "--package", str(package),
               "--rna", str(expression), "--scale", "bulk_log2_tpm_plus_1", "--out", str(output)]
    if method is not None:
        command.extend(["--model", method])
    if compounds is not None:
        command.extend(["--compounds", str(compounds)])
    started = perf_counter()
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        if expected_error is None or expected_error not in exc.stderr or output.exists():
            raise RuntimeError(exc.stderr) from exc
        return None, {"seconds": perf_counter() - started, "input_rejected": expected_error}
    if expected_error is not None:
        raise ValueError("CLI accepted a registered input with an invalid expression scale")
    elapsed = perf_counter() - started
    frame = pd.read_csv(output, sep="\t", dtype={name: str for name in PAIR})
    return frame, {
        "seconds": elapsed, "profiles": frame.sample_id.nunique(),
        "counts": frame.status.value_counts().to_dict(),
        "min_gene_coverage": float(frame.rna_gene_coverage.min()),
        "min_selected_coverage": float(frame.selected_gene_coverage.min()),
        "file_sha256": sha256(output),
    }


def diagnostics(result, expected, method, held):
    valid = result[result.status.eq("predicted")].set_index(PAIR).sensitivity_score
    checks = expected[expected.method.eq("corrected/" + method)].set_index(["cell_id", "compound_id"])
    left = valid.rename("cli").rename_axis(["cell_id", "compound_id"])
    pair = left.to_frame().join(checks.predicted_value.rename("archived"), how="outer", validate="one_to_one")
    if len(pair) != 3400 or not np.isfinite(pair.to_numpy()).all():
        raise ValueError(f"Incomplete LIMORE CPU replay for {method}")
    error = float((pair.cli - pair.archived).abs().max())
    if error > TOLERANCES[method]:
        raise ValueError(f"LIMORE CPU replay error for {method}: {error}")
    for name in held:
        statuses = result[result.sample_id.eq(name)].status
        if len(statuses) != 68 or not statuses.eq("insufficient_rna_coverage").all():
            raise ValueError("Unsupported expression unexpectedly produced predictions")
    return error


def replay(args):
    card = json.loads((args.package / "release.json").read_text())
    if (card["data_revision"], card["external_score_sha256"], card["prediction_freeze_sha256"]) != (
            REVISION, SCORE, FORECAST):
        raise ValueError("Unexpected package")
    expected, seal = verify_forecast(args.reference)
    if seal["sha256"] != FORECAST:
        raise ValueError("Unexpected LIMORE freeze")
    _, query, rna = registered(args, "corrected")
    expected = expected[expected.method.isin(["corrected/" + name for name in TOLERANCES])]
    shared_support(expected, query, ["corrected/" + name for name in TOLERANCES])
    if checked_cold(args.licob, ("query", "rna")) != COLD:
        raise ValueError("Unexpected LICOB RNA provenance")
    require_hash(args.wong / "RNA.tsv", WONG_RNA)
    require_hash(args.wong / "compounds.tsv", WONG_PANEL)

    with new_output(args.out) as work:
        limore = rna.loc[sorted(query.cell_id.unique())].rename_axis("sample_id")
        zero, missing = "qc_zero_expression", "qc_partial_expression"
        expression = [limore, pd.DataFrame(0., index=pd.Index([zero], name="sample_id"), columns=limore.columns),
                      limore.iloc[:1, :int(.8 * len(limore.columns))].rename(index=lambda _: missing)]
        block = work / "limore_rna.tsv"
        pd.concat(expression).to_csv(block, sep="\t")
        limore_res = {}
        for method, tolerance in TOLERANCES.items():
            output, info = run_cli(args.package, block, work / f"limore_{method}.tsv",
                                   method=None if method == card["default_model"] else method)
            error = diagnostics(output, expected, method, (zero, missing))
            limore_res[method] = info | {"max_abs_difference": error, "tolerance": tolerance}
            if method == "structured":
                batch = output[output.sample_id.eq(limore.index[0])].set_index(PAIR).sensitivity_score

        single_path = work / "single.tsv"
        limore.iloc[:1].to_csv(single_path, sep="\t")
        item, _ = run_cli(args.package, single_path, work / "single_result.tsv")
        delta = float((item.set_index(PAIR).sensitivity_score - batch).abs().max())
        if (len(item) != 68 or not item.status.eq("predicted").all() or not np.isfinite(delta)
                or delta > TOLERANCES["structured"]):
            raise ValueError("Unrelated query profiles changed a registered prediction")

        missing_panel = work / "unknown_parent.tsv"
        pd.DataFrame({"compound_id": ["ZZZZZZZZZZZZZZ-ZZZZZZZZZZ-Z"]}).to_csv(missing_panel, sep="\t", index=False)
        unknown, unknown_info = run_cli(args.package, single_path, work / "unknown_result.tsv",
                                        compounds=missing_panel)
        if len(unknown) != 1 or not unknown.status.eq("compound_outside_training_panel").all():
            raise ValueError("Unexpected support for unregistered parent")

        wong, wong_info = run_cli(args.package, args.wong / "RNA.tsv", work / "wong_tumour.tsv",
                                  compounds=args.wong / "compounds.tsv")
        licob_query = pd.read_parquet(args.licob / "query.parquet", columns=["cell_id", "compound_id"])
        licob_rna = pd.read_parquet(args.licob / "rna.parquet").loc[sorted(licob_query.cell_id.unique())]
        licob_input = work / "licob_rna.tsv"
        licob_panel = work / "licob_compounds.tsv"
        licob_rna.rename_axis("sample_id").to_csv(licob_input, sep="\t")
        licob_query[["compound_id"]].drop_duplicates().to_csv(licob_panel, sep="\t", index=False)
        negative = licob_rna.lt(0)
        coverage = licob_rna.reindex(columns=rna.columns).notna().mean(axis=1)
        if (len(wong) != 620 or len(licob_query) != 43 * 76
                or not (wong.status.eq("predicted") == np.isfinite(wong.sensitivity_score)).all()
                or not negative.any(axis=1).all() or coverage.ge(.85).any()):
            raise ValueError("Unexpected Wong or LICOB external input coverage")
        _, licob_info = run_cli(
            args.package, licob_input, work / "licob_organoid.tsv", compounds=licob_panel,
            expected_error="RNA measurements must be nonnegative finite numbers or missing",
        )
        registry = pd.read_csv(args.package / "compounds.tsv", sep="\t", usecols=["compound_id"])
        licob_info.update(
            profiles=len(licob_rna), requests=len(licob_query), predictions=0,
            exact_parent_drugs=licob_query.loc[licob_query.compound_id.isin(registry.compound_id),
                                             "compound_id"].nunique(),
            min_gene_coverage=float(coverage.min()), max_gene_coverage=float(coverage.max()),
            profiles_with_negative_values=int(negative.any(axis=1).sum()), negative_values=int(negative.sum().sum()),
        )
        publish_json(work / "receipt.json", {
            "release_sha256": sha256(args.package / "release.json"), "reference_forecast_sha256": FORECAST,
            "licob_input_sha256": COLD, "wong_rna_sha256": WONG_RNA, "wong_compounds_sha256": WONG_PANEL,
            "limore": limore_res, "wong_primary_tumor": wong_info, "licob_organoid": licob_info,
            "single_vs_batch_max_abs": delta, "outside_parent": unknown_info,
            "max_cli_child_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    paths = {"pack": ("core", "settings", "score", "priors", "hallmark", "targets", "auxiliary"),
             "replay": ("package", "original", "core", "reference", "wong", "licob")}
    for stage, names in paths.items():
        command = commands.add_parser(stage)
        for name in (*names, "out"):
            command.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    {"pack": pack, "replay": replay}[args.stage](args)


if __name__ == "__main__":
    main()
