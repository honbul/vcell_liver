"""Package and replay the fixed 76-drug organoid research panel without outcomes."""

import argparse
from contextlib import closing
from dataclasses import replace
import json
from pathlib import Path
import resource

import joblib
import numpy as np
import pandas as pd
from openpyxl import load_workbook

from .expanded_panel_cv import expanded_targets
from .function_data import new_output
from .io import publish_json, sha256, verify_forecast
from .licob_cold import ARMS, PRIMARY, checked, fit_panel
from .licob_dose_sources import COLD
from .licob_full_axis import RNA_HASH
from .limore_corrected import shared_support
from .pdo_transfer_cv import device_cuda
from .structured_release import PAIR, run_cli
from .v0plus_cli import BULK_FORMATS, RANK_FORMAT
from .v0plus_cv import molecular_inputs
from .v0plus_licob import ASSET_HASHES, sample_metadata
from .yang_hcc_external import require_hash

REVISION = "public-organoid-76"
FORECAST = "4cc7e3afa71fee91c0ca9a12ce06002d7a1016f9d4951b1833237830e3d7a2a1"
SCORE = "36194b0a43d89976e55d284b1dacdae2f5c0203f0471ea34e86acac5da4782ac"
TCGA_RECEIPT = "498ccd2d1984681afe033ca4497944237fb7ec6ef4f3c141ab0e07f4e9574a4b"
LICOB_MODELS = 43
PANEL_SIZE = 76
PARITY_LIMIT = 5e-5
SINGLE_LIMIT = 5e-6


def checked_benchmark(inputs, reference, result):
    if checked(inputs, ("query",)) != COLD:
        raise ValueError("Unexpected LICOB registry")
    require_hash(result, SCORE)
    past = json.loads(result.read_text())
    previous, stamp = verify_forecast(reference)
    if (stamp["sha256"], stamp["provenance"]["input_sha256"], past["forecast_sha256"]) != (FORECAST, COLD, FORECAST):
        raise ValueError("Original score and export require the same forecast registry")
    return previous, past


def pack(args):
    from .v0plus_neural import load_auxiliary

    _, current_score = checked_benchmark(args.inputs, args.reference, args.results)
    checked(args.inputs, ("train", "rna"))
    train = pd.read_parquet(args.inputs / "train.parquet")
    ids = sorted(train.cell_id.unique())
    rna = pd.read_parquet(args.inputs / "rna.parquet", filters=[("cell_id", "in", ids)]).loc[ids]
    keys = ["compound_id", "isomeric_smiles", "source_compound_label", "drug_target_source_labels",
            "in_source", "remaining_unspecified_stereo"]
    registry = pd.read_parquet(args.inputs / "query.parquet", columns=keys).drop_duplicates()
    registry = registry.sort_values("compound_id").reset_index(drop=True)
    if (len(train), train.compound_id.nunique(), len(rna), rna.shape[1]) != (670830, 1186, 1014, 14421):
        raise ValueError("Expected the original source rows and expression axis")
    if len(registry) != PANEL_SIZE or not registry.compound_id.is_unique or int(registry.in_source.sum()) != 57:
        raise ValueError("Incorrect registered 76-compound support")
    if not registry.in_source.equals(registry.compound_id.isin(train.compound_id)):
        raise ValueError("Mismatched compound training support")
    edges, targets, priors = molecular_inputs(args.priors, args.hallmark, args.targets)
    auxiliary, aux = load_auxiliary(args.auxiliary)
    annotations = pd.concat([train, registry], ignore_index=True)
    targets, _ = expanded_targets(annotations[~annotations.compound_id.isin(targets)], targets, rna.columns)
    fitted, notes = fit_panel(train, registry, rna, edges, targets, auxiliary, device_cuda())
    basis = fitted[PRIMARY].basis.cpu()
    portable = {name: replace(model, basis=basis,
                             projection=None if model.projection is None else model.projection.cpu(),
                             cell_center=None if model.cell_center is None else model.cell_center.cpu())
                for name, model in fitted.items()}
    registry["training_support"] = np.where(registry.in_source, "same_parent_key", "structure_transfer")
    registry["nearest_training_tanimoto"] = registry.compound_id.map(notes["nearest_tanimoto"])
    if not np.isfinite(registry.nearest_training_tanimoto).all():
        raise ValueError("Missing chemistry support metadata")

    with new_output(args.out) as package:
        checkpoint, compounds = package / "models.joblib", package / "compounds.tsv"
        joblib.dump(portable, checkpoint, compress=3)
        weights = sha256(checkpoint)
        registry.drop(columns=["drug_target_source_labels", "in_source"]).to_csv(compounds, sep="\t", index=False)
        publish_json(package / "release.json", {
            "schema_version": 1, "version": "V0+", "data_revision": REVISION, "fixed_registry": True,
            "allowed_formats": [*BULK_FORMATS, RANK_FORMAT], "inductive_query": True,
            "default_model": PRIMARY, "endpoint": "screen_aligned_sensitivity_score_higher_is_more_sensitive",
            "cohort": "LICOB_43_models_38_adult_donors", "clinical_response": False,
            "external_prior_gate": current_score["cold_gate"],
            "source_rows": len(train), "source_models": len(rna), "source_compounds": train.compound_id.nunique(),
            "expression_genes": rna.shape[1], "registered_compounds": PANEL_SIZE, "same_parent_count": 57,
            "models": {name: {"path": checkpoint.name, "checkpoint_entry": name,
                              "sha256": weights, "multiplier": 1.} for name in ARMS},
            "drug_registry_sha256": sha256(compounds), "prediction_freeze_sha256": FORECAST,
            "external_score_sha256": SCORE, "input_registration_sha256": COLD,
            "public_priors": priors, "auxiliary": aux, "fit": notes,
        })


def compare_expected(output, forecast, method):
    eligible = output[output.status.eq("predicted")].set_index(PAIR)
    truth = forecast[forecast.method.eq(method)].set_index(["cell_id", "compound_id"]).predicted_value
    truth.index = truth.index.set_names(PAIR)
    both = eligible.sensitivity_score.rename("cpu").to_frame().join(truth, how="outer", validate="one_to_one")
    if len(both) != PANEL_SIZE * LICOB_MODELS or not np.isfinite(both.to_numpy()).all():
        raise ValueError(f"CPU replay support differed for {method}")
    error = float((both.cpu - both.predicted_value).abs().max())
    if error > PARITY_LIMIT:
        raise ValueError(f"Archive replay failed for {method} at error {error:.8g}")
    support = eligible.groupby("training_support").size().to_dict()
    if support != {"same_parent_key": 57 * LICOB_MODELS, "structure_transfer": 19 * LICOB_MODELS}:
        raise ValueError("Missing forecast support labels")
    return error


def tcga_reference(paths, package, work):
    require_hash(paths / "receipt.json", TCGA_RECEIPT)
    receipt = json.loads((paths / "receipt.json").read_text())
    tpm, counts, lengths = [paths / f"{name}.tsv" for name in ("tpm", "counts", "gene-lengths")]
    require_hash(tpm, receipt["outputs"]["published_tpm"]["input_sha256"])
    require_hash(counts, receipt["outputs"]["counts_length_tpm"]["input_sha256"])
    require_hash(lengths, receipt["gene_lengths_sha256"])
    first, tpm_audit = run_cli(package, tpm, work / "tcga-tpm.tsv", scale="bulk_tpm")
    second, counts_audit = run_cli(package, counts, work / "tcga-counts.tsv", scale="bulk_counts", gene_lengths=lengths)
    matrices = []
    for result in (first, second):
        values = result.pivot(index="sample_id", columns="compound_id", values="sensitivity_score")
        if values.shape != (16, PANEL_SIZE) or not result.status.eq("predicted").all() or not np.isfinite(values).all(axis=None):
            raise ValueError("Incomplete RNA-seq replay")
        matrices.append(values)
    a, b = matrices
    if not a.index.equals(b.index) or not a.columns.equals(b.columns):
        raise ValueError("Mismatched published count and TPM identifiers")
    x, y = a.to_numpy(), b.to_numpy()
    error = float(np.abs(x - y).max())
    a_top = np.argsort(-x, axis=1, kind="stable")[:, :5]
    b_top = np.argsort(-y, axis=1, kind="stable")[:, :5]
    n_equal = sum(set(first) == set(second) for first, second in zip(a_top, b_top))
    if error > PARITY_LIMIT or n_equal != 16 or not second.gene_lengths_sha256.eq(sha256(lengths)).all():
        raise ValueError("RNA-seq input format reproduction did not pass")
    return {"max_abs_difference": error, "top5_exact": n_equal, "biosamples": 16,
            "published_tpm": tpm_audit, "counts_with_lengths": counts_audit,
            "gene_lengths_sha256": sha256(lengths), "expression_input_receipt": TCGA_RECEIPT}


def replay(args):
    expected, _ = checked_benchmark(args.inputs, args.reference, args.results)
    checked(args.inputs, ("rna",))
    query = pd.read_parquet(args.inputs / "query.parquet")
    shared_support(expected, query, ARMS)
    ids = sorted(query.cell_id.unique())
    rna = pd.read_parquet(args.inputs / "rna.parquet", filters=[("cell_id", "in", ids)]).loc[ids].rename_axis("sample_id")
    card = json.loads((args.package / "release.json").read_text())
    if (card["data_revision"], card["input_registration_sha256"], card["prediction_freeze_sha256"],
        card["external_score_sha256"]) != (REVISION, COLD, FORECAST, SCORE):
        raise ValueError("Unexpected offline package")
    if rna.shape != (43, 14421) or not rna.lt(0).any(axis=1).all():
        raise ValueError("Unexpected public rank-preserving expression support")
    with new_output(args.out) as work:
        partial, invariant = "qc_incomplete_genes", "qc_constant_rna"
        probes = pd.concat([
            rna, rna.iloc[:1, :int(.8 * rna.shape[1])].rename(index=lambda _: partial),
            pd.DataFrame(-1., index=pd.Index([invariant], name="sample_id"), columns=rna.columns),
        ])
        path = work / "licob-rna.tsv"
        probes.to_csv(path, sep="\t")
        summary, batch = {}, None
        for method in ARMS:
            rows, report = run_cli(args.package, path, work / f"licob-{method}.tsv", method=method, scale=RANK_FORMAT)
            denied = rows[rows.sample_id.isin([partial, invariant])]
            if len(denied) != 2 * PANEL_SIZE or not denied.status.eq("insufficient_rna_coverage").all():
                raise ValueError("Low coverage or constant RNA must not generate a score")
            summary[method] = report | {"max_abs_difference": compare_expected(rows, expected, method)}
            if method == PRIMARY:
                batch = rows[rows.sample_id.eq(ids[0])].set_index(PAIR).sensitivity_score

        single_rna = work / "single-rna.tsv"
        rna.iloc[:1].to_csv(single_rna, sep="\t")
        single, _ = run_cli(args.package, single_rna, work / "single.tsv", scale=RANK_FORMAT)
        single_score = single.set_index(PAIR).sensitivity_score
        error = float((single_score - batch).abs().max())
        if not single_score.index.equals(batch.index) or len(single_score) != PANEL_SIZE or not np.isfinite(error) or error > SINGLE_LIMIT:
            raise ValueError("Batch composition altered a registered forecast")

        outside = work / "outside.tsv"
        pd.DataFrame({"compound_id": ["ZZZZZZZZZZZZZZ-ZZZZZZZZZZ-Z"]}).to_csv(outside, sep="\t", index=False)
        unknown, unknown_info = run_cli(args.package, single_rna, work / "unknown.tsv", scale=RANK_FORMAT,
                                        compounds=outside)
        if len(unknown) != 1 or unknown.status.iloc[0] != "compound_outside_registered_panel":
            raise ValueError("Unregistered structure must not generate a score")
        _, legacy = run_cli(args.legacy, single_rna, work / "legacy.tsv", scale=RANK_FORMAT,
                            expected_error="is not enabled for this predictor release")
        public = tcga_reference(args.tcga, args.package, work)
        publish_json(work / "receipt.json", {
            "model_release_sha256": sha256(args.package / "release.json"), "input_registration_sha256": COLD,
            "forecast_sha256": FORECAST, "original_score_sha256": SCORE,
            "licob_models": LICOB_MODELS, "models": summary, "max_single_vs_batch_abs": error,
            "tolerances": {"archive": PARITY_LIMIT, "single": SINGLE_LIMIT},
            "unknown_compound": unknown_info, "legacy_scope": legacy, "tcga_input_replay": public,
            "max_child_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        })


def rna_only(args):
    if checked(args.inputs, ("query",)) != COLD:
        raise ValueError("Unexpected registered outcome-linked panel")
    require_hash(args.public_rna, RNA_HASH)
    require_hash(args.supplement, ASSET_HASHES["supplement"])
    with closing(load_workbook(args.supplement, read_only=True, data_only=True)) as book:
        registry = sample_metadata(book)
        row_ids = {
            name: {row[0] for row in book[name].iter_rows(min_col=1, max_col=1, values_only=True)}
            for name in ("S6. AUC", "S7. IC50", "S8. Emax")
        }
    rna = pd.read_parquet(args.public_rna)
    panel = pd.read_parquet(args.inputs / "query.parquet", columns=["cell_id"])
    matched = registry.index.intersection(rna.index)
    additional = matched.difference(panel.cell_id)
    if len(matched) != 44 or additional.tolist() != ["HCCO42"] or not registry.loc[additional, "adult_age_confirmed"].all():
        raise ValueError("Unexpected LICOB metadata-only HCC complement")
    if any(additional[0] in samples for samples in row_ids.values()):
        raise ValueError("The extra HCC model has a registered response row")
    card_path = args.package / "release.json"
    card = json.loads(card_path.read_text())
    if (card["data_revision"], card["prediction_freeze_sha256"]) != (REVISION, FORECAST):
        raise ValueError("Unexpected package for the RNA-only profile")
    with new_output(args.out) as work:
        path = work / "expression.tsv"
        rna.loc[additional].rename_axis("sample_id").to_csv(path, sep="\t")
        result, info = run_cli(args.package, path, work / "forecast.tsv", scale=RANK_FORMAT)
        if len(result) != PANEL_SIZE or not result.status.eq("predicted").all() or not np.isfinite(result.sensitivity_score).all():
            raise ValueError("Additional public RNA did not cover the offline panel")
        publish_json(work / "receipt.json", {
            "study_hcc_with_public_rna": len(matched), "hcc_models_with_scored_response": panel.cell_id.nunique(),
            "rna_only_model": additional[0], "response_presence_by_sheet": {key: False for key in row_ids},
            "public_rna_sha256": RNA_HASH, "supplement_sha256": ASSET_HASHES["supplement"],
            "model_release_sha256": sha256(card_path), "replay": info,
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    fields = {"pack": ("reference", "results", "priors", "hallmark", "targets", "auxiliary"),
              "replay": ("reference", "results", "package", "legacy", "tcga"),
              "rna-only": ("public-rna", "supplement", "package")}
    for stage, paths in fields.items():
        command = stages.add_parser(stage)
        for name in ("inputs", *paths, "out"):
            command.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    {"pack": pack, "replay": replay, "rna-only": rna_only}[args.stage](args)


if __name__ == "__main__":
    main()
