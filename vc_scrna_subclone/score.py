"""Evaluate the frozen single-cell transfer forecasts and donor diagnostics."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from .cohorts import file_record
from .diagnostics import (
    TECHNICAL, donor_partial_correlations, paired_site_contrast, ranks_to_residuals, summarize_effect,
)
from .singlecell_inputs import COHORTS, files_receipt
from .source_panels import frozen


def score_lineages(predictions, assay, models):
    quantitative = []
    rows = predictions[predictions.role.isin(("pseudo_single_cell", "pseudo_bulk", "external_real_single_cell"))]
    for (role, depth), group in rows.groupby(["role", "depth"], dropna=False):
        means = group.groupby(["cell_id", "cell_group", "drug"], as_index=False)[models].mean()
        joined = means.merge(assay, on=["cell_id", "cell_group", "drug"], validate="1:1", how="left")
        if joined.resistance_native.isna().any():
            raise ValueError("Frozen queries have no exact-lineage drug assay")
        # One independent biological lineage contributes once per drug.
        use = joined.groupby(["cell_group", "drug"], as_index=False)[models + ["resistance_native"]].mean()
        for name, drug in use.groupby("drug"):
            for method in models:
                count = drug.cell_group.nunique()
                estimate = drug[method].to_numpy()
                truth = drug.resistance_native.to_numpy()
                rho = None
                if count >= 5 and np.ptp(truth):
                    rho = float(spearmanr(estimate, truth).statistic) if np.ptp(estimate) else 0.0
                quantitative.append({
                    "query": role, "depth": None if pd.isna(depth) else int(depth), "drug": name, "model": method,
                    "lineages": count, "cells": len(group[group.drug.eq(name)]),
                    "within_drug_spearman": rho, "average_resistance_prediction": float(estimate.mean()),
                    "average_native_resistance": float(truth.mean()),
                })
    return pd.DataFrame(quantitative)


def score_organoids(predictions, sources, source, models, rng):
    query = predictions[predictions.role.eq("organoid") & predictions.drug.eq("Sorafenib")]
    observed = pd.read_parquet(frozen(sources, "organoids/labels.parquet", source))
    merged = query.merge(observed, on="id", validate="1:1")
    if len(merged) != len(observed) or len(merged) != len(query):
        raise ValueError("Missing pre-registered paired organoid prediction")
    pairs, turns = [], []
    for model in models:
        pivot = merged.groupby(["donor", "outcome"])[model].mean().unstack()
        y = pivot[1] - pivot[0]
        pairs.extend({"model": model, "donor": donor, "resistant_minus_sensitive": score}
                     for donor, score in y.items())
        turns.append({"model": model, "concordant_donors": int((y > 0).sum()),
                      **summarize_effect(y.to_numpy(), rng)})
    return pd.DataFrame(pairs), turns


def signature_correlations(frame, markers):
    result = []
    for donor, group in frame.groupby("donor", sort=True):
        if len(group) < 30:
            continue
        controls = rankdata(group[TECHNICAL].to_numpy(), axis=0)
        dummies = pd.get_dummies(group.source_sample, drop_first=True).to_numpy()
        design = np.column_stack((controls, dummies))
        outcome = ranks_to_residuals(rankdata(group.resistance), design)
        for marker in markers:
            signature = ranks_to_residuals(rankdata(group[f"marker_{marker}"]), design)
            denom = np.linalg.norm(signature) * np.linalg.norm(outcome)
            if denom > 1e-8:
                result.append({"donor": donor, "marker": marker,
                               "partial_rho": float(np.dot(outcome, signature) / denom)})
    return pd.DataFrame(result, columns=["donor", "marker", "partial_rho"])


def score_patients(predictions, counts, models, fit_manifest, longitudinal, rng):
    patient = predictions[predictions.role.eq("patient") & predictions.drug.eq("Sorafenib")].copy()
    metadata = []
    cell_receipt = json.loads((counts / "receipt.json").read_text())
    for name in COHORTS:
        table = pd.read_parquet(frozen(counts, f"{name}/scoring_metadata.parquet", cell_receipt))
        metadata.append(table[["id", "donor", "fold", "source_sample", "site", "cohort",
                               "malignant_candidate", *TECHNICAL]])
    measured = pd.concat(metadata, ignore_index=True)
    joined = patient.merge(measured, on=["id", "donor", "fold"], validate="1:1")
    if len(joined) != len(patient):
        raise ValueError("Missing cross-fitted malignant cohort and QC metadata")
    joined = joined[joined.malignant_candidate].copy()
    timing = pd.read_csv(longitudinal, sep="\t", dtype=str)
    populations = []
    pre = joined.loc[joined.cohort.eq("GSE151530")].merge(
        timing, on=["donor", "source_sample"], validate="many_to_one",
    )
    if timing.duplicated(["donor", "source_sample"]).any():
        raise ValueError("Conflicting NCI longitudinal identities")
    diagnostics = []
    summaries = []
    for model in models:
        thresholds = {int(f): evidence["models"][model]["sorafenib_high_resistance_threshold"]
                      for f, evidence in fit_manifest["folds"].items()}
        cancer = joined.assign(resistance=joined[model], high_resistance=joined[model].ge(joined.fold.map(thresholds)))
        for cohort, group in cancer.groupby("cohort", sort=True):
            tech = donor_partial_correlations(group)
            bio = signature_correlations(group, ("stemness", "EMT", "hypoxia"))
            if len(tech):
                diagnostics.append(tech.assign(model=model, cohort=cohort, kind="technical").rename(
                    columns={"covariate": "variable"}))
            if len(bio):
                diagnostics.append(bio.assign(model=model, cohort=cohort, kind="biological").rename(
                    columns={"marker": "variable"}))
            group_values = group.groupby(["donor", "source_sample", "site"], as_index=False).agg(
                cells=("id", "size"), median_forecast=("resistance", "median"),
                high_resistance_fraction=("high_resistance", "mean"),
            )
            populations.append(group_values.assign(model=model, cohort=cohort))
        primary = cancer[cancer.cohort.eq("GSE149614")].assign(resistance=lambda x: x.high_resistance.astype(float))
        timed = pre.assign(resistance=pre[model].ge(pre.fold.map(thresholds)).astype(float))
        for name, source, field, first, second in (
            ("paired_primary_to_pvtt", primary, "site", "Tumor", "PVTT"),
            ("paired_primary_to_lymph", primary, "site", "Tumor", "Lymph"),
            ("nci_paired_first_to_followup", timed, "period", "baseline", "first_followup"),
        ):
            contrast = paired_site_contrast(source, 0.5, field, first, second, min_cells=20)
            summaries.append({"model": model, "contrast": name, **summarize_effect(contrast.difference, rng)})
    return joined, pd.concat(diagnostics, ignore_index=True), pd.concat(populations, ignore_index=True), summaries


def run(a):
    manifest_file = a.predictions / "receipt.json"
    receipt = json.loads(manifest_file.read_text())
    for role in ("sources", "singlecell"):
        if receipt["inputs"][role] != file_record(getattr(a, role) / "receipt.json"):
            raise ValueError(f"Forecast uses a different registered {role} bundle")
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    source = json.loads((a.sources / "receipt.json").read_text())
    rows, assays = [], []
    models = sorted(receipt["folds"]["0"]["models"])
    for i in range(5):
        role = f"fold-{i}"
        frame = pd.read_parquet(frozen(a.predictions, role + ".parquet", receipt))
        if frame.duplicated(["id", "drug"]).any() or not frame.fold.eq(i).all():
            raise ValueError(f"Duplicate or wrong-fold predictions in {role}")
        rows.append(frame)
        assay = pd.read_parquet(frozen(a.sources, role + "/query.parquet", source))
        truth = pd.read_parquet(frozen(a.sources, role + "/labels.parquet", source))
        assays.append(assay.merge(truth, on="id", validate="1:1"))
    predictions = pd.concat(rows, ignore_index=True)
    if predictions.duplicated(["id", "drug"]).any():
        raise ValueError("A cell or replicate was predicted in multiple assay folds")
    assay = pd.concat(assays, ignore_index=True).groupby(
        ["cell_id", "cell_group", "drug"], as_index=False,
    ).resistance_native.mean()
    rng = np.random.default_rng(a.seed)
    cell_metrics = score_lineages(predictions, assay, models)
    organ_metrics, organ_summary = score_organoids(predictions, a.sources, source, models, rng)
    patients, diagnoses, samples, contrasts = score_patients(
        predictions, a.singlecell, models, receipt, a.longitudinal, rng,
    )
    distributions = []
    for (model, cohort, kind, variable), values in diagnoses.groupby(["model", "cohort", "kind", "variable"]):
        distributions.append({
            "model": model, "cohort": cohort, "kind": kind, "variable": variable,
            **summarize_effect(values.partial_rho, rng),
        })
    with tempfile.TemporaryDirectory(prefix=f".{a.output.name}.", dir=a.output.parent) as directory:
        stage = Path(directory)
        for name, data in (("transfer.tsv", cell_metrics), ("organoid_pairs.tsv", organ_metrics),
                           ("donor_partial_correlations.tsv", diagnoses), ("sample_fractions.tsv", samples)):
            data.to_csv(stage / name, sep="\t", index=False)
        (stage / "metrics.json").write_text(json.dumps({
            "organoid_donor_summary": organ_summary, "observational_paired_contrasts": contrasts,
            "partial_correlation_summary": distributions, "held_out_candidate_cells": len(patients),
            "held_out_candidate_donors": patients.donor.nunique(),
            "nci_role": "Prior and on-ICI biopsies. No corresponding sorafenib response outcome.",
        }, indent=2, sort_keys=True, allow_nan=False) + "\n")
        (stage / "receipt.json").write_text(json.dumps({
            "predictions": file_record(manifest_file),
            "inputs": {k: file_record(getattr(a, k) / "receipt.json") for k in ("sources", "singlecell")},
            "longitudinal_outcome_design": file_record(a.longitudinal), "files": files_receipt(stage),
        }, indent=2, sort_keys=True) + "\n")
        stage.rename(a.output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("predictions", "sources", "singlecell", "longitudinal", "output"):
        p.add_argument("--" + key, required=True, type=Path)
    p.add_argument("--seed", type=int, default=51226)
    run(p.parse_args())


if __name__ == "__main__":
    main()
