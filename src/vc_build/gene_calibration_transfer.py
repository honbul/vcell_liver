"""Test outcome-free calibration of RNA gene marginals across 2D and 3D HCC."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .evaluate import join_observations
from .expanded_panel_cv import RECEIPTS
from .external import KEY, _validate
from .io import freeze_table, publish_json, verify_forecast
from .licob_full_axis import validated
from .licob_node_recovery import joined_outcomes
from .limore_corrected import OBSERVATIONS, shared_support
from .measured_driver_transfer import LIMORE_ARCHIVE, PENALTY, REFERENCES
from .structured_kernel import KernelGrid, Standardization, dot_kernel, radial, ranked_expression, solve_residuals
from .v0plus_cv import molecular_inputs, panel_targets, training_inputs
from .v0plus_licob import ASSET_HASHES, adult_donor_outcomes
from .v0plus_neural import load_auxiliary
from .v0plus_release import load_query
from .v0plus_score import axis_metrics, compound_strata, crossed_patient_drug_ci
from .yang_hcc_external import require_hash

CLINICAL_COHORTS = ("LIMORE", "LICOB")
LICOB_ARCHIVE = "fc5ab2b2902c2ee0fdca5b4027544208f6630fe2ec7b7304615ab36389c1dde7"
LICOB_REGISTRATION = "1f37cdff695ad7c9cbac06430a86b77d5ef5dfca5f9fa03d4557954b464102a0"
EXTERNAL_ARCHIVES = {"LIMORE": LIMORE_ARCHIVE, "LICOB": LICOB_ARCHIVE}
LABELS = {"LIMORE": OBSERVATIONS, "LICOB": ASSET_HASHES["supplement"]}
BASELINES = (*REFERENCES, "structured")
MODELS = ("augmented_raw", "cohort_centered", "source_ecdf", "cohort_ecdf")
PRIMARY = "cohort_ecdf"
CONTROLS = ("source_ecdf", "cohort_centered", "structured")


def roster(study, directory, train=None):
    if study == "LIMORE":
        require_hash(directory / "receipt.json", RECEIPTS["original"])
        if train is None:
            return RECEIPTS["original"], None, None
        query, held, _ = load_query(directory, train)
        return RECEIPTS["original"], query, held
    registration = validated(directory)
    if registration != LICOB_REGISTRATION:
        raise ValueError("Larger LICOB RNA registration changed")
    return registration, pd.read_parquet(directory / "query.parquet"), pd.read_parquet(directory / "rna.parquet")


def donor_ranks(values, ids, metadata, group):
    identities = metadata[["cell_id", group]].drop_duplicates().set_index("cell_id")
    groups = identities.reindex(ids)[group]
    if not identities.index.is_unique or groups.isna().any():
        raise ValueError("RNA has missing or ambiguous donor identities")
    return pd.DataFrame(values).groupby(groups.to_numpy(), sort=True).mean().to_numpy()


def ecdf(values, reference):
    """Evaluate column-wise empirical mid-CDF on fixed reference profiles."""
    if values.ndim != 2 or reference.ndim != 2 or values.shape[1] != reference.shape[1] or not len(reference):
        raise ValueError("Invalid gene-wise CDF dimensions")
    ordered = np.sort(reference, axis=0).T.copy()
    output = np.empty(values.shape, dtype=np.float32, order="F")
    for j, observed in enumerate(ordered):
        lower = np.searchsorted(observed, values[:, j], side="left")
        upper = np.searchsorted(observed, values[:, j], side="right")
        output[:, j] = (lower + upper) * (0.5 / len(observed))
    return output


def extra_kernel(source, queries, device):
    scaler = Standardization.fit(source, .05)
    reference = scaler.apply(source, device)
    linear = dot_kernel(reference, reference)
    scale = linear.diagonal().mean().item()
    if scale <= 1e-8:
        raise ValueError("Degenerate RNA calibration kernel")
    linear = linear / scale
    energy = linear.diagonal()
    covariance = .5 * (linear + radial(linear, energy, energy, 2.))
    cross = {}
    for name, raw in queries.items():
        held = scaler.apply(raw, device)
        between = dot_kernel(held, reference) / scale
        held_energy = held.square().mean(dim=1) / scale
        cross[name] = .5 * (between + radial(between, held_energy, energy, 2.))
    return covariance, cross


def predict(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Gene calibration transfer requires CUDA")
    device = torch.device("cuda")
    require_hash(args.core / "receipt.json", RECEIPTS["core"])
    train, fit, _ = training_inputs(args.core)
    registration, query, held = roster(args.study, args.cohort, train)
    _validate(train, query)
    expected = (50, 50, 68) if args.study == "LIMORE" else (43, 38, 53)
    group = "lineage_group" if args.study == "LIMORE" else "donor_id"
    if (query.cell_id.nunique(), query[group].nunique(), query.compound_id.nunique()) != expected:
        raise ValueError("External cohort does not match prespecified support")

    original, stamp = verify_forecast(args.archive)
    archive_sha = EXTERNAL_ARCHIVES[args.study]
    if stamp["sha256"] != archive_sha:
        raise ValueError("Unexpected comparison archive")
    prefix = "corrected/" if args.study == "LIMORE" else "wide/"
    archived = original[original.method.isin([prefix + name for name in REFERENCES])].copy()
    archived.method = archived.method.str.removeprefix(prefix)
    shared_support(archived, query, REFERENCES)

    genes = fit.columns.intersection(held.columns).sort_values()
    fit = fit.loc[sorted(train.cell_id.unique()), genes]
    held = held.loc[sorted(query.cell_id.unique()), genes]
    if not all(np.isfinite(frame.to_numpy()).all() for frame in (fit, held)):
        raise ValueError("Nonfinite input RNA")
    raw_fit, raw_held = (ranked_expression(frame) for frame in (fit, held))
    ref_fit = donor_ranks(raw_fit, fit.index, train, "lineage_group")
    ref_held = donor_ranks(raw_held, held.index, query, group)
    displacement = ref_held.mean(axis=0) - ref_fit.mean(axis=0)
    query_ranks = {"source_ecdf": ecdf(raw_held, ref_fit), "cohort_ecdf": ecdf(raw_held, ref_held)}
    raw_queries = {"augmented_raw": raw_held, "cohort_centered": raw_held - displacement}
    calibration = (
        extra_kernel(raw_fit, raw_queries, device),
        extra_kernel(ecdf(raw_fit, ref_fit), query_ranks, device),
    )

    edges, targets, priors = molecular_inputs(args.priors, args.hallmark, args.targets)
    auxiliary, aux_receipt = load_auxiliary(args.auxiliary)
    _, anchor, source, _ = panel_targets(train)
    grid = KernelGrid.prepare(source, fit, held, edges, targets, auxiliary, device=device)
    cell, cross = grid.cell["full"]
    drug = grid.drug["structured"]
    index = grid.cells.get_indexer(query.cell_id), grid.drugs.get_indexer(query.compound_id)
    offset = query.compound_id.map(anchor).to_numpy()
    base, base_steps, base_error = solve_residuals(cell, drug, grid.weights, grid.response, PENALTY)
    estimates = {"structured": (cross @ base @ drug).cpu().numpy()[index]}
    for fit_kernel, cross_views in calibration:
        coef, _, _ = solve_residuals(.60 * cell + .40 * fit_kernel, drug, grid.weights, grid.response, PENALTY)
        for name, view in cross_views.items():
            estimates[name] = ((.60 * cross + .40 * view) @ coef @ drug).cpu().numpy()[index]

    parts = [archived[[*KEY, "method", "predicted_value"]]]
    parts.extend(query[KEY].assign(method=name, predicted_value=offset + values) for name, values in estimates.items())
    forecasts = pd.concat(parts, ignore_index=True).assign(track=args.study, fold=0)
    shared_support(forecasts, query, {*BASELINES, *MODELS})
    freeze_table(forecasts, args.out, provenance={
        "study": args.study, "registration_sha256": registration, "archive_sha256": archive_sha,
        "labels_sha256": LABELS[args.study], "core_sha256": RECEIPTS["core"], "genes": len(genes),
        "source_reference_groups": len(ref_fit), "query_reference_groups": len(ref_held),
        "median_absolute_gene_mean_shift": float(np.median(np.abs(displacement))),
        "source_ecdf_query_tail_fraction": float(((query_ranks["source_ecdf"] < .05) |
                                                 (query_ranks["source_ecdf"] > .95)).mean()),
        "cohort_ecdf_query_tail_fraction": float(((query_ranks["cohort_ecdf"] < .05) |
                                                 (query_ranks["cohort_ecdf"] > .95)).mean()),
        "priors": priors, "auxiliary": aux_receipt, "baseline_solver": {"steps": base_steps, "residual": base_error},
    })


def score(args):
    forecast, seal = verify_forecast(args.forecast)
    provenance = seal["provenance"]
    registration, query, _ = roster(args.study, args.cohort)
    if (provenance["study"], provenance["registration_sha256"], provenance["core_sha256"],
        provenance["labels_sha256"], provenance["archive_sha256"]) != (
            args.study, registration, RECEIPTS["core"], LABELS[args.study], EXTERNAL_ARCHIVES[args.study]):
        raise ValueError("Unexpected frozen cohort lineage")
    require_hash(args.labels, LABELS[args.study])
    if args.study == "LICOB":
        shared_support(forecast, query, {*BASELINES, *MODELS})
        joined = adult_donor_outcomes(joined_outcomes(query, forecast, args.labels))
    else:
        joined = join_observations(forecast, pd.read_parquet(args.labels))
        query = pd.read_parquet(args.cohort / "test_inputs/rows.parquet")

    pairs = [(PRIMARY, name) for name in (*CONTROLS, "augmented_raw", "drug_mean", "drug_ridge", "cellhit")]
    pairs += [("source_ecdf", name) for name in ("structured", "augmented_raw")]
    overall, _, _ = axis_metrics(joined, pairs)
    across_drugs = {(row["candidate"], row["baseline"], row["metric"]): row["ci95"]
                    for row in overall["paired_contrasts"]}
    across_cells = {(a, b): crossed_patient_drug_ci(joined, a, b) for a, b in pairs}
    gates = {name: all(across_cells[PRIMARY, name][metric][0] > 0 for metric in ("spearman", "pearson"))
             and all(across_drugs[PRIMARY, name, metric][0] > -.005 for metric in ("spearman", "ndcg_at_5"))
             for name in CONTROLS}
    strata = {}
    for name, compounds in compound_strata(query).items():
        if len(compounds) < 2:
            continue
        panel, _, _ = axis_metrics(joined[joined.compound_id.isin(compounds)], ())
        if len(compounds) <= 5:
            for row in panel["scores"]:
                row.pop("macro_ndcg_at_5", None)
                row.pop("macro_top5_overlap", None)
        strata[name] = {"drugs": len(compounds), "scores": panel}
    publish_json(args.out, {
        "study": args.study, "forecast_sha256": seal["sha256"], "labels_sha256": LABELS[args.study],
        "scope": "previously scored technical comparison", "lineages": joined.lineage_group.nunique(),
        "rows": joined.row_id.nunique(), "overall": overall, "strata": strata,
        "paired_crossed_ci95": [{"candidate": a, "baseline": b, **value} for (a, b), value in across_cells.items()],
        "gate_by_comparison": gates, "gate_pass": all(gates.values()),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    fields = {"predict": ("core", "archive", "priors", "hallmark", "targets", "auxiliary"),
              "score": ("forecast", "labels")}
    for action, names in fields.items():
        sub = stages.add_parser(action)
        for name in (*names, "cohort", "out"):
            sub.add_argument(f"--{name}", type=Path, required=True)
        sub.add_argument("--study", choices=CLINICAL_COHORTS, required=True)
    args = parser.parse_args()
    {"predict": predict, "score": score}[args.stage](args)


if __name__ == "__main__":
    main()
