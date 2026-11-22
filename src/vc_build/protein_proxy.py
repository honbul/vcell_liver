"""Distill source proteomics into an RNA-only HCC drug response model."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import ndtri
from scipy.stats import rankdata

from .expanded_panel_cv import RECEIPTS, registered_inputs, same_split
from .external import KEY, _validate
from .io import freeze_table, sha256
from .limore_corrected import shared_support
from .pdo_transfer_cv import device_cuda
from .protein_data import SOURCES, prepare
from .response_component_benchmark import (
    BASELINES, EXTERNAL_TRACKS, FORECAST_SCHEMA, PUBLIC, external_inputs, external_reference, source_reference,
)
from .response_components import component_scores, interaction_intervals
from .rna_state_projection import assess, score_external, score_source
from .structured_cv import ASSAYS, with_assay
from .structured_kernel import drug_kernels, fit_cell_kernels, solve_residuals, training_grid
from .v0plus_cv import molecular_inputs, panel_targets, validation_roster
from .v0plus_neural import load_auxiliary
from .v0plus_score import axis_metrics, crossed_patient_drug_ci
from .yang_hcc_external import require_hash

CIS = "protein_cis_rna"
PRIMARY = "protein_proxy_rna"
OBSERVED = "protein_measured_cohort"
CALIBRATED = "protein_proxy_cohort"
ARMS = (CIS, PRIMARY)
TECHNICAL = (CALIBRATED, OBSERVED)
PRESENT = .75


def normalized_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    n = np.isfinite(values).sum(axis=0)
    ranks = rankdata(values, axis=0, nan_policy="omit")
    return ndtri((ranks - .5) / np.maximum(n, 1)).astype(np.float32)


def donor_calibration(values, donors):
    reference = values.groupby(donors).mean()
    result = values.copy()
    for key in values:
        fit = np.sort(reference[key].dropna())
        observed = values[key].notna()
        query = values.loc[observed, key].to_numpy()
        mid = np.searchsorted(fit, query, side="left") + np.searchsorted(fit, query, side="right")
        q = (mid / (2 * len(fit))).clip(.5 / len(fit), 1 - .5 / len(fit))
        result.loc[observed, key] = ndtri(q)
    return result


def substitutions(measured, genes, variants, query, held, targets):
    donors = query.drop_duplicates("cell_id").set_index("cell_id").donor_id.reindex(held.index)
    profile = measured.reindex(held.index).reindex(columns=genes)
    observed_donor = profile.groupby(donors).mean()
    columns = observed_donor.columns[observed_donor.notna().mean().ge(PRESENT) & observed_donor.std().gt(1e-7)]
    if donors.nunique() != 38 or len(columns) < 100:
        raise ValueError("Insufficient independent measured LICOB protein features")
    profile = profile[columns]
    protein_index = torch.as_tensor(genes.get_indexer(columns), device=variants[PRIMARY].device)
    proxy = pd.DataFrame(variants[PRIMARY][:, protein_index].cpu().numpy(), index=held.index, columns=columns)
    direct_rna = pd.DataFrame(variants[CIS][:, protein_index].cpu().numpy(), index=held.index, columns=columns)
    reference = observed_donor[columns]
    proxy_corr = proxy.where(profile.notna()).groupby(donors).mean().corrwith(reference, method="spearman").dropna()
    rna_corr = direct_rna.where(profile.notna()).groupby(donors).mean().corrwith(reference, method="spearman").dropna()
    target_panel = sorted(set().union(*(targets.get(key, ()) for key in query.compound_id.unique())))
    common = proxy_corr.index.intersection(rna_corr.index)
    audit = {
        "cohort_aligned_proteins": len(columns), "model_protein_observed_fraction": float(profile.notna().mean().mean()),
        "donors": donors.nunique(), "rank_agreement_genes": len(common),
        "direct_rna_median_spearman": float(rna_corr.loc[common].median()),
        "protein_proxy_median_spearman": float(proxy_corr.loc[common].median()),
        "target_correlations": {key: {"protein_proxy": float(proxy_corr[key]), "direct_rna": float(rna_corr[key])}
                                for key in common.intersection(target_panel)},
    }
    for name, value in ((OBSERVED, profile), (CALIBRATED, proxy.where(profile.notna()))):
        aligned = donor_calibration(value.astype("float64"), donors)
        replacement = variants[PRIMARY].new_tensor(aligned.to_numpy())
        variants[name] = variants[PRIMARY].clone()
        variants[name][:, protein_index] = torch.where(replacement.isfinite(), replacement,
                                                        variants[name][:, protein_index])
    return audit


def protein_features(basis, source, learned, held, fit, query, measured, targets, *, replay):
    rna, pathways, _ = basis.reference
    held_vectors = basis.vectors(held)
    observed = learned.reindex(source.index).dropna(how="all")
    lineages = fit.drop_duplicates("cell_id").set_index("cell_id").lineage_group.reindex(observed.index)
    donor_values = observed.groupby(lineages).mean()
    eligible = donor_values.columns[donor_values.notna().sum().ge(max(100, len(donor_values) * PRESENT))
                                    & donor_values.std().gt(1e-7)]
    genes = eligible.intersection(source.columns).sort_values()
    if len(genes) < 100:
        raise ValueError("Insufficient fit-only protein targets")
    device = rna.device
    rank = normalized_ranks(donor_values[genes])
    rank = pd.DataFrame(rank, index=donor_values.index).loc[lineages].to_numpy()
    y = rna.new_tensor(np.nan_to_num(rank))
    measured_row = torch.as_tensor(source.index.get_indexer(observed.index), device=device)
    select = rna.var(dim=0).argsort(descending=True, stable=True)[:256]

    def design(vectors):
        return torch.cat((vectors[0][:, select], vectors[1]), dim=1)

    x = design(basis.reference)
    x_bar = x[measured_row].mean(dim=0)
    centered = x[measured_row] - x_bar
    coef = torch.linalg.solve(centered.T @ centered + 1000 * torch.eye(x.shape[1], device=device),
                              centered.T @ (y - y.mean(dim=0)))
    train = (x - x_bar) @ coef
    mu = train.mean(dim=0)
    sigma = train.std(dim=0, correction=0).clamp_min(.05)

    def standardized(value):
        return ((value - mu) / sigma).clamp(-8, 8)

    direct = torch.as_tensor(source.columns.get_indexer(genes), device=device)
    variants = {CIS: held_vectors[0][:, direct], PRIMARY: standardized((design(held_vectors) - x_bar) @ coef)}
    audit = {"fit_models": len(observed), "fit_lineages": len(donor_values), "eligible_proteins": len(genes),
             "supervised_features": x.shape[1], "observed_fraction": float(np.isfinite(rank).mean())}
    if replay:
        single = standardized((design(basis.vectors(held.iloc[:1])) - x_bar) @ coef)
        error = float((single - variants[PRIMARY][:1]).abs().max())
        if error > 2e-5:
            raise ValueError(f"Single-RNA protein estimate changed with batch size: {error}")
        audit["single_query_max_error"] = error
    if measured is not None:
        audit["measured_technical"] = substitutions(measured, genes, variants, query, held, targets)
    return {CIS: rna[:, direct], PRIMARY: standardized(train)}, variants, audit


def new_forecasts(fit, query, rna, edges, targets, auxiliary, learned, measured, *, replay):
    _validate(fit, query)
    device = device_cuda()
    source = rna.loc[sorted(fit.cell_id.unique())]
    held = rna.loc[sorted(query.cell_id.unique())]
    basis, standard = fit_cell_kernels(source, edges, auxiliary, device=device)
    old_cross = basis.cross(held)["full"]
    fit_features, new_features, audit = protein_features(
        basis, source, learned, held, fit, query, measured, targets, replay=replay,
    )
    _, anchor, labels, _ = panel_targets(fit)
    drugs, kernel = drug_kernels(labels, targets, device=device)
    chemistry = kernel["structured"]
    w, y = training_grid(labels, source.index, drugs, device=device)
    idx = held.index.get_indexer(query.cell_id), drugs.get_indexer(query.compound_id)
    if any((axis < 0).any() for axis in idx):
        raise ValueError("Missing assay compound or RNA")
    baseline = query.compound_id.map(anchor).to_numpy()
    output = []
    original = standard["full"]
    for arm, x in fit_features.items():
        gram = x @ x.T
        scale = original.diagonal().mean() / gram.diagonal().mean()
        weights, steps, err = solve_residuals(.75 * original + .25 * scale * gram, chemistry, w, y, 4.)
        selected = (arm,) if arm == CIS else tuple(name for name in new_features if name != CIS)
        for name in selected:
            cross = .75 * old_cross + .25 * scale * (new_features[name] @ x.T)
            prediction = (cross @ weights @ chemistry).cpu().numpy()[idx]
            output.append(query[KEY].assign(method=name, predicted_value=baseline + prediction))
        audit[arm] = {"iterations": steps, "solver_residual": err}
    return pd.concat(output, ignore_index=True), audit


def predict(args):
    external = args.mode in EXTERNAL_TRACKS
    if external:
        fit, query, rna, registered = external_inputs(args, args.mode)
        previous, reference = external_reference(args, args.mode, query)
    else:
        original, clean, rna, _ = registered_inputs(args)
        _, fit, check = same_split(original, clean, args.fold, args.mode)
        query = validation_roster(check)
        previous, reference = source_reference(args.cv_archive, args.mode, args.fold)
        registered = RECEIPTS["core"]
    edges, targets, priors = molecular_inputs(args.priors, args.hallmark, args.targets)
    auxiliary, aux = load_auxiliary(args.auxiliary)
    receipt_path = args.protein / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if any(receipt["input_hashes"].get(k) != digest for k, digest in SOURCES.items()):
        raise ValueError("Unexpected proteomics registration")
    data = {}
    for name in ("source", "licob") if args.mode == "LICOB" else ("source",):
        path = args.protein / (name + ".parquet")
        require_hash(path, receipt["files"][path.name])
        data[name] = pd.read_parquet(path)
    current, audit = new_forecasts(
        fit, query, rna, edges, targets, auxiliary, data["source"], data.get("licob"), replay=external,
    )
    current = current.assign(track=args.mode, fold=0) if external else with_assay(current, check, args.fold)
    joint = pd.concat([previous, current], ignore_index=True)[FORECAST_SCHEMA]
    if external:
        extra = TECHNICAL if args.mode == "LICOB" else ()
        shared_support(joint, query, (*BASELINES, *PUBLIC, *ARMS, *extra))
    else:
        support = joint.pivot(index=[*KEY, "track", "fold"], columns="method", values="predicted_value")
        if set(support) != {*BASELINES, *ARMS} or not np.isfinite(support).all(axis=None):
            raise ValueError("Source forecasts differ from comparison support")
    freeze_table(joint, args.out, provenance={"mode": args.mode, "fold": args.fold, "inputs_sha256": registered,
        "core_sha256": RECEIPTS["core"], "reference_sha256": reference, "priors": priors, "auxiliary": aux,
        "protein_sha256": sha256(receipt_path), "protein_diagnostics": audit})


def assess_protein(frame, *, external):
    result = assess(frame, external=external, primary=PRIMARY, controls=("structured", CIS), require_precision=True)
    if external:
        result["interaction_ci95"] = interaction_intervals(frame, PRIMARY, "structured")
        result["gates"]["structured"] &= result["interaction_ci95"][0] > 0
        result["components"] = component_scores(frame, cell_axis="lineage_group")
    if OBSERVED in frame.method.unique():
        result["measured_technical"] = {
            "across_cell_ci95": crossed_patient_drug_ci(frame, OBSERVED, CALIBRATED),
            "within_cell": axis_metrics(frame, [(OBSERVED, CALIBRATED)])[0]["paired_contrasts"],
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    fields = {
        "prepare": ("licob", "procan-models", "proteins", "hgnc", "models", "node"),
        "predict": ("licob", "archive", "cv-archive", "priors", "hallmark", "targets", "auxiliary", "protein"),
        "source": ("forecasts",), "external": ("licob", "forecasts", "limore-labels", "licob-labels"),
    }
    for stage, names in fields.items():
        sub = stages.add_parser(stage)
        for name in ("original", "core", *names, "out"):
            sub.add_argument(f"--{name}", type=Path, required=True)
        if stage == "predict":
            sub.add_argument("--mode", choices=(*ASSAYS, *EXTERNAL_TRACKS), required=True)
            sub.add_argument("--fold", choices=range(3), type=int, default=0)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args)
    elif args.stage == "predict":
        predict(args)
    elif args.stage == "source":
        score_source(args, assessor=assess_protein)
    else:
        score_external(args, assessor=assess_protein, arms=ARMS, extra_by_mode={"LICOB": TECHNICAL})


if __name__ == "__main__":
    main()
