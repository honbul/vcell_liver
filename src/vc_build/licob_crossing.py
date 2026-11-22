"""Predict the probability of crossing the LICOB response threshold at each dose."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from sklearn.isotonic import isotonic_regression
from sklearn.metrics import roc_auc_score

from .expanded_panel_cv import expanded_targets
from .external import KEY
from .io import freeze_table, publish_json, verify_forecast
from .licob_cold import PENALTY, checked as checked_cold
from .licob_curves import DOSE_REGISTRATION, DOSES, FRACTIONS, read_sources
from .licob_dose_sources import COLD, registration
from .licob_endpoints import drug_gram, paired_donors, read_outcomes, summarize
from .limore_corrected import shared_support
from .pdo_transfer_cv import device_cuda
from .response_components import centered_kernels
from .v0plus_cv import molecular_inputs
from .v0plus_licob import ASSET_HASHES
from .v0plus_score import axis_metrics, compound_strata
from .yang_hcc_external import require_hash

PREVIOUS = "2278e41c2e87a7dc86265c907b3a8cb60a4672ad9fd12a140853095a42cabb5f"
PRIOR_METHODS = ("drug_mean", "direct_endpoint", "curve_joint")
CDF_METHODS = ("crossing_mean", "crossing_linear", "crossing_logit")
TRACKS = tuple(f"cross_by_dose_{index}" for index in range(len(FRACTIONS)))
POTENCY = "ic50_bounded"
CONTROLS = (*PRIOR_METHODS, "crossing_linear")


def dose_events(curves):
    return np.stack([isotonic_regression(curve, y_min=0, y_max=1, increasing=False) <= .5 for curve in curves])


def monotone_cdf(probabilities):
    return np.stack([isotonic_regression(row, y_min=0, y_max=1, increasing=True) for row in probabilities])


def restricted_potency(cdf, windows):
    bounds = np.log10(windows[["minimum", "maximum"]].to_numpy(dtype=float))
    area = trapezoid(np.asarray(cdf, dtype=float), x=FRACTIONS, axis=1)
    return -bounds[:, 1] + (bounds[:, 1] - bounds[:, 0]) * area


def kernel_features(gram):
    import torch

    eigenvalues, vectors = torch.linalg.eigh(gram)
    active = eigenvalues > 1e-6 * eigenvalues[-1]
    vectors, root = vectors[:, active], eigenvalues[active].sqrt()
    return vectors * root, vectors / root


def fit_logit(cell, cross, drug_dose, event_index, targets, balance):
    import torch

    cell_features, transform = kernel_features(cell)
    cross_features = cross @ transform
    drug_features, _ = kernel_features(drug_dose)
    theta = torch.zeros((cell_features.shape[1], drug_features.shape[1]), device=cell.device, requires_grad=True)
    reference = torch.logit(balance.flatten())
    optimizer = torch.optim.LBFGS([theta], max_iter=150, line_search_fn="strong_wolfe", tolerance_grad=1e-7)
    iterations = []

    def objective():
        logits = cell_features @ theta @ drug_features.T + reference
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[event_index], targets, reduction="sum")
        return (loss + .5 * PENALTY * theta.square().sum()) / targets.numel()

    def closure():
        optimizer.zero_grad()
        loss = objective()
        loss.backward()
        iterations.append(float(loss.detach().item()))
        return loss

    optimizer.step(closure)
    final = closure().item()
    if not np.isfinite(final) or final >= iterations[0]:
        raise ValueError("Penalized crossing likelihood failed to improve")
    diagnostics = {"initial_objective": iterations[0], "final_objective": final,
                   "closure_evaluations": len(iterations), "gradient_max": theta.grad.abs().max().item(),
                   "positive_cell_axes": cell_features.shape[1], "positive_drug_dose_axes": drug_features.shape[1]}
    with torch.no_grad():
        probabilities = (cross_features @ theta @ drug_features.T + reference).sigmoid()
    return probabilities.cpu().numpy(), diagnostics


def forecast_cdf(source, roster, expression, edges, targets, auxiliary):
    import torch
    from .structured_kernel import cell_kernels, solve_residuals

    device = device_cuda()
    cells = pd.Index(sorted(source.cell_id.unique()))
    held = pd.Index(sorted(roster.cell_id.unique()))
    if not cells.intersection(held).empty or set(source.lineage_group) & set(roster.lineage_group):
        raise ValueError("Cell and lineage train/holdout sets must be disjoint")
    drugs, drug = drug_gram(source, targets, device)
    terms, feature_count = cell_kernels(expression.loc[cells], expression.loc[held], edges, auxiliary, device=device)
    cell, cross = centered_kernels(*terms["full"])
    n_doses = len(FRACTIONS)
    row = torch.as_tensor(cells.get_indexer(source.cell_id), device=device)
    col = torch.as_tensor(drugs.get_indexer(source.compound_id), device=device)
    col_dose = col[:, None] * n_doses + torch.arange(n_doses, device=device)
    obs_index = (row[:, None], col_dose)
    observed = torch.as_tensor(dose_events(source[DOSES].to_numpy()), dtype=cell.dtype, device=device)
    counts = torch.bincount(col, minlength=len(drugs))
    success = torch.zeros((len(drugs), n_doses), device=device).index_add_(0, col, observed)
    balance = (success + .5) / (counts[:, None] + 1)
    frac = torch.as_tensor(FRACTIONS, dtype=cell.dtype, device=device)
    dose = torch.exp(-.5 * ((frac[:, None] - frac[None, :]) / .25).square())
    drug_dose = torch.kron(drug.contiguous(), dose)

    probability, fit_info = fit_logit(cell, cross, drug_dose, obs_index, observed, balance)
    weight = torch.zeros((len(cells), len(drugs) * n_doses), device=device)
    weight[obs_index] = 1.
    response = torch.zeros_like(weight)
    response[obs_index] = observed - balance[col]
    coef, iterations, residue = solve_residuals(cell, drug_dose, weight, response, PENALTY)
    linear = (cross @ coef @ drug_dose + balance.flatten()).cpu().numpy()

    query_index = held.get_indexer(roster.cell_id), drugs.get_indexer(roster.compound_id)
    if any((ix < 0).any() for ix in query_index):
        raise ValueError("Each query must match the trained roster")
    constant = balance.cpu().numpy()[query_index[1]]
    raw = {"crossing_mean": constant,
           "crossing_linear": linear.reshape(len(held), len(drugs), n_doses)[query_index],
           "crossing_logit": probability.reshape(len(held), len(drugs), n_doses)[query_index]}
    return {name: monotone_cdf(value) for name, value in raw.items()}, {
        "logit": fit_info, "linear_iterations": iterations, "linear_residual_norm": residue,
        "auxiliary_features": feature_count, "train_event_by_dose": observed.mean(dim=0).cpu().tolist(),
    }


def fit(args):
    from .v0plus_neural import load_auxiliary

    if checked_cold(args.cold, ("query", "rna")) != COLD:
        raise ValueError("Unexpected LICOB registration")
    source, dose_hash, stamps = read_sources(SimpleNamespace(run=args.previous, inputs=args.inputs))
    frozen, baseline = verify_forecast(args.previous / "forecast.parquet")
    if baseline["sha256"] != PREVIOUS or stamps != baseline["provenance"]["extracts"]:
        raise ValueError("Previous curve model and source archive disagree")
    if (len(source), source.compound_id.nunique(), set(source.source_id)) != (23482, 38, {"CTRPv2"}):
        raise ValueError("Unexpected curve support")
    if source.duplicated(["cell_id", "compound_id"]).any():
        raise ValueError("Curve likelihood requires one record per cell–compound pair")
    roster = pd.read_parquet(args.cold / "query.parquet")
    roster = roster[roster.compound_id.isin(source.compound_id)]
    windows = pd.read_parquet(args.inputs / "dose.parquet").set_index("compound_id").loc[roster.compound_id]
    expression = pd.read_parquet(args.cold / "rna.parquet")
    edges, targets, priors = molecular_inputs(args.priors, args.hallmark, args.targets)
    annotation = pd.concat([source, roster], ignore_index=True)
    targets, _ = expanded_targets(annotation[~annotation.compound_id.isin(targets)], targets, expression.columns)
    auxiliary, aux_receipt = load_auxiliary(args.auxiliary)
    retained = frozen[frozen.track.eq(POTENCY) & frozen.method.isin(PRIOR_METHODS)]
    shared_support(retained, roster, PRIOR_METHODS)
    emax = frozen[frozen.track.eq("emax") & frozen.method.eq("curve_joint")].assign(
        method="curve_emax", track="emax_reference")
    probabilities, diagnostics = forecast_cdf(source, roster, expression, edges, targets, auxiliary)
    parts = [retained, emax]
    for name, cdf in probabilities.items():
        grid = dict(zip(TRACKS, cdf.T))
        grid[POTENCY] = restricted_potency(cdf, windows)
        parts.extend(roster[KEY].assign(track=track, method=name, fold=0, predicted_value=values)
                     for track, values in grid.items())
    freeze_table(pd.concat(parts, ignore_index=True), args.run / "forecast.parquet", provenance={
        "previous_sha256": PREVIOUS, "dose_sha256": dose_hash, "licob_sha256": COLD,
        "labels_sha256": ASSET_HASHES["supplement"], "compound_ids": sorted(roster.compound_id.unique()),
        "fit_rows": len(source), "source_curves": stamps, "priors": priors, "auxiliary": aux_receipt,
        "fit": diagnostics,
    })


def verify_probabilities(forecast, roster, windows):
    probabilities = forecast[forecast.track.isin(TRACKS)]
    for track in TRACKS:
        shared_support(probabilities[probabilities.track.eq(track)], roster, CDF_METHODS)
    points = probabilities.pivot(index=["row_id", "method"], columns="track", values="predicted_value")
    values = points[list(TRACKS)].to_numpy()
    if np.any((values < 0) | (values > 1)) or np.any(np.diff(values, axis=1) < 0):
        raise ValueError("Invalid monotone dose CDF")
    compounds = roster.set_index("row_id").compound_id.reindex(points.index.get_level_values("row_id"))
    expected = restricted_potency(values, windows.loc[compounds])
    potency = forecast[forecast.track.eq(POTENCY) & forecast.method.isin(CDF_METHODS)]
    actual = potency.set_index(["row_id", "method"]).predicted_value.reindex(points.index).to_numpy()
    if not np.allclose(expected, actual, rtol=0, atol=1e-12):
        raise ValueError("Frozen rank score differs from the verified CDF integral")
    return probabilities[probabilities.track.eq(TRACKS[-1])]


def brier_intervals(squared_errors, baseline):
    change = (squared_errors.crossing_logit - squared_errors[baseline]).unstack("compound_id").to_numpy()
    rng = np.random.default_rng(15839)
    patients = rng.integers(change.shape[0], size=(2400, change.shape[0]))
    drugs = rng.integers(change.shape[1], size=(2400, change.shape[1]))
    differences = change[patients[:, :, None], drugs[:, None, :]]
    return np.quantile(np.nanmean(differences, axis=(1, 2)), [.025, .975]).tolist()


def score_events(probabilities, observed):
    truth = observed.dropna(subset="endpoint_value")
    truth = truth[truth.adult_age_confirmed].assign(event=lambda x: x.bound_status.ne("above").astype(float))
    merged = probabilities.merge(truth[["row_id", "donor_id", "event"]], on="row_id", validate="many_to_one")
    patients = merged.groupby(["donor_id", "compound_id", "method"])[["event", "predicted_value"]].mean()
    squared = (patients.predicted_value - patients.event).pow(2).unstack("method")
    pure = patients[patients.event.isin([0., 1.])].reset_index()
    auc = {}
    for name, arm in pure.groupby("method"):
        per_drug = [roc_auc_score(group.event, group.predicted_value) for _, group in arm.groupby("compound_id")
                    if set(group.event) == {0., 1.} and group.event.value_counts().min() >= 2]
        auc[name] = {"eligible_drugs": len(per_drug), "auroc": float(np.mean(per_drug)) if per_drug else None}
    return {"patient_drug_rows": len(squared), "rate": float(patients.event.mean()),
            "mean_brier": squared.mean().to_dict(), "mean_predicted": patients.groupby("method").predicted_value.mean().to_dict(),
            "brier_logit_difference_ci95": {x: brier_intervals(squared, x) for x in CDF_METHODS[:2]},
            "within_drug_auroc": auc}


def gate(result):
    if min(x["n_drugs"] for x in result["metrics"]["across_cell_scores"]) < 10:
        return {key: "insufficient_support" for key in CONTROLS}
    rank = {(x["baseline"], x["metric"]): x["ci95"][0] for x in result["metrics"]["paired_contrasts"]}
    return {name: all(result["within_drug_ci95"][f"crossing_logit/{name}"][k][0] > 0 for k in ("spearman", "pearson"))
            and all(rank[name, k] > -.005 for k in ("spearman", "ndcg_at_5", "top5_overlap"))
            for name in CONTROLS}


def scoring_inputs(args):
    forecast, seal = verify_forecast(args.run / "forecast.parquet")
    proof = seal["provenance"]
    if (proof["previous_sha256"], proof["dose_sha256"], proof["licob_sha256"], proof["labels_sha256"]) != (
        PREVIOUS, DOSE_REGISTRATION, COLD, ASSET_HASHES["supplement"]
    ) or (registration(args.inputs), checked_cold(args.cold, ("query",))) != (DOSE_REGISTRATION, COLD):
        raise ValueError("Unexpected crossing forecast or input hashes")
    roster = pd.read_parquet(args.cold / "query.parquet")
    dose = pd.read_parquet(args.inputs / "dose.parquet")
    require_hash(args.supplement, proof["labels_sha256"])
    return forecast, seal, roster, dose


def limited_emax(forecast, emax, selected):
    potency = forecast[forecast.track.eq(POTENCY)]
    reference = forecast[forecast.track.eq("emax_reference")].assign(track=POTENCY)
    predicted = pd.concat([potency, reference], ignore_index=True)
    limited = paired_donors(predicted[predicted.row_id.isin(selected)], emax[emax.row_id.isin(selected)])
    anchor = potency[potency.method.eq("crossing_mean")].groupby("compound_id").predicted_value.first()
    mask = limited.method.eq("crossing_mean")
    limited.loc[mask, "predicted_value"] = limited.loc[mask, "compound_id"].map(anchor)
    metrics, _, _ = axis_metrics(limited, ())
    return {"organoid_pairs": len(selected), "donor_pairs": int(limited.row_id.nunique()),
            "emax_within_drug": metrics["across_cell_scores"]}


def score(args):
    forecast, seal, roster, dose = scoring_inputs(args)
    roster = roster[roster.compound_id.isin(seal["provenance"]["compound_ids"])]
    probabilities = verify_probabilities(forecast, roster, dose.set_index("compound_id"))
    potency = forecast[forecast.track.eq(POTENCY)]
    shared_support(potency, roster, (*PRIOR_METHODS, *CDF_METHODS))
    reference = forecast[forecast.track.eq("emax_reference")]
    shared_support(reference, roster, ("curve_emax",))
    outcome = read_outcomes(args.supplement, roster, dose, selected=(POTENCY, "emax"))
    truth = outcome[POTENCY]
    donors = paired_donors(potency, truth)
    anchor = potency[potency.method.eq("crossing_mean")].groupby("compound_id").predicted_value.first()
    same = donors.method.eq("crossing_mean")
    donors.loc[same, "predicted_value"] = donors.loc[same, "compound_id"].map(anchor)
    result = summarize(donors, [("crossing_logit", name) for name in CONTROLS], compound_strata(roster))
    result["gate"] = gate(result)
    above = truth[truth.adult_age_confirmed & truth.bound_status.eq("above")].row_id
    publish_json(args.run / "score.json", {"forecast_sha256": seal["sha256"], "binomial": score_events(probabilities, truth),
        "scope": result, "actual_above_boundary": limited_emax(forecast, outcome["emax"], above)})


def audit(args):
    forecast, seal, roster, dose = scoring_inputs(args)
    compounds = seal["provenance"]["compound_ids"]
    supported = roster[roster.compound_id.isin(compounds)]
    probability = verify_probabilities(forecast, supported, dose.set_index("compound_id"))
    outcome = read_outcomes(args.supplement, roster, dose, selected=(POTENCY, "emax"))
    emax = outcome["emax"].assign(emax_viability=lambda r: -r.endpoint_value)
    ic50 = outcome[POTENCY]
    table = ic50[ic50.adult_age_confirmed].merge(emax[["row_id", "emax_viability"]], on="row_id", validate="one_to_one")
    summaries = {}
    for name, group in table.groupby(np.where(table.compound_id.isin(compounds), "supported", "outside_support")):
        paired = group.dropna(subset=["endpoint_value", "emax_viability"])
        cross = paired.groupby([paired.bound_status.eq("above").rename("ic50_above"),
                                paired.emax_viability.gt(.5).rename("emax_above")]).size()
        summaries[name] = {"donors": group.donor_id.nunique(), "drugs": group.compound_id.nunique(),
                           "pairs": len(paired), "joint_counts": {f"ic50_above_{int(i)}/emax_above_{int(j)}": int(cross.get((i,j), 0))
                                                                 for i in (False, True) for j in (False, True)}}
    supported_emax = emax[emax.compound_id.isin(compounds)].assign(
        bound_status=lambda r: np.where(r.emax_viability.gt(.5), "above", "within"))
    subset = supported_emax[supported_emax.adult_age_confirmed & supported_emax.bound_status.eq("above")].row_id
    publish_json(args.run / "endpoint_audit.json", {"forecast_sha256": seal["sha256"], "paired": summaries,
        "emax_binomial": score_events(probability, supported_emax),
        "emax_above_half_viability": limited_emax(forecast, supported_emax, subset)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    for name, function, paths in (
        ("fit", fit, ("previous", "priors", "hallmark", "targets", "auxiliary")),
        ("score", score, ("supplement",)),
        ("audit", audit, ("supplement",)),
    ):
        action = actions.add_parser(name)
        for field in ("run", "inputs", "cold", *paths):
            action.add_argument(f"--{field}", type=Path, required=True)
        action.set_defaults(execute=function)
    args = parser.parse_args()
    args.execute(args)


if __name__ == "__main__":
    main()
