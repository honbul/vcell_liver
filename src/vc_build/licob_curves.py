"""Fit a joint dose-curve kernel in verified 2D ranges and score LICOB separately."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from sklearn.isotonic import isotonic_regression

from .expanded_panel_cv import expanded_targets
from .external import KEY, SOURCE
from .function_data import new_output
from .io import freeze_table, publish_json, sha256, verify_forecast
from .licob_cold import PENALTY, checked as checked_cold
from .licob_dose_sources import COLD, SCREENS, extract_curves, registration
from .licob_endpoints import drug_gram, paired_donors, read_outcomes, summarize
from .limore_corrected import shared_support
from .pdo_transfer_cv import device_cuda
from .response_components import centered_kernels, orthogonal_parts
from .v0plus_cv import molecular_inputs
from .v0plus_licob import ASSET_HASHES
from .v0plus_score import compound_strata
from .yang_hcc_external import require_hash

FRACTIONS = np.linspace(0, 1, 9)
DOSES = [f"viability_{j}" for j in range(len(FRACTIONS))]
ENDPOINTS = ("ic50_bounded", "auc", "emax")
CONTROLS = ("drug_mean", "direct_endpoint", "curve_independent")
METHODS = (*CONTROLS, "curve_joint")
DOSE_REGISTRATION = "61ebf8b95d4a12eb3a23619b5e67ba0a683c025185f8de4a35c615e4abf7f84b"


def grid_values(dose, viability, *, minimum, second, maximum):
    log_dose = np.log10(dose)
    bounds = np.log10([minimum, maximum])
    if len(dose) < 5 or log_dose[0] > bounds[0] + 1e-5 or log_dose[-1] < bounds[1] - 1e-5:
        return None
    return np.interp(np.linspace(*bounds, len(FRACTIONS)), log_dose, np.clip(viability, 0, 1))


def extract(args):
    table, digest, quality = extract_curves(args, grid_values, DOSES)
    if digest != DOSE_REGISTRATION or not np.isfinite(table[DOSES].to_numpy()).all():
        raise ValueError("Dose grid must be complete and match the fixed registration")
    with new_output(args.run / args.assay) as output:
        table.to_parquet(output / "curves.parquet", index=False)
        publish_json(output / "receipt.json", {"input_sha256": digest, "source": args.assay,
            "curve_sha256": sha256(output / "curves.parquet"), "rows": len(table),
            "compounds": int(table.compound_id.nunique()), "quality": quality})


def read_sources(args):
    digest = registration(args.inputs)
    if digest != DOSE_REGISTRATION:
        raise ValueError("Unexpected registered dose grid")
    frames, stamps = [], {}
    for source in SCREENS:
        directory = args.run / source
        stamp = json.loads((directory / "receipt.json").read_text())
        if (stamp["input_sha256"], stamp["source"]) != (digest, source):
            raise ValueError("Source grid does not match its registered dose metadata")
        require_hash(directory / "curves.parquet", stamp["curve_sha256"])
        table = pd.read_parquet(directory / "curves.parquet")
        if len(table) != stamp["rows"] or not table.source_id.eq(source).all():
            raise ValueError("Inconsistent source curve counts")
        frames.append(table)
        stamps[source] = stamp
    rows = pd.concat(frames, ignore_index=True)
    counts = rows.groupby("compound_id").cell_id.nunique()
    return rows[rows.compound_id.isin(counts[counts.ge(30)].index)], digest, stamps


def endpoints(curves, windows):
    """Summarize monotone projected viability in the registered physical window."""
    values = np.array([isotonic_regression(row, y_min=0, y_max=1, increasing=False) for row in curves])
    bounds = np.log10(windows[["minimum", "maximum"]].to_numpy(dtype=float))
    lower, upper = values[:, 0] <= 0.5, values[:, -1] > 0.5
    crossing = np.full(len(values), 1.0)
    crossing[lower] = 0.0
    active = np.flatnonzero(~lower & ~upper)
    right = (values[active] <= 0.5).argmax(axis=1)
    left = right - 1
    high, low = values[active, left], values[active, right]
    crossing[active] = FRACTIONS[left] + ((high - 0.5) / (high - low)) * (FRACTIONS[right] - FRACTIONS[left])
    return {
        "ic50_bounded": -(bounds[:, 0] + crossing * (bounds[:, 1] - bounds[:, 0])),
        "auc": -trapezoid(values, x=FRACTIONS, axis=1),
        # A monotone projection makes the last dose the smaller of the last two.
        "emax": -values[:, -1],
    }


def fit_models(fit, roster, dose, rna, edges, targets, aux):
    import torch
    from .structured_kernel import cell_kernels, solve_residuals, training_grid

    device = device_cuda()
    cells, held = (pd.Index(sorted(f.cell_id.unique())) for f in (fit, roster))
    if len(cells.intersection(held)) or set(fit.lineage_group) & set(roster.lineage_group):
        raise ValueError("LICOB must be disjoint from 2D training by cell and lineage")
    drugs, drug = drug_gram(fit, targets, device)
    terms, count = cell_kernels(rna.loc[cells], rna.loc[held], edges, aux, device=device)
    fit_cell, held_cell = centered_kernels(*terms["full"])
    row_ix, col_ix = held.get_indexer(roster.cell_id), drugs.get_indexer(roster.compound_id)

    def prepare(values):
        frame = fit.assign(value=values)
        groups = frame.groupby([*SOURCE, "compound_id"]).value
        frame["residual"] = frame.value - groups.transform("mean")
        mean = groups.mean().groupby("compound_id").mean().reindex(drugs).to_numpy()
        weight, response = training_grid(frame, cells, drugs, device=device)
        _, _, residual = orthogonal_parts(response.cpu().numpy(), weight.cpu().numpy())
        return weight, torch.as_tensor(residual, dtype=weight.dtype, device=device), mean

    def apply(kernel, weight, response, row_mean):
        coefficient, steps, residual = solve_residuals(fit_cell, kernel, weight, response, PENALTY)
        estimate = (held_cell @ coefficient @ kernel).cpu().numpy() + row_mean
        return estimate, {"iterations": steps, "residual_norm": residual}

    ordered = dose.set_index("compound_id")
    fit_values = endpoints(fit[DOSES].to_numpy(), ordered.loc[fit.compound_id])
    results = {}
    solves = {}
    for endpoint in ENDPOINTS:
        weight, response, mean = prepare(fit_values[endpoint])
        estimate, solves[endpoint] = apply(drug, weight, response, mean)
        results[endpoint, "direct_endpoint"] = estimate[row_ix, col_ix]
        results[endpoint, "drug_mean"] = mean[col_ix]

    weight, response, mean = zip(*(prepare(fit[name]) for name in DOSES))
    width = len(FRACTIONS)
    shape = (len(cells), len(drugs) * width)
    weight = torch.stack(weight, dim=-1).reshape(shape)
    response = torch.stack(response, dim=-1).reshape(shape)
    mean = np.stack(mean, axis=-1).ravel()
    f = torch.as_tensor(FRACTIONS, dtype=drug.dtype, device=device)
    kernels = {"curve_joint": torch.exp(-0.5 * ((f[:, None] - f[None, :]) / 0.25).square()),
               "curve_independent": torch.eye(width, device=device)}
    windows = ordered.loc[roster.compound_id]
    for name, concentration in kernels.items():
        estimate, solves[name] = apply(torch.kron(drug.contiguous(), concentration), weight, response, mean)
        curves = estimate.reshape(len(held), len(drugs), width)[row_ix, col_ix]
        results.update({(endpoint, name): values for endpoint, values in endpoints(curves, windows).items()})
    forecast = pd.concat([roster[KEY].assign(track=endpoint, method=method, predicted_value=value, fold=0)
                          for (endpoint, method), value in results.items()], ignore_index=True)
    return forecast, {"auxiliary_features": count, "solver": solves}


def fit(args):
    from .v0plus_neural import load_auxiliary

    if checked_cold(args.cold, ("query", "rna")) != COLD:
        raise ValueError("Unexpected LICOB registration")
    query = pd.read_parquet(args.cold / "query.parquet")
    dose = pd.read_parquet(args.inputs / "dose.parquet")
    source, digest, stamps = read_sources(args)
    compounds = sorted(source.compound_id.unique())
    if len(compounds) < 5:
        raise ValueError("Fewer than five drugs on the registered full curve support")
    target = query[query.compound_id.isin(compounds)]
    rna = pd.read_parquet(args.cold / "rna.parquet")
    edges, targets, priors = molecular_inputs(args.priors, args.hallmark, args.targets)
    annotation = pd.concat([source, target], ignore_index=True)
    targets, _ = expanded_targets(annotation[~annotation.compound_id.isin(targets)], targets, rna.columns)
    aux, auxiliary = load_auxiliary(args.auxiliary)
    forecasts, diagnostics = fit_models(source, target, dose, rna, edges, targets, aux)
    for endpoint in ENDPOINTS:
        shared_support(forecasts[forecasts.track.eq(endpoint)], target, METHODS)
    counts = source.groupby(["source_id", "compound_id"]).size().unstack(fill_value=0)
    freeze_table(forecasts, args.run / "forecast.parquet", provenance={
        "cold_sha256": COLD, "registration_sha256": digest, "labels_sha256": ASSET_HASHES["supplement"],
        "extracts": stamps, "support": counts.to_dict(), "sources": source.source_id.value_counts().to_dict(),
        "priors": priors, "auxiliary": auxiliary, **diagnostics,
    })


def decision(result):
    metrics = result["metrics"]
    if min(row["n_drugs"] for row in metrics["across_cell_scores"]) < 10:
        return {name: "insufficient_support" for name in CONTROLS}
    rank = {(x["baseline"], x["metric"]): x["ci95"][0] for x in metrics["paired_contrasts"]}
    skill = result["within_drug_ci95"]
    return {name: all(skill[f"curve_joint/{name}"][metric][0] > 0 for metric in ("spearman", "pearson"))
            and all(rank[name, metric] > -.005 for metric in ("spearman", "ndcg_at_5", "top5_overlap"))
            for name in CONTROLS}


def score(args):
    forecast, seal = verify_forecast(args.run / "forecast.parquet")
    proof = seal["provenance"]
    if (proof["registration_sha256"], proof["cold_sha256"], proof["labels_sha256"]) != (
        DOSE_REGISTRATION, COLD, ASSET_HASHES["supplement"]
    ) or (registration(args.inputs), checked_cold(args.cold, ("query",))) != (DOSE_REGISTRATION, COLD):
        raise ValueError("Unregistered model, dose grid or LICOB outcome")
    require_hash(args.supplement, proof["labels_sha256"])
    roster = pd.read_parquet(args.cold / "query.parquet")
    roster = roster[roster.compound_id.isin(proof["support"])]
    dose = pd.read_parquet(args.inputs / "dose.parquet")
    for endpoint in ENDPOINTS:
        shared_support(forecast[forecast.track.eq(endpoint)], roster, METHODS)
    outcomes = read_outcomes(args.supplement, roster, dose)
    comparisons = [("curve_joint", name) for name in CONTROLS]
    results = {}
    for endpoint, truth in outcomes.items():
        predicted = forecast[forecast.track.eq(endpoint)]
        result = summarize(paired_donors(predicted, truth), comparisons, compound_strata(roster))
        if endpoint == "ic50_bounded":
            bounds = np.log10(dose.set_index("compound_id").loc[predicted.compound_id, ["minimum", "maximum"]])
            limits = predicted.assign(at_min=np.isclose(predicted.predicted_value, -bounds.minimum.to_numpy()),
                                      at_max=np.isclose(predicted.predicted_value, -bounds.maximum.to_numpy()))
            result["predicted_bound_fraction"] = limits.groupby("method")[["at_min", "at_max"]].mean().to_dict("index")
            result["gate"] = decision(result)
        results[endpoint] = result
    publish_json(args.run / "score.json", {"forecast_sha256": seal["sha256"],
        "labels_sha256": proof["labels_sha256"], "support": proof["support"], "results": results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    tasks = parser.add_subparsers(dest="stage", required=True)
    options = {
        "extract": (extract, ("original", "ctrp", "gdsc", "prism-treatments")),
        "fit": (fit, ("cold", "priors", "hallmark", "targets", "auxiliary")),
        "score": (score, ("cold", "supplement")),
    }
    for name, (function, fields) in options.items():
        command = tasks.add_parser(name)
        for field in ("run", "inputs", *fields):
            command.add_argument(f"--{field}", type=Path, required=True)
        if name == "extract":
            command.add_argument("--assay", choices=SCREENS, required=True)
        command.set_defaults(execute=function)
    args = parser.parse_args()
    args.execute(args)


if __name__ == "__main__":
    main()
