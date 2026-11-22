"""Test paired RNA adaptation using donor-disjoint folds and a fixed response model."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .evaluate import join_observations
from .expanded_panel_cv import PENALTY, expanded_targets
from .external import KEY, _validate
from .function_data import new_output
from .io import freeze_table, publish_json, sha256, verify_forecast
from .limore_corrected import shared_support
from .v0plus_cv import molecular_inputs, panel_targets
from .v0plus_score import axis_metrics, compound_strata, crossed_patient_drug_ci
from .v0plus_wong import WORKBOOK_HASH
from .wong_expanded import COHORT, checked, observations
from .yang_hcc_external import require_hash

REGISTRATIONS = {
    "organoid": "f7ad39cd33ca3f945cd38a9a31793193a4a5a9f23ecb6f9b939ff0cdb4f70b90",
    "tissue": "8e3a299929a0664c11859b38bb33b8420709a581449f15f27640c32a7fa7c1d7",
}
ORIGINAL_FORECASTS = {
    "organoid": "9330a821ea14ab99bde652f7ff66e84879d92ef63bd9e97a3fb3b41e38af9965",
    "tissue": "0537484b825b84e43b0d91efe57c11ed114a18831009f7a6b6373664aa075b1c",
}
PRIMARY = "paired_ridge"
CONTROLS = ("raw_tissue", "mean_shift", "shuffled_ridge")
VIEWS = (*CONTROLS, PRIMARY, "observed_pdo")
REFERENCES = ("drug_mean", "drug_ridge", "cellhit")
METHODS = (*VIEWS, *REFERENCES)


def registration(directory, view, names):
    inputs = directory / "inputs"
    require_hash(inputs / "receipt.json", REGISTRATIONS[view])
    checked(inputs, names)
    return inputs


def rank_values(expression):
    size = expression.shape[1]
    return (rankdata(expression, axis=1) - 0.5) / size - 0.5


def held_views(tissue, pdo):
    if not tissue.index.equals(pdo.index) or not tissue.columns.equals(pdo.columns):
        raise ValueError("Each donor and gene needs a matched tissue and organoid profile")
    if not np.isfinite(tissue).all(axis=None) or not np.isfinite(pdo).all(axis=None):
        raise ValueError("Paired RNA profiles must be finite")
    x, y = [rank_values(frame.to_numpy()) for frame in (tissue, pdo)]
    outputs = {name: np.empty_like(x) for name in (*CONTROLS[1:], PRIMARY)}
    fit_donors = {}
    for held, donor in enumerate(tissue.index):
        fit = np.arange(len(x)) != held
        fit_donors[donor] = tissue.index[fit].tolist()
        train = x[fit]
        centered = train - train.mean(axis=0)
        gram = centered @ centered.T / x.shape[1]
        scale = np.trace(gram) / len(gram)
        if scale <= 0:
            raise ValueError("The fit-donor RNA ranks must have positive variance")
        cross = (x[held] - train.mean(axis=0)) @ centered.T / x.shape[1] / scale
        influence = np.linalg.solve(gram / scale + np.eye(len(gram)), cross)
        delta = y[fit] - train
        mean = delta.mean(axis=0)
        variation = delta - mean
        outputs["mean_shift"][held] = x[held] + 0.5 * mean
        outputs[PRIMARY][held] = x[held] + 0.5 * (mean + influence @ variation)
        outputs["shuffled_ridge"][held] = x[held] + 0.5 * (mean + influence @ np.roll(variation, 1, axis=0))
    adjusted = {name: pd.DataFrame(values, index=tissue.index, columns=tissue.columns)
                for name, values in outputs.items()}
    return {"raw_tissue": tissue, "observed_pdo": pdo, **adjusted}, fit_donors


def historical(directory, view, query):
    frame, receipt = verify_forecast(directory / "final.parquet")
    origin = receipt["provenance"]
    if (receipt["sha256"], origin["registration_sha256"], origin["workbook_sha256"]) != (
        ORIGINAL_FORECASTS[view], REGISTRATIONS[view], WORKBOOK_HASH
    ):
        raise ValueError(f"Unexpected historical {view} forecast")
    methods = ("structured", *REFERENCES) if view == "tissue" else ("structured",)
    frame = frame[frame.method.isin(methods)]
    shared_support(frame, query, methods)
    return frame


def predict(args):
    from .pdo_transfer_cv import device_cuda
    from .structured_kernel import drug_kernels, fit_cell_kernels, solve_residuals, training_grid
    from .v0plus_neural import load_auxiliary

    organoid = registration(args.organoid, "organoid", ("train.parquet", "query.parquet", "rna.parquet"))
    tissue = registration(args.tissue, "tissue", ("query.parquet", "rna.parquet"))
    query = pd.read_parquet(organoid / "query.parquet")
    train = pd.read_parquet(organoid / "train.parquet")
    _validate(train, query)
    source_rna = pd.read_parquet(organoid / "rna.parquet")
    tissue_rna = pd.read_parquet(tissue / "rna.parquet")
    cells = sorted(query.cell_id.unique())
    if (len(cells), len(query), len(source_rna.columns)) != (10, 620, 17741):
        raise ValueError("The frozen Wong support changed")
    pdo_rna = source_rna.loc[cells]
    source_rna = source_rna.loc[sorted(train.cell_id.unique())]
    if not source_rna.equals(tissue_rna.loc[source_rna.index]):
        raise ValueError("The response model must use the same source RNA in every view")
    views, fit_donors = held_views(tissue_rna.loc[cells], pdo_rna)
    edges, targets, priors = molecular_inputs(args.priors, args.hallmark, args.targets)
    targets, _ = expanded_targets(train[~train.compound_id.isin(targets)], targets, source_rna.columns)
    auxiliary, auxiliary_stamp = load_auxiliary(args.auxiliary)
    device = device_cuda()
    _, anchor, aligned, _ = panel_targets(train)
    basis, cell_kernels = fit_cell_kernels(source_rna, edges, auxiliary, device=device)
    drugs, drug_kernels_by_name = drug_kernels(aligned, targets, device=device)
    weight, response = training_grid(aligned, source_rna.index, drugs, device=device)
    drug = drug_kernels_by_name["structured"]
    coefficients, steps, residual = solve_residuals(cell_kernels["full"], drug, weight, response, PENALTY)
    ix = pd.Index(cells).get_indexer(query.cell_id), drugs.get_indexer(query.compound_id)
    if any((index < 0).any() for index in ix):
        raise ValueError("All query IDs must belong to the trained kernels")
    tape = []
    for name, expression in views.items():
        response = (basis.cross(expression)["full"] @ coefficients @ drug).cpu().numpy()
        tape.append(query[KEY].assign(method=name, predicted_value=query.compound_id.map(anchor) + response[ix]))
    results = pd.concat(tape, ignore_index=True)
    old_tissue = historical(args.tissue, "tissue", query)
    old_pdo = historical(args.organoid, "organoid", query)
    replay = {}
    for control, old in (("raw_tissue", old_tissue), ("observed_pdo", old_pdo)):
        reference = old[old.method.eq("structured")].set_index("row_id").predicted_value
        values = results[results.method.eq(control)].set_index("row_id").predicted_value
        replay[control] = float((values - reference).abs().max())
        if replay[control] > 1e-5:
            raise ValueError(f"{control} did not reproduce the historically frozen response model")
    results = pd.concat([results, old_tissue[old_tissue.method.isin(REFERENCES)]], ignore_index=True)
    shared_support(results, query, METHODS)
    with new_output(args.out) as out:
        adjusted = pd.concat({name: views[name] for name in CONTROLS[1:] + (PRIMARY,)}, names=["method"])
        adjusted.to_parquet(out / "adjusted_rna.parquet")
        freeze_table(results.assign(track=COHORT, fold=0), out / "forecast.parquet", provenance={
            "registrations": REGISTRATIONS, "references": ORIGINAL_FORECASTS, "workbook_sha256": WORKBOOK_HASH,
            "adjusted_rna_sha256": sha256(out / "adjusted_rna.parquet"), "fit_donors": fit_donors,
            "prior_sources": priors, "auxiliary": auxiliary_stamp,
            "kernel_iterations": steps, "kernel_solver_residual": residual, "historical_replay_max_abs": replay,
        })


def molecular_scores(args, query):
    cells = sorted(query.cell_id.unique())
    matrices = {view: pd.read_parquet(registration(getattr(args, view), view, ("rna.parquet",)) / "rna.parquet",
                                    filters=[("cell_id", "in", cells)]).loc[cells]
                for view in REGISTRATIONS}
    adjusted = pd.read_parquet(args.freeze / "adjusted_rna.parquet")
    original = matrices["tissue"]
    target = rank_values(matrices["organoid"].to_numpy())
    methods = {"raw_tissue": original, **{name: adjusted.loc[name].loc[cells] for name in (*CONTROLS[1:], PRIMARY)}}
    scores = {}
    for name, rna in methods.items():
        if not rna.columns.equals(original.columns):
            raise ValueError("Molecular score requires the registered gene order")
        values = rank_values(rna.to_numpy())
        corr = np.corrcoef(values, target)[:len(cells), len(cells):]
        paired = np.diag(corr)
        scores[name] = {"mean_spearman": float(paired.mean()), "median_spearman": float(np.median(paired)),
                        "rank_rmse": float(np.sqrt(np.mean((values - target) ** 2))),
                        "top1_pairs": int(np.sum(paired == corr.max(axis=1)))}
    return scores


def assess(frame, *, main):
    pairs = [(PRIMARY, baseline) for baseline in (*CONTROLS, "drug_mean", "observed_pdo")]
    metrics, _, _ = axis_metrics(frame, pairs if main else ())
    crossed = {baseline: crossed_patient_drug_ci(frame, PRIMARY, baseline) for _, baseline in pairs} if main else {}
    rankings = {(row["baseline"], row["metric"]): row["ci95"] for row in metrics["paired_contrasts"]}
    gate = {baseline: all(crossed[baseline][metric][0] > 0 for metric in ("spearman", "pearson"))
            and all(rankings[baseline, metric][0] > -.005 for metric in ("spearman", "ndcg_at_5"))
            for baseline in CONTROLS} if main else {}
    return {"observed_rows": int(frame.row_id.nunique()), "scores": metrics["scores"],
            "across_cell_scores": metrics["across_cell_scores"], "paired_rank_contrasts": metrics["paired_contrasts"],
            "paired_drug_ci95": crossed, "gate": gate}


def score(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    forecasts, seal = verify_forecast(args.freeze / "forecast.parquet")
    provenance = seal["provenance"]
    if (provenance["registrations"], provenance["references"], provenance["workbook_sha256"]) != (
        REGISTRATIONS, ORIGINAL_FORECASTS, WORKBOOK_HASH
    ):
        raise ValueError("The freeze does not belong to this protocol")
    require_hash(args.freeze / "adjusted_rna.parquet", provenance["adjusted_rna_sha256"])
    organoid = registration(args.organoid, "organoid", ("query.parquet",))
    query = pd.read_parquet(organoid / "query.parquet")
    shared_support(forecasts, query, METHODS)
    cells = set(query.cell_id)
    folds = provenance["fit_donors"]
    if set(folds) != cells or any(len(fit) != 9 or set(fit) != cells - {donor} for donor, fit in folds.items()):
        raise ValueError("Each RNA adapter fold must hold the paired donor out")
    rna = molecular_scores(args, query)
    observed = observations(query, args.workbook)
    if observed.lineage_group.nunique() != 10 or observed.groupby("compound_id").size().ge(5).sum() < 60:
        raise ValueError("The independent measured support is below the protocol minimum")
    joined = join_observations(forecasts[forecasts.row_id.isin(observed.row_id)], observed)
    all_results = assess(joined, main=True)
    groups = compound_strata(query)
    groups.update({"previous_27": query.loc[query.previously_scored, "compound_id"].unique(),
                   "additional_35": query.loc[~query.previously_scored, "compound_id"].unique()})
    strata = {name: assess(joined[joined.compound_id.isin(drugs)], main=False)
              for name, drugs in groups.items() if joined.compound_id.isin(drugs).sum()}
    publish_json(args.out, {"forecast_sha256": seal["sha256"], "workbook_sha256": WORKBOOK_HASH,
                            "molecular_scores": rna, "all_drugs": all_results, "strata": strata,
                            "passed": all(all_results["gate"].values())})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    for name, function, options in (
        ("predict", predict, ("priors", "hallmark", "targets", "auxiliary")),
        ("score", score, ("freeze", "workbook")),
    ):
        command = actions.add_parser(name)
        for option in ("organoid", "tissue", "out", *options):
            command.add_argument(f"--{option}", type=Path, required=True)
        command.set_defaults(run=function)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
