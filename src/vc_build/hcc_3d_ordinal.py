"""Learn within-study donor comparisons and transfer to a fully held HCC organoid study."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .evaluate import join_observations
from .external import KEY
from .hcc_study_pool import DRUGS, REFERENCE, STUDIES, data, donor_scales, registered
from .io import freeze_table, publish_json, verify_forecast
from .limore_corrected import shared_support
from .pdo_transfer_cv import device_cuda, domain_kernel_terms
from .structured_kernel import drug_kernels
from .v0plus_cv import molecular_inputs
from .v0plus_score import axis_metrics, crossed_patient_drug_ci

INPUT = "bd27132354dcf5537b717227099147514851ec0218d0c96920ec4c8a00a96a4c"
ARCHIVE = "d3cfcde3c4d4e71d29edacf1200176b2abbe5828b5ad0bddc7d266fc466f0616"
PRIMARY = "pairwise_ordinal"
ARMS = (PRIMARY, "pairwise_squared", "ordinal_without_2d")
REFERENCES = (*REFERENCE, "balanced_residual")
PENALTY = .02
CONTROLS = ("2d_teacher", "pairwise_squared")


def gram_features(gram, cross):
    fit = gram.double()
    values, basis = torch.linalg.eigh((fit + fit.T) / 2)
    supported = values > values[-1] * 1e-7
    if not supported.any():
        raise ValueError("No positive kernel feature axes")
    basis, scale = basis[:, supported], values[supported].sqrt()
    fit = basis * scale
    held = None if cross is None else cross.double() @ (basis / scale)
    return fit, held


def donor_pool(train, fit_ids, features):
    identities = train[["cell_id", "lineage_group", "study"]].drop_duplicates()
    if not identities.cell_id.is_unique or identities.groupby("lineage_group").study.nunique().max() != 1:
        raise ValueError("Ambiguous donor RNA identity across studies")
    ids = pd.Index(sorted(identities.lineage_group.unique()))
    per_cell = identities.set_index("cell_id").reindex(fit_ids)
    by_donor = ids.get_indexer(per_cell.lineage_group)
    if (by_donor < 0).any():
        raise ValueError("Missing donor RNA support")
    index = torch.as_tensor(by_donor, device=features.device)
    count = torch.bincount(index, minlength=len(ids))
    pooled = torch.zeros((len(ids), features.shape[1]), dtype=features.dtype, device=features.device)
    pooled.index_add_(0, index, features)
    return ids, pooled / count[:, None]


def paired_rows(train, calibration):
    donors = train.groupby(["study", "compound_id", "lineage_group"], as_index=False)[["endpoint_value", "contrast"]].mean()
    parts, counts = [], {}
    for (study, drug), rows in donors.groupby(["study", "compound_id"]):
        if len(rows) < 5:
            raise ValueError("Too few donor response ranks for a study and drug")
        a, b = np.triu_indices(len(rows), k=1)
        outcome = rows.endpoint_value.to_numpy()
        delta = outcome[a] - outcome[b]
        valid = delta != 0
        if not valid.any():
            raise ValueError("Constant measured outcome in training study and drug")
        contrast = rows.contrast.to_numpy()
        ids = rows.lineage_group.to_numpy()
        pairs = pd.DataFrame({
            "study": study, "compound_id": drug, "left": ids[a[valid]], "right": ids[b[valid]],
            "target": delta[valid] / outcome.std(), "prior": (contrast[a[valid]] - contrast[b[valid]]) / calibration[drug],
        })
        parts.append(pairs)
        counts[study] = counts.get(study, 0) + len(pairs)
    rows = pd.concat(parts, ignore_index=True)
    n_studies = donors.groupby("compound_id").study.nunique()
    pairs_per_block = rows.groupby(["study", "compound_id"]).left.transform("size")
    rows["weight"] = 1 / (pairs_per_block * rows.compound_id.map(n_studies) * len(calibration))
    if not np.isclose(rows.weight.sum(), 1.):
        raise ValueError("Broken donor-pair weighting")
    return rows, counts


def train_rank_model(first, second, weights, target, prior, *, logistic):
    theta = torch.zeros((first.shape[1], second.shape[1]), dtype=first.dtype, device=first.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([theta], max_iter=140, tolerance_grad=1e-8, tolerance_change=1e-11,
                                  line_search_fn="strong_wolfe")
    direction = target.sign()

    def objective():
        values = ((first @ theta) * second).sum(dim=1) + prior
        losses = F.softplus(-direction * values) if logistic else .5 * (values - target).square()
        return (weights * losses).sum() + .5 * PENALTY * theta.square().sum()

    def closure():
        optimizer.zero_grad()
        loss = objective()
        loss.backward()
        return loss

    initial = float(objective().detach())
    optimizer.step(closure)
    final = float(closure().detach())
    if not np.isfinite(final) or final > initial + 1e-8:
        raise RuntimeError("Rank fit did not decrease its objective")
    return theta.detach(), {
        "iterations": optimizer.state[theta]["n_iter"], "objective_initial": initial, "objective_final": final,
        "gradient_max_abs": float(theta.grad.abs().max()),
    }


def candidates(train, query, rna, edges, targets, device):
    if set(train.lineage_group) & set(query.lineage_group) or "endpoint_value" in query:
        raise ValueError("Held donors or labels entered training")
    train = train.assign(contrast=train["2d_teacher"] - train["2d_mean"])
    calibration = donor_scales(train).contrast_scale.groupby("compound_id").mean().clip(lower=.02)
    pairs, pair_count = paired_rows(train, calibration)
    fit_ids, held_ids = (pd.Index(sorted(item.cell_id.unique())) for item in (train, query))
    cell, cross = domain_kernel_terms(rna, fit_ids, held_ids, edges, targets, device)["nonlinear"]
    fit_features, held_features = gram_features(cell, cross)
    donors, donor_features = donor_pool(train, fit_ids, fit_features)
    drugs, kernels = drug_kernels(train, targets, device=device)
    drug_features, _ = gram_features(kernels["structured"], None)
    pair_index = [donors.get_indexer(pairs[name]) for name in ("left", "right")]
    pair_drugs = drugs.get_indexer(pairs.compound_id)
    q = held_ids.get_indexer(query.cell_id), drugs.get_indexer(query.compound_id)
    if any((index < 0).any() for index in (*pair_index, pair_drugs, *q)):
        raise ValueError("Unregistered donor/compound support")
    first = donor_features[pair_index[0]] - donor_features[pair_index[1]]
    second = drug_features[pair_drugs]
    weights, target, prior = (torch.as_tensor(pairs[name].to_numpy(), dtype=first.dtype, device=device)
                              for name in ("weight", "target", "prior"))
    factor = .5 * query.compound_id.map(calibration).to_numpy()
    results, diagnostics = [], {}
    for name, logistic, offset, baseline in (
        (PRIMARY, True, prior, "2d_teacher"), ("pairwise_squared", False, prior, "2d_teacher"),
        ("ordinal_without_2d", True, torch.zeros_like(prior), "2d_mean"),
    ):
        theta, diagnostics[name] = train_rank_model(first, second, weights, target, offset, logistic=logistic)
        correction = (held_features @ theta @ drug_features.T).cpu().numpy()[q]
        results.append(query[KEY].assign(method=name, predicted_value=query[baseline] + factor * correction))
    return pd.concat(results, ignore_index=True), {
        "pairs_by_study": pair_count, "fit_donors": train.groupby("study").lineage_group.nunique().to_dict(),
        "cell_kernel_rank": fit_features.shape[1], "drug_kernel_rank": drug_features.shape[1], "solves": diagnostics,
    }


def predict(args):
    if registered(args.inputs) != INPUT:
        raise ValueError("Unexpected archived whole-study design")
    if any(args.holdout == study and path is not None
           for study, path in (("LICOB", args.licob_book), ("Wong", args.wong_book))):
        raise ValueError("A held response file was passed to the predictor")
    train = pd.concat([data(args, name, full=False) for name in STUDIES if name != args.holdout], ignore_index=True)
    query = pd.read_parquet(args.inputs / f"{args.holdout}.parquet")
    rna = pd.read_parquet(args.inputs / "rna.parquet")
    edges, targets, prior_hash = molecular_inputs(args.priors, args.hallmark, args.targets)
    result, info = candidates(train, query, rna, edges, targets, device_cuda())
    previous, stamp = verify_forecast(args.archive)
    if (stamp["sha256"], stamp["provenance"]["input_sha256"]) != (ARCHIVE, INPUT):
        raise ValueError("Archived comparisons use a different cohort")
    matched = previous[previous.track.eq(args.holdout) & previous.method.isin(REFERENCES)]
    forecasts = pd.concat([matched, result.assign(track=args.holdout, fold=0)], ignore_index=True)
    shared_support(forecasts, query, (*REFERENCES, *ARMS))
    freeze_table(forecasts, args.out, provenance={
        "holdout": args.holdout, "inputs_sha256": INPUT, "archive_sha256": ARCHIVE,
        "training_studies": sorted(set(STUDIES) - {args.holdout}), "priors": prior_hash, **info,
    })


def scores(frame):
    compounds = frame.compound_id.nunique()
    constants = frame[frame.method.eq("2d_mean")].groupby("compound_id").predicted_value
    if not constants.nunique().eq(1).all():
        raise ValueError("The 2D anchor varies by held sample")
    block = frame.groupby(["lineage_group", "compound_id", "track", "fold", "method"], as_index=False)[
        ["predicted_value", "endpoint_value"]].mean()
    mean_rows = block.method.eq("2d_mean")
    block.loc[mean_rows, "predicted_value"] = block.loc[mean_rows, "compound_id"].map(constants.first())
    block = block.assign(cell_id=block.lineage_group, row_id=block.lineage_group + "|" + block.compound_id)
    comparisons = [(PRIMARY, name) for name in (*CONTROLS, "balanced_residual", "ordinal_without_2d")]
    overall = axis_metrics(block, comparisons)[0]
    intervals = {b: crossed_patient_drug_ci(block, a, b) for a, b in comparisons}
    if compounds <= 5:
        for row in overall["scores"]:
            row.pop("macro_ndcg_at_5", None)
            row.pop("macro_top5_overlap", None)
        overall["paired_contrasts"] = [r for r in overall["paired_contrasts"] if r["metric"] == "spearman"]
        gates = None
    else:
        within = {(r["baseline"], r["metric"]): r["ci95"] for r in overall["paired_contrasts"]}
        gates = {b: all(intervals[b][metric][0] > 0 for metric in ("spearman", "pearson"))
                 and all(within[b, metric][0] > -.005 for metric in ("spearman", "ndcg_at_5"))
                 for b in CONTROLS}
    return {"groups": block.lineage_group.nunique(), "drugs": compounds, "metrics": overall,
            "paired_donor_drug_ci95": intervals, "gate": gates}


def score(args):
    if registered(args.inputs) != INPUT:
        raise ValueError("Unexpected study registry")
    reports, hashes = {}, {}
    for study in STUDIES:
        forecast, stamp = verify_forecast(args.forecasts / f"{study}.parquet")
        info = stamp["provenance"]
        if (info["holdout"], info["inputs_sha256"], info["archive_sha256"], set(info["training_studies"])) != (
                study, INPUT, ARCHIVE, set(STUDIES) - {study}):
            raise ValueError("Predictor did not hold out the whole scoring study")
        truth = data(args, study, full=True)
        shared_support(forecast, truth, (*REFERENCES, *ARMS))
        joined = join_observations(forecast, truth)
        scopes = {"all": joined, "tki": joined[joined.compound_id.isin(DRUGS)]}
        if study == "LICOB":
            earlier = truth[truth.new_donor].lineage_group
            scopes["nine_prior_confirmation"] = joined[joined.lineage_group.isin(earlier)]
        reports[study] = {name: scores(frame) for name, frame in scopes.items()}
        hashes[study] = stamp["sha256"]
    publish_json(args.out, {"primary": PRIMARY, "input_sha256": INPUT, "forecast_sha256": hashes, "studies": reports,
                            "pass": all(all(reports[study]["all"]["gate"].values()) for study in ("LICOB", "Wong"))})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    fields = {"predict": ("priors", "hallmark", "targets", "archive"), "score": ("forecasts",)}
    for action, paths in fields.items():
        stage = stages.add_parser(action)
        for name in ("inputs", "transfer", *paths, "out"):
            stage.add_argument(f"--{name}", type=Path, required=True)
        for name in ("licob-book", "wong-book"):
            stage.add_argument(f"--{name}", type=Path)
        if action == "predict":
            stage.add_argument("--holdout", required=True, choices=STUDIES)
    args = parser.parse_args()
    {"predict": predict, "score": score}[args.stage](args)


if __name__ == "__main__":
    main()
