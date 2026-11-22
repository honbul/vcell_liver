"""Isolate dose supervision from source scale and forecast amplitude."""

import argparse
from contextlib import closing
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from openpyxl import load_workbook
from sklearn.model_selection import GroupKFold

from .evaluate import join_observations
from .expanded_panel_cv import expanded_targets
from .external import KEY, SOURCE, _validate, native_sensitivity
from .io import freeze_table, publish_json, sha256, verify_forecast
from .licob_cold import PENALTY
from .licob_endpoints import drug_gram
from .limore_corrected import shared_support
from .pdo_transfer_cv import device_cuda
from .response_components import centered_kernels, orthogonal_parts
from .structured_cv import paired_intervals, with_assay
from .structured_kernel import cell_kernels, solve_residuals, training_grid
from .v0plus_cv import molecular_inputs, validation_roster
from .v0plus_neural import load_auxiliary
from .v0plus_score import axis_metrics, compound_strata, crossed_patient_drug_ci
from .yang_hcc_external import require_hash
from .zhu_dose_transfer import IDENTITY, MIN_FIT_CELLS, REPLAY, SCREENS, TABLE
from .zhu_hcc_confirmation import ASSETS, ZHU, annotated_roster, observed, read_rosters, training_bundle

REFERENCE = "fd797857c40f220ea8c2869b2e924820b50b87631cf72255e0984d2fd58b996d"
DOSE, AUC, BLEND, MEAN = "balanced_dose", "balanced_auc", "balanced_blend", "drug_mean"
METHODS = (DOSE, AUC, BLEND, MEAN)
EXTERNAL_REF = (*REPLAY, "matched_auc", "matched_endpoint")
EXTERNAL = "ZHU10"
SEEDS = 2400


def sealed_inputs(args):
    prior, seal = verify_forecast(args.prior / "final.parquet")
    provenance = seal["provenance"]
    if seal["sha256"] != REFERENCE or provenance["cohort"] != ZHU:
        raise ValueError("The original dose comparison freeze has changed")
    physical = []
    for source in SCREENS:
        directory = args.prior / "extracts" / source
        receipt = directory / "receipt.json"
        require_hash(receipt, provenance["extracts"][source])
        require_hash(directory / TABLE, json.loads(receipt.read_text())["curve_sha256"])
        physical.append(pd.read_parquet(directory / TABLE))
    selected = prior[prior.track.eq("dose_matched")]
    if len(selected) != 8 * 672 or selected.compound_id.nunique() != 32:
        raise ValueError("Expected the same 32 supported Zhu compounds")
    fit, _, rna, _, _, added = training_bundle(args)
    observed = pd.concat(physical, ignore_index=True)
    observed = observed[observed.compound_id.isin(selected.compound_id)]
    fit = fit.merge(observed, on=[*IDENTITY, "endpoint_value"], validate="one_to_one").sort_values("row_id")
    if len(fit) != len(observed) or len(fit) != 22824:
        raise ValueError("The compared training labels must share the same exact source rows")
    return fit, rna, added, selected


def robust_labels(fit, values):
    levels = [*SOURCE, "compound_id"]
    frame = fit.assign(value=values)
    group = frame.groupby(levels).value
    gap = group.quantile(.75) - group.quantile(.25)
    source_scale = gap[gap > 1e-10].groupby(level=SOURCE).median()
    fallback = source_scale.reindex(gap.index.droplevel("compound_id")).fillna(0).to_numpy()
    scale = gap.clip(lower=pd.Series(np.maximum(fallback * .25, 1e-6), index=gap.index))
    rows = frame.join(scale.rename("scale"), on=levels, validate="many_to_one")
    normalized = (frame.value - group.transform("mean")) / rows.scale
    return rows.assign(residual=normalized), scale.groupby(level="compound_id").median()


def forecast(fit, query, rna, edges, targets, auxiliary):
    _validate(fit, query)
    device = device_cuda()
    cells = pd.Index(sorted(fit.cell_id.unique()))
    held = pd.Index(sorted(query.cell_id.unique()))
    drugs, gram = drug_gram(fit, targets, device)
    blocks, _ = cell_kernels(rna.loc[cells], rna.loc[held], edges, auxiliary, device=device)
    cell, cross = centered_kernels(*blocks["full"])
    index = held.get_indexer(query.cell_id), drugs.get_indexer(query.compound_id)
    physical = -fit.dose_viability
    anchor = (fit.assign(value=physical).groupby([*SOURCE, "compound_id"]).value.mean()
              .groupby(level="compound_id").mean().reindex(drugs).to_numpy())
    train, transfer, details = {}, {}, {}
    for arm, truth in ((DOSE, physical), (AUC, native_sensitivity(fit))):
        rows, scale = robust_labels(fit, truth)
        w, y = training_grid(rows, cells, drugs, device=device)
        _, _, interaction = orthogonal_parts(y.cpu().numpy(), w.cpu().numpy())
        coef, steps, error = solve_residuals(cell, gram, w, torch.as_tensor(interaction, device=device,
                                                          dtype=w.dtype), PENALTY)
        projected = coef @ gram.T
        train[arm], transfer[arm] = cell @ projected, cross @ projected
        details[arm] = {"steps": steps, "solver_error": error}
        if arm == DOSE:
            units = torch.as_tensor(scale.reindex(drugs).to_numpy(), device=device, dtype=w.dtype)
    rms = {arm: torch.sqrt((v.square() * w).sum(0) / w.sum(0)) for arm, v in train.items()}
    target_rms = rms[DOSE] * units
    if any(bool((v <= 1e-8).any()) for v in rms.values()):
        raise ValueError("A matched drug lacks fitted response variation")
    for arm in (DOSE, AUC):
        scale = target_rms / rms[arm]
        train[arm], transfer[arm] = train[arm] * scale, transfer[arm] * scale
    mixed = .5 * (train[DOSE] + train[AUC])
    mix_rms = torch.sqrt((mixed.square() * w).sum(0) / w.sum(0))
    if bool((mix_rms <= 1e-8).any()):
        raise ValueError("A matched drug lacks blended response variation")
    transfer[BLEND] = .5 * (transfer[DOSE] + transfer[AUC]) * (target_rms / mix_rms)
    draws = [query[KEY].assign(method=MEAN, predicted_value=anchor[index[1]])]
    rms_error = {}
    for arm in (DOSE, AUC, BLEND):
        values = transfer[arm].cpu().numpy()[index]
        draws.append(query[KEY].assign(method=arm, predicted_value=anchor[index[1]] + values))
        fitted = mixed * (target_rms / mix_rms) if arm == BLEND else train[arm]
        fitted_rms = torch.sqrt((fitted.square() * w).sum(0) / w.sum(0))
        rms_error[arm] = float((fitted_rms - target_rms).abs().max())
    return pd.concat(draws, ignore_index=True), {
        "fit_rows": len(fit), "fit_lineages": fit.lineage_group.nunique(), "rows": len(query),
        "test_lineages": query.lineage_group.nunique(), "test_drugs": query.compound_id.nunique(),
        "fit_amplitude_max_error": rms_error, "min_physical_rms": float(target_rms.min()),
        "median_physical_rms": float(target_rms.median()), "solves": details,
    }


def source_splits(fit):
    for fold, (a, b) in enumerate(GroupKFold(n_splits=3).split(fit, groups=fit.lineage_group)):
        train, test = fit.iloc[a], fit.iloc[b]
        for source in SCREENS:
            training = train[train.source_id.ne(source)]
            counts = training.groupby("compound_id").lineage_group.nunique()
            held = test[test.source_id.eq(source) & test.compound_id.isin(counts[counts.ge(MIN_FIT_CELLS)].index)]
            check = held.groupby("compound_id").lineage_group.nunique()
            held = held[held.compound_id.isin(check[check.ge(5)].index)]
            if held.compound_id.nunique() < 2 or held.lineage_group.nunique() < 20:
                raise ValueError(f"Insufficient registered held-screen support for {source} fold {fold}")
            yield fold, source, training, held


def predict(args):
    fit, rna, additions, old = sealed_inputs(args)
    edges, priors, _ = molecular_inputs(args.priors, args.hallmark, args.targets)
    auxiliary, _ = load_auxiliary(args.auxiliary)
    sections, traces = [], {}
    for fold, source, training, held in source_splits(fit):
        query = validation_roster(held)
        targets, _ = expanded_targets(training[training.compound_id.isin(additions)], priors, rna.columns)
        rows, traces[f"{source}:{fold}"] = forecast(training, query, rna, edges, targets, auxiliary)
        shared_support(rows, query, METHODS)
        sections.append(with_assay(rows, held, fold))
    query = read_rosters(args.zhu)["single_dose"]
    query = query[query.compound_id.isin(fit.compound_id)]
    targets, _ = expanded_targets(fit[fit.compound_id.isin(additions)], priors, rna.columns)
    rows, traces[EXTERNAL] = forecast(fit, query, rna, edges, targets, auxiliary)
    reference = old[old.method.isin(EXTERNAL_REF)]
    a = rows[rows.method.eq(MEAN)].set_index("row_id").sort_index().predicted_value
    b = old[old.method.eq(MEAN)].set_index("row_id").sort_index().predicted_value
    if not a.index.equals(b.index) or not np.allclose(a, b, rtol=0, atol=1e-12):
        raise ValueError("The prior and corrected comparison must share the drug mean")
    external = pd.concat([rows, reference], ignore_index=True).assign(track=EXTERNAL, fold=0)
    shared_support(external, query, (*METHODS, *EXTERNAL_REF))
    freeze_table(pd.concat([*sections, external], ignore_index=True), args.out, provenance={
        "cohort": ZHU, "reference_sha256": REFERENCE, "diagnostics": traces,
    })


def assessed(rows, comparisons, ci):
    report, per_drug, _ = axis_metrics(rows, comparisons)
    ranks = {(p["candidate"], p["baseline"], p["metric"]): p["ci95"][0] for p in report["paired_contrasts"]}
    decisions = []
    for a, b in comparisons:
        bounds = ci(rows, a, b, repeats=SEEDS)
        passed = all(bounds[m][0] > 0 for m in ("spearman", "pearson")) and all(
            ranks[a, b, m] > -.005 for m in ("spearman", "ndcg_at_5", "top5_overlap"))
        decisions.append({"candidate": a, "baseline": b, "intervals": bounds, "pass": passed})
    return report | {"gates": decisions}, per_drug


def score(args):
    predictions, stamp = verify_forecast(args.forecast)
    provenance = stamp["provenance"]
    if (provenance["cohort"], provenance["reference_sha256"]) != (ZHU, REFERENCE):
        raise ValueError("Expected a frozen same-support scale benchmark")
    fit, _, _, _ = sealed_inputs(args)
    held = []
    for fold, source, _, check in source_splits(fit):
        track = source + "/" + check.endpoint_id
        held.append(check.assign(track=track, fold=fold, row_id=check.cell_id + "|" + check.compound_id,
                                 sensitivity=-check.dose_viability, auc=native_sensitivity(check)))
    key = [*KEY, "track", "fold"]
    source_truth = pd.concat(held, ignore_index=True)
    source = predictions[predictions.track.ne(EXTERNAL)].merge(
        source_truth[key + ["sensitivity", "auc"]], on=key, how="left", validate="many_to_one",
    )
    if len(source) != len(source_truth) * len(METHODS) or source[["sensitivity", "auc"]].isna().any().any():
        raise ValueError("Source predictions must match the registered held identities")
    contrasts = [(DOSE, AUC), (DOSE, MEAN), (BLEND, AUC), (BLEND, DOSE), (BLEND, MEAN)]
    panels = {}
    for track, block in source.groupby("track"):
        panels[track] = {label: assessed(block.assign(endpoint_value=block[outcome]), contrasts, paired_intervals)[0]
                         for label, outcome in (("10uM", "sensitivity"), ("native_auc", "auc"))}
    query = read_rosters(args.zhu)["single_dose"]
    require_hash(args.labels, ASSETS["zhu-s4.xlsx"])
    with closing(load_workbook(args.labels, read_only=True, data_only=True)) as book:
        truth = observed(book, "single_dose", query)
        strata = compound_strata(annotated_roster(book, "single_dose", query))
    joined = join_observations(predictions[predictions.track.eq(EXTERNAL)], truth)
    other = [(arm, baseline) for arm in (DOSE, BLEND) for baseline in EXTERNAL_REF]
    external, drug = assessed(joined, contrasts + other, crossed_patient_drug_ci)
    subset = {}
    for name, drugs in strata.items():
        current = drug[drug.compound_id.isin(drugs)]
        if not current.empty:
            summary = current.groupby("method")[["spearman", "pearson"]].mean()
            subset[name] = {"drugs": current.compound_id.nunique(),
                            "mean": summary.astype(object).where(summary.notna(), None).to_dict("index")}
    joint = {}
    for arm in (DOSE, BLEND):
        count = sum(all(g["pass"] for g in v["10uM"]["gates"] if g["candidate"] == arm)
                    for v in panels.values())
        passed = all(g["pass"] for g in external["gates"] if g["candidate"] == arm)
        joint[arm] = {"held_sources_passed": count, "zhu_pass": passed, "joint": count >= 2 and passed}
    publish_json(args.out, {
        "forecast_sha256": stamp["sha256"], "labels_sha256": ASSETS["zhu-s4.xlsx"],
        "source": panels, "zhu": external, "zhu_strata": subset, "joint": joint,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    tasks = parser.add_subparsers(dest="stage", required=True)
    training = ("zhu", "core", "expanded", "icc", "licob", "prior")
    for stage, fields in {
        "predict": ("priors", "hallmark", "targets", "auxiliary"),
        "score": ("forecast", "labels"),
    }.items():
        task = tasks.add_parser(stage)
        for name in (*training, "out", *fields):
            task.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    {"predict": predict, "score": score}[args.stage](args)


if __name__ == "__main__":
    main()
