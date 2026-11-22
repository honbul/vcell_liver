"""Analyze frozen adult HCC predictions using paired cell and drug-pair cluster weights."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vc_fm_bench.artifacts import digest, file_record, new_directory, verify_record, write_json

ROOT = Path("/mnt/S1/sdata/workbench/Virtual_Cell/gssong/nextgen_5track")
MODELS = ("pair_mean", "mono_bliss_independent", "lightgbm", "deepsynergy", "matchmaker", "cross_attention")
BASELINES = ("lightgbm", "pair_mean")
FOLDS = {"lco_hcc": 1, "lpo": 5, "ldo": 5}
ENDPOINTS = ("bliss", "added_inhibition", "zip")
BOUNDS = (5, 10, 20)
DRAWS = 5000
SEED = 20261123
SCORE_RECEIPT_SHA = "55ded4ceafd4406d892534c4cd38bb7c0c8660f609515f9bfe40df39b89abb9c"
SCREEN_TSV_SHA = "448f1667bb1b9b769740e6284e36c48133fe1f09e832d757637d574de41f67a8"


def read(path):
    return json.loads(path.read_text())


def get_scope(scope, score_record):
    panel = ROOT / "prepared/combo-v1/panels" / scope
    frozen = ROOT / "results/combo-v1/predictions" / scope
    p, f = panel / "receipt.json", frozen / "receipt.json"
    verify_record(p, score_record["panel"])
    verify_record(f, score_record["frozen"])
    trained = read(f)
    verify_record(p, trained["panel"])
    contract = read(p)
    for name in ("labels.parquet", "query.parquet"):
        verify_record(panel / name, contract["files"][name])
    verify_record(panel / "labels.parquet", score_record["labels"])
    verify_record(frozen / "predictions.parquet", trained["files"]["predictions.parquet"])
    q = pd.read_parquet(panel / "query.parquet").set_index("id", verify_integrity=True)
    y = pd.read_parquet(panel / "labels.parquet").set_index("id", verify_integrity=True)
    predictions = pd.read_parquet(frozen / "predictions.parquet")
    model = predictions.loc[predictions.model.isin(MODELS)]
    wide = model.pivot(index="id", columns="model", values=list(ENDPOINTS))
    wide.columns = [f"{m}/{e}" for e, m in wide.columns]
    covered = [set(group.id) for _, group in model.groupby("model")]
    if len(covered) != len(MODELS) or not all(set(wide.index) == ids for ids in covered):
        raise ValueError(f"Unequal frozen model roster: {scope}")
    if len(wide) != score_record["scored"] or set(q.index) != set(y.index):
        raise ValueError(f"Unexpected held panel support: {scope}")
    if not set(wide.index) <= set(q.index):
        raise ValueError(f"Unregistered forecast row: {scope}")
    rows = q.join(y, validate="one_to_one").join(wide, how="inner", validate="one_to_one")
    rows = rows.loc[rows.source.str.startswith("Sanger:") & rows.adult_hcc]
    pairs = rows[["connectivity_a", "connectivity_b"]].apply(lambda values: "|".join(sorted(values)), axis=1)
    if not pairs.equals(rows.pair):
        raise ValueError(f"Drug order is not invariant in {scope}")
    evidence = {**score_record, "predictions": file_record(frozen / "predictions.parquet")}
    return rows, evidence


def complete_cases(frame, arms):
    fields = list(ENDPOINTS) + [m + "/" + e for m in arms for e in ENDPOINTS]
    return frame.loc[np.isfinite(frame[fields]).all(axis=1), ["cell_group", "pair", *fields]]


def compact(rows):
    fields = rows.drop(columns=["cell_group", "pair"]).columns
    values = rows.groupby(["cell_group", "pair"], sort=True)[list(fields)].mean().reset_index()
    return values


def confidence(values):
    return np.quantile(values[np.isfinite(values)], [.025, .975])


def widths(x):
    # Group ties exactly, including the common zero-Bliss predictions.
    cuts = np.flatnonzero(x[:-1] != x[1:]) + 1
    return np.r_[0, cuts], np.r_[cuts, len(x)]


def apo(scores, truth, weights):
    if scores.ndim == 1:
        scores = np.broadcast_to(scores, weights.shape)
    order = np.argsort(-scores, axis=1, kind="stable")
    p, y, w = (np.take_along_axis(v, order, axis=1) for v in (
        scores, np.broadcast_to(truth, weights.shape), weights))
    hits = np.cumsum(w*y, axis=1)
    mass = np.cumsum(w, axis=1)
    ends = np.c_[p[:, :-1] != p[:, 1:], np.ones(len(p), dtype=bool)]
    lower = np.c_[np.zeros(len(p)), np.maximum.accumulate(np.where(ends, hits, 0), axis=1)[:, :-1]]
    precision = np.divide(hits,mass,out=np.zeros_like(hits),where=mass > 0)
    return np.divide(((hits-lower)*precision*ends).sum(axis=1), hits[:, -1],
                     out=np.full(len(hits), np.nan), where=hits[:, -1] > 0)


def weighted_midranks(x, w):
    order = np.argsort(x, kind="stable")
    sorted_x = x[order]
    start, end = widths(sorted_x)
    cumulative = np.c_[np.zeros(len(w)), np.cumsum(w[:, order], axis=1)]
    means = .5*(cumulative[:, start] + cumulative[:, end])
    ranked = np.repeat(means, end-start, axis=1)
    return ranked[:, np.argsort(order)]


def correlation(pred, truth, weight):
    x = weighted_midranks(pred, weight)
    y = weighted_midranks(truth, weight)
    norm = weight / weight.sum(axis=1, keepdims=True)
    x = x - (x*norm).sum(axis=1, keepdims=True)
    y = y - (y*norm).sum(axis=1, keepdims=True)
    cov = (x*y*norm).sum(axis=1)
    denom = np.sqrt((x*x*norm).sum(axis=1)*(y*y*norm).sum(axis=1))
    return np.divide(cov, denom, out=np.zeros_like(cov), where=denom>0)


def mean_rows(values, weights):
    return (weights*values).sum(axis=-1) / weights.sum(axis=-1)


def constrained_slots(scores, outcomes, weights, indices, cell_weights):
    n = len(scores)
    slates = []
    for ids in indices:
        # Indices are sorted lexically by pair inside each lineage.
        selected = ids[np.argsort(-scores[ids], kind="stable")[:max(BOUNDS)]]
        if len(selected) < max(BOUNDS):
            raise ValueError("At least 20 distinct pair candidates per cell are needed")
        slates.append(selected)
    chosen = np.asarray(slates)
    if chosen.max() >= n:
        raise ValueError("Invalid cell-pair slate")
    for k in BOUNDS:
        current = chosen[:, :k]
        w = weights[:, current]
        normalized = w / w.sum(axis=2, keepdims=True)
        fields = {name: mean_rows((normalized * v[current]).sum(axis=2), cell_weights)
                  for name, v in outcomes.items()}
        fields["strong_hits"] = k*fields["strong_precision"]
        fields["efficacy_hits"] = k*fields["efficacy_precision"]
        yield k, fields


def pair_matrix(values, cell, pair, n_cells, n_pairs):
    result = np.zeros((n_cells, n_pairs), dtype=np.float64)
    result[cell, pair] = values
    return result


def summarize(data, axis, cohort, models):
    cells, ci = np.unique(data.cell_group, return_inverse=True)
    pairs, pi = np.unique(data.pair, return_inverse=True)
    nc, npair = len(cells), len(pairs)
    groups = [np.flatnonzero(ci == i) for i in range(nc)]
    measured_bliss = data.bliss.to_numpy()
    measured_added = data.added_inhibition.to_numpy()
    measured_zip = data.zip.to_numpy()
    strong = (measured_bliss >= 10) & (measured_added >= 10)
    efficacy = measured_added >= 10
    outcomes = {"strong_precision":strong, "efficacy_precision":efficacy,
                "mean_added_pp":measured_added,"mean_bliss_pp":measured_bliss,"mean_zip_pp":measured_zip}
    presence = pair_matrix(np.ones(len(data)), ci, pi, nc, npair)
    rng = np.random.default_rng(SEED)
    # Include a unit-weight row for the exact observed statistic, followed by the paired draws.
    cw = np.vstack([np.ones(nc), rng.exponential(size=(DRAWS, nc))])
    pw = np.vstack([np.ones(npair), rng.exponential(size=(DRAWS, npair))])
    cell_pair_weights = cw[:, ci] * pw[:, pi]
    denominator = cw @ presence
    any_strong = np.bincount(pi, weights=strong) > 0
    any_efficacy = np.bincount(pi, weights=efficacy) > 0
    report = {}
    for m in models:
        bliss, added, z = [data[m+"/"+e].to_numpy() for e in ENDPOINTS]
        for objective, score in (("joint", np.minimum(bliss - 10, added - 10)), ("efficacy", added)):
            positives, pair_positive = (strong, any_strong) if objective=="joint" else (efficacy, any_efficacy)
            report[m, objective, "row", 0, "auprc"] = apo(score, positives, cell_pair_weights)
            if objective=="joint":
                for name,pred,y in (("zip_spearman",z,measured_zip),("added_spearman",added,measured_added)):
                    report[m,objective,"row",0,name] = correlation(pred,y,cell_pair_weights)
            for k, perf in constrained_slots(score,outcomes,pw[:, pi],groups,cw):
                for name, weights in perf.items():
                    report[m,objective,"lineage",k,name] = weights
            score_matrix = pair_matrix(score,ci,pi,nc,npair)
            prediction = (cw @ score_matrix)/denominator
            report[m,objective,"pair",0,"auprc"] = apo(prediction,pair_positive,pw)
            top = np.argsort(-prediction, axis=1, kind="stable")[:, :20]
            for k in (10,20):
                kept = top[:,:k]
                weight = np.take_along_axis(pw, kept, axis=1)
                for name, target in (("strong_hits",any_strong),("efficacy_hits",any_efficacy)):
                    report[m,objective,"pair",k,name] = k*mean_rows(target[kept],weight)
    point, matched = [], []
    for (model, objective, unit, k, metric), sample in report.items():
        lo,hi = confidence(sample[1:])
        point.append(dict(axis=axis, support=cohort, model=model, objective=objective,
                          unit=unit, k=k, metric=metric, point=sample[0], lo95=lo, hi95=hi,
                          draws=int(np.isfinite(sample[1:]).sum())))
        if model=="cross_attention":
            for comparator in BASELINES:
                delta = sample-report[comparator,objective,unit,k,metric]
                low,high = confidence(delta[1:])
                matched.append(dict(axis=axis,support=cohort,comparator=comparator,objective=objective,
                                    unit=unit,k=k,metric=metric,delta=delta[0],lo95=low,hi95=high,
                                    draws=int(np.isfinite(delta[1:]).sum())))
    support = dict(model_families=list(models), cells=nc, distinct_pairs=npair, cell_pair_records=len(data),
                   per_cell_min=min(map(len,groups)), per_cell_max=max(map(len,groups)),
                   strong_rows=int(strong.sum()), efficacy_rows=int(efficacy.sum()),
                   strong_pairs=int(any_strong.sum()), efficacy_pairs=int(any_efficacy.sum()))
    return point,matched,support


def collect(out):
    archive = ROOT / "results/combo-v1"
    score_file = archive / "scores/all/receipt.json"
    if digest(score_file) != SCORE_RECEIPT_SHA:
        raise ValueError("Historical all-axis combination score receipt changed")
    scored = read(score_file)["held_panels"]
    evidence, points, differences, supports = {}, [], [], {}
    for axis, count in FOLDS.items():
        frames = []
        for i in range(count):
            key = f"{axis}-{i}"
            rows, audit = get_scope(key, scored[key])
            evidence[key] = audit
            frames.append(rows)
        data = pd.concat(frames, verify_integrity=True)
        supports[axis] = {}
        for tier, models in (("full", tuple(m for m in MODELS if m != "mono_bliss_independent")),
                             ("all_models", MODELS)):
            frame = complete_cases(data, models)
            result, pairs, support = summarize(compact(frame), axis, tier, models)
            if support["cells"] != 11:
                raise ValueError(f"Unexpected eligible HCC lineage count in {axis}/{tier}")
            support["held_complete_assay_rows"] = len(frame)
            points.extend(result)
            differences.extend(pairs)
            supports[axis][tier] = support
            print(axis, tier, support, flush=True)
    screen = archive / "report-v1/top20.tsv"
    if digest(screen) != SCREEN_TSV_SHA:
        raise ValueError("Organoid prediction table differs from historical report")
    fields = ["anchor","candidate","dili_concern","organoid_donors","bliss","added_inhibition",
              "added_inhibition_donor_low","added_inhibition_donor_high"]
    hypotheses = pd.read_csv(screen,sep="\t",usecols=fields)
    if len(hypotheses)!=20 or not hypotheses.bliss.max()<10:
        raise ValueError("Unexpected frozen organoid candidate table")
    with new_directory(out) as stage:
        pd.DataFrame(points).to_csv(stage / "combo_metrics.tsv",sep="\t",index=False)
        pd.DataFrame(differences).to_csv(stage / "combo_paired95.tsv",sep="\t",index=False)
        hypotheses.to_csv(stage / "combo_organoid_hypotheses.tsv",sep="\t",index=False)
        write_json(stage / "combo_receipt.json",dict(
            source_git_commit="96e6a7a128ac0c7ea53e64d7cb9741a8e9a4ec33",source_score_sha256=SCORE_RECEIPT_SHA,
            folds=evidence,supports=supports,bootstrap_draws=DRAWS,seed=SEED,
            bootstrap="paired crossed lineage and connectivity-pair positive exponential weights",
            pair_aggregation="equal-lineage mean of predictions with any qualifying observed lineage for activity",
            matched_hit_budget="fixed distinct model slates within each lineage; normalize sampled pair mass to the same K",
            organoid_top20_source=file_record(screen), organoid_max_bliss=float(hypotheses.bliss.max()),
            organoid_measured_combo=False))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", required=True,type=Path)
    collect(cli.parse_args().out)
