"""Test fit-only drug anchor calibration on sealed source development predictions."""

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from .evaluate import score, score_across_cells
from .external import KEY, SOURCE, align_endpoints
from .io import freeze_table, publish_json, sha256, verify_forecast
from .structured_cv import ASSAYS, attach, observations, split
from .v0plus_cv import training_inputs

PREVIOUS_SCORE = "10af6062118b6b3ad9eb205b2b465e7beeaad03d1be90035d6bef8cfacfdd521"
BLEND = "ridge_structured_average"
PRIMARY = "pairwise_tempered"
INDEX = [*KEY, "track", "fold"]
TEMPERATURE = 0.5
PRIOR = 0.05


def paired_evidence(aligned, compounds):
    """Compare two drugs only in cells with both within-assay observations."""
    pairs = list(combinations(range(len(compounds)), 2))
    scores = []
    for assay, rows in aligned.groupby(SOURCE, sort=True):
        values = rows.pivot(index=["cell_id", "lineage_group"], columns="compound_id", values="model_value")
        values = values.reindex(columns=compounds)
        lineage = pd.factorize(values.index.get_level_values("lineage_group"))[0]
        matrix = values.to_numpy()
        for a, b in pairs:
            valid = np.isfinite(matrix[:, a]) & np.isfinite(matrix[:, b])
            lineage_counts = np.bincount(lineage[valid], minlength=lineage.max() + 1)
            n = int(np.count_nonzero(lineage_counts))
            if n < 30:
                continue
            difference = matrix[valid, a] - matrix[valid, b]
            weights = 1 / lineage_counts[lineage[valid]]
            win = np.sum(weights * (0.5 + 0.5 * np.sign(difference))) / n
            # Collapsing siblings stabilizes the signed difference across assays.
            donor_diff = np.bincount(lineage[valid], weights=difference * weights,
                                    minlength=len(lineage_counts))
            delta = np.median(donor_diff[lineage_counts > 0])
            scores.append((a, b, assay[0], n, win, delta))
    if not scores:
        raise ValueError("No drug pair has support in the fit assays")
    return pd.DataFrame(scores, columns=["left", "right", "source", "lineages", "probability", "delta"])


def fit_anchors(fit):
    aligned, _ = align_endpoints(fit)
    old = aligned.groupby([*SOURCE, "compound_id"]).model_value.mean().groupby(level="compound_id").mean()
    center = old.median()
    scale = old.quantile(.75) - old.quantile(.25)
    if scale <= 1e-8:
        raise ValueError("The source anchor has no measurable spread")
    reference = (old.to_numpy() - center) / scale
    evidence = paired_evidence(aligned, old.index)
    pairs = evidence.groupby(["left", "right"], sort=True)[["probability", "delta"]].mean().reset_index()
    i = pairs.left.to_numpy()
    j = pairs.right.to_numpy()
    design = np.eye(len(old))[i] - np.eye(len(old))[j]
    p = pairs.probability.to_numpy()
    normalizer = len(design) / len(old)
    conditional_temperature = 0.7

    def objective(location):
        z = design @ location / conditional_temperature
        shift = location - reference
        loss = np.logaddexp(0, z) - p * z
        gradient = design.T @ (expit(z) - p) / (len(design) * conditional_temperature)
        return loss.mean() + PRIOR * np.mean(shift ** 2) / 2, gradient + PRIOR * shift / len(old)

    ordinal = minimize(objective, reference, jac=True, method="L-BFGS-B",
                       options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 300})
    if not ordinal.success or not np.isfinite(ordinal.x).all():
        raise ValueError(f"Pairwise anchor did not converge: {ordinal.message}")
    delta = np.linalg.solve(design.T @ design + PRIOR * normalizer * np.eye(len(old)),
                            design.T @ (pairs.delta.to_numpy() / scale) + PRIOR * normalizer * reference)
    candidates = pd.DataFrame({"source_anchor": old, "pairwise_anchor": center + scale * ordinal.x,
                               "difference_anchor": center + scale * delta}, index=old.index)
    diagnostics = {
        "drugs": len(candidates), "pairs": len(pairs), "iterations": int(ordinal.nit),
        "pair_observations_by_source": evidence.groupby("source").size().to_dict(),
        "fit_anchor_spearman": {
            name: float(candidates.source_anchor.corr(candidates[name], method="spearman"))
            for name in ("pairwise_anchor", "difference_anchor")
        },
    }
    return candidates, diagnostics


def old_forecasts(root, assay, fold, expected):
    path = root / "forecasts" / f"{assay}-{fold}.parquet"
    table, receipt = verify_forecast(path)
    if receipt["sha256"] != expected[path.stem] or receipt["provenance"]["assay"] != assay:
        raise ValueError("Sealed source predictions failed the previous score receipt")
    return table, receipt["sha256"]


def fit_and_freeze(args):
    previous = args.baseline / "score.json"
    if sha256(previous) != PREVIOUS_SCORE:
        raise ValueError("The fixed source development baseline has changed")
    hashes = json.loads(previous.read_text())["forecast_sha256"]
    train, _, _ = training_inputs(args.bundle)
    fit, _ = split(train, args.fold, args.assay)
    anchors, report = fit_anchors(fit)
    table, digest = old_forecasts(args.baseline, args.assay, args.fold, hashes)
    selected_methods = {BLEND, "drug_ridge", "drug_mean", "cellhit", "v0plus_inductive"}
    old = table[table.method.isin(selected_methods)]
    old_matrix = old.pivot(index=INDEX, columns="method", values="predicted_value")
    if not np.isfinite(old_matrix.to_numpy()).all():
        raise ValueError("Incomplete registered baseline forecasts")
    new = old_matrix.reset_index()
    new = new.join(anchors, on="compound_id", validate="many_to_one")
    if not np.allclose(new.source_anchor, new.drug_mean, rtol=1e-8, atol=1e-9):
        raise ValueError("Fit anchor does not match the sealed baseline")
    residual = new[BLEND] - new.source_anchor
    variants = {
        "tempered_blend": new.source_anchor + TEMPERATURE * residual,
        "pairwise_blend": new.pairwise_anchor + residual,
        PRIMARY: new.pairwise_anchor + TEMPERATURE * residual,
        "pairwise_mean": new.pairwise_anchor,
        "difference_tempered": new.difference_anchor + TEMPERATURE * residual,
    }
    parts = [new[INDEX].assign(method=name, predicted_value=value) for name, value in variants.items()]
    forecast = pd.concat([old, *parts], ignore_index=True)
    if not np.isfinite(forecast.predicted_value).all():
        raise ValueError("A registered drug anchor is missing")
    freeze_table(forecast, args.out, provenance={
        "assay": args.assay, "fold": args.fold, "baseline_forecast_sha256": digest,
        "train_receipt_sha256": sha256(args.bundle / "receipt.json"), "anchor_diagnostics": report,
    })


def score_arm(frame, assay):
    families = [PRIMARY, "tempered_blend", "pairwise_blend", "pairwise_mean", "difference_tempered"]
    comparisons = [(name, base) for name in families for base in ("drug_mean", BLEND)]
    cell, _ = score(frame, endpoint_direction="higher", rank_only=True, comparisons=comparisons)
    per_drug, _ = score_across_cells(frame)
    tables = (pd.DataFrame(cell["scores"]).merge(
        per_drug.groupby(["track", "method"], as_index=False)[["spearman", "pearson"]].mean()
        .rename(columns={"spearman": "within_drug_spearman", "pearson": "within_drug_pearson"}),
        on=["track", "method"], validate="one_to_one",
    ))
    keys = ["within_drug_spearman", "within_drug_pearson"]
    metrics = [*keys, "macro_spearman", "macro_ndcg_at_5", "macro_top5_overlap"]
    reference = tables[tables.method.eq(BLEND)].set_index("track")[keys].sort_index()
    invariance = {}
    for name in (PRIMARY, "tempered_blend", "pairwise_blend", "difference_tempered"):
        actual = tables[tables.method.eq(name)].set_index("track")[keys].sort_index()
        invariance[name] = float(np.abs(reference - actual).to_numpy().max())
    if max(invariance.values()) > 1e-6:
        raise ValueError("Calibration changed the within-drug associations")
    critical = ("ndcg_at_5", "spearman")
    pri = [row for row in cell["paired_contrasts"] if row["candidate"] == PRIMARY
           and row["metric"] in critical]
    baseline = "drug_mean" if assay == "PRISM20Q2" else BLEND
    endpoints = [row["ci95"][0] for row in pri if row["baseline"] == baseline]
    floor = 0.0 if assay == "PRISM20Q2" else -0.005
    return {
        "table": tables.groupby("method", as_index=False)[metrics].mean().to_dict("records"),
        "by_source": tables.to_dict("records"), "contrasts": cell["paired_contrasts"],
        "within_drug_max_absolute_change": invariance,
        "registered_anchor_gate": len(endpoints) == 2 * frame.track.nunique() and min(endpoints) > floor,
    }


def score_frozen(args):
    train, _, _ = training_inputs(args.bundle)
    modes, hashes, diagnostics = {}, {}, {}
    for assay in ASSAYS:
        folds = []
        diagnostics[assay] = []
        for fold in range(3):
            path = args.forecasts / f"{assay}-{fold}.parquet"
            table, receipt = verify_forecast(path)
            meta = receipt["provenance"]
            if meta["assay"] != assay or meta["fold"] != fold or (
                meta["train_receipt_sha256"] != sha256(args.bundle / "receipt.json")
            ):
                raise ValueError("Wrong source or outer fold in calibration freeze")
            _, check = split(train, fold, assay)
            expected = observations(check).set_index([*KEY, "track"]).index
            for _, part in table.groupby("method"):
                actual = part.set_index([*KEY, "track"]).index
                if len(actual) != len(expected) or set(actual) != set(expected):
                    raise ValueError("Mismatch in registered source score support")
            folds.append(attach(table, check))
            hashes[path.stem] = receipt["sha256"]
            diagnostics[assay].append(meta["anchor_diagnostics"])
        modes[assay] = score_arm(pd.concat(folds, ignore_index=True), assay)
    publish_json(args.out, {"results": modes, "fit_anchor_support": diagnostics, "forecast_sha256": hashes})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    fit = sub.add_parser("predict")
    for option in ("bundle", "baseline", "out"):
        fit.add_argument("--" + option, required=True, type=Path)
    fit.add_argument("--assay", required=True, choices=ASSAYS)
    fit.add_argument("--fold", required=True, type=int, choices=range(3))
    score_command = sub.add_parser("score")
    for option in ("bundle", "forecasts", "out"):
        score_command.add_argument("--" + option, required=True, type=Path)
    args = parser.parse_args()
    {"predict": fit_and_freeze, "score": score_frozen}[args.action](args)


if __name__ == "__main__":
    main()
