"""Verify frozen ICI predictions and score untouched HCC response labels."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .cohorts import record
from .reference import checked


METRICS = ("auroc", "auprc", "balanced_accuracy")
BASELINES = ("mean", "cd274", "ifng", "ifng6", "ayers_gep18", "nnls_purity",
             "tide_author", "ridge", "elasticnet", "lightgbm", "fraction_path_ridge")


def metric_arrays(y, scores, predicted):
    # Each row is one stratified bootstrap. Average precision uses grouped
    # thresholds so a constant baseline has exactly the observed prevalence.
    pos = int(y.sum())
    ranks = rankdata(scores, method="average", axis=-1)
    auc = (ranks[:, y == 1].sum(axis=1) - pos * (pos + 1) / 2) / (pos * (len(y) - pos))
    order = np.argsort(-scores, kind="stable", axis=1)
    sorted_scores = np.take_along_axis(scores, order, axis=1)
    sorted_y = y[order]
    boundary = np.column_stack([sorted_scores[:, :-1] != sorted_scores[:, 1:], np.ones(len(scores), dtype=bool)])
    tp = np.cumsum(sorted_y, axis=1)
    unique_tp = tp * boundary
    previous = np.column_stack([np.zeros(len(scores)), np.maximum.accumulate(unique_tp, axis=1)[:, :-1]])
    ap = ((unique_tp - previous) * boundary * tp / np.arange(1, len(y) + 1)).sum(axis=1) / pos
    acc = .5 * (predicted[:, y == 1].mean(axis=1) + (1 - predicted[:, y == 0]).mean(axis=1))
    return np.stack([auc, ap, acc], axis=-1)


def bootstrap(outcomes, models, n, rng):
    positive = np.flatnonzero(outcomes == 1)
    negative = np.flatnonzero(outcomes == 0)
    index = np.r_[positive, negative]
    draws = np.column_stack([
        rng.choice(positive, size=(n, len(positive)), replace=True),
        rng.choice(negative, size=(n, len(negative)), replace=True),
    ])
    y = outcomes[index]
    results, sampled = [], {}
    for name, group in models.items():
        score = group.score.to_numpy()
        pred = group.classification.to_numpy()
        point = metric_arrays(y, score[index][None, :], pred[index][None, :])[0]
        boot = metric_arrays(y, score[draws], pred[draws])
        sampled[name] = boot
        row = {"model": name, "participants": len(y), "responders": int(y.sum())}
        for m, metric in enumerate(METRICS):
            low, high = np.quantile(boot[:, m], [.025, .975])
            row.update({metric: point[m], metric + "_low": low, metric + "_high": high})
        results.append(row)
    best = next(row for row in results if row["model"] == "tme_gnn")
    baseline_results = []
    for row in results:
        name = row["model"]
        if name == "tme_gnn":
            continue
        draws = sampled["tme_gnn"] - sampled[name]
        record_out = {"comparison": name, "comparison_type": "baseline" if name in BASELINES else "ablation"}
        for m, metric in enumerate(METRICS):
            low, high = np.quantile(draws[:, m], [.025, .975])
            record_out.update({metric: best[metric] - row[metric],
                               metric + "_low": low, metric + "_high": high})
        baseline_results.append(record_out)
    return results, baseline_results


def fractions_summary(fractions, cohort):
    moments = []
    for name in fractions:
        values = fractions[name]
        for quantile, value in values.quantile([.10, .25, .50, .75, .90]).items():
            moments.append({"cohort": cohort, "cell_type": name, "quantile": quantile, "fraction": value})
    return moments


def counterfactual_summary(pred, table, truth, draws, rng):
    targets = table.merge(truth, on="id", validate="many_to_one").merge(
        pred.loc[pred.model.eq("tme_gnn"), ["id", "classification"]], on="id", validate="many_to_one"
    )
    masks = {"all": np.ones(len(targets), dtype=bool),
             "model_predicted_cold": targets.classification.eq(0),
             "observed_nonresponders": targets.outcome.eq(0)}
    results = []
    for role, mask in masks.items():
        subset = targets.loc[mask]
        for (axis, attenuation), group in subset.groupby(["axis", "attenuation"]):
            data = group.delta_probability.to_numpy()
            boot = rng.choice(data, size=(draws, len(data)), replace=True).mean(axis=1)
            low, high = np.quantile(boot, [.025, .975])
            results.append({"group": role, "axis": axis, "attenuation": attenuation,
                            "participants": len(group), "delta_probability": data.mean(),
                            "delta_low": low, "delta_high": high,
                            "mean_between_seed_sd": group.perturbed_ensemble_sd.mean()})
    return results


def markdown(data, columns):
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in data[columns].itertuples(index=False, name=None):
        rows.append("| " + " | ".join(f"{a:.4f}" if isinstance(a, (float, np.floating))
                                    else str(a) for a in row) + " |")
    return "\n".join(rows)


def score(a):
    registry_path = a.cohorts / "receipt.json"
    receipt_path = a.predictions / "receipt.json"
    registry = json.loads(registry_path.read_text())
    frozen = json.loads(receipt_path.read_text())
    if record(registry_path) != frozen["inputs"]["registry"]:
        raise ValueError("Predictions were frozen against a different external key registry")
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    performance, contrasts, fractions, perturbations, records = [], [], [], [], {}
    for cohort, spec in sorted(frozen["targets"].items()):
        base = a.predictions / cohort
        pred = pd.read_parquet(checked(base, "predictions.parquet", spec))
        frac = pd.read_parquet(checked(base, "fractions.parquet", spec))
        for p in spec["files"]:
            checked(base, p, spec)
        if "outcome" in pred or pred.duplicated(["id", "model"]).any():
            raise ValueError(f"Invalid frozen predictions for {cohort}")
        fractions.extend(fractions_summary(frac, cohort))
        label = f"external/{cohort}/labels.parquet"
        if label not in registry["files"]:
            records[cohort] = {"scored_response_patients": 0, "exploratory_profiles": len(frac)}
            continue
        labels_path = checked(a.cohorts, label, registry)
        truth = pd.read_parquet(labels_path).sort_values("id")
        if not truth.id.is_unique or set(truth.outcome.unique()) != {0, 1}:
            raise ValueError(f"Unexpected held response registry for {cohort}")
        models = {}
        ids = truth.id.to_numpy()
        for name, group in pred.groupby("model", sort=True):
            model = group.sort_values("id")
            if not np.array_equal(ids, model.id.to_numpy()):
                raise ValueError(f"Outcomes and frozen predictions disagree for {cohort}/{name}")
            models[name] = model
        results, comparison = bootstrap(truth.outcome.to_numpy(), models, a.bootstrap, rng)
        performance += [dict(cohort=cohort, **m) for m in results]
        contrasts += [dict(cohort=cohort, **m) for m in comparison]
        records[cohort] = {"labels": record(labels_path), "scored_response_patients": len(truth)}
        if "counterfactuals.parquet" in spec["files"]:
            cf = pd.read_parquet(checked(base, "counterfactuals.parquet", spec))
            if any(set(g.id) != set(ids) or g.id.duplicated().any()
                   for _, g in cf.groupby(["axis", "attenuation"])):
                raise ValueError("Counterfactual and untouched response IDs do not align")
            perturbations += counterfactual_summary(pred, cf, truth, a.bootstrap, rng)
    scores, paired = pd.DataFrame(performance), pd.DataFrame(contrasts)
    simulations = pd.DataFrame(perturbations)
    with tempfile.TemporaryDirectory(prefix=f".{a.output.name}.", dir=a.output.parent) as d:
        out = Path(d)
        for name, table in (("external_metrics", scores), ("paired_deltas", paired),
                            ("fraction_summary", pd.DataFrame(fractions)), ("counterfactuals", simulations)):
            table.to_csv(out / (name + ".tsv"), sep="\t", index=False)
        sections = [
            "# Track 7 — HCC response to immune checkpoint blockade",
            "External predictions were frozen before outcome scoring. Classification thresholds came from "
            "leave-one-cohort-out non-HCC predictions. Confidence intervals use stratified patient bootstrap.",
        ]
        for cohort, group in (scores.groupby("cohort") if len(scores) else ()):
            columns = ["model", "participants", "responders"]
            for value in METRICS:
                group = group.assign(**{
                    value + " (95% CI)": [
                        f"{r[0]:.3f} ({r[1]:.3f}–{r[2]:.3f})"
                        for r in group[[value, value + "_low", value + "_high"]].to_numpy()
                    ],
                })
                columns.append(value + " (95% CI)")
            sections.extend(["## " + cohort, markdown(group, columns)])
        if not simulations.empty:
            sub = simulations.loc[simulations.group.eq("observed_nonresponders") & simulations.attenuation.eq(.90)]
            sub = sub.sort_values("delta_probability", ascending=False)
            sections.extend([
                "## Graph perturbation hypotheses",
                "These are simulated changes in graph-model probabilities among observed nonresponders. "
                "They do not measure an effect of treatment. The two attenuation levels, all-patient results "
                "and the model-defined cold subgroup are also available in the TSV.",
                markdown(sub, ["axis", "participants", "delta_probability", "delta_low", "delta_high",
                               "mean_between_seed_sd"]),
            ])
        (out / "RESULTS.md").write_text("\n\n".join(sections) + "\n")
        (out / "receipt.json").write_text(json.dumps({
            "predictions": record(receipt_path), "cohorts": records, "bootstrap": a.bootstrap,
            "seed": a.seed, "outputs": {path.name: record(path) for path in sorted(out.iterdir())},
        }, indent=2, sort_keys=True) + "\n")
        out.rename(a.output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("cohorts", "predictions", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=4000)
    p.add_argument("--seed", type=int, default=30213)
    a = p.parse_args()
    if a.bootstrap < 100:
        p.error("At least 100 draws are required to compute bootstrap intervals")
    score(a)


if __name__ == "__main__":
    main()
