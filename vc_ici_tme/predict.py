"""Fit across public non-HCC ICI cohorts and freeze blinded HCC predictions."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
import torch

from .cohorts import record
from .features import EDGE_CHANNELS, construct, published_markers, tide_assets, tide_fit_center
from .models import CANDIDATES, classical_fit, graph_fit, sample_weights
from .reference import TYPES, checked


SOURCE_COHORTS = ("IMvigor210", "Gide", "Hugo", "Riaz")
EXTERNAL_COHORTS = ("GSE202069", "GSE140901", "GSE141198")
PERTURBATION_LEVELS = (.50, .90)


def assay_tables(root, receipt, scope, name):
    prefix = f"{scope}/{name}/"
    expr = pd.read_parquet(checked(root, prefix + "expression.parquet", receipt))
    keys_key = prefix + "keys.parquet"
    if keys_key in receipt["files"]:
        keys = pd.read_parquet(checked(root, keys_key, receipt))
    else:
        keys = pd.DataFrame({"id": expr.index, "donor": expr.index, "cohort": name})
    if list(keys.columns) != ["id", "donor", "cohort"] or not keys.id.is_unique or not keys.donor.is_unique:
        raise ValueError(f"Duplicate IDs or endpoint data in {name} assay keys")
    if set(expr.index) != set(keys.id) or keys.cohort.unique().tolist() != [name]:
        raise ValueError(f"Expression and registered donor keys differ for {name}")
    return expr.loc[keys.id], keys


def source_tables(root, receipt):
    expressions, cohorts = [], []
    for name in SOURCE_COHORTS:
        expr, keys = assay_tables(root, receipt, "sources", name)
        labels = pd.read_parquet(checked(root, f"sources/{name}/labels.parquet", receipt))
        merged = keys.merge(labels, on="id", how="left", validate="one_to_one")
        if set(merged.outcome.unique()) != {0, 1}:
            raise ValueError(f"Invalid pre-registered training endpoint in {name}")
        expressions.append(expr)
        cohorts.append(merged)
    return expressions, pd.concat(cohorts, ignore_index=True)


def raw_tide(expression, fit_rows, query_rows):
    return tide_fit_center(expression.iloc[fit_rows], expression.iloc[query_rows])


def calibration(raw, y, labels):
    # A positive scale preserves the published biomarker orientation.
    # Model thresholds and recalibration use out-of-cohort source scores.
    offset, spread = float(np.mean(raw)), max(float(np.std(raw)), 1e-5)
    standardized = (raw - offset) / spread
    weights = sample_weights(labels)
    weights /= weights.sum()

    def loss(coefs):
        a, b = coefs
        logits = a * standardized + b
        objective = np.dot(weights, np.logaddexp(0, logits) - y * logits) + .01 * a ** 2
        error = weights * (expit(logits) - y)
        derivative = (np.dot(error, standardized) + .02 * a, error.sum())
        return objective, np.array(derivative)

    optimization = minimize(loss, [.1, 0.], jac=True, bounds=[(.001, 10.), (-8., 8.)])
    if not optimization.success:
        raise RuntimeError(f"Source calibration failed: {optimization.message}")
    fpr, tpr, thresholds = roc_curve(y, raw, sample_weight=weights)
    threshold = thresholds[np.argmax((tpr - fpr)[1:]) + 1]
    a, b = map(float, optimization.x)
    return {"center": offset, "scale": spread, "slope": a, "intercept": b, "threshold": float(threshold)}


def probability(output, fit):
    return expit(fit["slope"] * (output - fit["center"]) / fit["scale"] + fit["intercept"])


def evaluate(v, y, cohort):
    rows = []
    for name in SOURCE_COHORTS:
        mask = cohort == name
        rows.append({"cohort": name, "auroc": float(roc_auc_score(y[mask], v[mask])),
                     "auprc": float(average_precision_score(y[mask], v[mask])), "n": int(mask.sum())})
    return float(np.mean([r["auroc"] for r in rows])), rows


def graph_arrays(name, profiles):
    if name == "tme_no_fractions":
        return profiles.unweighted_nodes, profiles.unweighted_edges
    return profiles.nodes, profiles.edges


def build_model(name, config, profiles, train, y, cohort, seed, device):
    if name.startswith("tme_"):
        node, edge = graph_arrays(name, profiles)
        return graph_fit(name, config, node[train], edge[train], y[train], cohort[train], seed, device)
    return classical_fit(name, config, profiles, train, y[train], cohort[train], seed)


def use_model(name, model, profiles, query):
    if name.startswith("tme_"):
        x, e = graph_arrays(name, profiles)
        return model.predict(x[query], e[query])
    return model.predict(profiles, query)


def learn(name, profiles, y, cohort, folds, query, a):
    candidates = []
    for config in CANDIDATES[name]:
        oof = np.full(len(y), np.nan)
        for i, (fit, left) in enumerate(folds):
            model = build_model(name, config, profiles, fit, y, cohort, a.seed + i, a.device)
            oof[left] = use_model(name, model, profiles, left)
        score, measures = evaluate(oof, y, cohort)
        candidates.append((score, config, oof, measures))
    score, config, oof, _ = max(candidates, key=lambda candidate: candidate[0])
    n_seeds = a.ensemble if name == "tme_gnn" else 1
    final = [
        build_model(name, config, profiles, np.arange(len(y)), y, cohort, a.seed + i, a.device)
        for i in range(n_seeds)
    ]
    pred = np.stack([use_model(name, model, profiles, query) for model in final]).mean(axis=0)
    print(f"{name} LOCO mean AUROC={score:.5f} config={config}", flush=True)
    return pred, oof, {
        "config": config, "loco_mean_auroc": score,
        "candidates": [{"config": h, "loco_mean_auroc": v, "per_cohort": m} for v, h, _, m in candidates],
        "ensemble": n_seeds,
    }, final


def counterfactuals(profiles, query, names, models, calibrator, base):
    payload = []
    for axis in EDGE_CHANNELS[1:]:
        if not (profiles.audit["ligand_receptor_pairs"][axis] or
                (axis == "WNT" and profiles.wnt_path_node is not None)):
            continue
        for level in PERTURBATION_LEVELS:
            node, edge = profiles.counterfactual(axis, level)
            partial = np.stack([model.predict(node[query], edge[query]) for model in models])
            effect = probability(partial.mean(axis=0), calibrator) - base
            uncertainty = probability(partial, calibrator).std(axis=0)
            payload.append(pd.DataFrame({
                "id": names, "axis": axis, "attenuation": level, "delta_probability": effect,
                "perturbed_ensemble_sd": uncertainty,
            }))
    return pd.concat(payload, ignore_index=True)


def target_predictions(a, root_receipt, source, source_markers, source_keys, target, directory):
    query, keys = assay_tables(a.cohorts, root_receipt, "external", target)
    if set(source_keys.donor) & set(keys.donor) or set(source.index) & set(query.index):
        raise ValueError(f"External patient would be present in fit for {target}")
    axis = source.columns.intersection(query.columns).sort_values()
    restricted = target == "GSE140901"
    minimum = 400 if restricted else 9000
    if len(axis) < minimum:
        raise ValueError(f"Only {len(axis)} genes shared with {target}. Expected at least {minimum}.")
    expression = pd.concat([source[axis], query[axis]])
    profiles = construct(expression, a.reference, a.hallmark, restricted)
    print(target, "features", json.dumps(profiles.audit, sort_keys=True), flush=True)
    cohort = source_keys.cohort.to_numpy()
    y = source_keys.outcome.to_numpy(np.int8)
    query_rows = np.arange(len(y), len(expression))
    folds = [(np.flatnonzero(cohort != c), np.flatnonzero(cohort == c)) for c in SOURCE_COHORTS]
    mean_oof = np.empty(len(y))
    for fit, left in folds:
        mean_oof[left] = np.average(y[fit], weights=sample_weights(cohort[fit]))
    raw = {"mean": (np.full(len(query), np.average(y, weights=sample_weights(cohort))), mean_oof)}
    assays, audit = published_markers(query)
    profiles.audit["biomarker_support"] = audit
    purity = -profiles.fractions[:, TYPES.index("malignant")]
    raw["nnls_purity"] = (purity[query_rows], purity[:len(y)])
    for name in assays:
        raw[name] = (assays[name].to_numpy(), source_markers[name].to_numpy())
    if not restricted:
        tide_oof = np.empty(len(y))
        for fit, left in folds:
            tide_oof[left] = raw_tide(expression, fit, left)
        raw["tide_author"] = (raw_tide(expression, np.arange(len(y)), query_rows), tide_oof)
    config, graph = {}, None
    for name in CANDIDATES:
        pred, oof, config[name], models = learn(name, profiles, y, cohort, folds, query_rows, a)
        raw[name] = (pred, oof)
        if name == "tme_gnn":
            graph = models
    if not restricted and set(["cd274", "ifng6", "ayers_gep18", "tide_author"]) - set(raw):
        raise ValueError(f"Incomplete required biomarker comparison for {target}")
    calibrators = {}
    outputs, out_of_fold = [], []
    for name, (forecasts, oof) in raw.items():
        if not np.isfinite(forecasts).all():
            raise ValueError(f"Nonfinite frozen score for {target}/{name}")
        usable = np.isfinite(oof)
        if y[usable].min() == y[usable].max():
            raise ValueError(f"Incomplete source biomarker reference for {name}")
        cal = calibration(oof[usable], y[usable], cohort[usable])
        cal["source_calibration_patients"] = int(usable.sum())
        if name == "mean":
            cal["threshold"] = float(np.nextafter(max(forecasts.max(), oof.max()), np.inf))
        calibrators[name] = cal
        outputs.append(pd.DataFrame({
            "id": keys.id, "model": name, "score": forecasts, "probability": probability(forecasts, cal),
            "classification": (forecasts >= cal["threshold"]).astype(np.int8),
        }))
        out_of_fold.append(pd.DataFrame({
            "id": source_keys.id, "cohort": cohort, "model": name, "score": oof,
        }))
    results = pd.concat(outputs, ignore_index=True)
    results.to_parquet(directory / "predictions.parquet", index=False)
    pd.concat(out_of_fold, ignore_index=True).to_parquet(directory / "source_loco.parquet", index=False)
    pd.DataFrame(profiles.fractions[query_rows], columns=TYPES, index=keys.id).to_parquet(
        directory / "fractions.parquet"
    )
    if target == "GSE202069":
        base = results.loc[results.model.eq("tme_gnn"), "probability"].to_numpy()
        counterfactuals(profiles, query_rows, keys.id, graph, calibrators["tme_gnn"], base).to_parquet(
            directory / "counterfactuals.parquet", index=False
        )
    for i, model in enumerate(graph):
        torch.save({
            "network": model.network.state_dict(),
            "node_center": torch.from_numpy(model.node_center),
            "node_scale": torch.from_numpy(model.node_scale),
            "edge_scale": torch.from_numpy(model.edge_scale),
        }, directory / f"tme_gnn_seed_{i}.pt")
    return {
        "features": profiles.audit, "source_cohort_counts": {k: int((cohort == k).sum()) for k in SOURCE_COHORTS},
        "model_selection": config, "calibrators": calibrators, "external_rows": len(keys),
        "restricted_assay": restricted, "files": {p.name: record(p) for p in sorted(directory.iterdir())},
    }


def predict(a):
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    registry = json.loads((a.cohorts / "receipt.json").read_text())
    source_profiles, keys = source_tables(a.cohorts, registry)
    source = pd.concat(source_profiles, join="inner")
    source_markers = pd.concat([published_markers(p)[0] for p in source_profiles]).reindex(source.index)
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix=f".{a.output.name}.", dir=a.output.parent) as d:
        stage = Path(d)
        target_audit = {}
        for target in a.target:
            directory = stage / target
            directory.mkdir()
            target_audit[target] = target_predictions(a, registry, source, source_markers, keys, target, directory)
        tide_records = {p.name: record(p) for p in tide_assets()}
        (stage / "receipt.json").write_text(json.dumps({
            "inputs": {"registry": record(a.cohorts / "receipt.json"),
                       "reference": record(a.reference / "receipt.json"), "hallmark": record(a.hallmark),
                       "tidepy": {"version": version("tidepy"), "assets": tide_records}},
            "seed": a.seed, "device": a.device, "fold_design": "leave one non-HCC ICI source cohort out",
            "targets": target_audit,
        }, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stage.rename(a.output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("cohorts", "reference", "hallmark", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--target", choices=EXTERNAL_COHORTS, nargs="+", default=EXTERNAL_COHORTS)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--seed", type=int, default=1927)
    p.add_argument("--ensemble", type=int, default=3)
    a = p.parse_args()
    if a.ensemble < 1 or len(a.target) != len(set(a.target)):
        p.error("Specify a positive ensemble size and unique external targets")
    predict(a)


if __name__ == "__main__":
    main()
