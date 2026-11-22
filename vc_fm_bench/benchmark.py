"""Fit matched heads, freeze predictions and score in a separate invocation."""
import argparse
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge, RidgeClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

from .artifacts import file_record, new_directory, verify_record, write_json
from .features import rank_block, read_vectors
from .metrics import classification_metrics, response_metrics


def panel_input(path, role, receipt):
    file = path / f"{role}.parquet"
    verify_record(file, receipt["files"][file.name])
    return pd.read_parquet(file)


def by_source_scale(training):
    # AUC response may vary by assay and readout. Equalize scales using only
    # the current fit subset, preserving drug-level and cell-level variation.
    y = training.outcome.copy()
    if "source_id" in training:
        by_source = training.groupby("source_id").outcome
        center = training.source_id.map(by_source.mean())
        width = training.source_id.map(by_source.std()).clip(lower=0.05)
        y = (y - center) / width
    return y.to_numpy()


def choose_validation(train, group_key):
    key = train[group_key].astype(str)
    groups = sorted(key.unique(), key=lambda s: hashlib.sha256(f"train-val-2801:{s}".encode()).digest())
    for phase in range(5):
        validation = key.isin(groups[phase::5])
        others = ~validation
        if validation.any() and others.any() and train.loc[others, "outcome"].nunique() > 1:
            if train.loc[validation, "outcome"].nunique() > 1:
                return train.loc[others].copy(), train.loc[validation].copy()
    raise ValueError("Could not construct grouped validation")


def candidates(classify, seed):
    from lightgbm import LGBMClassifier, LGBMRegressor

    if classify:
        ridge = [RidgeClassifier(alpha=a) for a in (1, 20, 100)]
        elastic = [LogisticRegression(C=c, penalty="elasticnet", l1_ratio=0.25, solver="saga",
                                      max_iter=4500, random_state=seed) for c in (0.01, 0.1)]
        tree = LGBMClassifier
        mlp = MLPClassifier
    else:
        ridge = [Ridge(alpha=a) for a in (20, 200, 2000)]
        elastic = [ElasticNet(alpha=a, l1_ratio=0.25, max_iter=3000)
                   for a in (0.01, 0.05)]
        tree = LGBMRegressor
        mlp = MLPRegressor
    return {
        "ridge": ridge, "elasticnet": elastic,
        "lightgbm": [tree(n_estimators=n, num_leaves=leaves, max_bin=63, min_child_samples=12,
                          learning_rate=0.05, random_state=seed, n_jobs=4, verbosity=-1)
                     for n, leaves in ((140, 15), (260, 31))],
        "mlp": [mlp(hidden_layer_sizes=(128, 64), max_iter=24, batch_size=256,
                    learning_rate_init=0.001, alpha=alpha, random_state=seed)
                for alpha in (0.001, 0.03)],
    }


def fit_head(model, fit, eval_x, y, family):
    if family != "lightgbm":
        scaler = StandardScaler().fit(fit)
        fit, eval_x = scaler.transform(fit), scaler.transform(eval_x)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(fit, y)
    if hasattr(model, "predict_proba"):
        output = model.predict_proba(eval_x)[:, 1]
    elif hasattr(model, "decision_function"):
        output = model.decision_function(eval_x)
    else:
        output = model.predict(eval_x)
    if not np.isfinite(output).all():
        raise ValueError(f"Nonfinite {family} predictions")
    notes = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    return output, notes


def validation_score(y, pred, rows, classify, panel_name):
    if classify:
        return classification_metrics(y, pred)["auroc"]
    measures = response_metrics(rows.assign(p=pred), "p")
    target = "within_drug_spearman" if "cell-cold" in panel_name else "cell_macro_ndcg5"
    value = measures[target]
    if value is None:
        raise ValueError(f"Validation group too sparse for {target}")
    return value


def matrix(rows, drug_frames, cell_frame, cell_fit=None):
    x = [f.loc[rows.drug_id].to_numpy(dtype=np.float32) for f in drug_frames]
    if cell_frame is not None:
        x.append(cell_frame.loc[rows.cell_id].to_numpy(dtype=np.float32))
    if cell_fit is not None:
        x.append(cell_fit.loc[rows.cell_id].to_numpy(dtype=np.float32))
    if not x:
        raise ValueError("No selected features")
    return np.concatenate(x, axis=1)


def load_profiles(files):
    tables = [pd.read_parquet(f) for f in files]
    if any(not f.columns.equals(tables[0].columns) for f in tables):
        raise ValueError("RNA axes differ")
    result = pd.concat(tables)
    if not result.index.is_unique or not result.columns.is_unique:
        raise ValueError("Duplicate RNA row or feature identity")
    return result


def per_drug_ridge(training, queries, train_y, cell_features, alpha):
    fit = matrix(training, (), cell_features)
    held_out = matrix(queries, (), cell_features)
    scaler = StandardScaler().fit(fit)
    fit, held_out = scaler.transform(fit), scaler.transform(held_out)
    prediction = np.full(len(queries), np.mean(train_y))
    domains = training.groupby("drug_id").indices
    for key, rows in queries.groupby("drug_id").indices.items():
        if key not in domains:
            continue
        seen = domains[key]
        if len(seen) < 12:
            prediction[rows] = train_y[seen].mean()
            continue
        model = Ridge(alpha=alpha).fit(fit[seen], train_y[seen])
        prediction[rows] = model.predict(held_out[rows])
    return prediction


def nearest_tanimoto(train, query, ecfp):
    x = ecfp.loc[sorted(set(train.drug_id))].to_numpy(dtype=np.float32)[:, :2048]
    y = ecfp.loc[sorted(set(query.drug_id))].to_numpy(dtype=np.float32)[:, :2048]
    intersection = y @ x.T
    denom = np.maximum(y.sum(1)[:, None] + x.sum(1)[None, :] - intersection, 1)
    scores = (intersection / denom).max(1)
    return dict(zip(("min", "median", "q90", "max"),
                    map(float, np.quantile(scores, [0, 0.5, 0.9, 1]))))


def predict(args):
    contract = json.loads((args.panel / "receipt.json").read_text())
    train, query = [panel_input(args.panel, role, contract) for role in ("train", "query")]
    if "outcome" in query or train.id.duplicated().any() or query.id.duplicated().any():
        raise ValueError("Invalid fit or query interface")
    classify = contract["task"] == "classification"
    if not classify and not args.rna:
        raise ValueError("Provide offline RNA files for response modelling")
    drug = {key: read_vectors(args.chem / key) for key in args.chem_models}
    if "ecfp4" not in drug:
        raise ValueError("ECFP4 is required for common support and the structure audit")
    covered = set.intersection(*(set(d.index) for d in drug.values()))
    original_query_rows = len(query)
    train = train.loc[train.drug_id.isin(covered)].copy()
    query = query.loc[query.drug_id.isin(covered)].copy()
    cells, rna = {}, None
    if not classify:
        rna = load_profiles(args.rna)
        cells = {key: read_vectors(args.cell / key) for key in args.cell_models}
        covered_rna = set(rna.index)
        for frame in cells.values():
            covered_rna &= set(frame.index)
        train = train.loc[train.cell_id.isin(covered_rna)].copy()
        query = query.loc[query.cell_id.isin(covered_rna)].copy()
    if not len(query) or not len(train):
        raise ValueError("No common support")

    inner, valid = choose_validation(train, contract["split_group"])
    inner_y, outer_y = by_source_scale(inner), by_source_scale(train)
    prior = float(outer_y.mean())
    predictions = query.copy()
    predictions["mean"] = prior
    if classify:
        desc = drug["ecfp4"].loc[query.drug_id].to_numpy()
        predictions["clogp"] = desc[:, 2049]
        predictions["mw_logp_rule"] = ((desc[:, 2048] >= 400) & (desc[:, 2049] >= 3)).astype(float)
    else:
        mean_by_drug = train.assign(scaled=outer_y).groupby("drug_id").scaled.mean()
        predictions["drug_mean"] = query.drug_id.map(mean_by_drug).fillna(prior).to_numpy()
        predictions["abcb1"] = -rna.loc[query.cell_id, "ABCB1"].to_numpy()
        predictions["mki67"] = rna.loc[query.cell_id, "MKI67"].to_numpy()

    cell_groups = {key: (value, value) for key, value in cells.items()}
    if not classify:
        for mode in ("rank", "paths"):
            cell_groups[mode] = tuple(rank_block(rna, subset.cell_id, mode, args.weights, args.hallmark)
                                      for subset in (inner, train))
    reference_cell = None if classify else "rank"
    variants = [(key, reference_cell, (key,)) for key in drug]
    variants += [(f"ecfp4+{key}", key, ("ecfp4",)) for key in cell_groups if key != reference_cell]
    variants += [(f"ecfp4+{key}", reference_cell, ("ecfp4", key)) for key in drug if key != "ecfp4"]
    if cells:
        variants.extend((f"ecfp4+rank+{key}", key, ("ecfp4",)) for key in cells)
    selected = {}
    if not classify and "cell-cold" in args.panel.name:
        inner_rna, outer_rna = cell_groups["rank"]
        controls = []
        for alpha in (50, 250, 1250):
            forecast = per_drug_ridge(inner, valid, inner_y, inner_rna, alpha)
            validation = validation_score(valid.outcome.to_numpy(), forecast, valid, False, args.panel.name)
            controls.append({"alpha": alpha, "score": validation})
        best = max(controls, key=lambda row: row["score"])
        predictions["per_drug_ridge"] = per_drug_ridge(train, query, outer_y, outer_rna, best["alpha"])
        selected["per_drug_ridge"] = {"alpha": best["alpha"], "validation": controls}
    for label, cell_name, drug_names in variants:
        molecular = [drug[name] for name in drug_names]
        inner_cell, outer_cell = cell_groups.get(cell_name, (None, None))
        inner_rank, outer_rank = cell_groups["rank"] if "+rank+" in label else (None, None)
        a, b = [matrix(rows, molecular, inner_cell, inner_rank) for rows in (inner, valid)]
        x, q = [matrix(rows, molecular, outer_cell, outer_rank) for rows in (train, query)]
        for family, models in candidates(classify, args.seed).items():
            results = []
            for i, model in enumerate(models):
                pred, notes = fit_head(model, a, b, inner_y, family)
                score = validation_score(valid.outcome.to_numpy(), pred, valid, classify, args.panel.name)
                results.append({"index": i, "score": score, "warnings": notes})
            winner = max(results, key=lambda h: h["score"])
            model = candidates(classify, args.seed)[family][winner["index"]]
            key = f"{label}/{family}"
            predictions[key], notes = fit_head(model, x, q, outer_y, family)
            selected[key] = {"candidate": winner["index"], "validation": results,
                             "refit_warnings": notes}
            print(f"{args.panel.name} {key} val={winner['score']:.5f}", flush=True)
    with new_directory(args.output) as out:
        path = out / "predictions.parquet"
        predictions.to_parquet(path, index=False)
        write_json(out / "receipt.json", {
            "predictions": file_record(path), "panel": file_record(args.panel / "receipt.json"),
            "task": contract["task"], "selection": selected,
            "columns": sorted(set(predictions) - set(query)),
            "feature_rows_original": original_query_rows, "predictions_rows": len(query),
            "train_rows": len(train), "validation_rows": len(valid),
            "nearest_training_tanimoto": nearest_tanimoto(train, query, drug["ecfp4"]),
            "fit_source_roles": {str(k): int(v) for k, v in train.source_id.value_counts().items()} if "source_id" in train else {},
            "features": {"rna": [file_record(f) for f in args.rna] if args.rna else [],
                         "chemistry": {m: file_record(args.chem / m / "receipt.json") for m in drug},
                         "cell": {m: file_record(args.cell / m / "receipt.json") for m in cells},
                         "pathway_weights": file_record(args.weights) if not classify else {},
                         "hallmark": file_record(args.hallmark) if not classify else {}},
        })


def score(args):
    directory = args.predictions
    receipt_path = directory / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    verify_record(args.panel / "receipt.json", receipt["panel"])
    contract = json.loads((args.panel / "receipt.json").read_text())
    path = directory / "predictions.parquet"
    verify_record(path, receipt["predictions"])
    predictions = pd.read_parquet(path)
    if args.external:
        prepared = json.loads((args.prepared / "receipt.json").read_text())
        verify_record(args.external, prepared["cell_cold_outcomes"])
        key_name = "limore_cell_cold_keys.parquet"
        verify_record(args.prepared / key_name, prepared["files"][key_name])
        keys = pd.read_parquet(args.prepared / key_name)
        truth = pd.read_parquet(args.external)
        labels = keys.merge(truth, on="record_id", validate="one_to_one")
        labels = labels[["id", "endpoint_value"]].rename(columns={"endpoint_value": "outcome"})
        label_receipt = file_record(args.external)
    else:
        labels = panel_input(args.panel, "outcomes", contract)
        label_receipt = file_record(args.panel / "outcomes.parquet")
    joined = predictions.merge(labels, on="id", validate="one_to_one")
    if len(joined) != len(predictions) or not np.isfinite(joined.outcome).all():
        raise ValueError("Scoring labels missing or mismatched")
    metrics = {}
    for key in receipt["columns"]:
        if contract["task"] == "classification":
            value = classification_metrics(joined.outcome.to_numpy(), joined[key].to_numpy())
        else:
            value = response_metrics(joined, key)
        metrics[key] = value
    with new_directory(args.output) as out:
        write_json(out / "metrics.json", metrics)
        write_json(out / "receipt.json", {
            "frozen_prediction_receipt": file_record(receipt_path),
            "frozen_predictions": file_record(path), "labels": label_receipt,
            "scored_rows": len(joined), "metrics": file_record(out / "metrics.json"),
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("predict")
    for flag in ("panel", "chem", "output"):
        fit.add_argument(f"--{flag}", type=Path, required=True)
    fit.add_argument("--chem-models", nargs="+", default=["ecfp4", "chemberta2", "molformer", "unimol"])
    fit.add_argument("--cell", type=Path)
    fit.add_argument("--cell-models", nargs="*", default=["scgpt", "geneformer"])
    fit.add_argument("--rna", type=Path, nargs="+")
    fit.add_argument("--weights", type=Path)
    fit.add_argument("--hallmark", type=Path)
    fit.add_argument("--seed", type=int, default=20261122)
    evaluation = commands.add_parser("score")
    for flag in ("panel", "predictions", "output"):
        evaluation.add_argument(f"--{flag}", type=Path, required=True)
    evaluation.add_argument("--external", type=Path)
    evaluation.add_argument("--prepared", type=Path)
    args = parser.parse_args()
    if args.command == "predict":
        predict(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
