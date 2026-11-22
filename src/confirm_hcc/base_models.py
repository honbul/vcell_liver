"""Corrected fixed baselines, including the upstream CellHit ensemble and Track 6 ElasticNet."""

import argparse
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from vc_build.evaluate import across_cell_metrics
from vc_build.external import KEY, SOURCE, native_sensitivity
from vc_build.io import freeze_table, publish_json, sha256
from vc_build.structured_release import fit_kernel
from vc_build.v0plus_cv import fit_baseline, molecular_inputs, panel_targets, validation_roster
from vc_build.v0plus_features import ExpressionFeatures
from vc_build.v0plus_neural import load_auxiliary
from vc_build.v0plus_sota import FILES, upstream_module
from vc_build.yang_hcc_external import SETTINGS as SETTINGS_HASH

from .contract import BUILD, FM, HALLMARK, NEXTGEN, RUNS, SETTINGS, TARGETS, TRAIN_SHA
from .contract import fold_indices, inputs, read_json, verified

UPSTREAM = BUILD / "v0plus-upstream"
SIGNED = NEXTGEN / "raw/pathways/signed_weights.tsv"
SIGNED_SHA = "11dd1ba95206c94397aaba6ef9ec8c76fc368e8cc65eaa70a2ff79114db21a59"
LINEAR = ("drug_mean", "additive", "drug_ridge")


@dataclass
class LinearPanel:
    models: dict

    def predict(self, query, expression):
        features = self.models["additive"][0].features
        projected = features.transform(expression)
        values = {}
        for name, (method, magnitude) in self.models.items():
            base, residual = method.predict_projected(query, projected)
            values[name] = base + magnitude * residual
        return values


@dataclass
class CellHitFit:
    features: ExpressionFeatures
    anchor: pd.Series
    models: dict
    amplitude: float

    def predict(self, query, expression):
        from xgboost import DMatrix

        x = self.features.transform(expression)
        result = query.compound_id.map(self.anchor).to_numpy()
        for drug, indices in query.groupby("compound_id").indices.items():
            if drug not in self.models:
                raise ValueError(f"CellHit lacks a source fit for {drug}")
            mx = DMatrix(x.loc[query.iloc[indices].cell_id], nthread=int(os.environ.get("OMP_NUM_THREADS", "4")))
            hold = np.stack([m.predict(mx) for m in self.models[drug]])
            result[indices] += self.amplitude * hold.mean(axis=0)
        return {"cellhit": result}


def fit_cellhit(features, fit_rna, source, anchor, setting):
    model_path = "CellHit/CellHit/models/model.py"
    raw = upstream_module(verified(UPSTREAM / model_path, FILES[model_path]), "cellhit_public_model")
    _, depth, _, amplitude = setting
    threads = min(5, int(os.environ.get("OMP_NUM_THREADS", "4")))
    x = features.transform(fit_rna)
    fits = {}
    for (*_, drug), rows in source.groupby([*SOURCE, "compound_id"], sort=True):
        if rows.lineage_group.nunique() < 35:
            continue
        x_drug = x.loc[rows.cell_id]
        y = rows.residual.to_numpy()
        partitions = [{"train_X": x_drug.iloc[f], "train_Y": y[f], "valid_X": x_drug.iloc[v], "valid_Y": y[v]}
                      for f, v in GroupShuffleSplit(n_splits=2, test_size=.2, random_state=811).split(
                          x_drug, groups=rows.lineage_group)]
        params = {"max_depth": int(depth), "eta": .045, "subsample": .8, "colsample_bytree": .7,
                  "n_estimators": 450, "early_stopping_rounds": 30, "reg_lambda": 6,
                  "objective": "reg:squarederror", "tree_method": "hist", "nthread": threads}
        model = raw.EnsembleXGBoost(params)
        model.fit(partitions, fix_seed=True)
        fits.setdefault(drug, []).extend(model.models)
    # Each contributing source has exactly two folds, hence equal booster and source weights.
    return CellHitFit(features, anchor, fits, amplitude)


def rank_expression(rna):
    return rankdata(rna.to_numpy(dtype=np.float32), method="average", axis=1).astype(np.float32) / rna.shape[1]


def pathway_weights(genes):
    book = pd.read_csv(verified(SIGNED, SIGNED_SHA), sep="\t")
    records = []
    for line in HALLMARK.read_text().splitlines():
        label, _, *members = line.split("\t")
        records.extend((gene, label, 1) for gene in members)
    table = pd.concat([book, pd.DataFrame(records, columns=["source", "target", "weight"])])
    table = table.loc[table.source.isin(genes)].copy()
    supported = table.groupby("target").source.transform("nunique").ge(10)
    overlap = table.loc[supported].pivot_table(index="source", columns="target", values="weight",
                                               aggfunc="mean", fill_value=0)
    weights = overlap.reindex(genes, fill_value=0).to_numpy(np.float32)
    return weights / np.maximum(np.abs(weights).sum(axis=0), 1e-6)


def chemistry():
    base = FM / "chemistry-v1/ecfp4"
    artifact = base / "vectors.npz"
    verified(artifact, read_json(base / "receipt.json")["output"]["sha256"])
    with np.load(artifact, allow_pickle=False) as f:
        matrix = pd.DataFrame(f["vectors"].astype(np.float32), index=f["id"].astype(str))
    if not matrix.index.is_unique or matrix.shape[1] != 2057:
        raise ValueError("Registered Track 6 ECFP4 plus descriptor axis changed")
    return matrix


@dataclass
class PathElasticNet:
    genes: pd.Index
    rank_scaler: StandardScaler
    pathway_weights: np.ndarray
    chemistry: pd.DataFrame
    scaler: StandardScaler
    model: ElasticNet

    def design(self, query, rna):
        paths = self.rank_scaler.transform(rank_expression(rna.loc[:, self.genes])) @ self.pathway_weights
        rows = rna.index.get_indexer(query.cell_id)
        if (rows < 0).any():
            raise ValueError("Missing pathway test RNA")
        return np.concatenate((self.chemistry.loc[query.compound_id].to_numpy(), paths[rows]), axis=1)

    def predict(self, query, expression):
        x = self.scaler.transform(self.design(query, expression))
        # Concatenated chemical and RNA features give a separable predictor.
        # Float64 dot products keep float32 accumulator noise out of interaction ranks.
        values = x.astype(np.float64) @ self.model.coef_.astype(np.float64) + float(self.model.intercept_)
        return {"ecfp4_path_elasticnet": values}


def standard_native(fit):
    y = native_sensitivity(fit)
    source = y.groupby(fit.source_id)
    return (y - source.transform("mean")) / source.transform("std").clip(lower=.05)


def tune_path(fit, rna):
    groups = sorted(fit.lineage_group.unique(), key=lambda s: hashlib.sha256(f"train-val-2801:{s}".encode()).digest())
    validation = fit.lineage_group.isin(groups[::5])
    inner, held = fit.loc[~validation], fit.loc[validation]
    ranked = rank_expression(rna)
    lookup = rna.index.get_indexer(inner.cell_id.unique())
    weights = pathway_weights(rna.columns)
    drug_bank = chemistry()
    rank_scaler = StandardScaler().fit(ranked[lookup])
    paths = rank_scaler.transform(ranked) @ weights

    def design(rows, matrix):
        return np.concatenate((drug_bank.loc[rows.compound_id].to_numpy(),
                               matrix[rna.index.get_indexer(rows.cell_id)]), axis=1)

    x, v = design(inner, paths), design(held, paths)
    scaler = StandardScaler().fit(x)
    x = np.asfortranarray(scaler.transform(x))
    v = scaler.transform(v)
    choices = []
    y = standard_native(inner)
    for alpha in (.01, .05):
        model = ElasticNet(alpha=alpha, l1_ratio=.25, max_iter=3000)
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(x, y)
        scores = held.assign(predicted_value=model.predict(v), endpoint_value=native_sensitivity(held))
        correlations = [across_cell_metrics(frame)["spearman"] for _, frame in scores.groupby("compound_id")]
        choices.append({"alpha": alpha, "within_drug_spearman": float(np.mean([r for r in correlations if r is not None])),
                        "iterations": int(model.n_iter_), "warnings": [str(n.message) for n in notices]})
    best = max(choices, key=lambda v: v["within_drug_spearman"])["alpha"]
    rank_scaler = StandardScaler().fit(ranked)
    paths = rank_scaler.transform(ranked) @ weights
    x = design(fit, paths)
    scaler = StandardScaler().fit(x)
    model = ElasticNet(alpha=best, l1_ratio=.25, max_iter=3000)
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(np.asfortranarray(scaler.transform(x)), standard_native(fit))
    diagnostics = {"selected_alpha": best, "inner": choices, "iterations": int(model.n_iter_),
                   "warnings": [str(n.message) for n in notices], "pathways": int(weights.shape[1])}
    return PathElasticNet(rna.columns, rank_scaler, weights, drug_bank, scaler, model), diagnostics


def fit(args):
    train, fit_rna, _, _ = inputs()
    edges, targets, hashes = molecular_inputs(BUILD / "v0plus-public-priors", HALLMARK, TARGETS)
    settings = read_json(verified(SETTINGS, SETTINGS_HASH))["settings"]
    args.out.mkdir(parents=True, exist_ok=False)
    held = None
    if args.fold != 3:
        train, held = list(fold_indices(train))[args.fold]
    rna = fit_rna.loc[sorted(train.cell_id.unique())]
    panel, anchor, aligned, _ = panel_targets(train)

    if args.part == "structured":
        auxiliary, aux_hashes = load_auxiliary(RUNS / "depmap-25q3-bb8a47b")
        model, details = fit_kernel(rna, aligned, anchor, edges, targets, auxiliary, device=args.device)
        details["auxiliary_sha256"] = aux_hashes
    elif args.part == "linear":
        features = ExpressionFeatures.fit(rna, edges, targets, n_variable=512, rank=True)
        lines = [x for x in settings["linear"] if x[0] in LINEAR]
        if sorted(x[0] for x in lines) != sorted(LINEAR):
            raise ValueError("Prior settings lack requested linear baselines")
        models = {name: (fit_baseline(name, (penalty, width), panel, anchor, rna, features, targets), amp)
                  for name, penalty, width, amp in lines}
        model, details = LinearPanel(models), {"settings": lines}
    elif args.part == "cellhit":
        features = ExpressionFeatures.fit(rna, edges, targets, n_variable=512, rank=True)
        setting, = settings["cellhit"]
        model = fit_cellhit(features, rna, aligned, anchor, setting)
        details = {"settings": setting, "fit_drugs": len(model.models),
                   "boosters": sum(map(len, model.models.values()))}
    else:
        model, details = tune_path(train, rna)
    checkpoint = args.out / "model.joblib"
    joblib.dump(model, checkpoint, compress=2)
    if held is not None:
        query = validation_roster(held)
        result = predict_part(args.part, model, query, fit_rna.loc[query.cell_id.unique()])
        freeze_table(as_forecast(query, result, fold=args.fold, track="source_oof"), args.out / "oof.parquet",
                     provenance={"train_data_sha256": TRAIN_SHA, "fit_rows": len(train),
                                 "held_rows": len(held), "checkpoint_sha256": sha256(checkpoint)})
    publish_json(args.out / "receipt.json", {"part": args.part, "fold": args.fold, "fit_rows": len(train),
                 "fit_lineages": train.lineage_group.nunique(), "train_data_sha256": TRAIN_SHA, "details": details,
                 "checkpoint_sha256": sha256(checkpoint), "prior_hashes": hashes})


def predict_part(name, model, query, rna):
    if name == "structured":
        anchor, residual = model.predict(query, rna)
        return {"structured": anchor + residual}
    return model.predict(query, rna)


def as_forecast(query, predictions, *, fold, track):
    if any(len(value) != len(query) or not np.isfinite(value).all() for value in predictions.values()):
        raise ValueError("All compared models require finite scores on the same panel")
    return pd.concat([query[KEY].assign(method=name, predicted_value=value, fold=fold, track=track)
                      for name, value in predictions.items()], ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=("linear", "cellhit", "structured", "elasticnet"), required=True)
    parser.add_argument("--fold", type=int, choices=range(4), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    fit(parser.parse_args())


if __name__ == "__main__":
    from .base_models import main

    main()
