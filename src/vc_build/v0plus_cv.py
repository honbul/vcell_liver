"""Train and compare public-source pharmacogenomic baselines on grouped splits."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import GroupKFold

from .context_features import HALLMARK_SHA256
from .evaluate import score, score_across_cells
from .external import KEY, SOURCE, _rna, align_endpoints, native_sensitivity
from .io import freeze_table, publish_json, read_table, sha256
from .v0plus_features import ExpressionFeatures, signatures, validated_public_prior

IDENTITY = ["cell_id", "lineage_group", "compound_id"]
LINEAR = [
    ("drug_ridge", alpha, n) for alpha, n in [(10., 256), (100., 256), (300., 512), (1000., 512)]
] + [
    ("drug_elasticnet", alpha, n) for alpha, n in [(0.002, 256), (0.007, 512), (0.02, 512)]
]
TREES = [(5, 450), (7, 650)]
GAMMAS = (0.25, 0.5, 1.0)
UPSTREAM_VERSIONS = {
    "CellHit": "9d05b0881768fedecb0842305ace51a3dabe798a",
    "Velodrome": "453dd5883337f32a3e667bf63cc53b681fdc363d",
    "CODE-AE": "46b448bcadfafb4c8f552233aa8392f47d2e6b33",
}


def training_inputs(bundle):
    receipt = json.loads((bundle / "receipt.json").read_text())
    for relative in ["train/rows.parquet", "train/rna.parquet"]:
        if sha256(bundle / relative) != receipt["artifacts"][relative]["sha256"]:
            raise ValueError(f"Training input differs from its receipt: {relative}")
    fit = read_table(bundle / "train/rows.parquet").rename(
        columns={"record_id": "row_id", "cell_group": "lineage_group"}
    )
    return fit, _rna(bundle / "train/rna.parquet"), receipt


def molecular_inputs(priors, hallmark, target_file):
    root = "https://github.com/saezlab/ConservedFootprints/tree/b93ffed985ba4b83e93cae9dc52a0f76f1a45755"
    sources = {
        "progeny": (priors / "progeny.csv", root, "11b440e307e1f1d58a09462ec30b356f5f9a3442fef594512e6d203b68220ae8"),
        "dorothea": (priors / "dorothea.csv", root, "35cb15ae6cd32c536b3acba6914a6cecc4a2b5fbfafc18217ba194b8ab1e1324"),
        "hallmark": (hallmark, "MSigDB 2026.1.Hs h.all symbols", HALLMARK_SHA256),
    }
    provenance = {name: validated_public_prior(path, kind=name, expected=url, digest=checksum)
                  for name, (path, url, checksum) in sources.items()}
    edges = signatures(hallmark, priors / "progeny.csv", priors / "dorothea.csv")
    targets = json.loads(target_file.read_text())
    return edges, targets, provenance | {"drug_targets_sha256": sha256(target_file)}


def panel_targets(fit):
    """Preserve equal assay weight on the compound anchor and learn cell-specific residuals."""
    aligned, calibration = align_endpoints(fit)
    means = aligned.groupby([*SOURCE, "compound_id"]).model_value.mean().rename("source_mean")
    anchor = means.groupby(level="compound_id").mean()
    aligned = aligned.join(means, on=[*SOURCE, "compound_id"], validate="many_to_one")
    aligned["residual"] = aligned.model_value - aligned.source_mean
    panel = aligned.groupby(IDENTITY, as_index=False).residual.mean()
    return panel, anchor, aligned, calibration


def validation_roster(validation):
    query = validation[[*IDENTITY, "isomeric_smiles"]].drop_duplicates()
    if query.duplicated(IDENTITY).any():
        raise ValueError("Multiple structures for a registered compound")
    return query.assign(row_id=query.cell_id + "|" + query.compound_id).reset_index(drop=True)


def attach_native(forecasts, validation, fold):
    actual = validation.assign(endpoint_value=native_sensitivity(validation))
    if actual.duplicated([*IDENTITY, *SOURCE]).any():
        raise ValueError("Different labels within one native assay")
    merged = forecasts.merge(actual[[*IDENTITY, *SOURCE, "endpoint_value"]], on=IDENTITY, validate="many_to_many")
    return merged.assign(track=merged.source_id + "/" + merged.endpoint_id, fold=fold)


def summarize(joined):
    within_cell, _ = score(joined, endpoint_direction="higher", rank_only=True)
    per_drug, _ = score_across_cells(joined)
    within_drug = (per_drug.groupby(["track", "method"])[["spearman", "pearson"]]
                   .mean().reset_index().to_dict("records"))
    cells = pd.DataFrame(within_cell["scores"])
    drugs = pd.DataFrame(within_drug)
    return cells.merge(drugs, on=["track", "method"], validate="1:1")


@dataclass
class FittedBaseline:
    features: ExpressionFeatures
    estimator: object
    method: str
    anchor: pd.Series
    drug_columns: dict

    def predict(self, query, rna):
        projected = self.features.transform(rna)
        base = query.compound_id.map(self.anchor).to_numpy()
        if self.method == "drug_mean":
            return base, np.zeros(len(query))
        if self.method == "additive":
            return base, self.estimator.predict(projected.loc[query.cell_id])
        if self.method in {"lightgbm", "catboost"}:
            x = pooled_design(query, projected, self.anchor, self.drug_columns)
            return base, self.estimator.predict(x)
        residual = np.full(len(query), np.nan)
        for drug, positions in query.groupby("compound_id").indices.items():
            if drug in self.estimator:
                x = projected.loc[query.iloc[positions].cell_id, self.drug_columns[drug]]
                residual[positions] = self.estimator[drug].predict(x.to_numpy())
        return base, residual


def pooled_design(rows, projected, anchor, drug_features):
    x = projected.loc[rows.cell_id].to_numpy(dtype=np.float32)
    chemistry = drug_features.loc[rows.compound_id].to_numpy(dtype=np.float32)
    return np.column_stack((x, chemistry, rows.compound_id.map(anchor).to_numpy(dtype=np.float32)))


def design_drugs(fit):
    from .external import _chemistry

    panel = fit[["compound_id", "isomeric_smiles"]].drop_duplicates()
    chemistry = _chemistry(panel)
    # One-hot identities permit compound-dependent splits even when fingerprints are sparse.
    identity = pd.get_dummies(chemistry.index, dtype=np.float32).set_index(chemistry.index)
    return pd.DataFrame(np.column_stack([chemistry, identity]), index=chemistry.index)


def fit_baseline(method, settings, panel, anchor, rna, features, targets, *, chemistry=None, gpu=False):
    projected = features.transform(rna)
    penalty, n_variable = settings
    if method == "drug_mean":
        estimator, columns = None, {}
    elif method == "additive":
        estimator = Ridge(alpha=penalty).fit(projected.loc[panel.cell_id], panel.residual)
        columns = {}
    elif method in {"lightgbm", "catboost"}:
        depth, trees = map(int, settings)
        if method == "catboost":
            from catboost import CatBoostRegressor

            estimator = CatBoostRegressor(iterations=trees, depth=depth, learning_rate=0.055, l2_leaf_reg=8,
                                          loss_function="RMSE", random_seed=11817, verbose=False,
                                          task_type="GPU" if gpu else "CPU", allow_writing_files=False)
        else:
            from lightgbm import LGBMRegressor

            estimator = LGBMRegressor(n_estimators=trees, max_depth=depth, num_leaves=min(47, 2**depth - 1),
                                      learning_rate=0.045, min_child_samples=25, reg_lambda=8,
                                      n_jobs=5, verbosity=-1, random_state=11817)
        columns = chemistry
        estimator.fit(pooled_design(panel, projected, anchor, chemistry), panel.residual)
    else:
        estimator, columns = {}, {}
        pathway = [col for col in projected if col.startswith(("HALLMARK_", "PROGENY_", "DOROTHEA_"))]
        for compound, rows in panel.groupby("compound_id", sort=True):
            if rows.lineage_group.nunique() < 30:
                continue
            selected = list(dict.fromkeys(projected.columns[:n_variable].tolist() + pathway
                                          + [gene for gene in targets.get(compound, []) if gene in projected]))
            model = (Ridge(alpha=penalty) if method == "drug_ridge" else
                     ElasticNet(alpha=penalty, l1_ratio=0.35, max_iter=1800, tol=3e-4))
            estimator[compound] = model.fit(projected.loc[rows.cell_id, selected].to_numpy(), rows.residual)
            columns[compound] = selected
    return FittedBaseline(features, estimator, method, anchor, columns)


def settings_for(arm):
    if arm == "linear":
        return [("drug_mean", 0., 256), ("additive", 30., 512), ("additive", 300., 512), *LINEAR]
    return [(arm, *setting) for setting in TREES]


def candidate_forecasts(arm, train, query, rna, edges, targets, *, gpu=False, selected=None):
    candidates = settings_for(arm) if selected is None else sorted({entry[:3] for entry in selected})
    gammas = GAMMAS if selected is None else tuple(sorted({entry[3] for entry in selected}))
    fit_rna = rna.loc[sorted(train.cell_id.unique())]
    query_rna = rna.loc[sorted(query.cell_id.unique())]
    panel, anchor, _, calibration = panel_targets(train)
    chemistry = design_drugs(train) if arm in {"lightgbm", "catboost"} else None
    features = ExpressionFeatures.fit(fit_rna, edges, targets, n_variable=512 if arm == "linear" else 256, rank=True)
    predictions = []
    for method, penalty, width in candidates:
        model = fit_baseline(method, (penalty, width), panel, anchor, fit_rna, features, targets,
                             chemistry=chemistry, gpu=gpu)
        ref, residual = model.predict(query, query_rna)
        for multiplier in (1.,) if method == "drug_mean" else gammas:
            if selected is not None and (method, penalty, width, multiplier) not in selected:
                continue
            name = f"{method}/{penalty}/{width}/{multiplier}"
            predictions.append(query[KEY].assign(method=name, predicted_value=ref + multiplier * residual))
    return pd.concat(predictions, ignore_index=True), calibration


def choose(metrics):
    """Equal source weight. Give patient discrimination half of setting selection."""
    utility = metrics.assign(utility=0.5 * metrics.spearman
                             + 0.25 * metrics.macro_spearman + 0.25 * metrics.macro_ndcg_at_5)
    utility = utility.groupby("method").utility.mean()
    picked = {}
    for name, _ in utility.sort_values(ascending=False, kind="stable").items():
        arm, penalty, width, multiplier = name.split("/")
        picked.setdefault(arm, (arm, float(penalty), int(width), float(multiplier)))
    return list(picked.values())


def run_fold(train, rna, edges, targets, arm, fold, *, gpu):
    folds = list(GroupKFold(n_splits=3).split(train, groups=train.lineage_group))
    outer_fit, outer_check = (train.iloc[indices] for indices in folds[fold])
    inner_forecasts = []
    for inner_fold, (fit_index, check_index) in enumerate(
        GroupKFold(n_splits=2).split(outer_fit, groups=outer_fit.lineage_group)
    ):
        fit, check = outer_fit.iloc[fit_index], outer_fit.iloc[check_index]
        prediction, _ = candidate_forecasts(arm, fit, validation_roster(check), rna, edges, targets, gpu=gpu)
        inner_forecasts.append(attach_native(prediction, check, inner_fold))
    inner = summarize(pd.concat(inner_forecasts, ignore_index=True))
    selected = choose(inner)
    outer, calibration = candidate_forecasts(
        arm, outer_fit, validation_roster(outer_check), rna, edges, targets, gpu=gpu, selected=selected,
    )
    outer.method = outer.method.str.split("/").str[0]
    results = attach_native(outer, outer_check, fold)
    return results, selected, calibration, inner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--priors", required=True, type=Path)
    parser.add_argument("--hallmark", required=True, type=Path)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--arm", choices=("linear", "catboost", "lightgbm"), required=True)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--fold", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    fit, rna, receipt = training_inputs(args.bundle)
    edges, targets, provenance = molecular_inputs(args.priors, args.hallmark, args.targets)
    outer, selected, calibration, inner = run_fold(fit, rna, edges, targets, args.arm, args.fold, gpu=args.gpu)
    report = {
        "fold": args.fold, "arm": args.arm, "gpu": args.gpu, "selected": selected, "calibration": calibration,
        "molecular_priors": provenance, "source_training_rows": receipt["train_rows"],
        "validation_lineages": int(outer.lineage_group.nunique()),
        "inner_metrics": inner.to_dict("records"), "outer_metrics": summarize(outer).to_dict("records"),
    }
    # Internal OOF observations were registered as source-only development data.
    args.out.mkdir(parents=True, exist_ok=False)
    outer.to_parquet(args.out / "validation.parquet", index=False)
    freeze_table(outer[["row_id", "track", "fold", "method", "predicted_value"]], args.out / "forecast.parquet",
                 provenance={"source_development_only": True, "selected": selected})
    publish_json(args.out / "metrics.json", report)


if __name__ == "__main__":
    main()
