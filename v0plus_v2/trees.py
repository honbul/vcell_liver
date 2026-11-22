"""Drug-specific boosted regressors, fitted only on the registered expression lineages."""

from collections import defaultdict

import numpy as np
import pandas as pd

from vc_build.external import KEY, SOURCE, native_sensitivity
from vc_build.v0plus_cv import panel_targets

from .adaptive import DrugHead, Projection, selectors
from .source import DrugModels, MIN_TRAIN, grouped, known_genes, selection_value

TREE_CONFIGS = {
    "drug_lgbm": ((256, 2, 180), (512, 3, 260)),
    "drug_catboost": ((256, 3, 190), (512, 4, 270)),
    "drug_xgboost": ((256, 2, 220), (512, 3, 300)),
}


def train_tree(kind, cfg, x, y):
    _, depth, n = cfg
    if kind == "drug_lgbm":
        from lightgbm import LGBMRegressor

        model = LGBMRegressor(max_depth=depth, num_leaves=2**depth, n_estimators=n, min_child_samples=17,
                              learning_rate=.035, reg_lambda=8., colsample_bytree=.45, verbosity=-1,
                              n_jobs=1, random_state=20261122)
        model.fit(x, y)
        return model.booster_
    if kind == "drug_catboost":
        from catboost import CatBoostRegressor

        model = CatBoostRegressor(depth=depth, iterations=n, learning_rate=.04, l2_leaf_reg=8.,
                                  verbose=False, allow_writing_files=False, random_seed=20261122,
                                  thread_count=1, rsm=.60)
    elif kind == "drug_xgboost":
        from xgboost import XGBRegressor

        model = XGBRegressor(max_depth=depth, n_estimators=n, learning_rate=.035, min_child_weight=11,
                             subsample=.85, colsample_bytree=.50, reg_lambda=10, tree_method="hist",
                             n_jobs=1, random_state=20261122)
    else:
        raise ValueError(f"Unknown per-drug tree {kind}")
    return model.fit(x, y)


def view(fit, expression):
    expr_fit = expression.loc[expression.index.intersection(fit.cell_id.unique())]
    projection = Projection.fit(expr_fit, "rank_z")
    return projection, projection.transform(expression)


def head(kind, cfg, projection, latent, ids, target, annotated):
    x = latent[ids]
    columns = selectors(x, target, projection.hvg, annotated, "target", cfg[0])
    return DrugHead("rank_z", columns, np.zeros(len(columns), dtype=np.float32),
                    train_tree(kind, cfg, x[:, columns], target))


def fit_models(fit, expression, targets, selected):
    panel, anchor, _, _ = panel_targets(fit)
    projection, values = view(fit, expression)
    models = {kind: {} for kind in TREE_CONFIGS}
    for drug, rows in panel.groupby("compound_id", sort=True):
        ids = expression.index.get_indexer(rows.cell_id)
        for kind, cfg in selected.get(drug, {}).items():
            models[kind][drug] = head(kind, cfg, projection, values, ids, rows.residual.to_numpy(),
                                      known_genes(expression, targets, drug))
    return DrugModels({"rank_z": projection}, models, anchor)


def tune(fit, expression, targets):
    scores = defaultdict(lambda: defaultdict(list))
    records = defaultdict(list)
    for i, (train, check) in enumerate(grouped(fit, 2)):
        view_model, values = view(train, expression)
        panel, anchor, _, _ = panel_targets(train)
        observed = check.assign(endpoint_value=native_sensitivity(check))
        for drug, rows in panel.groupby("compound_id", sort=True):
            if len(rows) < MIN_TRAIN:
                continue
            held = observed.loc[observed.compound_id.eq(drug)]
            if held.empty:
                continue
            cells = pd.Index(sorted(held.cell_id.unique()))
            tr, va = expression.index.get_indexer(rows.cell_id), expression.index.get_indexer(cells)
            annotations = known_genes(expression, targets, drug)
            for kind, settings in TREE_CONFIGS.items():
                for cfg in settings:
                    estimator = head(kind, cfg, view_model, values, tr, rows.residual.to_numpy(), annotations)
                    x = estimator.apply(values[va])
                    score = selection_value(held, cells, x[:, None])[0]
                    scores[drug][kind, cfg].append(score)
                    records[drug, kind, cfg].append(pd.DataFrame({
                        "inner": i, "cell_id": cells, "compound_id": drug, "method": kind,
                        "forecast": anchor[drug] + x,
                    }))
    selected = {}
    predictions = []
    for drug, choices in scores.items():
        selected[drug] = {}
        for method in TREE_CONFIGS:
            possible = {cfg: np.mean(v) for (kind, cfg), v in choices.items() if kind == method}
            if possible:
                best = max(possible, key=possible.get)
                selected[drug][method] = best
                predictions.extend(records[drug, method, best])
    return selected, pd.concat(predictions, ignore_index=True)


def development_targets(fit):
    """Use the configuration's source-only alignment to define measured residual error."""
    output = []
    for idx, (train, held) in enumerate(grouped(fit, 2)):
        _, anchor, aligned, transforms = panel_targets(train)
        means = aligned.groupby([*SOURCE, "compound_id"]).source_mean.first().rename("source_mean")
        held = held.merge(pd.DataFrame(transforms), on=SOURCE, validate="many_to_one")
        y = native_sensitivity(held) * held.slope + held.intercept
        held = held.assign(model_value=y).join(means, on=[*SOURCE, "compound_id"], validate="many_to_one")
        held["residual"] = held.model_value - held.source_mean
        held["anchor"] = held.compound_id.map(anchor)
        columns = ["cell_id", "compound_id", "lineage_group"]
        output.append(held.groupby(columns, as_index=False)[["residual", "anchor"]].mean().assign(inner=idx))
    return pd.concat(output, ignore_index=True)


def gains(oof, anchors, shrink=24.):
    joined = oof.merge(anchors, on=["cell_id", "compound_id", "inner"], validate="many_to_one")
    joined = joined.dropna(subset=["residual", "anchor"])
    joined["predicted_residual"] = joined.forecast - joined.anchor
    joined["cross"] = joined.residual * joined.predicted_residual
    joined["norm2"] = np.square(joined.predicted_residual)
    fits = {}
    for method, group in joined.groupby("method"):
        prior = np.clip(group.cross.sum() / max(.0001, group.norm2.sum()), .05, 1.8)
        drug_gain = {}
        for drug, part in group.groupby("compound_id"):
            penalty = shrink * max(.0001, np.median(group.norm2))
            value = (part.cross.sum() + penalty * prior) / (part.norm2.sum()+penalty)
            drug_gain[drug] = float(np.clip(value, .05, 2.2))
        fits[method] = drug_gain
    return fits


def apply_gain(values, query, anchor, fitted, cap):
    mean = query.compound_id.map(anchor).to_numpy()
    medication = query.compound_id.map(fitted).fillna(.6).to_numpy()
    return mean + cap * medication * (values - mean)
