"""Drug-specific boosted regressors, fitted only on the registered expression lineages."""

import numpy as np

from vc_build.v0plus_cv import panel_targets

from .adaptive import DrugHead, Projection, selectors
from .source import DrugModels, known_genes

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
