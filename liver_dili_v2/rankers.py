"""Rankers with scalar raw output and no fold-global normalization."""

from dataclasses import dataclass

import numpy as np
from scipy.special import logit
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .features import design, tanimoto

GRIDS = {
    "v1_lgbm": ((7, 80), (15, 120), (31, 160)),
    "lgbm": ((7, 250), (15, 350), (31, 250)),
    "catboost": ((4, 350), (6, 500)),
    "xgboost": ((2, 250), (4, 350)),
    "extra": ((3, .25), (7, .55)),
    "linear": (.005, .03, .2),
    "tsvm": (.15, .7, 3., 15.),
    "countsvm": (.15, .7, 3., 15.),
    "tlogit": (.15, 1., 6.),
    "minmaxsvm": (.3, 2., 8.),
    "rbfsvm": ((.4, .7), (2., 1.7), (6., .75)),
}

REPRESENTATIONS = {
    "v1_lgbm": ("v1",), "lgbm": ("structure", "chemberta2", "molformer", "unimol", "clamp", "all_fm"),
    "catboost": ("structure", "molformer", "all_fm"), "xgboost": ("structure", "all_fm"),
    "extra": ("structure", "all_fm"), "linear": ("structure", "all_fm"),
    "tsvm": ("structure",), "countsvm": ("structure",), "tlogit": ("structure",),
    "minmaxsvm": ("structure",), "rbfsvm": ("structure",),
}


@dataclass
class Ranker:
    name: str
    representation: str
    estimator: object
    kernel_reference: np.ndarray | None = None
    kernel_mode: str = ""
    bandwidth: float = 0.

    def raw(self, blocks):
        if self.kernel_reference is not None:
            values = blocks["base" if self.kernel_mode == "bits" else "counts"]
            value = tanimoto(values, self.kernel_reference, mode=self.kernel_mode)
            if self.bandwidth:
                value = np.exp(-self.bandwidth*(1.-value))
            return self.estimator.decision_function(value)
        x = design(blocks, self.representation)
        if self.name in ("linear", "tlogit"):
            return self.estimator.decision_function(x)
        return logit(np.clip(self.estimator.predict_proba(x)[:, 1], 1e-6, 1-1e-6))


def train(name, representation, cfg, blocks, y, gpu=False, sample_weight=None):
    if name in ("tsvm", "countsvm", "tlogit", "minmaxsvm", "rbfsvm"):
        mode = {"countsvm":"counts", "minmaxsvm":"minmax"}.get(name,"bits")
        penalty, bandwidth = cfg if name == "rbfsvm" else (cfg, 0.)
        ref = blocks["base" if mode == "bits" else "counts"]
        kernel = tanimoto(ref, ref, mode=mode)
        if bandwidth:
            kernel = np.exp(-bandwidth*(1.-kernel))
        estimator = (LogisticRegression(C=penalty, max_iter=500) if name == "tlogit"
                     else SVC(C=penalty, kernel="precomputed"))
        estimator.fit(kernel, y, sample_weight=sample_weight)
        return Ranker(name, representation, estimator, ref, mode, bandwidth)

    if name in ("v1_lgbm", "lgbm"):
        from lightgbm import LGBMClassifier

        leaves, n = cfg
        if name == "v1_lgbm":
            estimator = LGBMClassifier(n_estimators=n, num_leaves=leaves, learning_rate=.04, min_child_samples=20, colsample_bytree=.7,
                                       reg_lambda=2., random_state=20261122, verbosity=-1, n_jobs=2)
        else:
            estimator = LGBMClassifier(n_estimators=n, num_leaves=leaves, learning_rate=.027,
                                       min_child_samples=12, reg_lambda=5., colsample_bytree=.65,
                                       random_state=20261122, verbosity=-1, n_jobs=2)
    elif name == "catboost":
        from catboost import CatBoostClassifier

        depth, n = cfg
        estimator = CatBoostClassifier(depth=depth, iterations=n, learning_rate=.045,
                                       l2_leaf_reg=7., verbose=False, allow_writing_files=False,
                                       task_type="GPU" if gpu else "CPU",
                                       random_seed=20261122, thread_count=3)
    elif name == "xgboost":
        from xgboost import XGBClassifier

        depth, n = cfg
        estimator = XGBClassifier(max_depth=depth, n_estimators=n, learning_rate=.035,
                                  min_child_weight=4, subsample=.8, colsample_bytree=.55,
                                  reg_lambda=5., n_jobs=2, tree_method="hist", random_state=20261122)
    elif name == "extra":
        size, fraction = cfg
        estimator = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                  ExtraTreesClassifier(n_estimators=420, min_samples_leaf=size,
                                                       max_features=fraction, n_jobs=2, random_state=20261122))
    elif name == "linear":
        estimator = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                  StandardScaler(), LogisticRegression(C=cfg, max_iter=500))
    else:
        raise ValueError(f"Unknown ranker {name}")
    x = design(blocks, representation)
    weights = {} if sample_weight is None else {"sample_weight": sample_weight}
    if sample_weight is not None and name in ("extra", "linear"):
        model = estimator.steps[-1][0]
        weights = {f"{model}__sample_weight": sample_weight}
    estimator.fit(x, y, **weights)
    return Ranker(name, representation, estimator)
