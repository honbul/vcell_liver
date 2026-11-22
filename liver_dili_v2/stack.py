"""Nonnegative stacking, mechanism OOF features and representation fallback."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .round1 import ranking


@dataclass
class ScoreTransform:
    features: list
    anchors: list

    @classmethod
    def fit(cls, frame, columns):
        return cls(list(columns), [np.sort(frame[name].to_numpy()) for name in columns])

    def transform(self, frame):
        scores = []
        for feature, observed in zip(self.features, self.anchors):
            point = frame[feature].to_numpy()
            low = np.searchsorted(observed, point, side="left")
            high = np.searchsorted(observed, point, side="right")
            u = (.5*(low+high)+.5)/(len(observed)+1)
            scores.append(norm.ppf(u))
        return np.column_stack(scores)


@dataclass
class MetaFit:
    transform: ScoreTransform
    coeff: np.ndarray
    recipe: str

    def raw(self, frame):
        transformed = self.transform.transform(frame)
        return self.coeff[0]+transformed@self.coeff[1:]


GRID = {
    "heterogeneous": ("v1_lgbm", "lgbm", "catboost", "xgboost", "extra", "linear", "tsvm", "countsvm",
                      "tlogit", "minmaxsvm", "rbfsvm", "dmpnn", "mechanism_dmpnn"),
    "kernel_graph": ("linear", "tsvm", "countsvm", "tlogit", "minmaxsvm", "rbfsvm", "dmpnn", "mechanism_dmpnn"),
    "foundation_tree": ("lgbm", "catboost", "xgboost", "extra", "linear"),
    "structure_only": ("tsvm", "countsvm", "tlogit", "minmaxsvm", "rbfsvm", "dmpnn"),
    "sparse": ("extra", "linear", "tsvm", "mechanism_dmpnn"),
    "extra_single": ("extra",),
    "extra_anchor": ("extra", "extra_reweighted", "xgboost", "rbfsvm", "dmpnn", "mechanism_dmpnn"),
    "kernel_tree": ("lgbm", "catboost", "extra", "tsvm", "tlogit", "minmaxsvm", "rbfsvm"),
}


def nonnegative(x, y, penalty):
    intercept = logit(np.mean(y))
    if penalty is None:
        weight = np.full(x.shape[1], 1./x.shape[1])
        return np.r_[intercept, weight]

    def evaluate(coef):
        p = coef[0] + x@coef[1:]
        loss = (np.logaddexp(0, p) - y*p).mean()
        err = expit(p)-y
        gradient = np.r_[err.mean(), (x.T@err)/len(y) + penalty*coef[1:]]
        return loss + .5*penalty*np.dot(coef[1:], coef[1:]), gradient

    opt = minimize(evaluate, np.r_[intercept, np.zeros(x.shape[1])], jac=True, method="L-BFGS-B",
                   bounds=[(None, None)]+[(0., None)]*x.shape[1])
    if not opt.success:
        raise RuntimeError(f"Nonnegative meta-estimator failed: {opt.message}")
    return opt.x


def construct(frame, columns, label, recipe, penalty):
    reference = ScoreTransform.fit(frame, columns)
    weights = nonnegative(reference.transform(frame), label, penalty)
    return MetaFit(reference, weights, recipe)


def mechanistic_features(train, query, endpoints):
    """Fit signed mechanism scores on training-only predictions with capped rank dimension."""
    combined = train[endpoints].to_numpy()
    comments = query[endpoints].to_numpy()
    scale = StandardScaler().fit(combined)
    x, q = scale.transform(combined), scale.transform(comments)
    ridge = LogisticRegression(C=.018, max_iter=500).fit(x, train.label)
    return ridge.decision_function(q)


def tune(frame, label, groups):
    records, forecasts, settings = [], {}, {}
    splits = list(GroupKFold(5).split(frame, groups=groups))
    variants = {name: [k for k in keys if k in frame] for name, keys in GRID.items()}
    variants["weighted_heterogeneous"] = [*variants["heterogeneous"], *[n for n in frame if n.endswith("_reweighted")]]
    variants["mechanism_oof"] = [*variants["kernel_tree"], *[n for n in frame if "__" in n or n == "signed_mechanism"]]
    for name, columns in variants.items():
        if not columns:
            continue
        for penalty in (None, .01, .065, .22):
            raw = np.empty(len(frame))
            for tr, va in splits:
                foam = construct(frame.iloc[tr], columns, label[tr], name, penalty)
                raw[va] = foam.raw(frame.iloc[va])
            record = {"recipe": name, "penalty": penalty, "n_predictors": len(columns), **ranking(label, raw)}
            record["utility"] = .6*record["auroc"] + .4*record["auprc"]
            key = name, penalty
            settings[key] = (columns, penalty)
            forecasts[key] = raw
            records.append(record)
    results = sorted(records, key=lambda x: -x["utility"])
    best = results[0]
    choice = (best["recipe"], best["penalty"])
    columns, penalty = settings[choice]
    model = construct(frame, columns, label, choice[0], penalty)
    return model, forecasts[choice], best, results
