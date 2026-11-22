"""Historical nested scaffold selection and per-model frozen calibration."""
from dataclasses import dataclass
import warnings

import numpy as np
from lightgbm import LGBMClassifier
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20261123
GRIDS = {
    "clogp": (1.0,),
    "ecfp_logistic": (.02, .2, 2.),
    "ecfp_ridge": (1., 10., 100.),
    "ecfp_lightgbm": ((7, 100), (15, 140), (31, 180)),
    "clamp_ridge": (1., 10., 100.),
    "clamp_lightgbm": ((7, 100), (15, 140), (31, 180)),
    "ecfp_clamp_elasticnet": ((.025, .20), (.08, .20), (.08, .65)),
}
CLAMP_METHODS = ("clamp_ridge", "clamp_lightgbm", "ecfp_clamp_elasticnet")
CALIBRATORS = ("platt", "temperature", "beta", "prior_em")
FEATURES = {name: "clamp" if name.startswith("clamp") else
            "combined" if name.endswith("elasticnet") else
            "clogp" if name == "clogp" else "bits" for name in GRIDS}


def group_splits(rows, n, seed):
    cv = StratifiedGroupKFold(n, shuffle=True, random_state=seed)
    for a, b in cv.split(rows, rows.label, rows.scaffold):
        source, query = rows.iloc[a], rows.iloc[b]
        if (set(source.connectivity) & set(query.connectivity)
                or set(source.scaffold) & set(query.scaffold)
                or source.label.nunique() != 2 or query.label.nunique() != 2):
            raise ValueError("Historical scaffold CV is not a valid grouped binary split")
        yield a, b


def fit_head(name, setting, x, y):
    if name.endswith("lightgbm"):
        leaves, rounds = setting
        return LGBMClassifier(num_leaves=leaves, n_estimators=rounds, learning_rate=.04,
                              min_child_samples=20, colsample_bytree=.75, reg_lambda=2,
                              verbosity=-1, n_jobs=2, random_state=SEED).fit(x, y)
    if name.endswith("ridge"):
        model = Ridge(alpha=setting, solver="lsqr", tol=1e-5)
        if name == "clamp_ridge":
            model = make_pipeline(StandardScaler(), model)
        return model.fit(x, y)
    if name.endswith("elasticnet"):
        strength, ratio = setting
        scale = ColumnTransformer([("bits", "passthrough", slice(0, 2048)),
                                   ("embedding", StandardScaler(), slice(2048, None))])
        model = make_pipeline(scale, LogisticRegression(C=strength, l1_ratio=ratio,
                              penalty="elasticnet", solver="saga", max_iter=1600,
                              tol=.001, random_state=SEED))
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            return model.fit(x, y)
    return LogisticRegression(C=setting, solver="liblinear", random_state=SEED).fit(x, y)


def raw_score(name, model, x):
    if name.endswith("lightgbm"):
        return np.asarray(model.booster_.predict(x, raw_score=True, num_threads=2))
    if name.endswith("ridge"):
        return 4.0 * (model.predict(x) - .5)
    return model.decision_function(x)


def design(method, raw):
    if method == "beta":
        p = np.clip(expit(raw), 1e-6, 1 - 1e-6)
        return np.stack([np.log(p), -np.log1p(-p)], axis=-1)
    return np.asarray(raw)[..., None]


@dataclass
class Calibrator:
    method: str
    coefficients: np.ndarray
    intercept: float
    threshold: float = .5

    def probability(self, raw):
        return expit(design(self.method, raw) @ self.coefficients + self.intercept)

    def interval(self, raw, parameters):
        values = expit(design(self.method, raw) @ parameters[:, :-1].T + parameters[:, -1])
        return np.quantile(values, [.05, .95], axis=1).T


def fit_calibrator(method, raw, y, weights=None):
    if method == "temperature":
        def loss(temperature):
            return log_loss(y, expit(raw / temperature), sample_weight=weights, labels=[0, 1])

        temperature = minimize_scalar(loss, bounds=(.2, 12.), method="bounded").x
        return Calibrator(method, np.array([1 / temperature]), 0.)
    predictor = LogisticRegression(C=5., solver="lbfgs", max_iter=400)
    fitted = predictor.fit(design(method, raw), y, sample_weight=weights)
    weights_out = fitted.coef_[0]
    offset = float(fitted.intercept_[0])
    if method == "prior_em":
        # Estimate one frozen prior from the historical calibration reference.
        # Every query uses the same offset, including when given individually.
        source = float(np.average(y, weights=weights))
        p = expit(raw * weights_out[0] + offset)
        estimated = source
        for _ in range(25):
            responsibility = expit(logit(np.clip(p, 1e-8, 1 - 1e-8)) + logit(estimated) - logit(source))
            updated = .75 * np.average(responsibility, weights=weights) + .25 * source
            if abs(estimated - updated) < 1e-7:
                break
            estimated = float(np.clip(updated, .03, .97))
        offset += logit(estimated) - logit(source)
    return Calibrator(method, weights_out, offset)


def decision_threshold(y, p):
    candidates = np.unique(np.r_[.5, np.quantile(p, np.linspace(.06, .94, 45))])
    return float(max(candidates, key=lambda v: (balanced_accuracy_score(y, p >= v), -abs(v - .5))))


def metrics(y, p, decision):
    bins = np.clip((p * 10).astype(int), 0, 9)
    ece = np.abs(np.bincount(bins, weights=y - p, minlength=10)).sum() / len(y)
    return dict(auroc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)),
                balanced_accuracy=float(balanced_accuracy_score(y, decision)),
                brier=float(np.mean(np.square(y - p))), ece=float(ece))


def utility(y, raw):
    return .5 * (roc_auc_score(y, raw) + average_precision_score(y, raw))


def crossfit(name, setting, x, y, folds):
    prediction = np.empty(len(y))
    for t, v in folds:
        model = fit_head(name, setting, x[t], y[t])
        prediction[v] = raw_score(name, model, x[v])
    return prediction


def select_setting(name, x, y, folds, focal=utility):
    candidates = [(setting, crossfit(name, setting, x, y, folds)) for setting in GRIDS[name]]
    return max(candidates, key=lambda value: focal(y, value[1]))


def dev_selection(history, values):
    chosen = {}
    folds = list(group_splits(history, 5, SEED))
    y = history.label.to_numpy()
    for name, feature in FEATURES.items():
        print("Historical selection", name, flush=True)
        x = values[feature]
        arms = {method: (np.empty(len(y)), np.empty(len(y), dtype=bool)) for method in CALIBRATORS}
        inner_choices = []
        for i, (a, b) in enumerate(folds):
            inner = list(group_splits(history.iloc[a], 4, SEED + i + 1))
            setting, raw_inner = select_setting(name, x[a], y[a], inner)
            inner_choices.append(setting)
            raw_out = raw_score(name, fit_head(name, setting, x[a], y[a]), x[b])
            for method, (scores, decisions) in arms.items():
                calibrated = fit_calibrator(method, raw_inner, y[a])
                cutoff = decision_threshold(y[a], calibrated.probability(raw_inner))
                scores[b] = calibrated.probability(raw_out)
                decisions[b] = scores[b] >= cutoff
        stats = {method: metrics(y, p, d) for method, (p, d) in arms.items()}
        # Evaluate each calibration strategy on the outer historical OOF observations.
        calibrated = min(stats, key=lambda method: stats[method]["brier"] + .2 * stats[method]["ece"])
        setting, oof = select_setting(name, x, y, folds)
        chosen[name] = dict(setting=setting, method=calibrated, metrics=stats[calibrated],
                            nested_calibration=stats, outer_inner_settings=inner_choices)
    rank = max(CLAMP_METHODS, key=lambda n: chosen[n]["metrics"]["auroc"])
    shortlist = max(CLAMP_METHODS, key=lambda n: chosen[n]["metrics"]["auprc"])
    deployment = max(CLAMP_METHODS, key=lambda n: tradeoff(chosen[n]["metrics"]))
    return dict(rank_auroc=rank, shortlist_auprc=shortlist, deployment=deployment, methods=chosen)


def tradeoff(s):
    return .5 * (s["auroc"] + s["auprc"]) - .10 * s["brier"] - .05 * s["ece"]


def calibration_draws(method, raw, history, count=400):
    y = history.label.to_numpy()
    _, g = np.unique(history.connectivity, return_inverse=True)
    rnd = np.random.default_rng(SEED)
    result = []
    for _ in range(count):
        weights = np.bincount(rnd.integers(g.max() + 1, size=g.max() + 1), minlength=g.max() + 1)[g]
        if np.unique(y[weights > 0]).size == 2:
            fitted = fit_calibrator(method, raw, y, weights)
            result.append(np.r_[fitted.coefficients, fitted.intercept])
    return np.stack(result)
