"""Cross-fit calibration and confidence under uncertain FDA prevalence."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

METHODS = ("platt", "temperature", "beta", "isotonic", "prior_em")
EPS = 1e-5


@dataclass
class Calibration:
    method: str
    fitted: object
    prior: float
    threshold: float = .5

    def probabilities(self, score, *, estimate_prior=False):
        if self.method == "temperature":
            result = expit(score/self.fitted)
        if self.method == "isotonic":
            result = self.fitted.predict(score)
        if self.method in ("platt", "prior_em"):
            result = self.fitted.predict_proba(score[:, None])[:, 1]
        if self.method == "beta":
            result = self.fitted.predict_proba(beta_input(score))[:, 1]
        result = np.clip(result, EPS, 1-EPS)
        estimate = self.prior
        if self.method == "prior_em" and estimate_prior and len(result) > 5:
            estimate = shift_prior(result, self.prior)
            result = correction(result, self.prior, estimate)
        return result, estimate


def beta_input(score):
    p = np.clip(expit(score), EPS, 1.-EPS)
    return np.c_[np.log(p), -np.log1p(-p)]


def correction(probability, original, new):
    original = np.clip(original, .02, .98)
    new = np.clip(new, .02, .98)
    return expit(logit(probability) + logit(new) - logit(original))


def shift_prior(probability, train_prior):
    estimate = train_prior
    for _ in range(50):
        e = correction(probability, train_prior, estimate).mean()
        target = .7*e + .3*train_prior
        if abs(target-estimate) < 1e-5:
            break
        estimate = np.clip(target, .08, .92)
    return float(estimate)


def fit(kind, scores, target):
    if kind in ("platt", "prior_em", "beta"):
        x = beta_input(scores) if kind == "beta" else scores[:, None]
        cal = LogisticRegression(C=1.5, solver="lbfgs", max_iter=500).fit(x, target)
    elif kind == "temperature":
        def objective(t):
            return log_loss(target, expit(scores/t))

        cal = minimize_scalar(objective, bounds=(.2, 12.), method="bounded").x
    elif kind == "isotonic":
        cal = IsotonicRegression(out_of_bounds="clip", y_min=.025, y_max=.975).fit(scores, target)
    else:
        raise ValueError(f"Unregistered calibration method {kind}")
    return Calibration(kind, cal, float(np.mean(target)))


def ece(y, p):
    groups = np.clip((p*10).astype(int), 0, 9)
    return float(sum(np.mean(groups == slot) * abs(y[groups == slot].mean()-p[groups == slot].mean())
                     for slot in np.unique(groups)))


def threshold(y, prob):
    values = np.unique(np.r_[.5, np.quantile(prob, np.linspace(.12, .88, 45))])
    return float(max(values, key=lambda val: (balanced_accuracy_score(y, prob >= val), -abs(val-.5))))


def diagnostics(y, prob, decision):
    from sklearn.metrics import average_precision_score

    return {"auroc": float(roc_auc_score(y, prob)), "auprc": float(average_precision_score(y, prob)),
            "balanced_accuracy": float(balanced_accuracy_score(y, decision)),
            "brier": float(brier_score_loss(y, prob)), "ece": ece(y, prob)}


def calibrate(scores, target, group):
    metrics, variants = {}, {}
    for method in METHODS:
        out = np.empty(len(target))
        simulation = []
        for fit_index, held in GroupKFold(5).split(scores, groups=group):
            fitted = fit(method, scores[fit_index], target[fit_index])
            out[held], _ = fitted.probabilities(scores[held], estimate_prior=True)
            for desired in (.30, .45, .72):
                # The synthetic shift is constructed only within held development rows.
                rng = np.random.default_rng(34401 + len(held) + round(desired*100))
                yes, no = held[target[held] == 1], held[target[held] == 0]
                n = max(30, len(held))
                if not len(yes) or not len(no):
                    continue
                simulated = np.r_[rng.choice(yes, round(n*desired), replace=True),
                                 rng.choice(no, max(1, round(n*(1-desired))), replace=True)]
                p, _ = fitted.probabilities(scores[simulated], estimate_prior=True)
                simulation.append(ece(target[simulated], p))
        t = threshold(target, out)
        result = diagnostics(target, out, out >= t)
        result["shift_ece"] = float(np.mean(simulation))
        metrics[method] = result
        variants[method] = out, t
    winner = min(METHODS, key=lambda k: metrics[k]["brier"] + .18*metrics[k]["ece"]
                 + .18*metrics[k]["shift_ece"])
    forecast, cutoff = variants[winner]
    fitted = fit(winner, scores, target)
    fitted.threshold = cutoff
    return fitted, metrics, forecast


def risk_coverage(y, probability, decision, neighbor):
    order = np.argsort(-neighbor, kind="stable")
    result = []
    for n in (1., .9, .8, .7, .6, .5):
        keep = order[:max(1, round(len(order)*n))]
        if len(np.unique(y[keep])) > 1:
            result.append({"coverage": n, "n": int(len(keep)), "tanimoto_min": float(neighbor[keep].min()),
                           **diagnostics(y[keep], probability[keep], decision[keep])})
    return result
