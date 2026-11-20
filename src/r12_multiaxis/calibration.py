"""Closed-form, group-weighted residual calibration on parent-fit cross-predictions."""

import numpy as np

from src.r11_chemical_probe.dataset import LINE, group_weights


def shrink_fit(x, target, w, scale, penalty, prior, family):
    a0, b0 = prior
    w = w / w.sum()
    mean_x = w @ x
    mean_t = w @ target
    if family != "S1":
        xx = np.dot(w, np.square(x).mean(axis=1))
        xt = np.dot(w, (x * target).mean(axis=1))
        if family == "S3":
            xx -= np.square(mean_x).mean() / (1 + penalty)
            xt -= np.mean(mean_x * (mean_t + penalty * b0)) / (1 + penalty)
        a = max(0.0, (xt / scale**2 + penalty * a0) / (xx / scale**2 + penalty))
    else:
        a = 1.0
    b = np.zeros_like(mean_t) if family == "S2" else (mean_t - a * mean_x + penalty * b0) / (1 + penalty)
    return a, b


class Calibration:
    def __init__(self, rows, base, anchors, observed, scale, penalty, family):
        self.coef = {}
        self.family = family
        self.penalty = penalty
        self.parent_n = len(rows)
        self.scale = scale
        prior = (1.0, np.zeros(observed.shape[1]))
        self.global_coef = prior
        if penalty == "identity":
            return
        x = base.astype(np.float64) - anchors
        t = observed.astype(np.float64) - anchors
        if family == "S1" or np.max(np.abs(x)) < 1e-12:
            self.family = "S1"
        self.global_coef = shrink_fit(x, t, group_weights(rows), scale, penalty, prior, self.family)
        for line, pos in rows.groupby(LINE, sort=True).indices.items():
            self.coef[line] = shrink_fit(x[pos], t[pos], group_weights(rows.iloc[pos]), scale,
                                         penalty, self.global_coef, self.family)

    def predict(self, rows, base, anchor):
        out = base.copy()
        if self.penalty == "identity":
            return out
        for line, pos in rows.groupby(LINE, sort=False).indices.items():
            a, b = self.coef.get(line, self.global_coef)
            out[pos] = anchor[pos] + a * (base[pos] - anchor[pos]) + b
        return out

    def record(self):
        global_a, global_b = self.global_coef
        return dict(family=self.family, penalty=self.penalty, parent_rows=self.parent_n, scale=self.scale,
                    global_scale=global_a, global_offset_rms=float(np.sqrt(np.mean(global_b**2))),
                    line_scale={line: a for line, (a, _) in self.coef.items()},
                    line_offset_rms={line: float(np.sqrt(np.mean(b**2))) for line, (_, b) in self.coef.items()})

    def vectors(self):
        lines = sorted(self.coef)
        coefs = [self.global_coef] + [self.coef[line] for line in lines]
        return dict(line=np.asarray(["<global>", *lines]), a=np.array([a for a, _ in coefs]),
                    b=np.stack([b for _, b in coefs]))
