"""Nested family-group Ridge used identically for every representation.

`fit_predict` receives training rows only (features, labels, family ids) plus the feature rows to
predict. Test labels never enter this module, so alpha selection, scalers, PCA and label scaling
cannot use them.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core import family_weights, group_folds

FEATURE_KINDS = ("A1", "A1r", "A2", "A3", "A4")


def weighted_mean_sd(Y: np.ndarray, w: np.ndarray) -> tuple:
    w = w / w.sum()
    mu = (w[:, None] * Y).sum(axis=0)
    sd = np.sqrt((w[:, None] * (Y - mu) ** 2).sum(axis=0))
    return mu, sd


def family_first_mean(values: np.ndarray, families: np.ndarray) -> np.ndarray:
    """Mean over families of the within-family mean (rows x columns -> columns)."""
    fam, inv = np.unique(families, return_inverse=True)
    out = np.zeros((len(fam), values.shape[1]))
    np.add.at(out, inv, values)
    out /= np.bincount(inv)[:, None]
    return out.mean(axis=0)


def make_transformer(kind: str, pca_dim: int | None) -> Pipeline:
    if kind == "A2":
        return Pipeline([("scale", StandardScaler()),
                         ("pca", PCA(n_components=pca_dim, svd_solver="full")),
                         ("scale2", StandardScaler())])
    assert kind in FEATURE_KINDS
    return Pipeline([("scale", StandardScaler())])


def fit_transformer(kind: str, pca_dim, X: np.ndarray, w: np.ndarray) -> Pipeline:
    t = make_transformer(kind, pca_dim)
    params = {"scale__sample_weight": w}
    if kind == "A2":
        params["scale2__sample_weight"] = w
    return t.fit(X, **params)


def centred_rank(X: np.ndarray) -> int:
    Xc = X - X.mean(axis=0)
    return int(np.linalg.matrix_rank(Xc @ Xc.T))


def fit_predict(X_tr, Y_tr, fam_tr, X_te, kind, alphas, n_inner, inner_seed, K, const_sd=1e-8):
    """Returns predictions (rows of X_te x drugs) and a dict of per-drug diagnostics.
    Y_tr must be complete (callers group drugs by their observed-row pattern)."""
    assert not np.isnan(Y_tr).any()
    fam_tr = np.asarray(fam_tr)
    alphas = list(alphas)
    w = family_weights(fam_tr)
    mu, sd = weighted_mean_sd(Y_tr, w)
    constant = sd < const_sd
    if kind == "A0":
        return np.tile(mu, (X_te.shape[0], 1)), dict(mu=mu, sd=sd, alpha=np.full(len(mu), np.nan),
                                                    constant=constant, pca_dim=None)
    inner = group_folds(fam_tr, n_inner, inner_seed)
    pca_dim = None
    if kind == "A2":
        n_min = min(int((inner != k).sum()) for k in range(n_inner))
        ranks = [centred_rank(X_tr[inner != k]) for k in range(n_inner)] + [centred_rank(X_tr)]
        pca_dim = int(min(K, n_min - 1, min(ranks)))
    loss = np.zeros((len(alphas), Y_tr.shape[1]))
    n_fam_total = 0
    for k in range(n_inner):
        tr, te = inner != k, inner == k
        assert not set(fam_tr[tr]) & set(fam_tr[te])
        w_in = family_weights(fam_tr[tr])
        mu_in, sd_in = weighted_mean_sd(Y_tr[tr], w_in)
        sd_safe = np.where(sd_in < const_sd, 1.0, sd_in)
        Z_tr, Z_te = (Y_tr[tr] - mu_in) / sd_safe, (Y_tr[te] - mu_in) / sd_safe
        tf = fit_transformer(kind, pca_dim, X_tr[tr], w_in)
        T_tr, T_te = tf.transform(X_tr[tr]), tf.transform(X_tr[te])
        n_fam = len(np.unique(fam_tr[te]))
        for a, alpha in enumerate(alphas):
            P = Ridge(alpha=alpha).fit(T_tr, Z_tr, sample_weight=w_in).predict(T_te)
            loss[a] += family_first_mean((P - Z_te) ** 2, fam_tr[te]) * n_fam
        n_fam_total += n_fam
    loss /= n_fam_total
    # smallest loss; ties resolved towards the largest alpha
    best = len(alphas) - 1 - np.argmin(loss[::-1], axis=0)
    sd_safe = np.where(constant, 1.0, sd)
    Z = (Y_tr - mu) / sd_safe
    tf = fit_transformer(kind, pca_dim, X_tr, w)
    T_tr, T_te = tf.transform(X_tr), tf.transform(X_te)
    pred = np.empty((X_te.shape[0], Y_tr.shape[1]))
    for a, alpha in enumerate(alphas):
        cols = np.where(best == a)[0]
        if len(cols) == 0:
            continue
        P = Ridge(alpha=alpha).fit(T_tr, Z[:, cols], sample_weight=w).predict(T_te)
        pred[:, cols] = P.reshape(X_te.shape[0], len(cols)) * sd_safe[cols] + mu[cols]
    return pred, dict(mu=mu, sd=sd, alpha=np.array(alphas, float)[best], constant=constant, pca_dim=pca_dim,
                      inner_loss=loss)


def fit_predict_missing(X_tr, Y_tr, fam_tr, X_te, **kw):
    """Wrapper for label matrices with missing entries: drugs sharing an observed-row pattern are
    fitted together on exactly those rows. Missing labels are never imputed."""
    n_d = Y_tr.shape[1]
    pred = np.full((X_te.shape[0], n_d), np.nan)
    info = dict(mu=np.full(n_d, np.nan), sd=np.full(n_d, np.nan), alpha=np.full(n_d, np.nan),
                constant=np.zeros(n_d, bool), n_train=np.zeros(n_d, int), pca_dim=[None] * n_d)
    obs = ~np.isnan(Y_tr)
    patterns: dict = {}
    for j in range(n_d):
        patterns.setdefault(obs[:, j].tobytes(), []).append(j)
    fam_tr = np.asarray(fam_tr)
    for cols in patterns.values():
        rows = obs[:, cols[0]]
        if len(np.unique(fam_tr[rows])) < 2 * kw["n_inner"]:
            continue                                    # too few labelled families: left NaN, reported
        p, i = fit_predict(X_tr[rows], Y_tr[np.ix_(rows, cols)], fam_tr[rows], X_te, **kw)
        pred[:, cols] = p
        for key in ("mu", "sd", "alpha", "constant"):
            info[key][cols] = i[key]
        info["n_train"][cols] = int(rows.sum())
        for c in cols:
            info["pca_dim"][c] = i["pca_dim"]
    return pred, info


def outer_folds(X, Y, fam, fold, kind, alphas, n_inner, inner_seed_of, K, const_sd=1e-8, train_subset=None):
    """Yield (fold k, train row indices, test row indices, predictions, info) for each outer fold.
    Only training rows of Y are passed on. `train_subset(k, train_idx)` may shrink the training set."""
    fam, fold = np.asarray(fam), np.asarray(fold)
    for k in sorted(set(fold.tolist())):
        tr_idx, te_idx = np.where(fold != k)[0], np.where(fold == k)[0]
        assert not set(fam[tr_idx]) & set(fam[te_idx]), "family in both train and test"
        if train_subset is not None:
            tr_idx = train_subset(k, tr_idx)
            assert set(tr_idx) <= set(np.where(fold != k)[0])
        pred, info = fit_predict_missing(X[tr_idx], Y[tr_idx], fam[tr_idx], X[te_idx], kind=kind, alphas=alphas,
                                         n_inner=n_inner, inner_seed=inner_seed_of(k), K=K, const_sd=const_sd)
        yield k, tr_idx, te_idx, pred, info
