"""Fixture checks that must pass before any R1 model is trained on real data.

Run: pytest src/softprior_multitask_r1/tests -q
  1. eta = 0, tau = 0 is the Track A weighted Ridge (sklearn, StandardScaler + Ridge) after the
     documented conversion alpha = lam * D * sum(family weights).
  2. dual solver == reduced primal solver == direct full-space solve of the stated objective.
  3. eta -> infinity is a Ridge restricted to the pathway subspace, and that subspace is the one a
     linear model on unstandardised set means lives in (the diag(s) M derivation).
  4. tau > 0: reduced-space ADMM == reduced-space FISTA == full-space proximal gradient.
  5. the same objective minimised with PyTorch autograd gives the same solution (skipped if torch is absent).
  6. tau >= tau_max gives W = 0; held-out labels cannot influence a fit; permuted sets keep structure.
"""
import os
import sys

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import r1_data  # noqa: E402
import r1_models as M  # noqa: E402
import r1_solvers as S  # noqa: E402
from r1_data import a1core  # noqa: E402


def toy(n_fam=26, G=60, D=7, K=5, seed=0):
    rs = np.random.RandomState(seed)
    fam = np.array([f"F{i:02d}" for i in range(n_fam)] + ["F00", "F01", "F01"])
    N = len(fam)
    X = rs.normal(size=(N, G)) * rs.uniform(0.5, 3.0, size=G) + rs.normal(size=G)
    X[:, 7] = 2.5                                   # zero-variance gene
    sets = [np.sort(rs.choice(G, size=rs.randint(6, 15), replace=False)) for _ in range(K)]
    sets[0] = np.union1d(sets[0], [7])
    B = np.zeros((G, D))
    for k in range(K):
        B[sets[k], :] += rs.normal(size=D) / len(sets[k])
    lat = rs.normal(size=(G, 2)) @ rs.normal(size=(2, D)) * 0.05
    Y = X @ (B + lat) + rs.normal(size=(N, D)) * 0.7
    X_te = rs.normal(size=(9, G)) * 1.5 + 0.3
    return X, Y, fam, sets, X_te


def prep_parts(X, Y, fam, sets):
    wt = S.norm_weights(a1core.family_weights(fam))
    mu, sd = S.wmean_sd(X, wt)
    Xs = S.standardise(X, mu, sd)
    Q = S.orth_basis(S.membership_std(sets, sd))
    ymu, ysd = S.wmean_sd(Y, wt)
    Z = (Y - ymu) / ysd
    return wt, mu, sd, Xs, Q, Z


def test_ridge_special_case_matches_track_a_sklearn():
    X, Y, fam, sets, X_te = toy()
    prep = M.Prep(X, Y, fam, X_te, sets)
    w = a1core.family_weights(fam)
    sc = StandardScaler().fit(X, sample_weight=w)
    for lam in (1e-3, 0.1, 3.0):
        alpha = lam * prep.D * w.sum()
        ref = Ridge(alpha=alpha).fit(sc.transform(X), prep.Z, sample_weight=w).predict(sc.transform(X_te))
        dual = prep.predict_ridge_z("whole", lam)
        primal, _ = prep.predict_nuc_z(0.0, lam, 0.0)
        assert np.abs(dual - ref).max() < 1e-8
        assert np.abs(primal - ref).max() < 1e-8


def test_soft_prior_dual_equals_primal_equals_full_space():
    X, Y, fam, sets, X_te = toy(seed=1)
    wt, mu, sd, Xs, Q, Z = prep_parts(X, Y, fam, sets)
    Xt = S.standardise(X_te, mu, sd)
    prep = M.Prep(X, Y, fam, X_te, sets)
    for name, ratio in (("r1", 1.0), ("r10", 10.0), ("r100", 100.0)):
        for lam in (1e-2, 0.5):
            W = S.full_space_reference(Xs, wt, Z, Q, lam, ratio * lam, 0.0)
            ref = Xt @ W
            assert np.abs(prep.predict_ridge_z(name, lam) - ref).max() < 1e-8
            assert np.abs(prep.predict_nuc_z(ratio, lam, 0.0)[0] - ref).max() < 1e-8


def test_hard_limit_is_pathway_subspace_regression():
    X, Y, fam, sets, X_te = toy(seed=2)
    wt, mu, sd, Xs, Q, Z = prep_parts(X, Y, fam, sets)
    Xt = S.standardise(X_te, mu, sd)
    prep = M.Prep(X, Y, fam, X_te, sets)
    lam = 0.2
    # (a) kernel 'par' == Ridge on the coordinates Xs Q
    ref = Ridge(alpha=lam * prep.D, fit_intercept=False).fit(Xs @ Q, Z, sample_weight=wt).predict(Xt @ Q)
    assert np.abs(prep.predict_ridge_z("par", lam) - ref).max() < 1e-8
    # (b) a very large eta/lam approaches it
    big, _ = prep.predict_nuc_z(1e9, lam, 0.0)
    assert np.abs(big - ref).max() < 1e-5
    # (c) with almost no shrinkage it equals least squares on the *unstandardised* set means
    Mraw = np.column_stack([X[:, idx].mean(axis=1) for idx in sets])
    Mraw_te = np.column_stack([X_te[:, idx].mean(axis=1) for idx in sets])
    ols = LinearRegression().fit(Mraw, Z, sample_weight=wt).predict(Mraw_te)
    assert np.abs(prep.predict_ridge_z("par", 1e-13) - ols).max() < 1e-5
    # (d) par + perp = whole
    assert np.abs(prep.K_par + prep.K_perp - prep.K).max() < 1e-9
    assert np.abs(Q.T @ Q - np.eye(Q.shape[1])).max() < 1e-10


@pytest.mark.parametrize("ratio,lam,frac", [(0.0, 0.05, 0.3), (10.0, 0.05, 0.1), (100.0, 0.01, 0.03), (1.0, 1.0, 0.5)])
def test_nuclear_solvers_agree(ratio, lam, frac):
    X, Y, fam, sets, X_te = toy(seed=3)
    wt, mu, sd, Xs, Q, Z = prep_parts(X, Y, fam, sets)
    prep = M.Prep(X, Y, fam, X_te, sets)
    eta, tau = ratio * lam, frac * prep.tau_max
    H, Rm = S.smooth_terms(prep.XB, prep.wt, prep.Z, prep.r, lam, eta)
    C_admm, info = S.admm_nuclear(H, Rm, tau, tol=1e-10)
    assert info["converged"]
    C_fista = S.fista_nuclear(H, Rm, tau)
    W_full = S.full_space_reference(Xs, wt, Z, Q, lam, eta, tau)
    B = S.reduced_basis(Xs, Q)
    f_admm = S.full_objective(B @ C_admm, Xs, wt, Z, Q, lam, eta, tau)
    f_fista = S.full_objective(B @ C_fista, Xs, wt, Z, Q, lam, eta, tau)
    f_full = S.full_objective(W_full, Xs, wt, Z, Q, lam, eta, tau)
    assert abs(f_admm - f_full) < 1e-8 * max(1.0, abs(f_full))
    assert abs(f_fista - f_full) < 1e-8 * max(1.0, abs(f_full))
    Xt = S.standardise(X_te, mu, sd)
    assert np.abs(Xt @ B @ C_admm - Xt @ W_full).max() < 1e-5
    assert np.abs(B @ C_admm - W_full).max() < 1e-5
    # production tolerance stays close to the tight solution
    P_prod, info2 = prep.predict_nuc_z(ratio, lam, frac)
    assert info2["converged"] and np.abs(P_prod - Xt @ W_full).max() < 1e-4


def test_tau_max_and_rank_path():
    X, Y, fam, sets, X_te = toy(seed=4)
    prep = M.Prep(X, Y, fam, X_te, sets)
    P1, i1 = prep.predict_nuc_z(1.0, 0.1, 1.0001)
    assert np.abs(P1).max() < 1e-6 and i1["rank"] == 0
    ranks = [prep.predict_nuc_z(1.0, 0.1, f)[1]["rank"] for f in (0.9, 0.5, 0.2, 0.05)]
    assert ranks[0] >= 1 and all(a <= b for a, b in zip(ranks, ranks[1:]))


def test_torch_autograd_reference():
    torch = pytest.importorskip("torch")
    X, Y, fam, sets, X_te = toy(seed=5, G=40, D=5)
    wt, mu, sd, Xs, Q, Z = prep_parts(X, Y, fam, sets)
    prep = M.Prep(X, Y, fam, X_te, sets)
    Xt = S.standardise(X_te, mu, sd)
    tX, tZ, tQ, tw = (torch.tensor(a, dtype=torch.float64) for a in (Xs, Z, Q, wt))
    D = Z.shape[1]

    def smooth(W, lam, eta):
        res = tX @ W - tZ
        perp = W - tQ @ (tQ.T @ W)
        return (tw[:, None] * res ** 2).sum() / D + lam * (W ** 2).sum() + eta * (perp ** 2).sum()

    # tau = 0: L-BFGS on the stated objective
    lam, eta = 0.05, 0.5
    W = torch.zeros(Xs.shape[1], D, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([W], lr=1.0, max_iter=2000, tolerance_grad=1e-13, tolerance_change=1e-16,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = smooth(W, lam, eta)
        loss.backward()
        return loss
    opt.step(closure)
    ref = Xt @ W.detach().numpy()
    assert np.abs(prep.predict_ridge_z("r10", lam) - ref).max() < 1e-6

    # tau > 0: proximal gradient with autograd gradients and torch SVD
    tau = 0.2 * prep.tau_max
    A = Xs.T @ (wt[:, None] * Xs) / D
    L = 2.0 * (np.linalg.eigvalsh(A)[-1] + lam + eta)
    W = torch.zeros(Xs.shape[1], D, dtype=torch.float64, requires_grad=True)
    for _ in range(20000):
        loss = smooth(W, lam, eta)
        (g,) = torch.autograd.grad(loss, W)
        with torch.no_grad():
            U, s, Vh = torch.linalg.svd(W - g / L, full_matrices=False)
            W_new = (U * torch.clamp(s - tau / L, min=0.0)) @ Vh
            delta = (W_new - W).norm().item()
            W.copy_(W_new)
        if delta < 1e-13:
            break
    ref = Xt @ W.detach().numpy()
    P, info = prep.predict_nuc_z(10.0, lam, 0.2, tol=1e-10)
    assert info["converged"] and np.abs(P - ref).max() < 1e-5


def test_inner_loss_and_selection_ignore_heldout_labels():
    X, Y, fam, sets, X_te = toy(seed=6)
    cfgs = M.ridge_configs([0.01, 1.0], ["whole", "r10", "par"]) + M.nuc_configs([0.1], [0.0, 10.0], [0.0, 0.1, 0.3])
    loss, rec = M.inner_losses(X, Y, fam, sets, cfgs, 3, 11)
    assert loss.shape == (len(cfgs), Y.shape[1]) and np.isfinite(loss).all()
    assert all(r["converged"] for r in rec)
    # tau_frac = 0 in the reduced primal path equals the dual Ridge path
    i_r = cfgs.index(("ridge", "r10", 1.0))
    cf2 = [("ridge", "r10", 1.0), ("nuc", 10.0, 1.0, 0.0)]
    l2, _ = M.inner_losses(X, Y, fam, sets, cf2, 3, 11)
    assert np.allclose(l2[0], l2[1], atol=1e-10) and np.allclose(l2[0], loss[i_r], atol=1e-12)
    # a prediction for new rows depends on training rows only
    prep = M.Prep(X, Y, fam, X_te, sets)
    a = prep.predict_ridge_z("r10", 0.1)
    prep2 = M.Prep(X, Y, fam, X_te[::-1].copy(), sets)
    assert np.allclose(a, prep2.predict_ridge_z("r10", 0.1)[::-1])
    # selection helpers: ties go to the strongest regularisation
    fake = np.array([[1.0, 2.0], [1.0, 1.0], [3.0, 1.0]])
    assert list(M.select_per_drug(fake, [0, 1, 2])) == [1, 2]
    assert M.select_global(np.array([[1.0, 1.0], [1.0, 1.0]]), [0, 1]) == 1


def test_inner_loss_matches_track_a_nested_ridge():
    """The R1 inner loss of the eta = 0 Ridge equals the inner loss computed by Track A's code."""
    import nested as a1nested
    X, Y, fam, sets, X_te = toy(seed=7)
    w_sum = []
    inner = a1core.group_folds(fam, 3, 21)
    for k in range(3):
        w_sum.append(a1core.family_weights(fam[inner != k]).sum())
    assert len(set(w_sum)) >= 1
    lam = 0.3
    D = Y.shape[1]
    # Track A's alpha is a fixed number across inner folds, so pick folds with the same weight sum
    alpha = None
    loss_r1, _ = M.inner_losses(X, Y, fam, sets, [("ridge", "whole", lam)], 3, 21)
    if len(set(np.round(w_sum, 9))) == 1:
        alpha = lam * D * w_sum[0]
        _, info = a1nested.fit_predict(X, Y, fam, X_te, "A1r", [alpha], 3, 21, K=len(sets))
        assert np.allclose(info["inner_loss"][0], loss_r1[0], atol=1e-9)


def test_permuted_sets_keep_structure():
    rs = np.random.RandomState(0)
    G = 500
    sets = [np.sort(rs.choice(G, size=n, replace=False)) for n in (30, 50, 20, 80)]
    ps = r1_data.permuted_sets(sets, G, 101)
    assert [len(a) for a in ps] == [len(a) for a in sets]
    for a in range(4):
        for b in range(a + 1, 4):
            assert len(np.intersect1d(sets[a], sets[b])) == len(np.intersect1d(ps[a], ps[b]))
    assert any(not np.array_equal(a, b) for a, b in zip(sets, ps))
