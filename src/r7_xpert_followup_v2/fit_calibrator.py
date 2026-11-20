"""Cell-line affine output calibration z = a_c * prediction + b_c (private plan 2026-11-19 section 4.2). numpy only, float64.

a_c > 0 is one scalar per cell line, b_c a 978-gene vector. Arms:
  C0  identity                      a = 1, b = 0
  Cg  global                        a = exp(alpha0), b = b0                (one pair for every cell line)
  Cs  per-cell scale                a_c = exp(alpha0 + v_c), b = 0
  Cb  per-cell offset               a = 1, b_c = b0 + u_c
  Csb per-cell scale + offset       a_c = exp(alpha0 + v_c), b_c = b0 + u_c
Two stages, all in units of one fixed RMS s of the fitting truths (s = sqrt(mean y^2) over the fitting source units):
  global prior (source = non-HCC units only, family-equal weights w_u = N_src / (n_families * n_units_of_family), so the
  weights sum to N_src):  J_g = sum_u w_u mean_g (a p + b0 - y)^2 + lam * alpha0^2 + lam * ||b0||^2 / G
  cell stage (every cell line of the fitting set, HCC train-drug OOF pairs included; global prior fixed):
                          J_c = sum_{u in c} mean_g (a_c p + b0 + u_c - y)^2 + lam * v_c^2 + lam * ||u_c||^2 / G
  (sum over the cell's units, not divided by its drug count: few drugs -> the prior dominates.)
For a fixed scale the offset is closed form, b = (m_y - a m_p) / (W + lam); the profile objective in the log-scale is
1-D and is minimised by a grid on [-6, 6] (step 0.01) plus golden-section refinement. lam = inf is the identity.
A cell line without fitting units falls back to the global prior (flagged)."""
import numpy as np

ARMS = ("C0", "Cg", "Cs", "Cb", "Csb")
LAMBDAS = (0.01, 0.1, 1.0, 10.0, 100.0)          # plus 'identity' (lam = inf), pre-registered grid
TIE_TOL = 1e-6                                    # CV RMSE within this of the minimum = tie -> strongest shrinkage
VGRID = np.round(np.arange(-6.0, 6.0 + 1e-9, 0.01), 2)
SPEC = dict(Cg=(True, True, False, False), Cs=(True, False, True, False), Cb=(False, True, False, True), Csb=(True, True, True, True))
#            global scale, global offset, cell scale, cell offset


def _stats(p, y, w):
    W = float(w.sum())
    q = dict(W=W, Spp=float(w @ (p * p).sum(1)), Spy=float(w @ (p * y).sum(1)), Syy=float(w @ (y * y).sum(1)))
    mp = w @ p; my = w @ y
    q.update(mp=mp, my=my, Mpp=float(mp @ mp), Mpy=float(mp @ my), Myy=float(my @ my))
    return q


def _objective(q, a, lam, offset, G):
    J = q["Syy"] - 2 * a * q["Spy"] + a * a * q["Spp"]
    if offset:
        J = J - (q["Myy"] - 2 * a * q["Mpy"] + a * a * q["Mpp"]) / (q["W"] + lam)
    return J / G


def _fit_scale(q, center, lam, offset, G):
    """min over v of J(a = exp(center + v)) + lam v^2. Returns v, a, flag_at_bound."""
    a = np.exp(center + VGRID)
    f = _objective(q, a, lam, offset, G) + lam * VGRID ** 2
    i = int(np.argmin(f))
    lo, hi = VGRID[max(i - 1, 0)], VGRID[min(i + 1, len(VGRID) - 1)]
    g = lambda v: _objective(q, np.exp(center + v), lam, offset, G) + lam * v * v
    r = (np.sqrt(5) - 1) / 2
    x1, x2 = hi - r * (hi - lo), lo + r * (hi - lo)
    f1, f2 = g(x1), g(x2)
    for _ in range(80):
        if f1 < f2:
            hi, x2, f2 = x2, x1, f1; x1 = hi - r * (hi - lo); f1 = g(x1)
        else:
            lo, x1, f1 = x1, x2, f2; x2 = lo + r * (hi - lo); f2 = g(x2)
    v = (lo + hi) / 2
    if g(v) > f[i]:
        v = float(VGRID[i])
    return float(v), float(np.exp(center + v)), bool(i in (0, len(VGRID) - 1))


def family_weights(fam):
    uf, inv, cnt = np.unique(fam, return_inverse=True, return_counts=True)
    return len(fam) / (len(uf) * cnt[inv])


def fit(arm, lam, P, Y, cells, fam, hcc):
    """P, Y: (n, G) predictions / truths of the fitting set (OOF). cells, fam: labels; hcc: bool per unit."""
    G = P.shape[1]
    src = ~np.asarray(hcc, bool)
    s = float(np.sqrt((Y[src] ** 2).mean()))
    coef = dict(arm=arm, lam=lam, s=s, alpha0=0.0, b0=np.zeros(G), cells={}, flags=[])
    if arm == "C0" or not np.isfinite(lam):
        coef["identity"] = True
        return coef
    coef["identity"] = False
    gs, go, cs, co = SPEC[arm]
    p = P[src] / s; y = Y[src] / s
    w = family_weights(np.asarray(fam)[src])
    q = _stats(p, y, w)
    if gs:
        v, a0, edge = _fit_scale(q, 0.0, lam, go, G)
        coef["alpha0"] = v
        if edge:
            coef["flags"].append("global_scale_at_grid_bound")
    a0 = np.exp(coef["alpha0"])
    if go:
        coef["b0"] = (q["my"] - a0 * q["mp"]) / (q["W"] + lam) * s
    coef["global_effective_n"] = q["W"]
    if arm == "Cg":
        return coef
    cells = np.asarray(cells)
    for c in np.unique(cells):
        k = cells == c
        p = P[k] / s; y = (Y[k] - coef["b0"]) / s
        q = _stats(p, y, np.ones(k.sum()))
        if cs:
            v, a, edge = _fit_scale(q, coef["alpha0"], lam, co, G)
        else:
            v, a, edge = 0.0, a0, False
        u = (q["my"] - a * q["mp"]) / (q["W"] + lam) * s if co else np.zeros(G)
        coef["cells"][c] = dict(v=v, a=a, b=coef["b0"] + u, n_units=int(k.sum()), u_rms=float(np.sqrt((u ** 2).mean())), at_bound=edge)
    return coef


def apply(coef, P, cells, hcc_global_only=None):
    """Calibrated predictions. Cell lines absent from the fitting set use the global prior (returned as fallback mask).
    hcc_global_only: optional bool mask of rows that must use the global prior (HCC no-label diagnostic)."""
    P = np.asarray(P, np.float64)
    if coef["identity"]:
        return P.copy(), np.zeros(len(P), bool)
    cells = np.asarray(cells)
    a0 = np.exp(coef["alpha0"])
    Z = a0 * P + coef["b0"]
    fb = np.ones(len(P), bool)
    if coef["arm"] != "Cg":
        for c, d in coef["cells"].items():
            k = cells == c
            if hcc_global_only is not None:
                k &= ~hcc_global_only
            Z[k] = d["a"] * P[k] + d["b"]; fb[k] = False
        if hcc_global_only is not None:
            fb &= ~hcc_global_only
    else:
        fb[:] = False
    return Z, fb


def unit_rmse(Z, Y):
    return np.sqrt(((Z - Y) ** 2).mean(1))


def fam_mean(v, fam):
    uf, inv = np.unique(fam, return_inverse=True)
    return float((np.bincount(inv, weights=v) / np.bincount(inv)).mean())


def select_lambda(arm, P, Y, cells, fam, hcc, cvfold):
    """Drug-group calibration CV inside the fitting set (cvfold = inner fold of each unit's drug group).
    Score = family-equal mean unit RMSE over source units of the pooled held-out calibrated predictions."""
    if arm == "C0":
        return float("inf"), {"identity": None}
    grid = [float("inf")] + sorted(LAMBDAS, reverse=True)          # strongest shrinkage first (tie order)
    src = ~np.asarray(hcc, bool)
    scores = {}
    for lam in grid:
        Z = np.empty_like(P)
        for f in np.unique(cvfold):
            te = cvfold == f; trm = ~te
            c = fit(arm, lam, P[trm], Y[trm], cells[trm], fam[trm], hcc[trm])
            Z[te], _ = apply(c, P[te], cells[te])
        scores[lam] = fam_mean(unit_rmse(Z[src], Y[src]), fam[src])
    best = min(scores.values())
    chosen = next(l for l in grid if scores[l] <= best + TIE_TOL)
    return chosen, {("identity" if not np.isfinite(k) else k): v for k, v in scores.items()}


def coef_summary(coef):
    if coef["identity"]:
        return dict(arm=coef["arm"], identity=True)
    out = dict(arm=coef["arm"], lam=coef["lam"], s=coef["s"], a0=float(np.exp(coef["alpha0"])), b0_rms=float(np.sqrt((coef["b0"] ** 2).mean())),
               flags=list(coef["flags"]))
    if coef["cells"]:
        a = np.array([d["a"] for d in coef["cells"].values()])
        out.update(n_cells=len(a), a_min=float(a.min()), a_median=float(np.median(a)), a_max=float(a.max()), all_a_positive=bool((a > 0).all()),
                   cells_at_bound=[c for c, d in coef["cells"].items() if d["at_bound"]],
                   u_rms_median=float(np.median([d["u_rms"] for d in coef["cells"].values()])),
                   n_units_min=int(min(d["n_units"] for d in coef["cells"].values())))
    return out


def coef_hash(coef):
    import hashlib
    h = hashlib.sha256()
    h.update(repr((coef["arm"], coef["lam"], coef["identity"])).encode())
    if not coef["identity"]:
        h.update(np.float64(coef["alpha0"]).tobytes()); h.update(np.asarray(coef["b0"], np.float64).tobytes())
        for c in sorted(coef["cells"]):
            d = coef["cells"][c]
            h.update(c.encode()); h.update(np.float64(d["a"]).tobytes()); h.update(np.asarray(d["b"], np.float64).tobytes())
    return h.hexdigest()
