"""Follow-up predictors. The original methods.py is imported unchanged (V0 reproduction uses it as is); everything new is here.

Notation (one episode, K supports, Q queries, 978 genes):
  f*        shrunk same-drug anchor, f0 * n / (n + kappa_f)
  E         Y_S - f*_S                          support residuals
  Omega     observation covariance (identity = iid; otherwise methods.omega with a covariance version V0..V3)
  w         Omega^-1 1 / 1' Omega^-1 1;  K_eff = 1' Omega^-1 1;  alpha(k) = k / (k + kappa_b)
  C0 paths  iid: alpha(K) mean(E) | weights-only: alpha(K) w'E | shrink-only: alpha(K_eff) mean(E) | both: alpha(K_eff) w'E
  C1        f* + b + eta * Kqs_c (Kss_c + lambda Omega)^-1 (E - w'E)     (kernel centred on the w-weighted support mean)
  J1        f* + a mean(E) + c (o_qS E) / K              o_qS = W_q W_S' (query-support plate overlap; planned metadata)
  J2/J3     f* + C_QS C_SS^-1 E with one joint covariance  tau_b 11' + tau_k K + v_s W W'  (+ D_S on the support block)
No query label enters any function here; truth is only used by score().
"""
import math
import numpy as np
import torch

from src.replicate_calibrated_r8 import methods as M

INF = float("inf")


def is_inf(x):
    return isinstance(x, str) or (isinstance(x, float) and math.isinf(x))


def alpha(k, kappa_b):
    return 0.0 if is_inf(kappa_b) else float(k / (k + kappa_b))


def fstar(inp, kappa_f):
    """Same rule as methods.anchor (kappa_f = 0 -> f0 where n > 0, zero vector where n = 0)."""
    f = lambda n: (n / (n + kappa_f)).nan_to_num(0.0)[:, None] if kappa_f > 0 else (n > 0).float()[:, None]
    return inp["f0_s"] * f(inp["n0_s"]), inp["f0_q"] * f(inp["n0_q"])


def omega(inp, cov):
    return M.omega(inp, cov)


def anchor_om(inp, kappa_f, kappa_b, Om):
    """methods.anchor with an explicit Omega (identical arithmetic; lets a rescaled Omega be passed)."""
    fs, fq = fstar(inp, kappa_f)
    E = inp["y_s"] - fs
    L = torch.linalg.cholesky(Om)
    a = torch.cholesky_solve(torch.ones(len(E), 1, device=E.device, dtype=torch.float64), L)[:, 0]
    keff = a.sum()
    w = (a / keff).float()
    ebar = (w[:, None] * E).sum(0, keepdim=True)
    shrink = 0.0 if is_inf(kappa_b) else float(keff / (keff + kappa_b))
    return dict(fs=fs, fq=fq, E=E, ebar=ebar, b=ebar * shrink, R=E - ebar, Om=Om, w=w, keff=float(keff), shrink=shrink)


def omega_trace_matched(inp, cov, cov_ref):
    """V3 rescaled per episode so that its trace equals that of the reference version (structure vs overall scale)."""
    Om, Or = M.omega(inp, cov), M.omega(inp, cov_ref)
    return Om * (torch.trace(Or) / torch.trace(Om))


def gls_weights(Om):
    L = torch.linalg.cholesky(Om)
    a = torch.cholesky_solve(torch.ones(Om.shape[0], 1, dtype=Om.dtype, device=Om.device), L)[:, 0]
    keff = a.sum()
    return (a / keff), float(keff)


def omega_diag(Om, w):
    K = Om.shape[0]
    ev = torch.linalg.eigvalsh(Om)
    return dict(eig_min=float(ev.min()), eig_max=float(ev.max()), cond=float(ev.max() / ev.min()), w_min_xK=float(w.min()) * K, w_max_xK=float(w.max()) * K,
                n_negative_w=int((w < 0).sum()), trace_over_K=float(torch.trace(Om)) / K)


def c0_paths(E, Om, kappa_b):
    """The four common-correction paths of plan section 4.4 on the same E. Returns dict of (1, G) vectors and diagnostics."""
    K = E.shape[0]
    w, keff = gls_weights(Om)
    wf = w.float()
    e_i = E.mean(0, keepdim=True)
    e_o = (wf[:, None] * E).sum(0, keepdim=True)
    aI, aO = alpha(K, kappa_b), alpha(keff, kappa_b)
    out = dict(iid=e_i * aI, weights_only=e_o * aI, shrink_only=e_i * aO, both=e_o * aO)
    d = omega_diag(Om, w)
    d.update(k_eff=keff, alpha_iid=aI, alpha_omega=aO)
    return out, d


# ---------------------------------------------------------------- kernels
BLOCKS = ("chem", "ctrl", "f0")
CONFIGS = {"chem": ("chem",), "f0": ("f0",), "ctrl": ("ctrl",), "chem+f0": ("chem", "f0"), "all": ("chem", "ctrl", "f0")}


def kernel_blocks(ft):
    """Original block definitions and scales: Tanimoto(ECFP), linear(within ctrl)/32, linear(f0 repr)/33 (float32 as in methods.py)."""
    def lin(name):
        a, b = ft[name + "_s"], ft[name + "_q"]
        return a @ a.T / a.shape[1], b @ a.T / a.shape[1]
    return dict(chem=(M.tanimoto(ft["fp_s"], ft["fp_s"]), M.tanimoto(ft["fp_q"], ft["fp_s"])), ctrl=lin("ctrl"), f0=lin("f0r"))


def centred_trace_over_k(Kss):
    K = Kss.shape[0]
    H = torch.eye(K, dtype=torch.float64) - 1.0 / K
    return float(torch.trace(H @ Kss.double() @ H)) / K


def combine(blocks, config, scales=None):
    """scales None: original scale (plain sum in the original order chem + ctrl + f0).
    scales = {"block": {b: s_b}, "comb": {config: s_c}}: each block / s_b (s_b == 0 -> zero kernel), then the sum / s_c."""
    names = CONFIGS[config]
    order = [b for b in ("chem", "ctrl", "f0") if b in names]
    if scales is None:
        ss, qs = blocks[order[0]]
        for b in order[1:]:
            ss, qs = ss + blocks[b][0], qs + blocks[b][1]
        return ss, qs
    ss, qs = 0.0, 0.0
    for b in order:
        s = scales["block"][b]
        if s > 0:
            ss, qs = ss + blocks[b][0].double() / s, qs + blocks[b][1].double() / s
        else:
            ss, qs = ss + 0.0 * blocks[b][0].double(), qs + 0.0 * blocks[b][1].double()
    sc = scales["comb"][config]
    return (ss / sc, qs / sc) if sc > 0 else (ss * 0.0, qs * 0.0)


def krr(Kss, Kqs, w, R, lam, Om):
    """methods.krr_correction with explicit kernels: centre on the w-weighted support mean, Cholesky GLS solve."""
    Kss_c, Kqs_c = M.centre_kernel(Kss.double(), Kqs.double(), w)
    return M.gls_solve(Kss_c, Kqs_c, R.double(), lam, Om).float()


def influence(Kss, Kqs, w, lam, Om, eta):
    """Q x K matrix A with correction = A E (includes the residual centring R = (I - 1 w') E)."""
    K = Kss.shape[0]
    Kss_c, Kqs_c = M.centre_kernel(Kss.double(), Kqs.double(), w)
    L = torch.linalg.cholesky(Kss_c + lam * Om)
    P = torch.eye(K, dtype=torch.float64) - torch.ones(K, 1, dtype=torch.float64) @ w.double()[None, :]
    return eta * (Kqs_c @ torch.cholesky_solve(P, L))


def derangement(K, j):
    """Pre-fixed residual permutation j (1, 2, 3) for support size K: cyclic derangement of a fixed random order."""
    order = np.random.RandomState(31000 * j + K).permutation(K)
    p = np.empty(K, int)
    p[order] = np.roll(order, 1)
    assert (p != np.arange(K)).all()
    return p


def permute_inputs(inp, p):
    """Full-row permutation of the support: labels, features, metadata and overlap together (prediction must not change)."""
    out = dict(inp)
    t = torch.as_tensor(p)
    for k in ("x0_s", "f0_s", "n0_s", "y_s", "q_s"):
        out[k] = inp[k][t]
    out["g_s"] = np.asarray(inp["g_s"])[p]
    out["o_s"] = inp["o_s"][t][:, t]
    if "o_qs" in inp:
        out["o_qs"] = inp["o_qs"][:, t]
    return out


# ---------------------------------------------------------------- observation-target models
J1_GRID = [(a, c) for a in (0.0, 0.25, 0.5, 0.75, 1.0) for c in (0.0, 0.25, 0.5, 0.75, 1.0) if a + c <= 1.0 + 1e-12]


def j1(fq, E, o_qs, a, c):
    K = E.shape[0]
    return fq + a * E.mean(0, keepdim=True) + c * (o_qs.float() @ E) / K


def joint_blocks(o_ss, o_qs, q_s, tau_b, v_s, v_e, sig0, tau_k=0.0, Kss=None, Kqs=None, jitter=0.0):
    K, Q = o_ss.shape[0], o_qs.shape[0]
    d = torch.float64
    Css = tau_b * torch.ones(K, K, dtype=d) + v_s * o_ss.double() + torch.diag(sig0 + v_e * q_s.double()) + jitter * torch.eye(K, dtype=d)
    Cqs = tau_b * torch.ones(Q, K, dtype=d) + v_s * o_qs.double()
    if tau_k > 0:
        Css = Css + tau_k * Kss.double()
        Cqs = Cqs + tau_k * Kqs.double()
    return Css, Cqs


def joint(fq, E, o_ss, o_qs, q_s, tau_b, v_s, v_e, sig0, tau_k=0.0, Kss=None, Kqs=None, jitter=0.0):
    """Gaussian conditional mean with one joint working covariance. Replaces (never adds to) the common correction b_S."""
    Css, Cqs = joint_blocks(o_ss, o_qs, q_s, tau_b, v_s, v_e, sig0, tau_k, Kss, Kqs, jitter)
    if Css.abs().max() == 0:
        return fq.clone()
    L = torch.linalg.cholesky(Css)
    return fq + (Cqs @ torch.cholesky_solve(E.double(), L)).float()


def joint_full_matrix(W_s, W_q, q_s, tau_b, v_s, v_e, sig0, tau_k=0.0, K_full=None, jitter=0.0):
    """The full (K+Q) x (K+Q) joint covariance (test / PSD check): SS, QS, QQ are blocks of ONE matrix."""
    W = torch.cat([W_s, W_q]).double()
    n, K = W.shape[0], W_s.shape[0]
    C = tau_b * torch.ones(n, n, dtype=torch.float64) + v_s * W @ W.T
    if tau_k > 0:
        C = C + tau_k * K_full.double()
    D = torch.zeros(n, dtype=torch.float64)
    D[:K] = sig0 + v_e * q_s.double() + jitter
    D[K:] = sig0 + v_e * 1.0
    return C + torch.diag(D)


# ---------------------------------------------------------------- scoring
def _rowcorr(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    den = np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (a * b).sum(1) / den
    r[den < 1e-12] = np.nan
    return r


def _harm(n):
    return np.concatenate([[0.0], np.cumsum(1.0 / np.arange(1, n + 1))])


def retrieval(pred, truth, ks=(1, 5, 10)):
    """Rank of each query's own truth among the episode's query truths by cosine to its prediction.
    legacy: better + (equal + 1) / 2 (mean rank, R2 rule).  tie-corrected: RR = (1/t) sum_{j=1..t} 1/(a+j); Top-k = min(max(k-a,0),t)/t."""
    pn = np.linalg.norm(pred, axis=1, keepdims=True)
    tn = np.linalg.norm(truth, axis=1, keepdims=True)
    cos = (pred @ truth.T) / np.maximum(pn, 1e-12) / np.maximum(tn.T, 1e-12)
    cos[pn[:, 0] < 1e-12] = 0.0
    cos = np.round(cos, 10)
    own = np.diag(cos)[:, None]
    a = (cos > own).sum(1)
    t = (cos == own).sum(1)
    H = _harm(len(truth) + 1)
    rr = (H[a + t] - H[a]) / t
    legacy = 1.0 / (a + (t + 1) / 2.0)
    top = {k: np.minimum(np.maximum(k - a, 0), t) / t for k in ks}
    return rr, legacy, top


def score(pred, truth):
    """Episode-level means over the query units (truth only used here). Undefined Pearson values are NaN and counted."""
    p, t = np.asarray(pred, np.float64), np.asarray(truth, np.float64)
    assert p.shape == t.shape and np.isfinite(p).all()
    m = len(t)
    d = np.sqrt(((p - t) ** 2).mean(1))
    pc, tc = p - p.mean(0, keepdims=True), t - t.mean(0, keepdims=True)
    dc = np.sqrt(((pc - tc) ** 2).mean(1))
    r, rc = _rowcorr(p, t), _rowcorr(pc, tc)
    rr, leg, top = retrieval(p, t)
    rrc, legc, topc = retrieval(pc, tc)
    nm = lambda x: float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")
    return dict(delta_rmse=float(d.mean()), centred_rmse=float(dc.mean()), delta_pearson=nm(r), centred_pearson=nm(rc),
                n_pearson_na=int((~np.isfinite(r)).sum()), n_centred_pearson_na=int((~np.isfinite(rc)).sum()),
                mrr_tie=float(rr.mean()), mrr_tie_centred=float(rrc.mean()), mrr_legacy=float(leg.mean()), mrr_legacy_centred=float(legc.mean()),
                top5_tie_centred=float(topc[5].mean()), mrr_chance=float(_harm(m)[m] / m),
                pred_rms=float(np.sqrt((p ** 2).mean())), truth_rms=float(np.sqrt((t ** 2).mean())), n_query=m)


def rmse(pred, truth):
    return torch.sqrt(((pred - truth) ** 2).mean(1)).mean().item()
