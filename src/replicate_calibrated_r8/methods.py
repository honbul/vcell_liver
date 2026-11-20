"""Shared anchor, observation models, closed-form residual regression, the state-conditioned adapter and the generic CNP.

Every method maps a method-visible input dict (Episode.inputs(K): no query label) to query deltas.

Shared anchor (identical for C0 / C1 / C2 inside one observation model):
    f*(p)  = f0(p) * n_p / (n_p + kappa_f)                 n_p = number of source families behind f0
    E_i    = y_i - f*(p_i)
    e_bar  = (1' Omega^-1 E) / (1' Omega^-1 1)             Omega = I  ->  plain mean
    b      = e_bar * K_eff / (K_eff + kappa_b)             K_eff = 1' Omega^-1 1  (= K for Omega = I)
    R_i    = E_i - e_bar
Residual regression:  f*(q) + b + eta * k_q' (K_ss + lambda * Omega)^-1 R     (Cholesky solve; no explicit inverse)
"""
import math
import torch
import torch.nn as nn

FLOOR = 0.05          # diagonal floor added to the normalised Omega
OFFDIAG_KEEP = 0.9    # diagonal shrinkage: off-diagonal entries are multiplied by this


def omega(inp, obs):
    """K x K observation covariance of the support deltas, normalised so that a single-row unit alone on its plate has ~1.

    obs None / {"kind": "iid"} -> identity. Otherwise {"kind": "cov", "v_r", "v_e", "v_s"} from source replicates:
      Omega_ij = [ delta_ij (v_r + v_e q_i) + v_s o_ij * (1 or OFFDIAG_KEEP) ] / (v_r + v_e + v_s) + FLOOR * delta_ij
    """
    K = len(inp["y_s"])
    eye = torch.eye(K, device=inp["y_s"].device, dtype=torch.float64)
    if obs is None or obs["kind"] == "iid":
        return eye
    o, q = inp["o_s"].double(), inp["q_s"].double()
    v_r = obs["v_r"] * obs.get("v_r_scale", 1.0)             # revision R-2 diagnostics only; the defaults reproduce the pre-registered model
    structure = obs.get("structure", "full")
    assert structure in ("full", "diag_only", "block_only")
    od = torch.diagonal(o)
    if structure == "block_only":                            # equal diagonal: only the shared-plate block differs between supports
        q, od = torch.ones_like(q), torch.ones_like(od)
    off = obs["v_s"] * o * OFFDIAG_KEEP * (0.0 if structure == "diag_only" else 1.0)
    diag = v_r + obs["v_e"] * q + obs["v_s"] * od
    Om = off - torch.diag(torch.diagonal(off)) + torch.diag(diag)
    return Om / (v_r + obs["v_e"] + obs["v_s"]) + FLOOR * eye


def anchor(inp, kappa_f, kappa_b, obs=None):
    """The common starting point. Returns a dict; nothing in it depends on a query label."""
    rs = (inp["n0_s"] / (inp["n0_s"] + kappa_f)).nan_to_num(0.0)[:, None] if kappa_f > 0 else (inp["n0_s"] > 0).float()[:, None]
    rq = (inp["n0_q"] / (inp["n0_q"] + kappa_f)).nan_to_num(0.0)[:, None] if kappa_f > 0 else (inp["n0_q"] > 0).float()[:, None]
    fs, fq = inp["f0_s"] * rs, inp["f0_q"] * rq
    E = inp["y_s"] - fs
    Om = omega(inp, obs)
    L = torch.linalg.cholesky(Om)
    a = torch.cholesky_solve(torch.ones(len(E), 1, device=E.device, dtype=torch.float64), L)[:, 0]
    keff = a.sum()
    w = (a / keff).float()
    ebar = (w[:, None] * E).sum(0, keepdim=True)
    shrink = 0.0 if (isinstance(kappa_b, str) or math.isinf(kappa_b)) else float(keff / (keff + kappa_b))
    return dict(fs=fs, fq=fq, E=E, ebar=ebar, b=ebar * shrink, R=E - ebar, Om=Om, w=w, keff=float(keff), shrink=shrink)


def no_change(inp):
    return torch.zeros_like(inp["f0_q"])


def same_drug_mean(inp):
    return inp["f0_q"]


def mean_offset(inp):
    """Unshrunk reference of the predecessor study: f0 + plain mean support residual."""
    return inp["f0_q"] + (inp["y_s"] - inp["f0_s"]).mean(0, keepdim=True)


def c0(anc):
    return anc["fq"] + anc["b"]


def tanimoto(a, b):
    inter = a @ b.T
    den = a.sum(1)[:, None] + b.sum(1)[None, :] - inter
    return inter / den.clamp(min=1e-9)


def centre_kernel(Kss, Kqs, w):
    """Centre both kernels in feature space on the w-weighted support mean (w sums to 1; w = 1/K is the plain mean)."""
    w = w.double()
    m = w @ Kss                                   # (K,)
    mm = m @ w
    Kss_c = Kss - m[None, :] - m[:, None] + mm
    Kqs_c = Kqs - (Kqs @ w)[:, None] - m[None, :] + mm
    return Kss_c, Kqs_c


def gls_solve(Kss, Kqs, R, lam, Om):
    """Kqs (Kss + lam * Omega)^-1 R by Cholesky."""
    L = torch.linalg.cholesky(Kss + lam * Om)
    return Kqs @ torch.cholesky_solve(R, L)


def allfeat_kernels(ft):
    """Untrained kernel on the within-only inputs: Tanimoto(ECFP) + linear(ctrl)/32 + linear(f0 repr)/33 (as in R2)."""
    def lin(name):
        a, b = ft[name + "_s"], ft[name + "_q"]
        return a @ a.T / a.shape[1], b @ a.T / a.shape[1]
    k1, k2 = lin("ctrl"), lin("f0r")
    return tanimoto(ft["fp_s"], ft["fp_s"]) + k1[0] + k2[0], tanimoto(ft["fp_q"], ft["fp_s"]) + k1[1] + k2[1]


def krr_correction(anc, ft, lam):
    Kss, Kqs = allfeat_kernels(ft)
    Kss_c, Kqs_c = centre_kernel(Kss.double(), Kqs.double(), anc["w"])
    return gls_solve(Kss_c, Kqs_c, anc["R"].double(), lam, anc["Om"]).float()


def c1(anc, ft, lam, eta=1.0):
    return anc["fq"] + anc["b"] + eta * krr_correction(anc, ft, lam)


# ---------------------------------------------------------------- learned models
class StateAdapter(nn.Module):
    """Learned embedding for the residual regression on top of the shared anchor.

    state 'none'   : within-only inputs (R2 feature set): phi([fp, f0r, ctrl])
          'film'   : the drug block [fp, f0r] is embedded and multiplied per dimension by 1 + gamma(z_global) BEFORE support centring
          'concat' : z_global is added as an extra linear input (ablation: constant per episode)
          'dummy'  : as 'film' but the context is a per-cell pseudo-random vector (same parameter count, no state information)
    The first layer of 'none' equals the R2 first layer (a linear map of the concatenated blocks).
    """

    def __init__(self, state="none", hidden=128, out=64, d_glob=16, k_values=(8, 16, 32), lam_init=1.0, lam_bounds=(1e-3, 1e3), film_init=0.0):
        super().__init__()
        assert state in ("none", "film", "concat", "dummy")
        self.state, self.out, self.k_values = state, out, tuple(k_values)
        self.lin_d = nn.Linear(2048 + 33, hidden)
        self.lin_w = nn.Linear(32, hidden, bias=False)
        if state in ("film", "dummy"):
            self.gamma = nn.Linear(d_glob, hidden)
            nn.init.normal_(self.gamma.weight, std=film_init) if film_init > 0 else nn.init.zeros_(self.gamma.weight)
            nn.init.zeros_(self.gamma.bias)
        if state == "concat":
            self.lin_g = nn.Linear(d_glob, hidden, bias=False)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, out), nn.LayerNorm(out))
        self.log_lam = nn.Parameter(torch.tensor(math.log(lam_init)))
        self.lam_bounds = (math.log(lam_bounds[0]), math.log(lam_bounds[1]))
        self.eta_logit = nn.Parameter(torch.full((len(k_values),), 2.0))     # residual trust per K, in (0, 1)

    def lam(self):
        return torch.exp(self.log_lam.clamp(*self.lam_bounds))

    def eta(self, K):
        return torch.sigmoid(self.eta_logit[self.k_values.index(K)])

    def embed(self, ft, side):
        hd = self.lin_d(torch.cat([ft["fp_" + side], ft["f0r_" + side]], 1))
        if self.state in ("film", "dummy"):
            hd = hd * (1.0 + self.gamma(ft["glob"]))
        h = hd + self.lin_w(ft["ctrl_" + side])
        if self.state == "concat":
            h = h + self.lin_g(ft["glob"])
        return self.head(h) / math.sqrt(self.out)

    def correction(self, anc, ft, R=None):
        zs, zq = self.embed(ft, "s"), self.embed(ft, "q")
        m = (anc["w"][:, None] * zs).sum(0, keepdim=True)
        zs, zq = zs - m, zq - m
        R = anc["R"] if R is None else R
        corr = gls_solve((zs @ zs.T).double(), (zq @ zs.T).double(), R.double(), self.lam().double(), anc["Om"]).float()
        return corr * self.eta(len(zs))

    def forward(self, anc, ft, mode="normal", perm=None):
        if mode == "zero_residual":
            return anc["fq"] + anc["b"]
        R = anc["R"][perm] if mode == "shuffle_residual" else None
        return anc["fq"] + anc["b"] + self.correction(anc, ft, R)


class GenericCNP(nn.Module):
    """Conditional neural process on the within-only inputs (as in R2): support set -> mean representation -> decoder -> delta.

    No f0 skip connection, no anchor, no offset/residual split.
    """

    def __init__(self, hidden=512, r_dim=128, fp_dim=128):
        super().__init__()
        self.fp = nn.Linear(2048, fp_dim)
        d_feat = fp_dim + 32 + 33
        self.enc = nn.Sequential(nn.Linear(d_feat + 978 + 978, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, r_dim))
        self.dec = nn.Sequential(nn.Linear(d_feat + 978 + r_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.LayerNorm(hidden),
                                 nn.GELU(), nn.Linear(hidden, 978))

    def feat(self, ft, side):
        return torch.cat([self.fp(ft["fp_" + side]), ft["ctrl_" + side], ft["f0r_" + side]], 1)

    def forward(self, inp, ft):
        r = self.enc(torch.cat([self.feat(ft, "s"), inp["f0_s"], inp["y_s"]], 1)).mean(0, keepdim=True)
        fq = self.feat(ft, "q")
        return self.dec(torch.cat([fq, inp["f0_q"], r.expand(len(fq), -1)], 1))


def rmse_loss(pred, truth):
    return torch.sqrt(((pred - truth) ** 2).mean(1) + 1e-8).mean()
