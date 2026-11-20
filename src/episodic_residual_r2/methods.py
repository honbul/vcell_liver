"""Closed-form baselines, the residual adapter and the generic CNP. Every method maps a method-visible
input dict (Episode.inputs(K): no query label) to query deltas."""
import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------- closed-form pieces
def split_residual(inp):
    """e_i = y_i - f0_i ; b_S = mean e_i ; R_i = e_i - b_S."""
    e = inp["y_s"] - inp["f0_s"]
    b = e.mean(0, keepdim=True)
    return b, e - b


def no_change(inp):
    return torch.zeros_like(inp["f0_q"])


def same_drug_mean(inp):
    return inp["f0_q"]


def mean_offset(inp):
    b, _ = split_residual(inp)
    return inp["f0_q"] + b


def krr_solve(Kss, Kqs, R, lam):
    """Kqs (Kss + lam I)^-1 R by Cholesky; no explicit inverse."""
    A = Kss + lam * torch.eye(Kss.shape[0], device=Kss.device, dtype=Kss.dtype)
    L = torch.linalg.cholesky(A)
    return Kqs @ torch.cholesky_solve(R, L)


def centre_kernel(Kss, Kqs):
    """Centre both kernels in feature space on the support mean."""
    cm = Kss.mean(0, keepdim=True)
    Kss_c = Kss - cm - cm.T + Kss.mean()
    Kqs_c = Kqs - Kqs.mean(1, keepdim=True) - cm + Kss.mean()
    return Kss_c, Kqs_c


def tanimoto(a, b):
    inter = a @ b.T
    den = a.sum(1)[:, None] + b.sum(1)[None, :] - inter
    return inter / den.clamp(min=1e-9)


def kernel_residual(inp, Kss, Kqs, lam):
    b, R = split_residual(inp)
    Kss_c, Kqs_c = centre_kernel(Kss.double(), Kqs.double())
    return inp["f0_q"] + b + krr_solve(Kss_c, Kqs_c, R.double(), lam).float()


def ecfp_ridge(inp, ft, lam):
    return kernel_residual(inp, ft["fp_s"] @ ft["fp_s"].T, ft["fp_q"] @ ft["fp_s"].T, lam)


def ecfp_krr(inp, ft, lam):
    return kernel_residual(inp, tanimoto(ft["fp_s"], ft["fp_s"]), tanimoto(ft["fp_q"], ft["fp_s"]), lam)


def allfeat_kernels(ft):
    """Untrained kernel on the same inputs as phi: Tanimoto(ECFP) + linear(ctrl)/32 + linear(f0 repr)/33."""
    def lin(name):
        a, b = ft[name + "_s"], ft[name + "_q"]
        return a @ a.T / a.shape[1], b @ a.T / a.shape[1]
    k1 = lin("ctrl")
    k2 = lin("f0r")
    return tanimoto(ft["fp_s"], ft["fp_s"]) + k1[0] + k2[0], tanimoto(ft["fp_q"], ft["fp_s"]) + k1[1] + k2[1]


def allfeat_krr(inp, ft, lam):
    Kss, Kqs = allfeat_kernels(ft)
    return kernel_residual(inp, Kss, Kqs, lam)


CLOSED_FORM = dict(ecfp_ridge=ecfp_ridge, ecfp_krr=ecfp_krr, allfeat_krr=allfeat_krr)


# ---------------------------------------------------------------- learned models
def mlp(d_in, hidden, d_out):
    return nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, d_out), nn.LayerNorm(d_out))


def shrunk_prediction(inp, kf, kb):
    """Post-hoc added baseline (diagnostic S1): f0 * n/(n+kf) plus the support mean residual * K/(K+kb); kb=inf -> no support label used."""
    wq = (inp["n0_q"] / (inp["n0_q"] + kf))[:, None]
    ws = (inp["n0_s"] / (inp["n0_s"] + kf))[:, None]
    base = inp["f0_q"] * wq
    if kb == "inf" or math.isinf(float(kb)):
        return base
    K = len(inp["y_s"])
    b = (inp["y_s"] - inp["f0_s"] * ws).mean(0, keepdim=True)
    return base + b * (K / (K + float(kb)))


class ResidualAdapter(nn.Module):
    """f0(q) + b_S + z_q^T Z_S^T (Z_S Z_S^T + lambda I)^-1 R_S with a learned embedding phi and one global lambda.

    Options (all default to the initial integrated design):
      use_fp / use_ctrl / use_f0r : input blocks of phi
      offset_mode 'split' : b_S separated, embeddings centred on the support mean (the proposal)
                  'joint' : no separation; ridge on the uncentred residual e with a learned constant feature
      svd_basis : optional (978, r) orthonormal source-only basis; the per-drug correction is restricted to its span
      shrink_by_k : optional learned per-K multiplier in (0, 1) on the per-drug correction
      shrink_offset : optional learned per-K multiplier in (0, 1) on the support mean offset b_S ('split' mode only)
    """

    def __init__(self, hidden=128, out=64, use_fp=True, use_ctrl=True, use_f0r=True, offset_mode="split", svd_basis=None,
                 shrink_by_k=False, shrink_offset=False, k_values=(8, 16, 32), lam_init=1.0, lam_bounds=(1e-3, 1e3)):
        super().__init__()
        self.use = dict(fp=use_fp, ctrl=use_ctrl, f0r=use_f0r)
        d_in = 2048 * use_fp + 32 * use_ctrl + 33 * use_f0r
        assert d_in > 0 and offset_mode in ("split", "joint")
        self.phi = mlp(d_in, hidden, out)
        self.out = out
        self.offset_mode = offset_mode
        self.log_lam = nn.Parameter(torch.tensor(math.log(lam_init)))
        self.lam_bounds = (math.log(lam_bounds[0]), math.log(lam_bounds[1]))
        if offset_mode == "joint":
            self.const = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("svd_basis", svd_basis if svd_basis is not None else torch.zeros(0))
        self.k_values = tuple(k_values)
        if shrink_by_k:
            self.shrink_logit = nn.Parameter(torch.full((len(k_values),), 2.0))
        self.shrink_by_k = shrink_by_k
        assert not (shrink_offset and offset_mode != "split")
        if shrink_offset:
            self.offset_logit = nn.Parameter(torch.full((len(k_values),), 2.0))
        self.shrink_offset = shrink_offset

    def lam(self):
        return torch.exp(self.log_lam.clamp(*self.lam_bounds))

    def embed(self, ft, side):
        x = torch.cat([ft[f"{k}_{side}"] for k in ("fp", "ctrl", "f0r") if self.use[k]], 1)
        return self.phi(x) / math.sqrt(self.out)

    def correction(self, inp, ft, R=None, zero_residual=False):
        """Per-drug correction term (beyond f0 + b_S in 'split' mode)."""
        zs, zq = self.embed(ft, "s"), self.embed(ft, "q")
        if self.offset_mode == "split":
            b, R0 = split_residual(inp)
            R = R0 if R is None else R
            m = zs.mean(0, keepdim=True)
            zs, zq = zs - m, zq - m
        else:
            b = torch.zeros_like(inp["f0_q"][:1])
            R = (inp["y_s"] - inp["f0_s"]) if R is None else R
            c = self.const.expand(1, 1)
            zs = torch.cat([zs, c.expand(len(zs), 1)], 1)
            zq = torch.cat([zq, c.expand(len(zq), 1)], 1)
        if zero_residual:
            R = torch.zeros_like(R)
        if self.svd_basis.numel():
            R = (R @ self.svd_basis) @ self.svd_basis.T
        corr = krr_solve((zs @ zs.T).double(), (zq @ zs.T).double(), R.double(), self.lam().double()).float()
        if self.shrink_by_k:
            K = len(zs)
            assert K in self.k_values
            corr = corr * torch.sigmoid(self.shrink_logit[self.k_values.index(K)])
        if self.shrink_offset:
            b = b * torch.sigmoid(self.offset_logit[self.k_values.index(len(zs))])
        return b, corr

    def forward(self, inp, ft, **kw):
        b, corr = self.correction(inp, ft, **kw)
        return inp["f0_q"] + b + corr


class GenericCNP(nn.Module):
    """Conditional neural process with the same inputs: support set -> mean representation -> decoder -> delta.

    No f0 skip connection and no offset/residual split; the aggregated support representation is query-independent.
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
