"""Arm definitions and the shared prediction path used by training, source evaluation and the HCC evaluation."""
import json
import numpy as np
import torch

from .common import man_file
from . import methods as M

ARMS = dict(B00=dict(state="none", obs="iid"), B01=dict(state="film", obs="iid"), B10=dict(state="none", obs="cov"), B11=dict(state="film", obs="cov"),
            G_concat=dict(state="concat", obs="iid"), G_dummy=dict(state="dummy", obs="iid"),
            # revision R-1: per-drug correction projected on the repeat-informed rank-16 output basis (common correction stays in gene space)
            B00r=dict(state="none", obs="iid", proj=True))


class Context:
    """Frozen source-side constants: anchor shrinkage per K and observation model, covariance constants, feature maps."""

    def __init__(self, fmap, fp_table, selection=None, committed_only=False, basis=None):
        self.sel = selection or json.load(open(man_file("anchor_selection.json", committed_only)))
        self.cov = self.sel["covariance_model"]
        self.fmap, self.FP = fmap, fp_table
        self.basis = None if basis is None else torch.as_tensor(basis, dtype=torch.float32, device=fp_table.device)     # (978, r) orthonormal columns

    def project(self, corr):
        assert self.basis is not None, "this arm needs the repeat-informed basis"
        return (corr @ self.basis) @ self.basis.T

    def anchor(self, inp, obs):
        e = self.sel["anchor"][obs][str(len(inp["y_s"]))]
        return M.anchor(inp, e["kappa_f"], e["kappa_b"], self.cov if obs == "cov" else None)

    def features(self, inp, anc, dummy=False):
        return self.fmap.build(inp, self.FP, anc["fs"], anc["fq"], dummy=dummy)

    def c0(self, inp, obs):
        return M.c0(self.anchor(inp, obs))

    def c1(self, inp, obs, proj=False, lam=None, eta=None):
        anc = self.anchor(inp, obs)
        e = self.sel["c1r" if proj and "c1r" in self.sel else "c1"][obs][str(len(inp["y_s"]))]
        lam, eta = e["lam"] if lam is None else lam, e["eta"] if eta is None else eta
        corr = M.krr_correction(anc, self.features(inp, anc), lam)
        return anc["fq"] + anc["b"] + eta * (self.project(corr) if proj else corr)

    def shrunk_same_drug_mean(self, inp):
        e = self.sel["shrunk_same_drug_mean"][str(len(inp["y_s"]))]
        return M.anchor(inp, e["kappa_f"], "inf")["fq"]

    def adapter(self, model, arm, inp, mode="normal", perm_seed=0):
        a = ARMS[arm]
        anc = self.anchor(inp, a["obs"])
        ft = self.features(inp, anc, dummy=a["state"] == "dummy")
        perm = None
        if mode == "shuffle_residual":
            K = len(inp["y_s"])
            order = np.random.RandomState(1000 * perm_seed + K).permutation(K)
            p = np.empty(K, int)
            p[order] = np.roll(order, 1)                      # cyclic derangement: no support keeps its own residual
            perm = torch.tensor(p, device=anc["R"].device)
        pred = model(anc, ft, mode=mode, perm=perm)
        if a.get("proj"):
            base = anc["fq"] + anc["b"]
            pred = base + self.project(pred - base)
        return pred

    def cnp(self, model, inp):
        return model(inp, self.fmap.build(inp, self.FP))
