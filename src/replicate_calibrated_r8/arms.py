"""Arm definitions and the shared prediction path used by training, source evaluation and the HCC evaluation."""
import json
import numpy as np
import torch

from .common import man_file
from . import methods as M

ARMS = dict(B00=dict(state="none", obs="iid"), B01=dict(state="film", obs="iid"), B10=dict(state="none", obs="cov"), B11=dict(state="film", obs="cov"),
            G_concat=dict(state="concat", obs="iid"), G_dummy=dict(state="dummy", obs="iid"))


class Context:
    """Frozen source-side constants: anchor shrinkage per K and observation model, covariance constants, feature maps."""

    def __init__(self, fmap, fp_table, selection=None, committed_only=False):
        self.sel = selection or json.load(open(man_file("anchor_selection.json", committed_only)))
        self.cov = self.sel["covariance_model"]
        self.fmap, self.FP = fmap, fp_table

    def anchor(self, inp, obs):
        e = self.sel["anchor"][obs][str(len(inp["y_s"]))]
        return M.anchor(inp, e["kappa_f"], e["kappa_b"], self.cov if obs == "cov" else None)

    def features(self, inp, anc, dummy=False):
        return self.fmap.build(inp, self.FP, anc["fs"], anc["fq"], dummy=dummy)

    def c0(self, inp, obs):
        return M.c0(self.anchor(inp, obs))

    def c1(self, inp, obs):
        anc = self.anchor(inp, obs)
        e = self.sel["c1"][obs][str(len(inp["y_s"]))]
        return M.c1(anc, self.features(inp, anc), e["lam"], e["eta"])

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
        return model(anc, ft, mode=mode, perm=perm)

    def cnp(self, model, inp):
        return model(inp, self.fmap.build(inp, self.FP))
