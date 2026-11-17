"""Every compared method behind one call, so source virtual targets, matched-plate episodes and the HCC targets share the code path."""
import json, os
import numpy as np
import torch

from .common import sha256, man_file, proc
from .data import Features
from .arms import ARMS, Context, load_anchor_selection
from . import methods as M
from .train import build_model
from src.episodic_residual_r2 import tsg as G
from src.transigen_adaptation_b1 import tsg_adapter as T


def load_learned(entry, dev):
    assert sha256(entry["checkpoint"]) == entry["checkpoint_sha256"], f"checkpoint changed: {entry['checkpoint']}"
    ck = torch.load(entry["checkpoint"], map_location="cpu", weights_only=False)
    m = build_model(ck["arm"])
    m.load_state_dict(ck["state_dict"])
    return ck["arm"], m.to(dev).eval()


def current_selection(committed_only=False, with_tsg=True):
    """Selection assembled from the source-side selection manifests."""
    f = lambda n: man_file(n, committed_only)
    out = dict(anchor_selection=load_anchor_selection(committed_only), learned=json.load(open(f("model_selection.json")))["final"])
    if with_tsg:
        out["tsg"] = json.load(open(f("tsg_selection.json")))
    rb = f("repeat_basis.json")
    if os.path.exists(rb) and any(ARMS.get(n, {}).get("proj") for n in out["learned"]):
        e = json.load(open(rb))["bases"]["all"]
        out["repeat_basis"] = dict(path=e["path"], sha256=e["sha256"])
    return out


class Suite:
    """selection = {"anchor_selection": {...}, "learned": {name: {seed: {checkpoint, checkpoint_sha256}}},
                    "tsg": {"checkpoints": {seed: {checkpoint, checkpoint_sha256}}, "ft": {K: {lr, bn, updates}}}}"""

    def __init__(self, selection, st, dev, with_tsg=True, shuffle_modes=("B00",)):
        self.st, self.dev = st, dev
        self.fmap = Features.load(proc("feature_basis.npz"), dev)
        rb = selection.get("repeat_basis")
        if rb:
            assert sha256(rb["path"]) == rb["sha256"], "repeat-informed basis changed"
        self.ctx = Context(self.fmap, st.FP, selection["anchor_selection"], basis=np.load(rb["path"]) if rb else None)
        self.learned = {name: {int(s): load_learned(e, dev) for s, e in seeds.items()} for name, seeds in selection.get("learned", {}).items()}
        self.shuffle_modes = set(shuffle_modes)
        self.tsg, self.tsg_ft = {}, None
        if with_tsg:
            self.tsg_ft = selection["tsg"]["ft"]
            for s, e in selection["tsg"]["checkpoints"].items():
                assert sha256(e["checkpoint"]) == e["checkpoint_sha256"]
                self.tsg[int(s)] = T.load_checkpoint(e["checkpoint"], dev, seed=int(s))[0]

    def closed_form(self, inp):
        c = self.ctx
        yield "no_change", M.no_change(inp)
        yield "same_drug_mean", M.same_drug_mean(inp)
        yield "mean_offset", M.mean_offset(inp)
        yield "shrunk_same_drug_mean", c.shrunk_same_drug_mean(inp)
        for obs in ("iid", "cov") + (("cov2",) if "cov2" in c.sel["anchor"] else ()):
            yield f"C0_{obs}", c.c0(inp, obs)
            yield f"C1_{obs}", c.c1(inp, obs)
        if c.basis is not None:
            yield "C1r_iid", c.c1(inp, "iid", proj=True)

    def run(self, ep, K, ra=None, tsg=True):
        """Yields (method, model_seed or -1, prediction for the query units). Query labels are never passed in."""
        inp = ep.inputs(K)
        with torch.no_grad():
            for name, p in self.closed_form(inp):
                yield name, -1, p
            for name, seeds in self.learned.items():
                for s, (arm, m) in seeds.items():
                    if arm == "cnp":
                        yield name, s, self.ctx.cnp(m, inp)
                        continue
                    yield name, s, self.ctx.adapter(m, arm, inp)
                    if name in self.shuffle_modes:
                        yield f"{name}__shuffle_residual", s, self.ctx.adapter(m, arm, inp, "shuffle_residual", perm_seed=max(ep.seed, 0))
        if self.tsg and tsg:
            assert ra is not None
            ft = self.tsg_ft[str(K)]
            for s, base in self.tsg.items():
                as_is = G.predict_units(base, ra, ep.g_q, self.dev)
                yield "transigen_as_is", s, as_is
                ps = G.predict_units(base, ra, ep.g_s[:K], self.dev)
                yield "transigen_offset", s, as_is + (inp["y_s"] - ps).mean(0, keepdim=True)
                preds, status, _ = G.fine_tune(base, ep, K, ra, ft["lr"], ft["bn"], [ft["updates"]], s, self.dev)
                yield "transigen_full_ft", s, preds[ft["updates"]] if ft["updates"] in preds else status
