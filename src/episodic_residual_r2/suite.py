"""Every compared method behind one call, so source virtual targets and the HCC targets go through identical code."""
import json, os
import numpy as np
import torch

from .common import MAN, sha256
from .data import Features, residual_basis
from . import methods as M
from . import tsg as G
from .train_meta import predict as learned_predict
from src.transigen_adaptation_b1 import tsg_adapter as T


def load_learned(entry, src, dev):
    """entry: dict(checkpoint, checkpoint_sha256). Rebuilds the module from the stored kind/variant."""
    assert sha256(entry["checkpoint"]) == entry["checkpoint_sha256"], f"checkpoint changed: {entry['checkpoint']}"
    ck = torch.load(entry["checkpoint"], map_location="cpu", weights_only=False)
    v = dict(ck["variant"])
    if ck["kind"] == "cnp":
        m = M.GenericCNP()
    else:
        if "svd_rank" in v:
            v["svd_basis"] = residual_basis(src, v.pop("svd_rank"))[0]
        m = M.ResidualAdapter(**v)
    m.load_state_dict(ck["state_dict"])
    return m.to(dev).eval()


class Suite:
    """selection: dict with baseline lambdas, TranSiGen fine-tuning settings and learned-model checkpoints.

    selection = {"lambda": {"ecfp_krr|16": {"lam": ..}, ..}, "tsg_ft": {"16": {"lr":.., "bn":.., "updates":..}, ..},
                 "learned": {"residual_adapter": {"42": {checkpoint, checkpoint_sha256}, ..}, "generic_cnp": {...}, ...},
                 "modes": {"residual_adapter": ["zero_residual", "shuffle_residual"]}}
    """

    def __init__(self, selection, src, dev, with_tsg=True):
        self.sel, self.src, self.dev = selection, src, dev
        self.fmap = Features.load(os.path.join(os.environ["R2_PROC"], "feature_basis.npz"), dev)
        self.learned = {name: {int(s): load_learned(e, src, dev) for s, e in seeds.items()} for name, seeds in selection.get("learned", {}).items()}
        self.tsg = {}
        if with_tsg:
            for s, e in G.checkpoints().items():
                self.tsg[s] = T.load_checkpoint(e["checkpoint"], dev, seed=s)[0]

    def run(self, ep, K, ra=None):
        """Yields (method, model_seed or -1, prediction for the query units). Query labels are never passed in."""
        inp = ep.inputs(K)
        with torch.no_grad():
            ft = self.fmap.build(inp, self.src.FP)
            yield "no_change", -1, M.no_change(inp)
            yield "source_same_drug_mean", -1, M.same_drug_mean(inp)
            yield "mean_offset", -1, M.mean_offset(inp)
            for name, f in M.CLOSED_FORM.items():
                yield name, -1, f(inp, ft, float(self.sel["lambda"][f"{name}|{K}"]["lam"]))
            for name, seeds in self.learned.items():
                for s, m in seeds.items():
                    yield name, s, learned_predict(m, ep, K, self.fmap, self.src.FP)
                    for mode in self.sel.get("modes", {}).get(name, []):
                        yield f"{name}__{mode}", s, learned_predict(m, ep, K, self.fmap, self.src.FP, mode, perm_seed=max(ep.seed, 0))
        if self.tsg:
            assert ra is not None
            ft_sel = self.sel["tsg_ft"][str(K)]
            for s, base in self.tsg.items():
                as_is = G.predict_units(base, ra, ep.g_q, self.dev)
                yield "transigen_as_is", s, as_is
                ps = G.predict_units(base, ra, ep.g_s[:K], self.dev)
                yield "transigen_offset", s, as_is + (inp["y_s"] - ps).mean(0, keepdim=True)
                preds, status, _ = G.fine_tune(base, ep, K, ra, ft_sel["lr"], ft_sel["bn"], [ft_sel["updates"]], s, self.dev)
                if ft_sel["updates"] in preds:
                    yield "transigen_full_ft", s, preds[ft_sel["updates"]]
                else:
                    yield "transigen_full_ft", s, status          # diverged: recorded by the caller, no prediction


def current_selection(extra_learned=None, modes=None):
    """Selection assembled from the committed manifests (source-side selections only)."""
    sel = dict(lam_file=os.path.join(MAN, "baseline_lambda_selection.json"), tsg_file=os.path.join(MAN, "tsg_ft_selection.json"))
    out = {"lambda": json.load(open(sel["lam_file"]))["selected"], "tsg_ft": json.load(open(sel["tsg_file"]))["selected"], "learned": {}, "modes": modes or {}}
    ms = os.path.join(MAN, "model_selection.json")
    if os.path.exists(ms):
        out["learned"] = json.load(open(ms))["final"]
    if extra_learned:
        out["learned"].update(extra_learned)
    return out
