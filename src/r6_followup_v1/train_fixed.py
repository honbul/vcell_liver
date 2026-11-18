"""Minimal retraining comparison of the R6 follow-up (plan 6.1): S / SD / M on one fixed view cache.

Arms (every arm: same fitting conditions, same view cache, same minibatch sequence, same
additive head  delta = g(z) + h(z, drug)  with the frozen R6 shrinkage on h, same optimiser,
same number of updates):
    SD   the frozen R6 B00: gene-token state encoder + drug ECFP4                 (state + drug)
    S    the same encoder; the drug fingerprint is replaced by ONE learned constant
         embedding, so h(z, e0) is a working state-only head                      (state only)
    M    z = LayerNorm(MLP(standardised mean log-normalised RNA of the 64 cells)); the same head
         and drug input as SD                                                     (mean RNA + drug)
The Ridge (R) is fitted by ridge_fixed.py on the same cache.

Views: `--views fixed` (default) draws every control set from the stored cache of the run's seed
(views.py). `--views fresh` is the original on-the-fly sampling and thinning (kept only for the
limited fixed-vs-fresh control). The sampler generator is separate from the global generator the
model uses, so S, SD and M see the identical sequence of (view, drugs) batches for a given seed.

Settings
    source  --fold k           fit: TRAIN drugs outside fold k; predict: fold k (out-of-fold)
            --fold k --inner   fit: outer-fit minus the inner development drugs; predict: inner dev
    final                      fit: all TRAIN drugs; no prediction here (final_eval.py, after the freeze)

Optional branch F1 (plan 7.1), off by default:
    --anchor-kappa K   prediction = shrunk pool mean (fitting drugs only; leave-one-drug-out for
                       training conditions) + network output; the target stays the full delta
    --pair-lambda X    adds X * mean over same-set drug pairs of |(o_d - o_e) - (y_d - y_e)|^2 / G on
                       the network outputs (a common anchor cancels in the difference, so no
                       per-drug anchor enters this term)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.depth_robust_r6 import data as D
from src.depth_robust_r6.count_view import set_features, thin_torch
from src.depth_robust_r6.encoder import ResponseModel, n_params
from src.depth_robust_r6.train import Sampler

from . import common as C
from .baselines import MeanBaseline
from .views import load_views

ARMS = ("S", "SD", "M")


class MeanEncoder(nn.Module):
    """R6-M: the set's mean log-normalised expression (the Ridge's control feature) -> small MLP -> z."""

    def __init__(self, n_genes: int, d: int = 128, hidden: int = 256, dropout: float = 0.1, x_mean=None, x_sd=None):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_genes, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, d))
        self.norm = nn.LayerNorm(d)
        self.register_buffer("x_mean", torch.zeros(n_genes) if x_mean is None else torch.as_tensor(x_mean).float())
        self.register_buffer("x_sd", torch.ones(n_genes) if x_sd is None else torch.as_tensor(x_sd).float())

    def forward(self, f: dict):
        return self.norm(self.net((f["x"].mean(1) - self.x_mean) / self.x_sd)), None


class StateOnlyModel(ResponseModel):
    """R6-S: the drug branch is removed; a learned constant takes the place of the drug embedding."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        d_drug = self.drug[-2].out_features
        del self.drug
        self.e0 = nn.Parameter(torch.zeros(1, 1, d_drug))

    def delta_parts(self, z: torch.Tensor, fp: torch.Tensor):
        e = self.e0.expand(z.shape[0], fp.shape[1], -1)
        h = self.head(torch.cat([z.unsqueeze(1).expand(-1, e.shape[1], -1), e], -1))
        g = self.state_head(z).unsqueeze(1) if self.state_head is not None else None
        return g, h


def build_model(d: D.R6Data, arm: str, hp: dict) -> ResponseModel:
    assert arm in ARMS
    kw = dict(obs_input=False, use_state=True, d=hp.get("d", 128), n_tok=hp.get("n_tok", 256), n_blocks=hp.get("n_blocks", 2),
              dropout=hp.get("dropout", 0.1), head_mode=hp.get("head_mode", "joint"), x_mean=d.x_mean, x_sd=d.x_sd, tiebreak=d.tiebreak)
    cls = StateOnlyModel if arm == "S" else ResponseModel
    m = cls(d.Y.shape[2], d.fp.shape[1], **kw)
    if arm == "M":
        m.encoder = MeanEncoder(d.Y.shape[2], d=hp.get("d", 128), hidden=hp.get("mean_hidden", 256), dropout=hp.get("dropout", 0.1),
                                x_mean=d.x_mean, x_sd=d.x_sd)
    return m


def param_report(m: nn.Module, arm: str) -> dict:
    by = {}
    for name, q in m.named_parameters():
        by[name.split(".")[0]] = by.get(name.split(".")[0], 0) + q.numel()
    return dict(total=n_params(m), by_module=by,
                note={"S": "no drug MLP; e0 is the learned constant that replaces the drug embedding",
                      "SD": "the frozen R6 B00 architecture", "M": "encoder = 2-layer MLP on the set mean; no gene tokens, no attention"}[arm])


class FixedSampler(Sampler):
    """(view of the condition's pool, K fitting drugs of that pool); views come from the stored cache."""

    def __init__(self, d, fit_drugs, dev, gen, n_views_per_pool: int):
        super().__init__(d, fit_drugs, dev, gen)
        self.nv = n_views_per_pool
        self.pool_t = torch.as_tensor(self.pools, device=dev)

    def draw(self, B: int, K: int, N: int = D.EVAL_N):
        pi = torch.multinomial(self.prob, B, replacement=True, generator=self.gen)
        vi = self.pool_t[pi] * self.nv + torch.randint(self.nv, (B,), device=self.dev, generator=self.gen)
        j = (torch.rand(B, K, device=self.dev, generator=self.gen) * self.count[pi][:, None]).long()
        ci = self.start[pi][:, None] + j
        return vi, self.L[ci], self.Dg[ci]


def anchors(d: D.R6Data, fit_drugs, kappa: float) -> np.ndarray:
    """[L, D, G] shrunk pool mean from fitting drugs only: leave-one-drug-out for fitting conditions,
    plain for every other valid condition (held / test). Invalid conditions stay 0."""
    Lf, Df = d.conditions(fit_drugs)
    mb = MeanBaseline(d.Y, Lf, Df, C.pool_of(d, Lf, Df), d.Y.shape[0], len(d.pools))
    A = np.zeros(d.Y.shape, np.float32)
    A[Lf, Df] = mb.predict(Lf, Df, C.pool_of(d, Lf, Df), kappa, exclude_self=True)[0]
    Lo, Do = np.nonzero(d.valid)
    keep = ~np.isin(Do, np.asarray(list(fit_drugs)))
    Lo, Do = Lo[keep], Do[keep]
    A[Lo, Do] = mb.predict(Lo, Do, C.pool_of(d, Lo, Do), kappa)[0]
    return A


@torch.no_grad()
def predict(model, d: D.R6Data, drugs, dev, y_scale: float, views=None, n_cells: int = D.EVAL_N, anchor=None,
            parts: bool = False, chunk: int = 256):
    """Predictions for every valid condition of `drugs` from the first `n_cells` fixed eval_input cells.

    The eval_input cells are stored in barcode-hash order, so the first n cells are a nested subset;
    thinned views are the stored draws of those same cells. Returns (L, Dg, {(p, seed): [n, G]})
    or, with parts=True, {(p, seed): (g [n, G], h [n, G])} without anchor.
    """
    model.eval()
    L, Dg = d.conditions(drugs)
    pid = C.pool_of(d, L, Dg)
    views = views or C.VIEWS
    ev_rows = np.concatenate([np.arange(r.eval_start, r.eval_end) for r in d.pools.itertuples()])
    npool, G = len(d.pools), d.Y.shape[2]
    fp = torch.as_tensor(d.fp, device=dev)
    out = {}
    for (p, s) in views:
        c, r = (d.cnt[ev_rows], d.crest[ev_rows]) if p == 1.0 else d.eval_views[(p, s)]
        zs = []
        for i in range(0, npool, 32):
            cc = torch.as_tensor(c[i * D.EVAL_N:(i + 32) * D.EVAL_N], device=dev).float().reshape(-1, D.EVAL_N, G)[:, :n_cells]
            rr = torch.as_tensor(r[i * D.EVAL_N:(i + 32) * D.EVAL_N], device=dev).float().reshape(-1, D.EVAL_N)[:, :n_cells]
            zs.append(model.state(set_features(cc, rr))[0])
        z = torch.cat(zs)
        Pg = np.zeros((len(L), G), np.float32); Ph = np.empty((len(L), G), np.float32)
        for i in range(0, len(L), chunk):
            zi = z[torch.as_tensor(pid[i:i + chunk], device=dev)]
            g, h = model.delta_parts(zi, fp[torch.as_tensor(Dg[i:i + chunk], device=dev)].unsqueeze(1))
            Ph[i:i + chunk] = (h[:, 0] * y_scale).float().cpu().numpy()
            if g is not None:
                Pg[i:i + chunk] = (g[:, 0] * y_scale).float().cpu().numpy()
        if parts:
            out[(p, s)] = (Pg, Ph)
        else:
            out[(p, s)] = Pg + Ph + (anchor[L, Dg] if anchor is not None else 0.0)
    return L, Dg, out


def run_dir(stage: str, run_id: str) -> Path:
    return C.assets() / "runs" / stage / run_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--setting", required=True, choices=["source", "plate", "final"])
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--inner", action="store_true")
    ap.add_argument("--stage", required=True, help="run group under $R6F_ASSETS/runs/ (plumbing, confirm, branch, final)")
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--hp", default="configs/r6_followup_v1/model.json")
    ap.add_argument("--views", default="fixed", choices=["fixed", "fresh"])
    ap.add_argument("--anchor-kappa", type=float, default=None)
    ap.add_argument("--pair-lambda", type=float, default=0.0)
    ap.add_argument("--np-aug", action="store_true", help="branch F3 only: also sample the number of cells N in {64, 32, 16} (fresh views)")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--drug-shrink", type=float, default=None, help="overrides hp.drug_shrink (branch F1 development only)")
    a = ap.parse_args()
    hp = json.loads(Path(a.hp).read_text())
    if a.drug_shrink is not None:
        hp["drug_shrink"] = a.drug_shrink
    if a.setting == "final":
        fm = Path("manifests/r6_followup_v1/freeze_manifest.json")
        assert fm.exists(), "freeze the follow-up configuration before any final-setting run"
        assert a.run_id in {r["run_id"] for r in json.loads(fm.read_text())["final_runs"]}, "run is not listed in the freeze manifest"
    assert not (a.np_aug and a.views == "fixed"), "N-and-p augmentation draws fresh views"
    run = run_dir(a.stage, a.run_id)
    run.mkdir(parents=True, exist_ok=False)                           # immutable run ids
    dev = "cuda"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    gen = torch.Generator(device=dev); gen.manual_seed(a.seed)
    d = C.load()
    fit_drugs, eval_drugs = C.split_spec(d, a.setting, a.fold, a.inner)
    assert not (set(fit_drugs) & set(eval_drugs)) and not (set(fit_drugs) & set(C.test_drugs(d)))
    Lf, Df = d.conditions(fit_drugs)
    y_scale = float(np.sqrt((d.Y[Lf, Df].astype(np.float64) ** 2).mean()))   # scalar, fitting drugs only
    Y = torch.as_tensor(d.Y, device=dev); fp = torch.as_tensor(d.fp, device=dev)
    anc_np = anchors(d, fit_drugs, a.anchor_kappa) if a.anchor_kappa is not None else None
    anc = torch.as_tensor(anc_np, device=dev) if anc_np is not None else None
    view_info = dict(mode=a.views)
    if a.views == "fixed":
        v = load_views(a.seed)
        nv = int(v["reps"]) * len(D.PS)
        vc = torch.as_tensor(v["cnt"], device=dev); vr = torch.as_tensor(v["crest"], device=dev)
        smp = FixedSampler(d, fit_drugs, dev, gen, nv)
        view_info.update(cache=str(C.assets() / "cache" / "views_v1" / f"views_s{a.seed}.npz"), sha256_counts=D.sha256_array(v["cnt"]),
                         sha256_cells=D.sha256_array(v["cells"]), views_per_pool=nv, distinct_views_available=int(nv * len(smp.pools)))
    else:
        cnt = torch.as_tensor(d.cnt, device=dev); crest = torch.as_tensor(d.crest, device=dev)
        smp = Sampler(d, fit_drugs, dev, gen)
        ps = torch.tensor(D.PS, device=dev)
        ns = torch.tensor([64, 32, 16], device=dev)
    model = build_model(d, a.arm, hp).to(dev)
    B, K = hp.get("sets_per_step", 16), hp.get("drugs_per_set", 12)
    shrink = float(hp.get("drug_shrink", 0.0))
    steps_per_epoch = math.ceil(smp.n_cond / (B * K))
    total = steps_per_epoch * a.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=hp.get("weight_decay", 1e-4))
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1.0, (t + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, t / total))))
    pr = param_report(model, a.arm)
    print(json.dumps(dict(arm=a.arm, setting=a.setting, fold=a.fold, inner=a.inner, lr=a.lr, seed=a.seed, params=pr, n_fit_cond=smp.n_cond,
                          n_fit_drugs=len(fit_drugs), n_eval_drugs=len(eval_drugs), steps_per_epoch=steps_per_epoch, epochs=a.epochs,
                          y_scale=y_scale, views=view_info, anchor_kappa=a.anchor_kappa, pair_lambda=a.pair_lambda)), flush=True)
    curve, t0, used = [], time.time(), set()
    torch.cuda.reset_peak_memory_stats()
    from .metrics import cond_rmse, drug_macro
    for ep in range(a.epochs):
        model.train()
        agg = {}
        for _ in range(steps_per_epoch):
            if a.views == "fixed":
                vi, Lb, Db = smp.draw(B, K)
                c, r = vc[vi].float(), vr[vi].float()
                used.update(vi.tolist())
            else:
                n_set = int(ns[torch.randint(len(ns), (1,), device=dev, generator=gen)]) if a.np_aug else D.EVAL_N
                cells, Lb, Db = smp.draw(B, K, n_set)
                p = ps[torch.randint(len(ps), (B,), device=dev, generator=gen)]
                c, r = thin_torch(cnt[cells].float(), crest[cells].float(), p[:, None].expand(B, cells.shape[1]), gen)
            z, _ = model.state(set_features(c, r), B)
            g_part, h_part = model.delta_parts(z, fp[Db])
            o = h_part if g_part is None else g_part + h_part
            y = Y[Lb, Db] / y_scale
            pred = o if anc is None else o + anc[Lb, Db] / y_scale
            parts = dict(supervised=torch.nn.functional.mse_loss(pred, y))
            if shrink > 0:
                parts["drug_shrink"] = (h_part ** 2).mean()
            if a.pair_lambda > 0:
                oc = o - o.mean(1, keepdim=True); yc = y - y.mean(1, keepdim=True)
                parts["pair"] = (2.0 * K / (K - 1)) * ((oc - yc) ** 2).mean()   # = mean over the set's drug pairs of |diff error|^2 / G
            loss = parts["supervised"] + shrink * parts.get("drug_shrink", 0.0) + a.pair_lambda * parts.get("pair", 0.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            for k_, v_ in parts.items():
                agg[k_] = agg.get(k_, 0.0) + float(v_.detach())
        row = dict(epoch=ep + 1, seconds=round(time.time() - t0, 1), **{f"train_{k_}": v_ / steps_per_epoch for k_, v_ in agg.items()})
        if a.setting in ("source",) and a.inner and ((ep + 1) % a.eval_every == 0 or ep + 1 == a.epochs):
            Le, De, pe = predict(model, d, eval_drugs, dev, y_scale, views=[(1.0, 0), (0.25, 0)], anchor=anc_np)
            T = d.Y[Le, De]
            row["inner_rmse_p1.0"] = drug_macro(cond_rmse(pe[(1.0, 0)], T), De)
            row["inner_rmse_p0.25"] = drug_macro(cond_rmse(pe[(0.25, 0)], T), De)
        curve.append(row)
        print(json.dumps(row), flush=True)
    train_s = time.time() - t0
    torch.save(dict(state_dict=model.state_dict(), arm=a.arm, setting=a.setting, fold=a.fold, inner=a.inner, hp=hp, lr=a.lr, seed=a.seed,
                    epochs=a.epochs, y_scale=y_scale, fit_drugs=list(fit_drugs), views=view_info, anchor_kappa=a.anchor_kappa,
                    pair_lambda=a.pair_lambda, np_aug=a.np_aug), run / "ckpt.pt")
    rec = dict(run_id=a.run_id, stage=a.stage, arm=a.arm, setting=a.setting, fold=a.fold, inner=a.inner, lr=a.lr, seed=a.seed, epochs=a.epochs,
               hp=hp, params=pr, y_scale=y_scale, n_fit_conditions=smp.n_cond, n_fit_drugs=len(fit_drugs), n_eval_drugs=len(eval_drugs),
               optimizer="AdamW, cosine schedule with 5% warm-up, grad-clip 1.0", updates=total, sets_per_update=B, drugs_per_set=K,
               control_views_processed=total * B, distinct_views_used=len(used) if a.views == "fixed" else None, views=view_info,
               anchor_kappa=a.anchor_kappa, pair_lambda=a.pair_lambda, np_aug=a.np_aug,
               train_seconds=round(train_s, 1), peak_vram_MB=round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
               gpu=torch.cuda.get_device_name(0), job=os.environ.get("SLURM_JOB_ID"), array=os.environ.get("SLURM_ARRAY_JOB_ID"),
               curve=curve, ckpt_sha256=D.sha256_file(run / "ckpt.pt"), commit=C.git_commit())
    if a.setting in ("source", "plate"):                               # the TEST drugs are never predicted here
        t1 = time.time()
        Le, De, pe = predict(model, d, eval_drugs, dev, y_scale, anchor=anc_np)
        rec["predict_seconds"] = round(time.time() - t1, 2)
        np.savez_compressed(run / "pred.npz", line=Le, drug=De, **{f"p{p}_s{s}": v_.astype(np.float16) for (p, s), v_ in pe.items()})
        T = d.Y[Le, De]
        rec["eval_role"] = "inner_dev" if a.inner else ("held_plate_drugs_all_conditions" if a.setting == "plate" else "held_fold")
        rec["scores"] = {f"p{p}_s{s}": drug_macro(cond_rmse(v_, T), De) for (p, s), v_ in pe.items()}
        rec["pred_sha256"] = D.sha256_file(run / "pred.npz")
    C.dump_json(rec, run / "record.json")
    print(json.dumps({k: v_ for k, v_ in rec.items() if k not in ("curve", "hp")}), flush=True)


if __name__ == "__main__":
    main()
