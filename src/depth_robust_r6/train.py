"""Train one arm of R6 in one setting. Every arm sees the same cells, drugs, genes and step budget.

Arms
    TEACH0   basic input, full observation only, supervised          (fixed teacher of B01)
    TEACH1   obs input,   full observation only, supervised          (fixed teacher of B11)
    B00      basic input, count-thinning augmentation, supervised    (the strong control)
    B10      obs input,   same augmentation, supervised
    B01      basic input, same augmentation, supervised + response/latent consistency to TEACH0
    B11      obs input,   same augmentation, supervised + response/latent consistency to TEACH1
    DEN      obs input,   same augmentation, supervised + full-observation expression reconstruction
    NOSTATE  no RNA state input (a learned constant), supervised

A training example: one control set (N=64 fit_input cells of one pool, one keep-rate p drawn from
{1, 0.5, 0.25}) with K fitting drugs measured in that pool; the target of each is the measured
full-observation delta. Evaluation-role drugs and eval_input cells are never touched here except
by `predict` after training in the dev setting (learning curve on the dev fold).
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

from . import data as D
from . import loss as LS
from .count_view import set_features, thin_torch
from .encoder import ResponseModel, n_params

ARMS = dict(
    TEACH0=dict(obs=False, aug=False, state=True, w={}),
    TEACH1=dict(obs=True, aug=False, state=True, w={}),
    B00=dict(obs=False, aug=True, state=True, w={}),
    B10=dict(obs=True, aug=True, state=True, w={}),
    B01=dict(obs=False, aug=True, state=True, w=dict(response=0.5, latent=0.5), teacher="TEACH0"),
    B11=dict(obs=True, aug=True, state=True, w=dict(response=0.5, latent=0.5), teacher="TEACH1"),
    DEN=dict(obs=True, aug=True, state=True, w=dict(reconstruction=1.0)),
    NOSTATE=dict(obs=False, aug=False, state=False, w={}),
)


def build_model(d: D.R6Data, arm: str, hp: dict) -> ResponseModel:
    a = ARMS[arm]
    return ResponseModel(d.Y.shape[2], d.fp.shape[1], obs_input=a["obs"], use_state=a["state"],
                         recon_head="reconstruction" in a["w"], latent_proj="latent" in a["w"],
                         d=hp.get("d", 128), n_tok=hp.get("n_tok", 256), n_blocks=hp.get("n_blocks", 2),
                         dropout=hp.get("dropout", 0.1), x_mean=d.x_mean, x_sd=d.x_sd, tiebreak=d.tiebreak) \
        if a["state"] else ResponseModel(d.Y.shape[2], d.fp.shape[1], use_state=False, d=hp.get("d", 128),
                                         dropout=hp.get("dropout", 0.1))


class Sampler:
    """Draws (control set, K drugs) training examples on the GPU."""

    def __init__(self, d: D.R6Data, fit_drugs, dev, gen):
        L, Dg = d.conditions(fit_drugs)
        pid = d.pool_id[d.plate[L, Dg], L]
        assert (pid >= 0).all()
        self.n_cond = len(L)
        order = np.argsort(pid, kind="stable")
        L, Dg, pid = L[order], Dg[order], pid[order]
        self.pools, start = np.unique(pid, return_index=True)
        cnt = np.diff(np.r_[start, len(pid)])
        self.start = torch.as_tensor(start, device=dev); self.count = torch.as_tensor(cnt, device=dev)
        self.prob = torch.as_tensor(cnt / cnt.sum(), dtype=torch.float32, device=dev)
        self.L = torch.as_tensor(L, device=dev); self.Dg = torch.as_tensor(Dg, device=dev)
        pt = d.pools.set_index("pool_id").loc[self.pools]
        self.fit_start = torch.as_tensor(pt.fit_start.to_numpy(), device=dev)
        self.fit_n = torch.as_tensor(pt.n_fit_input.to_numpy(), device=dev)
        self.dev, self.gen = dev, gen

    def draw(self, B: int, K: int, N: int = D.EVAL_N):
        pi = torch.multinomial(self.prob, B, replacement=True, generator=self.gen)
        r = torch.rand(B, int(self.fit_n.max()), device=self.dev, generator=self.gen)
        r = r.masked_fill(torch.arange(r.shape[1], device=self.dev)[None] >= self.fit_n[pi][:, None], 2.0)
        cells = self.fit_start[pi][:, None] + r.argsort(1)[:, :N]             # N distinct fit_input cells
        j = (torch.rand(B, K, device=self.dev, generator=self.gen) * self.count[pi][:, None]).long()
        ci = self.start[pi][:, None] + j
        return cells, self.L[ci], self.Dg[ci]


@torch.no_grad()
def predict(model, d: D.R6Data, drugs, dev, y_scale: float, views=None, chunk: int = 256):
    """Predictions for every valid condition of `drugs` from the fixed eval_input cells.

    Returns (L, Dg, {(p, seed): [n_cond, G] float32}). p = 1.0 has the single seed 0 (identity).
    """
    model.eval()
    L, Dg = d.conditions(drugs)
    pid = d.pool_id[d.plate[L, Dg], L]
    views = views or [(1.0, 0)] + [(p, s) for p in D.PS[1:] for s in D.THIN_SEEDS]
    ev_rows = np.concatenate([np.arange(r.eval_start, r.eval_end) for r in d.pools.itertuples()])
    npool = len(d.pools)
    fp = torch.as_tensor(d.fp, device=dev)
    out = {}
    for (p, s) in views:
        if p == 1.0:
            c, r = d.cnt[ev_rows], d.crest[ev_rows]
        else:
            c, r = d.eval_views[(p, s)]
        if model.use_state:
            zs = []
            for i in range(0, npool, 32):
                cc = torch.as_tensor(c[i * D.EVAL_N:(i + 32) * D.EVAL_N], device=dev).float().reshape(-1, D.EVAL_N, c.shape[1])
                rr = torch.as_tensor(r[i * D.EVAL_N:(i + 32) * D.EVAL_N], device=dev).float().reshape(-1, D.EVAL_N)
                zs.append(model.state(set_features(cc, rr))[0])
            z = torch.cat(zs)
        else:
            z = model.z0.expand(npool, -1)
        P = np.empty((len(L), d.Y.shape[2]), np.float32)
        for i in range(0, len(L), chunk):
            zi = z[torch.as_tensor(pid[i:i + chunk], device=dev)]
            P[i:i + chunk] = (model.delta(zi, fp[torch.as_tensor(Dg[i:i + chunk], device=dev)].unsqueeze(1))[:, 0] * y_scale).float().cpu().numpy()
        out[(p, s)] = P
    return L, Dg, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--setting", required=True, choices=["dev", "final"])
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--teacher", default=None, help="run dir of the fixed teacher (teacher arms only)")
    ap.add_argument("--hp", default="configs/depth_robust_r6/model.json")
    ap.add_argument("--weights", default=None, help="json overriding the arm's auxiliary loss weights (revisions)")
    ap.add_argument("--eval-every", type=int, default=5)
    a = ap.parse_args()
    hp = json.loads(Path(a.hp).read_text())
    arm = dict(ARMS[a.arm])
    if a.weights:
        arm["w"] = json.loads(a.weights)
    run = Path(os.environ["R6_ASSETS"]) / "runs" / a.setting / a.run_id
    run.mkdir(parents=True, exist_ok=False)                           # immutable run ids
    dev = "cuda"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    gen = torch.Generator(device=dev); gen.manual_seed(a.seed)
    d = D.load()
    fit_drugs, eval_drugs = d.split_spec(a.setting)
    Lf, Df = d.conditions(fit_drugs)
    y_scale = float(np.sqrt((d.Y[Lf, Df].astype(np.float64) ** 2).mean()))   # scalar, fitting drugs only
    cnt = torch.as_tensor(d.cnt, device=dev); crest = torch.as_tensor(d.crest, device=dev)
    Y = torch.as_tensor(d.Y, device=dev); fp = torch.as_tensor(d.fp, device=dev)
    smp = Sampler(d, fit_drugs, dev, gen)
    model = build_model(d, a.arm, hp).to(dev)
    teacher, t_info = None, None
    if "teacher" in arm:
        assert a.teacher, "teacher arm needs --teacher"
        ck = torch.load(Path(a.teacher) / "ckpt.pt", map_location="cpu", weights_only=False)
        assert ck["arm"] == arm["teacher"] and ck["setting"] == a.setting, (ck["arm"], ck["setting"])
        assert sorted(ck["fit_drugs"]) == sorted(fit_drugs), "teacher must be fitted on exactly the student's fitting drugs"
        teacher = build_model(d, ck["arm"], ck["hp"]).to(dev)
        teacher.load_state_dict(ck["state_dict"]); teacher.eval()
        for q in teacher.parameters():
            q.requires_grad_(False)
        t_info = dict(run=str(a.teacher), arm=ck["arm"], sha256=D.sha256_file(Path(a.teacher) / "ckpt.pt"))
    B, K = hp.get("sets_per_step", 16), hp.get("drugs_per_set", 12)
    steps_per_epoch = math.ceil(smp.n_cond / (B * K))
    total = steps_per_epoch * a.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=hp.get("weight_decay", 1e-4))
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1.0, (t + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, t / total))))
    ps = torch.tensor(D.PS if arm["aug"] else (1.0,), device=dev)
    x_mean, x_sd = torch.as_tensor(d.x_mean, device=dev), torch.as_tensor(d.x_sd, device=dev)
    print(json.dumps(dict(arm=a.arm, setting=a.setting, lr=a.lr, seed=a.seed, params=n_params(model), n_fit_cond=smp.n_cond,
                          n_fit_drugs=len(fit_drugs), steps_per_epoch=steps_per_epoch, epochs=a.epochs, y_scale=y_scale,
                          weights=arm["w"], teacher=t_info)), flush=True)
    curve, t0 = [], time.time()
    torch.cuda.reset_peak_memory_stats()
    for ep in range(a.epochs):
        model.train()
        agg = {}
        for _ in range(steps_per_epoch):
            cells, Lb, Db = smp.draw(B, K)
            c_full, r_full = cnt[cells].float(), crest[cells].float()
            p = ps[torch.randint(len(ps), (B,), device=dev, generator=gen)]
            c, r = thin_torch(c_full, r_full, p[:, None].expand(B, cells.shape[1]), gen)
            f = set_features(c, r) if model.use_state else None
            pred, z, hc = model(f, fp[Db])
            y = Y[Lb, Db] / y_scale
            parts = dict(supervised=LS.supervised(pred, y))
            if teacher is not None:
                with torch.no_grad():
                    tp, tz, _ = teacher(set_features(c_full, r_full), fp[Db])
                parts["response"] = LS.response_consistency(pred, tp)
                parts["latent"] = LS.latent_consistency(model.proj(z), tz)
            if model.recon is not None:
                x_full = set_features(c_full, r_full)["x"]
                parts["reconstruction"] = LS.reconstruction(model.recon(hc), x_full, x_mean, x_sd)
            loss = LS.total(parts, dict(supervised=1.0, **arm["w"]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            for k_, v_ in parts.items():
                agg[k_] = agg.get(k_, 0.0) + float(v_.detach())
        row = dict(epoch=ep + 1, seconds=round(time.time() - t0, 1), **{f"train_{k_}": v_ / steps_per_epoch for k_, v_ in agg.items()})
        if a.setting == "dev" and ((ep + 1) % a.eval_every == 0 or ep + 1 == a.epochs):
            Le, De, pr = predict(model, d, eval_drugs, dev, y_scale, views=[(1.0, 0), (0.25, 0)])
            from .evaluate import cond_rmse, drug_macro
            T = d.Y[Le, De]
            row["dev_rmse_p1.0"] = drug_macro(cond_rmse(pr[(1.0, 0)], T), De)
            row["dev_rmse_p0.25"] = drug_macro(cond_rmse(pr[(0.25, 0)], T), De)
        curve.append(row)
        print(json.dumps(row), flush=True)
    train_s = time.time() - t0
    torch.save(dict(state_dict=model.state_dict(), arm=a.arm, setting=a.setting, hp=hp, lr=a.lr, seed=a.seed,
                    epochs=a.epochs, y_scale=y_scale, fit_drugs=list(fit_drugs), weights=arm["w"], teacher=t_info), run / "ckpt.pt")
    rec = dict(run_id=a.run_id, arm=a.arm, setting=a.setting, lr=a.lr, seed=a.seed, epochs=a.epochs, hp=hp, weights=arm["w"],
               teacher=t_info, params=n_params(model), y_scale=y_scale, n_fit_conditions=smp.n_cond, n_fit_drugs=len(fit_drugs),
               steps=total, train_seconds=round(train_s, 1), peak_vram_MB=round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
               gpu=torch.cuda.get_device_name(0), job=os.environ.get("SLURM_JOB_ID"), curve=curve,
               ckpt_sha256=D.sha256_file(run / "ckpt.pt"),
               commit=os.popen("git rev-parse HEAD").read().strip())
    if a.setting == "dev":                                             # the held-out TEST drugs are never predicted here
        Le, De, pr = predict(model, d, eval_drugs, dev, y_scale)
        np.savez_compressed(run / "pred_dev.npz", line=Le, drug=De,
                            **{f"p{p}_s{s}": v.astype(np.float16) for (p, s), v in pr.items()})
        from .evaluate import cond_rmse, drug_macro
        T = d.Y[Le, De]
        rec["dev"] = {f"p{p}_s{s}": drug_macro(cond_rmse(v, T), De) for (p, s), v in pr.items()}
    D.dump_json(rec, run / "record.json")
    print(json.dumps({k: v for k, v in rec.items() if k not in ("curve", "hp")}), flush=True)


if __name__ == "__main__":
    main()
