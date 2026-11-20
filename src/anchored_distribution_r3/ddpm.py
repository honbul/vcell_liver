"""Conditional DDPMs on real treated cells: mean-anchored residual (proposed) and plain absolute.

target = "resid"   r = (y - mean of y's own training well) / s          (proposed)
target = "abs"     x = (y - grand mean) / total SD                      (plain conditional DDPM)

Both use the same denoiser, the same context c = (control mean, control log-variance, ECFP4), the
same v-prediction loss on a cosine schedule, the same hierarchical condition -> well -> cell
sampling and the same sampler. Treated statistics of a query well never enter the context.

Denoiser: input projection, `blocks` residual MLP blocks of width `width`, output projection, plus
a per-gene linear skip whose gain depends on (time, context). `cond`:
    film    context modulates every block (scale/shift), separate from the time embedding
    late    blocks see time only; the same context embedding is concatenated once before the output
`drug=False` replaces the fingerprint by zeros (drug-condition removal contrast).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import data as D
from .setting import build, run_dir


def alpha_bar(t):
    s = 0.008
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2 / math.cos(s / (1 + s) * math.pi / 2) ** 2
    return f.clamp(1e-5, 1.0)


def t_embed(t, dim=128):
    half = dim // 2
    freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    ang = t[:, None] * 1000.0 * freq[None]
    return torch.cat([ang.sin(), ang.cos()], 1)


class Block(nn.Module):
    def __init__(self, w, film):
        super().__init__()
        self.norm = nn.LayerNorm(w)
        self.film = nn.Linear(w, 2 * w) if film else None
        self.tproj = nn.Linear(w, w)
        self.l1, self.l2 = nn.Linear(w, w), nn.Linear(w, w)
        nn.init.zeros_(self.l2.weight); nn.init.zeros_(self.l2.bias)
        if film:
            nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)

    def forward(self, h, te, ce):
        z = self.norm(h)
        if self.film is not None:
            g, b = self.film(ce).chunk(2, -1)
            z = z * (1 + g) + b
        z = z + self.tproj(te)
        return h + self.l2(F.silu(self.l1(F.silu(z))))


class Denoiser(nn.Module):
    def __init__(self, G, fp_dim, width=256, blocks=2, cond="film", drug=True, selfcond=False, early=False):
        super().__init__()
        self.cond, self.drug, self.selfcond, self.early = cond, drug, selfcond, early
        self.pop = nn.Sequential(nn.Linear(2 * G, width), nn.LayerNorm(width))
        self.drg = nn.Sequential(nn.Linear(fp_dim, width), nn.LayerNorm(width))
        self.ctx = nn.Sequential(nn.SiLU(), nn.Linear(width, width), nn.LayerNorm(width))
        self.time = nn.Sequential(nn.Linear(128, width), nn.SiLU(), nn.Linear(width, width))
        self.inp = nn.Linear(G * (2 if selfcond else 1), width)
        self.early_proj = nn.Linear(width, width) if early else None
        self.blocks = nn.ModuleList([Block(width, cond == "film") for _ in range(blocks)])
        self.late = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU()) if cond == "late" else None
        self.out_norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, G)
        self.gain = nn.Linear(2 * width, G)
        nn.init.zeros_(self.gain.weight); nn.init.zeros_(self.gain.bias)

    def forward(self, x, t, pop, fp, x_sc=None):
        te = self.time(t_embed(t))
        ce = self.pop(pop)
        if self.drug:
            ce = ce + self.drg(fp)
        ce = self.ctx(ce)
        h = self.inp(torch.cat([x, x_sc if x_sc is not None else torch.zeros_like(x)], 1) if self.selfcond else x)
        if self.early_proj is not None:
            h = h + self.early_proj(ce)
        for b in self.blocks:
            h = b(h, te, ce)
        if self.late is not None:
            h = self.late(torch.cat([h, ce], 1))
        return self.out(self.out_norm(h)) + self.gain(torch.cat([te, ce], 1)) * x


# ------------------------------------------------------------------ context and scales
class Context:
    """Standardised control-population context per (plate, line), fitted on the setting."""

    def __init__(self, st, dev):
        keys = sorted(st.ctx)
        self.index = {k: i for i, k in enumerate(keys)}
        mu = np.stack([st.ctx[k][0] for k in keys]); lv = np.stack([st.ctx[k][1] for k in keys])
        fit_keys = sorted({(r.plate_index, r.line) for r in st.fit_wells.itertuples()})
        fi = [self.index[k] for k in fit_keys]
        self.mu_m, self.mu_s = mu[fi].mean(0), mu[fi].std(0) + 0.05
        self.lv_m, self.lv_s = lv[fi].mean(0), lv[fi].std(0) + 0.05
        pop = np.concatenate([(mu - self.mu_m) / self.mu_s, (lv - self.lv_m) / self.lv_s], 1)
        self.pop = torch.as_tensor(pop, dtype=torch.float32, device=dev)
        self.fp = torch.as_tensor(st.d.fp, dtype=torch.float32, device=dev)
        # control-variance residual scale (plan 5.3: the one alternative scale setting)
        v = np.exp(lv) - 1e-3
        s = np.sqrt(0.5 * np.maximum(v, 0) + 0.5 * np.median(st.scale["var_resid"]))
        s = np.maximum(s, D.residual_scale(st.scale["var_resid"]).min())   # same floor as the global scale
        self.s_ctrl = torch.as_tensor(s, dtype=torch.float32, device=dev)
        self.s_glob = torch.as_tensor(st.s_resid, dtype=torch.float32, device=dev)
        self.mean_abs = torch.as_tensor(st.scale["mean_abs"], dtype=torch.float32, device=dev)
        self.sd_total = torch.as_tensor(np.maximum(st.scale["sd_total"], 0.02), dtype=torch.float32, device=dev)

    def key_idx(self, wells, dev):
        return torch.as_tensor([self.index[(p, l)] for p, l in zip(wells.plate_index, wells.line)], device=dev)

    def scale(self, cfg, kidx):
        if cfg["target"] == "abs":
            return self.sd_total[None].expand(len(kidx), -1)
        return self.s_ctrl[kidx] if cfg["scale"] == "ctrl_var" else self.s_glob[None].expand(len(kidx), -1)


@torch.no_grad()
def sample(model, cfg, ctx, wells, n_each, seed, dev, steps=100, bs=16384):
    """Generate n_each[i] cells per well. Returns a list of tensors in log1p units:
    residual target -> scaled residuals (not yet centred or anchored); absolute target -> expression."""
    kidx = ctx.key_idx(wells, dev)
    reps = torch.as_tensor(np.asarray(n_each), device=dev)
    wi = torch.repeat_interleave(torch.arange(len(wells), device=dev), reps)
    drug = torch.as_tensor(wells.drug.to_numpy(), device=dev)
    g = torch.Generator(device=dev).manual_seed(int(seed))
    G = ctx.mean_abs.shape[0]
    out = torch.empty(len(wi), G, device=dev)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=dev)
    for b0 in range(0, len(wi), bs):
        w = wi[b0:b0 + bs]
        pop, fp = ctx.pop[kidx[w]], ctx.fp[drug[w]]
        x = torch.randn(len(w), G, generator=g, device=dev)
        x_sc = None
        for i in range(steps):
            t, s = ts[i], ts[i + 1]
            ab_t, ab_s = alpha_bar(t), alpha_bar(s) if i + 1 < steps else torch.tensor(1.0, device=dev)
            v = model(x, t.expand(len(w)), pop, fp, x_sc)
            x0 = ab_t.sqrt() * x - (1 - ab_t).sqrt() * v
            eps = (1 - ab_t).sqrt() * x + ab_t.sqrt() * v
            if model.selfcond:
                x_sc = x0
            if i + 1 == steps:
                x = x0
            else:
                sig = (((1 - ab_s) / (1 - ab_t)) * (1 - ab_t / ab_s)).clamp(min=0).sqrt()   # eta = 1
                x = ab_s.sqrt() * x0 + (1 - ab_s - sig ** 2).clamp(min=0).sqrt() * eps \
                    + sig * torch.randn(x.shape, generator=g, device=dev)
        sc = ctx.scale(cfg, kidx[w])
        out[b0:b0 + bs] = x * sc + (ctx.mean_abs if cfg["target"] == "abs" else 0.0)
    return list(torch.split(out, [int(n) for n in n_each]))


def load_model(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    cfg = ck["cfg"]
    m = Denoiser(ck["G"], ck["fp_dim"], cfg["width"], cfg["blocks"], cfg["cond"], cfg["drug"],
                 cfg.get("selfcond", False), cfg.get("early", False)).to(dev)
    m.load_state_dict(ck["ema"]); m.eval()
    return m, cfg


def run_id(cfg) -> str:
    h = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    return (f"{cfg['setting']}-{cfg['target']}-{cfg.get('scale','std')}-lr{cfg['lr']:g}-{cfg['cond']}"
            f"{'' if cfg['drug'] else '-nodrug'}{'-sc' if cfg.get('selfcond') else ''}"
            f"{'-early' if cfg.get('early') else ''}-w{cfg['width']}b{cfg['blocks']}-st{cfg['steps']}-s{cfg['seed']}-{h}")


def ledger(rec):
    p = Path(os.environ["R3_ASSETS"]) / "runs" / "ledger.jsonl"
    with open(p, "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", default="dev")
    ap.add_argument("--target", choices=["resid", "abs"], default="resid")
    ap.add_argument("--scale", choices=["resid_global", "ctrl_var", "std"], default="resid_global")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--cond", choices=["film", "late"], default="film")
    ap.add_argument("--no-drug", action="store_true")
    ap.add_argument("--selfcond", action="store_true")
    ap.add_argument("--early", action="store_true")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--wells-per-batch", type=int, default=256)
    ap.add_argument("--cells-per-well", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage", default="")
    a = ap.parse_args()
    cfg = dict(setting=a.setting, target=a.target, scale="std" if a.target == "abs" else a.scale, lr=a.lr,
               width=a.width, blocks=a.blocks, cond=a.cond, drug=not a.no_drug, selfcond=a.selfcond,
               early=a.early, steps=a.steps, wells_per_batch=a.wells_per_batch,
               cells_per_well=a.cells_per_well, seed=a.seed, loss="v", schedule="cosine", ema=0.999)
    rid = run_id(cfg)
    out = run_dir("ddpm", rid)
    if (out / "model.pt").exists() or (out / "FAILED").exists():
        raise SystemExit(f"run id {rid} already exists; run ids are immutable")
    dev = "cuda"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    t0 = time.time()
    st = build(a.setting, dev)
    ctx = Context(st, dev)
    fw = st.fit_wells.reset_index(drop=True)
    G = st.d.X.shape[1]
    # all fitting cells on the GPU, grouped by well
    Xs = torch.cat([torch.as_tensor(np.asarray(st.d.X[r.start:r.end])) for r in fw.itertuples()]).to(dev)
    n_w = torch.as_tensor(fw.n_cells.to_numpy(), device=dev)
    off = torch.cumsum(n_w, 0) - n_w
    wmean = torch.stack([Xs[o:o + n].mean(0) for o, n in zip(off.tolist(), n_w.tolist())])
    kidx = ctx.key_idx(fw, dev); drug = torch.as_tensor(fw.drug.to_numpy(), device=dev)
    conds = fw.groupby(["line", "drug"]).indices
    cw = [torch.as_tensor(v, device=dev) for v in conds.values()]
    maxw = max(len(v) for v in cw)
    cmat = torch.stack([F.pad(v, (0, maxw - len(v)), value=int(v[0])) for v in cw]); ccount = torch.as_tensor([len(v) for v in cw], device=dev)
    model = Denoiser(G, ctx.fp.shape[1], a.width, a.blocks, a.cond, not a.no_drug, a.selfcond, a.early).to(dev)
    ema = Denoiser(G, ctx.fp.shape[1], a.width, a.blocks, a.cond, not a.no_drug, a.selfcond, a.early).to(dev)
    ema.load_state_dict(model.state_dict())
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(i, a.steps) / a.steps)))
    g = torch.Generator(device=dev).manual_seed(a.seed)
    print(f"run={rid} fit_wells={len(fw)} conds={len(cw)} cells={len(Xs)} params={n_par} setup={time.time()-t0:.0f}s", flush=True)
    hist, run, status, first = [], 0.0, "ok", None
    t1 = time.time()
    for it in range(a.steps):
        ci = torch.randint(0, len(cw), (a.wells_per_batch,), generator=g, device=dev)
        wi = cmat[ci, (torch.rand(len(ci), generator=g, device=dev) * ccount[ci]).long()]
        wi = wi.repeat_interleave(a.cells_per_well)
        cell = off[wi] + (torch.rand(len(wi), generator=g, device=dev) * n_w[wi]).long()
        y = Xs[cell]
        sc = ctx.scale(cfg, kidx[wi])
        if a.target == "resid":
            nn_ = n_w[wi].float()[:, None]
            x0 = (y - wmean[wi]) * (nn_ / (nn_ - 1)).sqrt() / sc
        else:
            x0 = (y - ctx.mean_abs) / sc
        t = torch.rand(len(wi), generator=g, device=dev)
        ab = alpha_bar(t)[:, None]
        eps = torch.randn(x0.shape, generator=g, device=dev)
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        vt = ab.sqrt() * eps - (1 - ab).sqrt() * x0
        pop, fp = ctx.pop[kidx[wi]], ctx.fp[drug[wi]]
        x_sc = None
        if a.selfcond and float(torch.rand(1, generator=g, device=dev)) < 0.5:
            with torch.no_grad():
                v0 = model(xt, t, pop, fp, None)
                x_sc = (ab.sqrt() * xt - (1 - ab).sqrt() * v0).detach()
        loss = ((model(xt, t, pop, fp, x_sc) - vt) ** 2).mean()
        if not torch.isfinite(loss) or (first is not None and float(loss) > 20 * first):
            status = "diverged"; print(f"diverged at step {it}: loss={float(loss)}", flush=True); break
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sched.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - cfg["ema"])
        run += float(loss)
        if first is None:
            first = float(loss)
        if (it + 1) % 1000 == 0:
            hist.append(dict(step=it + 1, loss=run / 1000, wall_s=time.time() - t1)); run = 0.0
            print(hist[-1], flush=True)
    wall = time.time() - t1
    vram = torch.cuda.max_memory_allocated() / 2 ** 30
    rec = dict(run_id=rid, kind="ddpm_train", stage=a.stage, status=status, cfg=cfg, n_params=n_par,
               n_fit_wells=int(len(fw)), n_fit_cells=int(len(Xs)), wall_s=wall, setup_s=t1 - t0,
               peak_vram_gb=vram, gpu=torch.cuda.get_device_name(0), final_loss=hist[-1]["loss"] if hist else None,
               job=os.environ.get("SLURM_JOB_ID"), commit=os.popen("git rev-parse HEAD").read().strip(),
               time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    if status == "ok":
        torch.save(dict(cfg=cfg, ema=ema.state_dict(), G=G, fp_dim=ctx.fp.shape[1], hist=hist), out / "model.pt")
        rec["sha256_model"] = D.sha256_file(out / "model.pt"); rec["path"] = str(out / "model.pt")
    else:
        (out / "FAILED").write_text(status)
    D.dump_json(rec, out / "record.json")
    ledger(rec)
    print(json.dumps(rec, indent=1, default=float), flush=True)


if __name__ == "__main__":
    main()
