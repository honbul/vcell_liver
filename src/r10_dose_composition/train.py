"""B3: train the 2x2 heads (pi-head, Q-head) and the compact whole-distribution model W.

Training examples: eligible CROSS-replicate triplets of TRAIN drugs, both episodes, all lines.
Selection: DEVELOPMENT drugs (cross-replicate triplets), each head by its own dev objective:
  pi-head  drug-balanced dev TV between the head composition and the observed query composition
  Q-head   drug-balanced dev weighted state-mean residual MSE
  W        drug-balanced dev energy distance (all cells)
Grid (identical for every head): weight decay {1e-4, 1e-2}; hidden 64; dropout 0.1; Adam 1e-3;
early stopping on the dev objective. D11 = the selected pi-head + the selected Q-head (independent
heads, no shared trunk), so the 2x2 arms share one training budget per head.
Query cells/means/compositions of train drugs are targets; they never enter the features.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from .core import LINES, LineData, build_context, build_query, neff
from .methods import EPS, MLP, _mix, features


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def build_examples(cfg, trip, Ds, role, K_pad, drugs=None, keep_clouds=False, max_cells=None, seed=0,
                   direction="cross"):
    rng = np.random.default_rng(seed)
    t = trip[(trip.role == role) & (trip.direction == direction) & trip.eligible]
    if drugs is not None:
        t = t[t.drug.isin(drugs)]
    ex = []
    for r in t.itertuples():
        D = Ds[r.line]
        e = cfg["episodes"][r.episode]
        ctx = build_context(D, r.drug, r.episode, e, r.rep_endpoint)
        q = build_query(D, r.drug, e["query"], r.rep_query)
        f, I = features(ctx, cfg, K_pad)
        K, d = D.K, D.Z.shape[1]
        piQ = q.S.mean(0)
        nk = q.S.sum(0)
        mQ = (q.S.T @ q.Z) / np.maximum(nk[:, None], 1e-12)
        nq = np.array([neff(q.S[:, k]) for k in range(K)])
        wq = np.where(nq >= 3, nq / (nq + cfg["shrink"]["kappa_mean"]), 0.0)
        x = dict(line=r.line, drug=r.drug, episode=r.episode, f=f, K=K,
                 pi_int=np.pad(I["pi"], (0, K_pad - K)), pi_q=np.pad(piQ, (0, K_pad - K)),
                 mask=np.pad(np.ones(K), (0, K_pad - K)),
                 r_target=np.pad(mQ - I["m_raw"], ((0, K_pad - K), (0, 0))), wq=np.pad(wq, (0, K_pad - K)))
        if keep_clouds:
            X, w, g, _ = _mix(ctx)
            Q = q.Z
            if max_cells is not None:
                sel = np.concatenate([rng.permutation(np.where(g == j)[0])[:max_cells] for j in (0, 1)])
                X, w, g = X[sel], w[sel], g[sel]
                for j, tj in ((0, 1 - ctx.t), (1, ctx.t)):
                    w[g == j] = tj / (g == j).sum()
                Q = Q[rng.permutation(len(Q))[:max_cells]]
            x.update(X=X, w=w, g=g, Q=Q)
        ex.append(x)
    return ex


def drug_weights(ex):
    c = pd.Series([e["drug"] for e in ex]).value_counts()
    w = np.array([1.0 / c[e["drug"]] for e in ex])
    return w / w.sum()


def stack(ex, key):
    return torch.as_tensor(np.stack([e[key] for e in ex]), dtype=torch.float32)


def pi_loss_and_tv(out, B):
    lg = torch.log(B["pi_int"] + EPS) + out
    lg = lg.masked_fill(B["mask"] == 0, -1e9)
    logp = torch.log_softmax(lg, 1)
    p = logp.exp()
    kl = (B["pi_q"] * (torch.log(B["pi_q"] + 1e-12) - logp)).masked_fill(B["mask"] == 0, 0).sum(1)
    tvv = 0.5 * (p - B["pi_q"]).abs().sum(1)
    return kl, tvv


def q_loss(out, B, d):
    r = out.view(len(out), -1, d)
    se = ((r - B["r_target"]) ** 2).mean(2)
    return (B["wq"] * se).sum(1) / B["wq"].sum(1).clamp(min=1e-6)


def batch(ex, f_mean, f_sd):
    return dict(f=(stack(ex, "f") - f_mean) / f_sd, pi_int=stack(ex, "pi_int"), pi_q=stack(ex, "pi_q"),
                mask=stack(ex, "mask"), r_target=stack(ex, "r_target"), wq=stack(ex, "wq"),
                dw=torch.as_tensor(drug_weights(ex), dtype=torch.float32))


def fit_head(kind, tr, dv, d, K_pad, seed, wd, max_epochs=600, patience=60):
    torch.manual_seed(seed)
    n_out = K_pad if kind == "pi" else K_pad * d
    net = MLP(tr["f"].shape[1], n_out)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=wd)

    def objective(B, train):
        out = net(B["f"])
        if kind == "pi":
            kl, tvv = pi_loss_and_tv(out, B)
            return ((kl if train else tvv) * B["dw"]).sum()
        return (q_loss(out, B, d) * B["dw"]).sum()

    net.eval()
    with torch.no_grad():
        best, best_ep = float(objective(dv, False)), 0
    start = best
    best_state = {k: v.clone() for k, v in net.state_dict().items()}
    curve = [(0, float(objective(tr, True)), best)]
    for ep in range(1, max_epochs + 1):
        net.train()
        opt.zero_grad()
        loss = objective(tr, True)
        loss.backward()
        opt.step()
        if ep % 5 == 0:
            net.eval()
            with torch.no_grad():
                v = float(objective(dv, False))
                curve.append((ep, float(objective(tr, True)), v))
            if v < best:
                best, best_ep = v, ep
                best_state = {k: x.clone() for k, x in net.state_dict().items()}
            elif ep - best_ep >= patience:
                break
    return dict(state=best_state, dev=best, dev_at_init=start, epoch=best_ep, curve=curve)


def ed_torch(X, w, g, Q):
    """Differentiable version of core.cloud_stat (ED) for the two-group mixture vs a uniform query."""
    def blk(A, a, B, b, same):
        D = torch.cdist(A, B)
        W = a[:, None] * b[None, :]
        if same:
            W = W * (1 - torch.eye(len(A)))
        return (W * D).sum() / W.sum()
    out = 0.0
    tot = w.sum()
    m0, m1 = g == 0, g == 1
    a0, a1 = w[m0].sum() / tot, w[m1].sum() / tot
    s = (a0 ** 2 * blk(X[m0], w[m0], X[m0], w[m0], True) + a1 ** 2 * blk(X[m1], w[m1], X[m1], w[m1], True)
         + 2 * a0 * a1 * blk(X[m0], w[m0], X[m1], w[m1], False))
    u = torch.full((len(Q),), 1.0 / len(Q), dtype=X.dtype)
    cross = blk(X, w / tot, Q, u, False)
    qq = blk(Q, u, Q, u, True)
    return 2 * cross - s - qq


def fit_W(tr_ex, dv_ex, f_mean, f_sd, d, seed, wd, max_epochs=150, patience=30, bs=64):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = MLP(len(f_mean), 2 * d)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=wd)
    dw_tr = drug_weights(tr_ex)
    dw_dv = drug_weights(dv_ex)

    def ep_ed(e, o):
        X = torch.as_tensor(e["X"], dtype=torch.float32)
        w = torch.as_tensor(e["w"], dtype=torch.float32)
        g = torch.as_tensor(e["g"])
        mu = (w[:, None] * X).sum(0) / w.sum()
        Xt = mu + (X - mu) * torch.exp(o[d:]) + o[:d]
        return ed_torch(Xt, w, g, torch.as_tensor(e["Q"], dtype=torch.float32))

    def dev_obj():
        net.eval()
        with torch.no_grad():
            F = (torch.as_tensor(np.stack([e["f"] for e in dv_ex])) - f_mean) / f_sd
            O = net(F)
            return float(sum(dw_dv[i] * ep_ed(e, O[i]) for i, e in enumerate(dv_ex)))

    best = start = dev_obj()
    best_ep, best_state = 0, {k: v.clone() for k, v in net.state_dict().items()}
    curve = [(0, np.nan, best)]
    F_tr = (torch.as_tensor(np.stack([e["f"] for e in tr_ex])) - f_mean) / f_sd
    for ep in range(1, max_epochs + 1):
        net.train()
        perm = rng.permutation(len(tr_ex))
        tot = 0.0
        for s in range(0, len(perm), bs):
            idx = perm[s:s + bs]
            opt.zero_grad()
            O = net(F_tr[idx])
            ww = torch.as_tensor(dw_tr[idx] / dw_tr[idx].sum(), dtype=torch.float32)
            loss = sum(ww[i] * ep_ed(tr_ex[j], O[i]) for i, j in enumerate(idx))
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx) / len(perm)
        if ep % 5 == 0:
            v = dev_obj()
            curve.append((ep, tot, v))
            if v < best:
                best, best_ep = v, ep
                best_state = {k: x.clone() for k, x in net.state_dict().items()}
            elif ep - best_ep >= patience:
                break
    return dict(state=best_state, dev=best, dev_at_init=start, epoch=best_ep, curve=curve)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--triplets", required=True)
    ap.add_argument("--atlas_K", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--heads", default="pi,Q,W")
    ap.add_argument("--direction", default="cross", help="replicate contract of train and dev examples")
    ap.add_argument("--pilot_train_drugs", type=int, default=0)
    ap.add_argument("--pilot_dev_drugs", type=int, default=0)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    os.makedirs(args.out, exist_ok=True)
    for s_ in args.seeds.split(","):
        for k_ in args.heads.split(","):
            assert not os.path.exists(f"{args.out}/{k_}_seed{s_}.pt"), "immutable run: head exists"
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))
    Kmap = json.load(open(args.atlas_K))
    K_pad = max(Kmap.values())
    Ds = {l: LineData(cfg, l, Kmap[l]) for l in LINES}
    d = Ds["A549"].Z.shape[1]
    trip = pd.read_csv(args.triplets, sep="\t")
    rng = np.random.default_rng(cfg["split"]["seed"])
    tr_drugs = dv_drugs = None
    if args.pilot_train_drugs:
        tr_drugs = list(rng.permutation(sorted(trip[trip.role == "train"].drug.unique()))[:args.pilot_train_drugs])
        dv_drugs = list(rng.permutation(sorted(trip[trip.role == "development"].drug.unique()))[:args.pilot_dev_drugs])
    heads = args.heads.split(",")
    needW = "W" in heads
    tr = build_examples(cfg, trip, Ds, "train", K_pad, tr_drugs, keep_clouds=needW, max_cells=128,
                        direction=args.direction)
    dv = build_examples(cfg, trip, Ds, "development", K_pad, dv_drugs, keep_clouds=needW, max_cells=None,
                        direction=args.direction)
    log("examples train", len(tr), "dev", len(dv), "K_pad", K_pad)
    F = np.stack([e["f"] for e in tr])
    f_mean = torch.as_tensor(F.mean(0), dtype=torch.float32)
    f_sd = torch.as_tensor(F.std(0) + 1e-3, dtype=torch.float32)
    TR, DV = batch(tr, f_mean, f_sd), batch(dv, f_mean, f_sd)
    sel = dict(direction=args.direction, train_contract=args.direction, n_train_examples=len(tr), n_dev_examples=len(dv), n_train_drugs=len({e["drug"] for e in tr}),
               n_dev_drugs=len({e["drug"] for e in dv}), K_pad=K_pad, K=Kmap, heads={})
    for seed in [int(s) for s in args.seeds.split(",")]:
        for kind in heads:
            res = []
            for wd in (1e-4, 1e-2):
                t0 = time.time()
                if kind == "W":
                    r = fit_W(tr, dv, f_mean, f_sd, d, seed, wd)
                else:
                    r = fit_head(kind, TR, DV, d, K_pad, seed, wd)
                r["wd"], r["sec"] = wd, time.time() - t0
                res.append(r)
                log(kind, "seed", seed, "wd", wd, "dev", round(r["dev"], 5), "init", round(r["dev_at_init"], 5),
                    "epoch", r["epoch"], f"{r['sec']:.0f}s")
            b = min(res, key=lambda r: r["dev"])
            n_out = K_pad if kind == "pi" else (K_pad * d if kind == "Q" else 2 * d)
            torch.save(dict(kind=kind, state_dict=b["state"], f_mean=f_mean.numpy(), f_sd=f_sd.numpy(),
                            n_in=len(f_mean), n_out=n_out, K=K_pad, d=d, hidden=64, wd=b["wd"], epoch=b["epoch"],
                            train_contract=args.direction),
                       f"{args.out}/{kind}_seed{seed}.pt")
            sel["heads"][f"{kind}_seed{seed}"] = dict(
                chosen_wd=b["wd"], chosen_epoch=b["epoch"], dev_objective=b["dev"], dev_at_init=b["dev_at_init"],
                grid=[dict(wd=r["wd"], dev=r["dev"], epoch=r["epoch"], sec=r["sec"]) for r in res],
                curve=b["curve"])
    tag = "seeds" + args.seeds.replace(",", "-")
    json.dump(sel, open(f"{args.out}/selection_{tag}.json", "w"), indent=1, default=float)
    open(f"{args.out}/DONE_{tag}", "w").write(time.strftime("%Y-%m-%dT%H:%M:%S"))


if __name__ == "__main__":
    main()
