"""Training of the R7 arms (chem / concat / gated / fixed) with or without knowledge-missingness augmentation.
Validation units (never test) select the checkpoint; the validation score is the source panel only (no HCC number)."""
import argparse, json, os, time
import numpy as np
import pandas as pd
import torch

from .common import new_run_dir, run_record, dump, pman, sha_list, kg
from .data import Store, TrainStats, Knowledge
from .evaluate import unit_metrics, summarise
from .models import R7Model

SEED_COLS = Knowledge.SEED


def load_graph(K, variant="real"):
    G = dict(edge_index=K.edge_index, edge_rel=K.edge_rel, edge_feat=K.edge_feat, measured=K.measured, lm_idx=K.lm_idx)
    if variant == "rewired":
        G["edge_index"] = np.load(kg("edge_index_rewired.npy"))
    return G


class SeedBank:
    """Per-group seed arrays; builds (possibly augmented / masked) batch seeds."""

    def __init__(self, K, evidence="on"):
        self.K = K
        self.rows = {}
        for g, d in K.by_group.items():
            f = d[SEED_COLS].to_numpy(np.float32)
            if evidence == "off":     # attribution: evidence kinds / signs / mapping removed -> presence indicator only
                f = np.zeros_like(f); f[:, 0] = 1.0
            self.rows[g] = (d.node.to_numpy(np.int64), f, [set(str(t).split("|")) for t in d.tids])
        self.evidence = evidence

    def status(self, nodes, f):
        if len(nodes) == 0:
            return np.zeros(4, np.float32)
        if self.evidence == "off":
            return np.array([0, 0, min(np.log1p(len(nodes)) / np.log(201), 1.0), 0], np.float32)
        return np.array([f[:, 0].max(), max(f[:, 1].max(), f[:, 2].max()), min(np.log1p(len(nodes)) / np.log(201), 1.0),
                         1.0 - f[:, 5].mean()], np.float32)

    def augment(self, g, rng, p):
        nodes, f, tids = self.rows[g]
        u = rng.rand()
        if u < p["all"]:
            return nodes[:0], f[:0]
        u -= p["all"]
        f = f.copy()
        if u < p["source"]:
            has_m = f[:, 0].max() > 0; has_a = max(f[:, 1].max(), f[:, 2].max()) > 0
            if has_m and has_a:
                if rng.rand() < 0.5:     # drop mechanism evidence
                    f[:, [0, 3, 4]] = 0; f[:, 5] = 1
                else:                    # drop activity evidence
                    f[:, [1, 2]] = 0
                keep = f[:, :3].max(1) > 0
                return nodes[keep], f[keep]
            return nodes, f
        u -= p["source"]
        if u < p["block"]:
            blocks = sorted(set().union(*tids))
            if len(blocks) > 1:
                drop = set(rng.choice(blocks, len(blocks) // 2, replace=False))
                keep = np.array([not t <= drop for t in tids])
                return nodes[keep], f[keep]
        return nodes, f

    def mask_fraction(self, g, frac, rng):
        """Annotation stress: remove a fraction of the drug's tid blocks (1.0 removes everything)."""
        nodes, f, tids = self.rows[g]
        if frac >= 1.0:
            return nodes[:0], f[:0]
        if frac <= 0:
            return nodes, f
        blocks = sorted(set().union(*tids))
        drop = set(rng.choice(blocks, int(round(frac * len(blocks))), replace=False)) if len(blocks) > 1 else set()
        keep = np.array([not t <= drop for t in tids])
        return nodes[keep], f[keep]

    def batch(self, groups, dev, rng=None, p=None, mask_frac=0.0, mask_rng=None):
        bi, ni, sf, st, ma = [], [], [], [], []
        for b, g in enumerate(groups):
            if g in self.rows:
                if p is not None:
                    nodes, f = self.augment(g, rng, p)
                elif mask_frac > 0:
                    nodes, f = self.mask_fraction(g, mask_frac, mask_rng)
                else:
                    nodes, f = self.rows[g][0], self.rows[g][1]
            else:
                nodes, f = np.zeros(0, np.int64), np.zeros((0, len(SEED_COLS)), np.float32)
            bi.append(np.full(len(nodes), b)); ni.append(nodes); sf.append(f)
            st.append(self.status(nodes, f)); ma.append(float(len(nodes) > 0))
        seeds = (torch.as_tensor(np.concatenate(bi), dtype=torch.long, device=dev), torch.as_tensor(np.concatenate(ni), device=dev),
                 torch.as_tensor(np.concatenate(sf), device=dev))
        return seeds, torch.as_tensor(np.stack(st), device=dev), torch.as_tensor(np.array(ma, np.float32), device=dev)


def tensors(S, st, dev):
    return dict(X0S=torch.as_tensor(st.x0s(S, np.arange(len(S.units))).astype(np.float32), device=dev),
                FP=torch.as_tensor(S.group_fp, device=dev), GI=S.group_idx, CI=S.cell_idx,
                CM=torch.as_tensor(st.cell_mean, device=dev))


def forward_idx(model, T, bank, idx, dev, rng=None, p=None, mask_frac=0.0, mask_rng=None):
    j = torch.as_tensor(idx, device=dev)
    gi = T["GI"][idx]
    seeds, status, ma = bank.batch(gi, dev, rng, p, mask_frac, mask_rng) if bank is not None else (None, None, None)
    return model(T["FP"][torch.as_tensor(gi, device=dev)], T["X0S"][j], T["CM"][torch.as_tensor(T["CI"][idx], device=dev)], status, ma, seeds)


@torch.no_grad()
def predict(model, T, bank, idx, dev, bs=64, mask_frac=0.0, mask_seed=0, parts=False):
    model.eval()
    out, corr, gates = [], [], []
    mrng = np.random.RandomState(mask_seed)
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        d, r, a = forward_idx(model, T, bank, j, dev, mask_frac=mask_frac, mask_rng=mrng)
        out.append(d.float().cpu().numpy()); corr.append(r.float().cpu().numpy())
        if a is not None:
            gates.append(a.float().mean(1).cpu().numpy())
    model.train()
    res = np.concatenate(out)
    if parts:
        return res, np.concatenate(corr), (np.concatenate(gates) if gates else None)
    return res


def val_score(model, T, bank, S, val, dev, ytrue):
    p, r, g = predict(model, T, bank, val, dev, parts=True)
    uv = S.units.iloc[val].reset_index(drop=True)
    src = uv.is_hcc.to_numpy() == 0
    m = unit_metrics(p[src], ytrue[src], uv.cell.to_numpy()[src])
    s = summarise(m, uv[src].reset_index(drop=True), "source")
    dchem = p - r
    s["corr_rms"] = float(np.sqrt((r[src] ** 2).mean())); s["chem_rms"] = float(np.sqrt((dchem[src] ** 2).mean()))
    s["mean_gate"] = float(g[src].mean()) if g is not None else np.nan
    return s, p


def aug_rates(K, train_groups):
    st = K.status.set_index("group_idx").loc[sorted(set(train_groups))]
    return dict(all=float(np.clip((st.status == "missing").mean(), 0.10, 0.50)), source=0.25, block=0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["chem", "concat", "gated", "fixed"])
    ap.add_argument("--aug", type=int, default=0)
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--updates", type=int, default=20000)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--graph", default="real", choices=["real", "rewired"])
    ap.add_argument("--evidence", default="on", choices=["on", "off"])
    ap.add_argument("--max_updates_smoke", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    kind = f"{a.arm}"
    out = new_run_dir(kind, a.run_id)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda")
    S = Store(); st = TrainStats(S)
    K = Knowledge(len(S.groups))
    bank = SeedBank(K, a.evidence) if a.arm != "chem" else None
    model = R7Model(a.arm, load_graph(K, a.graph) if a.arm != "chem" else None).to(dev)
    T = tensors(S, st, dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    train, val = S.train, S.val
    ytrue_val = S.delta(val)
    Y = torch.zeros((len(S.units), 978), device=dev)
    Y[torch.as_tensor(train, device=dev)] = torch.as_tensor(S.delta(train), device=dev)
    p_aug = aug_rates(K, S.group_idx[train]) if a.aug else None
    cfg = vars(a).copy()
    cfg.update(n_train_units=int(len(train)), n_val_units=int(len(val)), train_units_sha=sha_list(train), val_units_sha=sha_list(val),
               aug_rates=p_aug, n_params=int(sum(p.numel() for p in model.parameters())))
    dump(cfg, os.path.join(out, "config.json"))
    by_cell = [x for x in (train[S.cell_idx[train] == c] for c in range(S.n_cells)) if len(x)]
    rng = np.random.RandomState(a.seed); arng = np.random.RandomState(a.seed + 1000)
    hist, best = [], dict(score=np.inf, step=-1)
    for step in range(1, a.updates + 1):
        c = by_cell[rng.randint(len(by_cell))]
        b = c[rng.choice(len(c), min(a.batch, len(c)), replace=False)]
        d_hat, r, _ = forward_idx(model, T, bank, b, dev, rng=arng, p=p_aug)
        loss = ((d_hat - Y[torch.as_tensor(b, device=dev)]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        last = step == a.updates or (a.max_updates_smoke and step >= a.max_updates_smoke)
        if step % a.eval_every == 0 or last:
            s, p = val_score(model, T, bank, S, val, dev, ytrue_val)
            hist.append(dict(step=step, train_loss=float(loss), val_rmse=s["rmse"], val_pearson=s["pearson"], val_centred_pearson=s["centred_pearson"],
                             val_mrr=s["mrr"], corr_rms=s["corr_rms"], chem_rms=s["chem_rms"], mean_gate=s["mean_gate"], sec=round(time.time() - t0)))
            print(hist[-1], flush=True)
            if s["rmse"] < best["score"]:
                best = dict(score=s["rmse"], step=step, val=s)
                torch.save(model.state_dict(), os.path.join(out, "best.pt")); np.save(os.path.join(out, "val_pred.npy"), p.astype(np.float32))
            if last:
                break
    pd.DataFrame(hist).to_csv(os.path.join(out, "history.tsv"), sep="\t", index=False)
    np.save(os.path.join(out, "val_idx.npy"), val)
    rec = run_record(a.run_id, kind, t0, config=cfg, best=best, gpu_max_mem_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2))
    dump(rec, os.path.join(out, "record.json"))
    dump(rec, pman("runs", f"{kind}_{a.run_id}.json"))
    pd.DataFrame(hist).to_csv(pman("runs", f"{kind}_{a.run_id}_history.tsv"), sep="\t", index=False)
    print(json.dumps(dict(run=a.run_id, best=best, seconds=rec["seconds"], mem=rec["gpu_max_mem_gb"]), default=float))


if __name__ == "__main__":
    main()
