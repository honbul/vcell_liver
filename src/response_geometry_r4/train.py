"""Training of W0 (upstream recipe), W1 (delta anchor), W2 (relational), latent regression and the delta-MLP.
Validation units (never test) select the checkpoint. The trained predictor is predict_delta(x0, fp) only."""
import argparse, json, os, time
import numpy as np
import pandas as pd
import torch

from .common import new_run_dir, run_record, dump, public, sha_list, proc
from .data import Store, TrainStats, reliability_weights, split_half_reliability
from .evaluate import unit_metrics, summarise
from .models import W0Native, W1DeltaAnchor, DeltaMLP, wave_loss, soft_targets, rel_loss, cross_half_similarity


def dev_subset(S, n_cells=12, n_groups=1000):
    """Metadata-only development subset: the 12 non-HCC cells with most training units, then the 1000 training groups
    with most training units in those cells (ties: group id)."""
    u = S.units
    tr = u.iloc[S.train]
    cells = tr[tr.is_hcc == 0].groupby("cell").size().sort_values(ascending=False, kind="mergesort")
    cells = sorted(cells.index[:n_cells])
    t = tr[tr.cell.isin(cells)]
    cnt = t.groupby("drug_group").size().reset_index(name="n").sort_values(["n", "drug_group"], ascending=[False, True])
    groups = set(cnt.drug_group.iloc[:n_groups])
    train = S.train[np.isin(u.cell.to_numpy()[S.train], cells) & np.isin(u.drug_group.to_numpy()[S.train], list(groups))]
    val = S.val[np.isin(u.cell.to_numpy()[S.val], cells)]
    return train, val, cells, groups


def predict(model, S, idx, dev, bs=4096):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]
            x0 = torch.from_numpy(S.x0[j]).to(dev); fp = torch.from_numpy(S.fp(j)).to(dev)
            out.append(model.predict_delta(x0, fp).float().cpu().numpy())
    model.train()
    return np.concatenate(out)


def val_score(model, S, val, dev, ytrue):
    p = predict(model, S, val, dev)
    m = unit_metrics(p, ytrue, S.units.cell.to_numpy()[val])
    s = summarise(m, S.units.iloc[val].reset_index(drop=True), "source")
    flat = np.corrcoef(p.ravel(), ytrue.ravel())[0, 1]       # upstream selection statistic (flattened delta Pearson)
    return s, flat, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["w0", "w1", "w2", "latreg", "mlp"])
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--subset", default="full", choices=["dev", "full"])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--rel", default="on", choices=["on", "off"])
    ap.add_argument("--pairing", default="cell", choices=["cell", "pooled"])
    ap.add_argument("--teacher_view", default="consensus", choices=["consensus", "cross_half"], help="modification 1: similarity for q from independent plate views")
    ap.add_argument("--tau_t", type=float, default=0.1)
    ap.add_argument("--tau_s", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--updates", type=int, default=20000)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=40, help="W0 only (upstream recipe, batch 32)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_updates_smoke", type=int, default=0, help="stop after this many updates (smoke run)")
    a = ap.parse_args()
    t0 = time.time()
    out = new_run_dir(a.arm, a.run_id)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda")
    S = Store()
    if a.subset == "dev":
        train, val, cells, groups = dev_subset(S)
    else:
        train, val = S.train, S.val
    ytrue_val = S.delta(val)
    stats = TrainStats(S)
    rr, known, glob = reliability_weights(S, stats)
    if a.rel == "off":
        rr = np.where(S.role == "train", 1.0, rr).astype(np.float32)
    hist = []
    best = dict(score=np.inf, step=-1)
    cfg = vars(a).copy()
    cfg.update(n_train_units=int(len(train)), n_val_units=int(len(val)), train_units_sha=sha_list(train), val_units_sha=sha_list(val),
               teacher_explained=stats.explained, reliability_global_median=glob, reliability_known_frac=float(known[train].mean()))
    dump(cfg, os.path.join(out, "config.json"))

    if a.arm == "w0":
        model = W0Native().to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=a.lr)
        tr_rows = pd.read_csv(proc("train_rows.tsv"), sep="\t")
        keep = np.ones(len(tr_rows), bool)
        if a.subset == "dev":
            keep = tr_rows.cell.isin(cells).to_numpy() & tr_rows.drug_group.isin(groups).to_numpy()
        X0 = torch.from_numpy(np.load(proc("train_rows_x0.npy"))[keep]).to(dev)
        X1 = torch.from_numpy(np.load(proc("train_rows_x1.npy"))[keep]).to(dev)
        fp_all = torch.from_numpy(np.load(os.path.join(os.environ["TB_PROC"], "ecfp4.npy")).astype(np.float32)).to(dev)
        didx = torch.from_numpy(tr_rows.drug_idx.to_numpy()[keep]).to(dev)
        n = X0.shape[0]; step = 0
        g = torch.Generator(device="cpu"); g.manual_seed(a.seed)
        for ep in range(a.epochs):
            perm = torch.randperm(n, generator=g).to(dev)
            tl = 0.0
            for i in range(0, n, 32):
                b = perm[i:i + 32]
                final, mu, logvar, _ = model(X0[b], fp_all[didx[b]])
                loss = wave_loss(final, X1[b], mu, logvar)
                opt.zero_grad(); loss.backward(); opt.step()
                tl += loss.item(); step += 1
                if a.max_updates_smoke and step >= a.max_updates_smoke:
                    break
            s, flat, p = val_score(model, S, val, dev, ytrue_val)
            hist.append(dict(epoch=ep + 1, step=step, train_loss=tl, val_flat_delta_pearson=flat, val_rmse=s["rmse"], val_pearson=s["pearson"], sec=round(time.time() - t0)))
            print(hist[-1], flush=True)
            if -flat < best["score"]:      # upstream rule: best validation delta Pearson
                best = dict(score=-flat, step=step, epoch=ep + 1, val=s, val_flat_delta_pearson=flat)
                torch.save(model.state_dict(), os.path.join(out, "best.pt")); np.save(os.path.join(out, "val_pred.npy"), p.astype(np.float32))
            if a.max_updates_smoke and step >= a.max_updates_smoke:
                break
    else:
        model = (DeltaMLP() if a.arm == "mlp" else W1DeltaAnchor()).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=a.lr)
        X0 = torch.from_numpy(S.x0).to(dev)
        FP = torch.from_numpy(S.group_fp).to(dev)
        gidx = torch.from_numpy(S.units.group_idx.to_numpy()).to(dev)
        Y = torch.zeros((len(S.units), 978), device=dev)
        Y[torch.from_numpy(train).to(dev)] = torch.from_numpy(S.delta(train)).to(dev)
        Ut = torch.zeros((len(S.units), 64), device=dev)
        Ut[torch.from_numpy(train).to(dev)] = torch.from_numpy(stats.teacher(S, train)).to(dev)
        R = torch.from_numpy(np.nan_to_num(rr, nan=0.0)).to(dev)
        if a.teacher_view == "cross_half":
            rep, _, ha, hb = split_half_reliability(S, stats)
            m_ = stats.cell_mean[S.cell_idx[rep]]
            UA = torch.zeros((len(S.units), 64), device=dev); UB = torch.zeros((len(S.units), 64), device=dev)
            HAS = torch.zeros(len(S.units), dtype=torch.bool, device=dev)
            rt = torch.from_numpy(rep).to(dev)
            UA[rt] = torch.from_numpy((((ha - m_) / stats.scale - stats.pca_mu) @ stats.U).astype(np.float32)).to(dev)
            UB[rt] = torch.from_numpy((((hb - m_) / stats.scale - stats.pca_mu) @ stats.U).astype(np.float32)).to(dev)
            HAS[rt] = True
        by_cell = [train[S.cell_idx[train] == c] for c in range(S.n_cells)]
        by_cell = [x for x in by_cell if len(x)]
        rng = np.random.RandomState(a.seed)
        use_rel = a.arm in ("w2", "latreg") and a.lam > 0
        for step in range(1, a.updates + 1):
            if a.pairing == "cell":
                c = by_cell[rng.randint(len(by_cell))]
                b = c[rng.choice(len(c), min(a.batch, len(c)), replace=False)]
            else:
                cs = rng.randint(len(by_cell), size=a.batch)
                b = np.unique(np.array([by_cell[k][rng.randint(len(by_cell[k]))] for k in cs]))
            bt = torch.from_numpy(b).to(dev)
            d_hat, z = model(X0[bt], FP[gidx[bt]])
            loss = ((d_hat - Y[bt]) ** 2).mean()
            lr_ = torch.zeros((), device=dev)
            if use_rel:
                u, r = Ut[bt], R[bt]
                if a.arm == "w2":
                    sim = cross_half_similarity(u, UA[bt], UB[bt], HAS[bt]) if a.teacher_view == "cross_half" else None
                    q = soft_targets(u, r, a.rho, a.tau_t, sim)
                    lr_ = rel_loss(z, u, q, r, a.tau_s)
                else:
                    lr_ = (r * ((z - u) ** 2).mean(1)).mean()
                loss = loss + a.lam * lr_
            opt.zero_grad(); loss.backward(); opt.step()
            if step % a.eval_every == 0 or step == a.updates or (a.max_updates_smoke and step >= a.max_updates_smoke):
                s, flat, p = val_score(model, S, val, dev, ytrue_val)
                hist.append(dict(step=step, train_loss=float(loss), rel_loss=float(lr_), val_rmse=s["rmse"], val_pearson=s["pearson"], val_mrr=s["mrr"],
                                 val_flat_delta_pearson=flat, sec=round(time.time() - t0)))
                print(hist[-1], flush=True)
                if s["rmse"] < best["score"]:
                    best = dict(score=s["rmse"], step=step, val=s, val_flat_delta_pearson=flat)
                    torch.save(model.state_dict(), os.path.join(out, "best.pt")); np.save(os.path.join(out, "val_pred.npy"), p.astype(np.float32))
                if a.max_updates_smoke and step >= a.max_updates_smoke:
                    break
    pd.DataFrame(hist).to_csv(os.path.join(out, "history.tsv"), sep="\t", index=False)
    np.save(os.path.join(out, "val_idx.npy"), val)
    rec = run_record(a.run_id, a.arm, t0, config=cfg, best=best, gpu_max_mem_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2))
    dump(rec, os.path.join(out, "record.json"))
    dump(rec, public("manifests", "response_geometry_r4", "runs", f"{a.arm}_{a.run_id}.json"))
    pd.DataFrame(hist).to_csv(public("manifests", "response_geometry_r4", "runs", f"{a.arm}_{a.run_id}_history.tsv"), sep="\t", index=False)
    print(json.dumps(dict(run=a.run_id, best=best, seconds=rec["seconds"], mem=rec["gpu_max_mem_gb"]), default=float))


if __name__ == "__main__":
    main()
