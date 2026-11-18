"""Training of the follow-up arms (chem / M / G0 / GS / GR / TD) inside one split context.

Context `foldK` (K = 0, 1, 2): inner-train / inner-validation / held drug groups of the ORIGINAL TRAIN drugs
(manifests/r7_followup_v1/source_folds.tsv, committed before any run). Cell means, X0 standardisation, the chemistry
backbone and every correction are fitted on the inner-train units only; the inner-validation source panel selects the
checkpoint; the held fold is only predicted (its deltas are never opened by this module).
Context `orig`: original train / validation (final re-evaluation models; the test units are not touched here).

All correction arms of one (context, model seed) share the frozen backbone, the batch stream (seed), the augmentation
stream (seed + 1000, drawn on the drug STUBS, which are identical across real / null / no-seed), the loss, the
learning rate and the update budget of the original modification-1 B11 (lr 1e-4, batch 64, 20000 updates, eval 500).
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
import torch

from src.evidence_gated_r7.data import Store, Knowledge
from src.evidence_gated_r7.evaluate import unit_metrics, summarise
from .common import MAN, new_run_dir, run_record, dump, pman, sha_list, sha256, fruns, orig_runs, cfg
from .kgraph import inner_role
from .models import FollowupModel, count_params, N_META
from .shuffle import load_mapping

SEED_COLS = Knowledge.SEED


class Context:
    def __init__(self, S, name):
        self.name = name
        if name == "orig":
            self.train, self.sel, self.held = S.train, S.val, None
        else:
            k = int(name.replace("fold", ""))
            f = pd.read_csv(os.path.join(MAN, "source_folds.tsv"), sep="\t")
            role = dict(zip(f.group_idx, inner_role(f.fold, f["sub"], k)))
            r = np.array([role.get(g, "") if ro == "train" else "" for g, ro in zip(S.group_idx, S.role)])
            self.train, self.sel, self.held = (np.where(r == x)[0] for x in ("inner_train", "inner_val", "held"))
            assert len(self.train) + len(self.sel) + len(self.held) == len(S.train)
        u = S.units.drug_group.to_numpy()
        sets = [set(u[x]) for x in (self.train, self.sel) + (() if self.held is None else (self.held,))]
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                assert not sets[i] & sets[j], "drug group shared between splits"


class FoldStats:
    """Cell means and X0 standardisation from the given training units only. A cell line without a training unit in
    this context gets the mean of the context's training units (flagged)."""

    def __init__(self, S, train_idx):
        y = S.delta(train_idx).astype(np.float64)
        c = S.cell_idx[train_idx]
        m = np.zeros((S.n_cells, 978)); n = np.bincount(c, minlength=S.n_cells)
        np.add.at(m, c, y)
        self.cells_without_train = np.where(n == 0)[0]
        cm = m / np.maximum(n, 1)[:, None]
        cm[n == 0] = y.mean(0)
        self.cell_mean = cm.astype(np.float32)
        self.x0_mu = S.x0[train_idx].mean(0).astype(np.float32)
        self.x0_sd = (S.x0[train_idx].std(0) + 1e-3).astype(np.float32)

    def x0s(self, S, idx):
        return (S.x0[idx] - self.x0_mu) / self.x0_sd


def train_targets(S, train_idx, dev):
    """Regression targets: rows of the training units only; every other row stays zero and is never indexed."""
    Y = torch.zeros((len(S.units), 978), device=dev)
    Y[torch.as_tensor(train_idx, device=dev)] = torch.as_tensor(S.delta(train_idx), device=dev)
    return Y


class Bank:
    """Per-group drug stubs. mode: real (original nodes) / null (nodes of a fixed null mapping) / noseed (stubs are
    used for status, metadata, m_available and the augmentation draw, then the seed list is emptied)."""

    def __init__(self, K, mode="real", null_map=None):
        assert mode in ("real", "null", "noseed")
        self.mode = mode
        self.rows = {}
        for g, d in K.by_group.items():
            nodes = d.node.to_numpy(np.int64)
            if mode == "null":
                nn_ = null_map[g]
                assert len(nn_) == len(nodes) and len(set(nn_.tolist())) == len(nn_)
                nodes = nn_
            self.rows[g] = (nodes, d[SEED_COLS].to_numpy(np.float32), [set(str(t).split("|")) for t in d.tids])

    @staticmethod
    def status(nodes, f):
        if len(nodes) == 0:
            return np.zeros(4, np.float32)
        return np.array([f[:, 0].max(), max(f[:, 1].max(), f[:, 2].max()), min(np.log1p(len(nodes)) / np.log(201), 1.0),
                         1.0 - f[:, 5].mean()], np.float32)

    @classmethod
    def meta(cls, nodes, f):
        if len(nodes) == 0:
            return np.zeros(N_META, np.float32)
        return np.concatenate([cls.status(nodes, f), [f[:, 6].mean(), f[:, 8].mean()]]).astype(np.float32)

    def augment(self, g, rng, p):
        """Identical to SeedBank.augment of the original study (same draws on the same stubs)."""
        nodes, f, tids = self.rows[g]
        u = rng.rand()
        if u < p["all"]:
            return nodes[:0], f[:0]
        u -= p["all"]
        f = f.copy()
        if u < p["source"]:
            has_m = f[:, 0].max() > 0; has_a = max(f[:, 1].max(), f[:, 2].max()) > 0
            if has_m and has_a:
                if rng.rand() < 0.5:
                    f[:, [0, 3, 4]] = 0; f[:, 5] = 1
                else:
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

    def batch(self, groups, dev, rng=None, p=None, override=None):
        """override: optional dict group -> (nodes, f) replacing the stubs of that group (sensitivity analysis)."""
        bi, ni, sf, st, me, ma = [], [], [], [], [], []
        for b, g in enumerate(groups):
            if g in self.rows:
                nodes, f = self.augment(g, rng, p) if p is not None else self.rows[g][:2]
            else:
                nodes, f = np.zeros(0, np.int64), np.zeros((0, len(SEED_COLS)), np.float32)
            st.append(self.status(nodes, f)); me.append(self.meta(nodes, f)); ma.append(float(len(nodes) > 0))
            if self.mode == "noseed":
                nodes, f = nodes[:0], f[:0]
            bi.append(np.full(len(nodes), b)); ni.append(nodes); sf.append(f)
        seeds = (torch.as_tensor(np.concatenate(bi), dtype=torch.long, device=dev), torch.as_tensor(np.concatenate(ni), dtype=torch.long, device=dev),
                 torch.as_tensor(np.concatenate(sf), dtype=torch.float32, device=dev))
        return seeds, torch.as_tensor(np.stack(st), device=dev), torch.as_tensor(np.stack(me), device=dev), torch.as_tensor(np.array(ma, np.float32), device=dev)


def tensors(S, st, dev):
    return dict(X0S=torch.as_tensor(st.x0s(S, np.arange(len(S.units))).astype(np.float32), device=dev),
                FP=torch.as_tensor(S.group_fp, device=dev), GI=S.group_idx, CI=S.cell_idx,
                CM=torch.as_tensor(st.cell_mean, device=dev))


def load_graph(K):
    return dict(edge_index=K.edge_index, edge_rel=K.edge_rel, edge_feat=K.edge_feat, measured=K.measured, lm_idx=K.lm_idx)


def forward_idx(model, T, bank, idx, dev, rng=None, p=None):
    j = torch.as_tensor(idx, device=dev)
    gi = T["GI"][idx]
    seeds, status, meta, ma = bank.batch(gi, dev, rng, p) if bank is not None else (None, None, None, None)
    return model(T["FP"][torch.as_tensor(gi, device=dev)], T["X0S"][j], T["CM"][torch.as_tensor(T["CI"][idx], device=dev)], status, meta, ma, seeds)


@torch.no_grad()
def predict(model, T, bank, idx, dev, bs=64, parts=False):
    model.eval()
    out, corr, gates = [], [], []
    for i in range(0, len(idx), bs):
        d, r, a = forward_idx(model, T, bank, idx[i:i + bs], dev)
        out.append(d.float().cpu().numpy()); corr.append(r.float().cpu().numpy())
        if a is not None:
            gates.append(a.float().mean(1).cpu().numpy())
    model.train()
    res = np.concatenate(out)
    if parts:
        return res, np.concatenate(corr), (np.concatenate(gates) if gates else None)
    return res


def sel_score(model, T, bank, S, sel, dev, ytrue):
    """Checkpoint selection score: family-balanced source-panel delta RMSE of the selection units (no HCC number)."""
    p, r, g = predict(model, T, bank, sel, dev, parts=True)
    uv = S.units.iloc[sel].reset_index(drop=True)
    src = uv.is_hcc.to_numpy() == 0
    m = unit_metrics(p[src], ytrue[src], uv.cell.to_numpy()[src])
    s = summarise(m, uv[src].reset_index(drop=True), "source")
    s["corr_rms"] = float(np.sqrt((r[src] ** 2).mean())); s["chem_rms"] = float(np.sqrt(((p - r)[src] ** 2).mean()))
    s["mean_gate"] = float(g[src].mean()) if g is not None else np.nan
    return s, p


def aug_rates(K, train_groups):
    st = K.status.set_index("group_idx").loc[sorted(set(train_groups))]
    return dict(all=float(np.clip((st.status == "missing").mean(), 0.10, 0.50)), source=0.25, block=0.25)


def build(arm, K, dev, null_map_path=""):
    """Model + bank of an arm (shared by training, scoring and the final evaluation)."""
    if arm == "chem":
        return FollowupModel("chem").to(dev), None
    mode = {"M": "real", "G0": "noseed", "GS": "null", "GR": "real", "TD": "real", "TDS": "null"}[arm]
    bank = Bank(K, mode, load_mapping(null_map_path) if mode == "null" else None)
    marm = "TD" if arm == "TDS" else arm
    model = FollowupModel(marm, load_graph(K) if marm != "M" else None, h2=cfg()["m_arm_h2"] if marm == "M" else None).to(dev)
    return model, bank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["chem", "M", "G0", "GS", "GR", "TD", "TDS"])
    ap.add_argument("--ctx", required=True, help="fold0 | fold1 | fold2 | orig")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--null_seed", type=int, default=0)
    ap.add_argument("--backbone_from", default="", help="chem run id of this study (same ctx and seed), or orig:<kind>/<run_id> for the original chem run")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--updates", type=int, default=20000)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--aug", type=int, default=1)
    ap.add_argument("--max_updates_smoke", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    out = new_run_dir(a.arm, a.run_id)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda")
    S = Store(); C = Context(S, a.ctx); st = FoldStats(S, C.train)
    K = Knowledge(len(S.groups))
    null_path, null_sha = "", ""
    if a.arm in ("GS", "TDS"):
        man = json.load(open(os.path.join(MAN, "target_shuffle_manifest.json")))["mappings"][f"{a.ctx}_null{a.null_seed}"]
        null_path, null_sha = man["path"], man["sha256"]
        assert sha256(null_path) == null_sha, "null mapping differs from the committed manifest"
    model, bank = build(a.arm, K, dev, null_path)
    T = tensors(S, st, dev)
    if a.arm != "chem":
        assert a.backbone_from, "correction arms need the frozen seed-matched backbone"
        if a.backbone_from.startswith("orig:"):
            assert a.ctx == "orig"
            bk, br = a.backbone_from[5:].split("/")
            bdir = orig_runs(bk, br)
            src_rec = json.load(open(os.path.join(bdir, "record.json")))
            assert src_rec["config"]["arm"] == "chem" and src_rec["config"]["seed"] == a.seed
        else:
            bdir = fruns("chem", a.backbone_from)
            src_rec = json.load(open(os.path.join(bdir, "record.json")))
            assert src_rec["config"]["arm"] == "chem" and src_rec["config"]["seed"] == a.seed and src_rec["config"]["ctx"] == a.ctx
            assert src_rec["config"]["train_units_sha"] == sha_list(C.train)
        sd = torch.load(os.path.join(bdir, "best.pt"), map_location=dev)
        model.backbone.load_state_dict({k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")})
        for p_ in model.backbone.parameters():
            p_.requires_grad_(False)
    opt = torch.optim.Adam([p_ for p_ in model.parameters() if p_.requires_grad], lr=a.lr)
    train, sel = C.train, C.sel
    ytrue_sel = S.delta(sel)
    Y = train_targets(S, train, dev)
    p_aug = aug_rates(K, S.group_idx[train]) if (a.aug and a.arm != "chem") else None
    cfg_ = vars(a).copy()
    cfg_.update(n_train_units=int(len(train)), n_sel_units=int(len(sel)), n_held_units=0 if C.held is None else int(len(C.held)),
                train_units_sha=sha_list(train), sel_units_sha=sha_list(sel), held_units_sha="" if C.held is None else sha_list(C.held),
                aug_rates=p_aug, n_params_total=count_params(model), n_params_correction=0 if model.corr is None else count_params(model.corr),
                n_params_trainable=int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
                cells_without_train_units=int(len(st.cells_without_train)), null_mapping=null_path, null_mapping_sha256=null_sha,
                samples_seen=int(a.updates * a.batch))
    dump(cfg_, os.path.join(out, "config.json"))
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
            s, p = sel_score(model, T, bank, S, sel, dev, ytrue_sel)
            hist.append(dict(step=step, train_loss=float(loss), sel_rmse=s["rmse"], sel_centred_pearson=s["centred_pearson"],
                             corr_rms=s["corr_rms"], chem_rms=s["chem_rms"], mean_gate=s["mean_gate"], sec=round(time.time() - t0)))
            print(hist[-1], flush=True)
            if s["rmse"] < best["score"]:
                best = dict(score=s["rmse"], step=step, sel=s)
                torch.save(model.state_dict(), os.path.join(out, "best.pt")); np.save(os.path.join(out, "sel_pred.npy"), p.astype(np.float32))
            if last:
                break
    train_sec = time.time() - t0
    pd.DataFrame(hist).to_csv(os.path.join(out, "history.tsv"), sep="\t", index=False)
    np.save(os.path.join(out, "sel_idx.npy"), sel)
    infer = {}
    if C.held is not None:                              # predictions only; held deltas are opened by the scoring job
        model.load_state_dict(torch.load(os.path.join(out, "best.pt"), map_location=dev))
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        ph = predict(model, T, bank, C.held, dev)
        infer = dict(held_units=int(len(C.held)), seconds=round(time.time() - t1, 2), sec_per_1000_units=round((time.time() - t1) / len(C.held) * 1000, 3))
        np.save(os.path.join(out, "held_pred.npy"), ph.astype(np.float32)); np.save(os.path.join(out, "held_idx.npy"), C.held)
    rec = run_record(a.run_id, a.arm, t0, config=cfg_, best=best, train_seconds=round(train_sec, 1), inference=infer,
                     gpu_max_mem_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), checkpoint_sha256=sha256(os.path.join(out, "best.pt")))
    dump(rec, os.path.join(out, "record.json"))
    dump(rec, pman("runs", f"{a.arm}_{a.run_id}.json"))
    pd.DataFrame(hist).to_csv(pman("runs", f"{a.arm}_{a.run_id}_history.tsv"), sep="\t", index=False)
    print(json.dumps(dict(run=a.run_id, best_step=best["step"], best=best["score"], seconds=rec["seconds"], mem=rec["gpu_max_mem_gb"]), default=float))


if __name__ == "__main__":
    main()
