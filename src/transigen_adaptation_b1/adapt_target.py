"""Target adaptation runs: B1 (same architecture from random init) and B2 (full fine-tune of the source checkpoint).

Both arms get exactly the same adapt rows, validation rows, update budget and evaluation schedule.
Test rows are never touched here: only X0/X1 of adapt and val rows are requested from the guarded loader.
"""
import argparse, hashlib, json, os, time
import numpy as np
import torch
import yaml

from . import tsg_adapter as T
from .data import Data, condition_delta, MAN_DIR
from .evaluate import per_condition

ARMS = ("B1_scratch", "B2_finetune")


def ids_hash(a):
    return hashlib.sha256(np.asarray(a, dtype=np.int64).tobytes()).hexdigest()[:16]


def run_id(target, split, k, arm, lr, seed, budget, tag=""):
    return f"adapt_{target}_s{split}_k{k}_{arm}_lr{lr:g}_seed{seed}_u{budget}{tag}"


def source_checkpoint(seed):
    sel = json.load(open(os.path.join(MAN_DIR, "source_selection.json")))
    e = sel["checkpoints"][str(seed)]
    assert T.sha256(e["checkpoint"]) == e["checkpoint_sha256"], "source checkpoint changed after selection"
    return e


class TargetSet:
    """Tensors of one target/split/k, shared by both arms."""

    def __init__(self, D, target, split, k, dev):
        self.target, self.split, self.k = target, split, k
        self.rows_a = D.target_rows(target, split, "adapt", k)
        self.rows_v = D.target_rows(target, split, "val")
        rows_t = D.target_rows(target, split, "test")
        assert not (set(self.rows_a) & set(self.rows_v)) and not (set(self.rows_a) & set(rows_t)) and not (set(self.rows_v) & set(rows_t))
        ga, gv, gt = (set(D.rows.drug_group.iloc[r]) for r in (self.rows_a, self.rows_v, rows_t))
        assert not (ga & gv) and not (ga & gt) and not (gv & gt), "a drug group straddles split roles"
        di = D.rows.drug_idx.to_numpy()
        f = lambda r: torch.tensor(np.asarray(D.x0[np.sort(r)]), device=dev)
        self.rows_a, self.rows_v = np.sort(self.rows_a), np.sort(self.rows_v)
        self.x0_a, self.x0_v = f(self.rows_a), f(self.rows_v)
        self.x1_a = torch.tensor(D.target_x1(self.rows_a, target, split, "adapt"), device=dev)
        x1_v = D.target_x1(self.rows_v, target, split, "val")
        self.fp_a = torch.tensor(D.fp[di[self.rows_a]], dtype=torch.float32, device=dev)
        self.fp_v = torch.tensor(D.fp[di[self.rows_v]], dtype=torch.float32, device=dev)
        self.meta_v = D.rows.iloc[self.rows_v]
        self.cond_v, self.truth_v = condition_delta(self.meta_v, x1_v - self.x0_v.cpu().numpy())
        self.n_adapt_groups = len(ga)
        self.n_adapt_conditions = int(D.rows.iloc[self.rows_a].pert_id.nunique())

    def val_pred(self, model):
        p = T.predict_x1_mean_path(model, self.x0_v, self.fp_v)
        _, d = condition_delta(self.meta_v, (p - self.x0_v).cpu().numpy())
        return d

    def val_score(self, model):
        d = self.val_pred(model)
        if not np.isfinite(d).all():
            return float("inf"), d
        return float(per_condition(d, self.truth_v).delta_rmse.mean()), d


def adapt_run(D, ts, arm, lr, seed, cfg, dev, max_updates, tag="", fp_perm=None, fixed_steps=None):
    """One adaptation run. fp_perm/fixed_steps are used only by the label-permutation control."""
    ad = cfg["adaptation"]
    rid = run_id(ts.target, ts.split, ts.k, arm, lr, seed, max_updates, tag)
    out = os.path.join(os.environ["TB_RUNS"], "adapt", rid)
    os.makedirs(out, exist_ok=False)                      # immutable run id
    init = dict(kind="random")
    if arm == "B2_finetune":
        e = source_checkpoint(seed)
        model, _ = T.load_checkpoint(e["checkpoint"], dev, seed=seed)
        init = dict(kind="source", run_id=e["run_id"], checkpoint_sha256=e["checkpoint_sha256"])
    else:
        model = T.build_model(seed, dev)
    T.seed_all(seed)
    init_hash = T.state_hash(model)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg["model"]["weight_decay"])
    n = len(ts.rows_a)
    bs = min(32, n)
    fp_a = ts.fp_a if fp_perm is None else ts.fp_a[torch.tensor(fp_perm, device=dev)]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    t0 = time.time()
    v, pred = ts.val_score(model)
    curve = [(0, v)]
    best, best_step, best_pred = v, 0, pred
    best_state = {k_: t_.detach().clone() for k_, t_ in model.state_dict().items()}
    step, stopped, perm, pos = 0, "max_updates", None, 0
    total = fixed_steps if fixed_steps is not None else max_updates
    while step < total:
        if perm is None or pos >= n:
            perm, pos = torch.randperm(n, generator=gen).to(dev), 0
        b = perm[pos:pos + bs]
        pos += bs
        if len(b) < 2:
            continue
        l = T.upstream_train_step(model, opt, ts.x0_a[b], ts.x1_a[b], fp_a[b])
        step += 1
        if not np.isfinite(l):                 # recorded as a diverged run; the best finite state so far is kept
            stopped = "non_finite_loss"
            break
        if step % ad["eval_every"] == 0 or step == total:
            v, pred = ts.val_score(model)
            curve.append((step, v))
            if fixed_steps is None:
                if v < best:
                    best, best_step, best_pred = v, step, pred
                    best_state = {k_: t_.detach().clone() for k_, t_ in model.state_dict().items()}
                if (step - best_step) >= ad["patience_evals"] * ad["eval_every"]:
                    stopped = "patience"
                    break
    if fixed_steps is not None and stopped != "non_finite_loss":      # control run: the state after exactly fixed_steps updates, no selection
        best, best_step, best_pred, stopped = v, step, pred, "fixed_steps"
        best_state = {k_: t_.detach().clone() for k_, t_ in model.state_dict().items()}
    model.load_state_dict(best_state)
    ck = os.path.join(out, "best.pt")
    T.save_checkpoint(model, ck, dict(run_id=rid, arm=arm, lr=lr, seed=seed, best_step=best_step))
    np.save(os.path.join(out, "pred_val_delta.npy"), best_pred.astype(np.float32))
    with open(os.path.join(out, "val_curve.tsv"), "w") as f:
        f.write("step\tval_delta_rmse\n" + "".join(f"{s}\t{x:.6f}\n" for s, x in curve))
    res = dict(run_id=rid, kind="adapt", target=ts.target, split_seed=ts.split, k=ts.k, arm=arm, lr=lr, seed=seed,
               max_updates=max_updates, batch_size=bs, steps_run=step, best_step=best_step, stopped=stopped,
               best_at_max=bool(best_step == max_updates), val_delta_rmse_best=best, val_delta_rmse_step0=curve[0][1],
               n_adapt_rows=n, n_adapt_conditions=ts.n_adapt_conditions, n_adapt_drug_groups=ts.n_adapt_groups,
               n_val_conditions=int(len(ts.cond_v)), adapt_rows_hash=ids_hash(ts.rows_a), val_rows_hash=ids_hash(ts.rows_v),
               init=init, init_state_hash=init_hash, state_hash=T.state_hash(model), checkpoint=ck,
               checkpoint_sha256=T.sha256(ck), seconds=time.time() - t0, control=bool(fp_perm is not None),
               commit=os.environ.get("TB_COMMIT", ""), slurm_job=os.environ.get("SLURM_JOB_ID", ""))
    json.dump(res, open(os.path.join(out, "result.json"), "w"), indent=1)
    print(json.dumps({k_: res[k_] for k_ in ("run_id", "best_step", "stopped", "val_delta_rmse_step0", "val_delta_rmse_best", "seconds")}), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--phase", required=True, choices=["search", "seeds", "curve"])
    ap.add_argument("--config", default="configs/transigen_adaptation_b1/protocol.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    ad = cfg["adaptation"]
    summ = json.load(open(os.path.join(MAN_DIR, "splits_summary.json")))["targets"][a.target]
    assert summ["status"] != "excluded"
    budget_file = os.path.join(MAN_DIR, "adapt_budget.json")
    max_updates = json.load(open(budget_file))["max_updates"] if os.path.exists(budget_file) else ad["max_updates"]
    dev = torch.device("cuda")
    D = Data()
    torch.cuda.reset_peak_memory_stats()
    sel = None
    if a.phase != "search":
        sel = json.load(open(os.path.join(MAN_DIR, "adapt_selection.json")))
        assert sel["max_updates"] == max_updates
    splits = cfg["splits"]["target"]["split_seeds"]
    done = []
    for split in (splits if a.phase != "curve" else splits[:1]):
        info = summ["splits"][str(split)]
        ks = [info["k_main"]] if a.phase != "curve" else info["k_curve"]
        for k in ks:
            ts = TargetSet(D, a.target, split, k, dev)
            hashes = set()
            for arm in ARMS:
                if a.phase == "search":
                    jobs = [(lr, ad["search_seed"]) for lr in ad["arms"][arm]["lr_grid"]]
                else:
                    lr = sel["selected"][f"{a.target}|{split}|{arm}"]["lr"]
                    seeds = [s for s in ad["seeds"] if not (a.phase == "seeds" and s == ad["search_seed"])]
                    jobs = [(lr, s) for s in seeds]
                for lr, seed in jobs:
                    r = adapt_run(D, ts, arm, lr, seed, cfg, dev, max_updates)
                    hashes.add((r["adapt_rows_hash"], r["val_rows_hash"]))
                    done.append(r["run_id"])
            assert len(hashes) == 1, "B1 and B2 did not see the same adapt/val rows"
    print(f"done {len(done)} runs; gpu peak MB {torch.cuda.max_memory_allocated() / 2 ** 20:.0f}")


if __name__ == "__main__":
    main()
