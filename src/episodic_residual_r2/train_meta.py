"""Episodic training of the residual adapter (and of the generic CNP) on meta-train families.

Selection uses meta-validation episodes only. No HCC value is read anywhere in this module.
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
import torch
import yaml

from .common import CFG, sha256, dump, new_run_dir, run_record
from .data import Source, Features, residual_basis
from . import methods as M
from .evalcore import score, family_mean


def build_model(kind, variant, src):
    v = dict(variant)
    if kind == "cnp":
        assert not v
        return M.GenericCNP()
    if "svd_rank" in v:
        basis, _ = residual_basis(src, v.pop("svd_rank"))
        v["svd_basis"] = basis
    return M.ResidualAdapter(**v)


def predict(model, ep, K, fmap, fp_table, mode="normal", perm_seed=0):
    """mode: normal | zero_residual | shuffle_residual (support residual rows permuted across support drugs)."""
    inp = ep.inputs(K)
    ft = fmap.build(inp, fp_table)
    if mode == "normal":
        return model(inp, ft)
    assert isinstance(model, M.ResidualAdapter) and model.offset_mode == "split"
    if mode == "zero_residual":
        return model(inp, ft, zero_residual=True)
    assert mode == "shuffle_residual"
    _, R = M.split_residual(inp)
    order = np.random.RandomState(1000 * perm_seed + K).permutation(K)
    p = np.empty(K, int)
    p[order] = np.roll(order, 1)                         # cyclic derangement: no support keeps its own residual
    return model(inp, ft, R=R[torch.tensor(p, device=R.device)])


def val_table(model, eps, ks, fmap, fp_table, modes=("normal",), full=False):
    rows = []
    model.eval()
    with torch.no_grad():
        for ep in eps:
            for K in ks:
                if K > len(ep.g_s):
                    continue
                for mode in modes:
                    p = predict(model, ep, K, fmap, fp_table, mode, perm_seed=ep.seed)
                    if full:
                        r, _ = score(ep, K, p, dict(mode=mode))
                    else:
                        r = dict(cell=ep.name, family=ep.family, support_seed=ep.seed, K=K, mode=mode,
                                 delta_rmse=torch.sqrt(((p - ep.y_q) ** 2).mean(1)).mean().item())
                    rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["adapter", "cnp"])
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--variant", default="{}", help="JSON kwargs of ResidualAdapter (e.g. offset_mode, use_ctrl, svd_rank, shrink_by_k)")
    ap.add_argument("--train-shuffle-residual", action="store_true", help="control: support residual rows permuted during training too")
    ap.add_argument("--max-steps", type=int, default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(CFG))
    oc = cfg["proposed"]["optimiser"]
    max_steps = a.max_steps or oc["max_steps"]
    variant = json.loads(a.variant)
    t0 = time.time()
    out = new_run_dir("meta", a.run_id)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    src = Source(dev)
    fpath = os.path.join(os.environ["R2_PROC"], "feature_basis.npz")
    fmap = Features.load(fpath, dev)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    model = build_model(a.kind, variant, src).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=oc["weight_decay"])
    sample = src.make_sampler(tuple(cfg["episodes"]["k_values"]))
    rs = np.random.RandomState(a.seed)
    ks = cfg["episodes"]["k_values"]
    val_eps = [src.eval_episode(ci, s) for ci in src.eval_cells("val") for s in cfg["episodes"]["meta_val_support_seeds"]]
    k_main = cfg["episodes"]["k_main"]

    def val_score():
        t = val_table(model, val_eps, ks, fmap, src.FP)
        agg, _ = family_mean(t, by=("K",))
        return {int(r.K): float(r.delta_rmse) for r in agg.itertuples()}

    log = open(os.path.join(out, "curve.tsv"), "w")
    log.write("step\ttrain_loss\t" + "\t".join(f"val_k{k}" for k in ks) + "\tlambda\n")
    v = val_score()
    log.write("0\tnan\t" + "\t".join(f"{v[k]:.6f}" for k in ks) + f"\t{model.lam().item() if a.kind == 'adapter' else float('nan'):.5g}\n")
    best, best_step, v0 = v[k_main], 0, dict(v)
    best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
    stopped, run_loss, step = "max_steps", [], 0
    while step < max_steps:
        model.train()
        opt.zero_grad()
        tot = 0.0
        for _ in range(oc["batch_episodes"]):
            K, ep = sample(rs)
            inp = ep.inputs(K)
            ft = fmap.build(inp, src.FP)
            if a.train_shuffle_residual:
                _, R = M.split_residual(inp)
                pred = model(inp, ft, R=R[torch.randperm(K, device=dev)])
            else:
                pred = model(inp, ft)
            loss = M.rmse_loss(pred, ep.y_q) / oc["batch_episodes"]
            loss.backward()
            tot += loss.item()
        step += 1
        if not np.isfinite(tot):
            stopped = "non_finite_loss"
            break
        opt.step()
        run_loss.append(tot)
        if step % oc["eval_every"] == 0 or step == max_steps:
            v = val_score()
            log.write(f"{step}\t{np.mean(run_loss):.6f}\t" + "\t".join(f"{v[k]:.6f}" for k in ks) + f"\t{model.lam().item() if a.kind == 'adapter' else float('nan'):.5g}\n")
            log.flush()
            run_loss = []
            if v[k_main] < best:
                best, best_step = v[k_main], step
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            if step - best_step >= oc["patience_evals"] * oc["eval_every"]:
                stopped = "patience"
                break
    log.close()
    model.load_state_dict(best_state)
    ck = os.path.join(out, "best.pt")
    torch.save(dict(state_dict={k: t.cpu() for k, t in model.state_dict().items()}, kind=a.kind, variant=variant, seed=a.seed, lr=a.lr, best_step=best_step), ck)
    modes = ("normal", "zero_residual", "shuffle_residual") if (a.kind == "adapter" and variant.get("offset_mode", "split") == "split") else ("normal",)
    full = val_table(model, val_eps, ks, fmap, src.FP, modes=modes, full=True)
    full.insert(0, "run_id", a.run_id)
    full.to_csv(os.path.join(out, "meta_val_episodes.tsv"), sep="\t", index=False, float_format="%.6g")
    agg, _ = family_mean(full, by=("K", "mode"))
    # reference on the same episodes: mean-offset
    ref = []
    with torch.no_grad():
        for ep in val_eps:
            for K in ks:
                ref.append(dict(cell=ep.name, family=ep.family, K=K, delta_rmse=torch.sqrt(((M.mean_offset(ep.inputs(K)) - ep.y_q) ** 2).mean(1)).mean().item()))
    ref, _ = family_mean(pd.DataFrame(ref), by=("K",))
    res = run_record(a.run_id, "meta_" + a.kind, t0, model=a.kind, variant=variant, train_shuffle_residual=a.train_shuffle_residual, lr=a.lr, seed=a.seed,
                     n_parameters=int(n_par), steps_run=step, best_step=best_step, stopped=stopped, max_steps=max_steps,
                     val_delta_rmse_step0={str(k): v0[k] for k in ks}, val_delta_rmse_best_k_main=best,
                     val_delta_rmse={f"{r.mode}|{int(r.K)}": float(r.delta_rmse) for r in agg.itertuples()},
                     val_mean_offset={str(int(r.K)): float(r.delta_rmse) for r in ref.itertuples()},
                     lam=float(model.lam().item()) if a.kind == "adapter" else None, n_sampler_families=len(sample.families),
                     n_val_episodes=len(val_eps), feature_basis_sha256=sha256(fpath), checkpoint=ck, checkpoint_sha256=sha256(ck),
                     gpu=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu")
    dump(res, os.path.join(out, "result.json"))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
