"""Episodic training of the state-conditioned residual adapter (and the generic CNP) on source-train families.

Two modes:
  --fold k   inner family cross-validation: fold-k families are virtual targets (bases refit without them); the curve of the
             fold score is written every `eval_every` steps; nothing is selected or early-stopped inside the run.
  (no fold)  final training on every source-train family for a FIXED number of steps (--steps, chosen by the cross-validation);
             the confirmation families are scored once at the end. No HCC value is read anywhere in this module.
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
import torch
import yaml

from .common import CFG, sha256, dump, new_run_dir, run_record, proc
from .data import Store, Features
from .arms import ARMS, Context
from . import methods as M
from src.episodic_residual_r2.evalcore import family_mean


def build_model(arm):
    return M.GenericCNP() if arm == "cnp" else M.StateAdapter(state=ARMS[arm]["state"])


def predict(model, arm, ctx, inp, mode="normal", perm_seed=0):
    if arm == "cnp":
        return ctx.cnp(model, inp)
    return ctx.adapter(model, arm, inp, mode, perm_seed)


def score_table(model, arm, ctx, eps, ks, modes=("normal",)):
    rows = []
    model.eval()
    with torch.no_grad():
        for ep in eps:
            for K in ks:
                if K > len(ep.g_s):
                    continue
                for mode in modes:
                    p = predict(model, arm, ctx, ep.inputs(K), mode, perm_seed=max(ep.seed, 0))
                    rows.append(dict(cell=ep.name, family=ep.family, support_seed=ep.seed, K=K, mode=mode,
                                     delta_rmse=torch.sqrt(((p - ep.y_q) ** 2).mean(1)).mean().item()))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS) + ["cnp"])
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None, help="final mode: fixed number of steps; fold mode: maximum (default from the protocol)")
    ap.add_argument("--basis-tag", default="v1")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(CFG))
    oc, ec = cfg["models"]["optimiser"], cfg["episodes"]
    ks = ec["k_values"]
    final = a.fold is None
    assert not final or a.steps, "final training needs the fixed number of steps chosen by the cross-validation"
    max_steps = a.steps or oc["cv_max_steps"]
    t0 = time.time()
    out = new_run_dir("cv" if not final else "final", a.run_id)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    st = Store(dev, fold=a.fold)
    if final:
        fpath = proc("feature_basis.npz")
        fmap = Features.load(fpath, dev)
    else:
        fmap, fpath = Features.fit(st), ""
    basis, bpath = None, ""
    if a.arm != "cnp" and ARMS[a.arm].get("proj"):
        bpath = proc(f"repeat_basis_{'all' if final else 'fold' + str(a.fold)}_{a.basis_tag}.npy")
        basis = np.load(bpath)
    ctx = Context(fmap, st.FP, basis=basis)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    model = build_model(a.arm).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=oc["weight_decay"])
    sample = st.make_sampler(tuple(ks))
    rs = np.random.RandomState(a.seed)
    val_eps = st.episodes("val", ec["confirmation_support_seeds"]) if final else st.episodes("val", ec["dev_support_seeds"], ec["dev_query_cap"])
    is_ad = a.arm != "cnp"

    def val_score():
        agg, _ = family_mean(score_table(model, a.arm, ctx, val_eps, ks), by=("K",))
        return {int(r.K): float(r.delta_rmse) for r in agg.itertuples()}

    extra = lambda: (f"\t{model.lam().item():.5g}\t" + "\t".join(f"{model.eta(k).item():.4f}" for k in ks)) if is_ad else "\tnan" + "\tnan" * len(ks)
    log = open(os.path.join(out, "curve.tsv"), "w")
    log.write("step\ttrain_loss\t" + "\t".join(f"val_k{k}" for k in ks) + "\tlambda\t" + "\t".join(f"eta_k{k}" for k in ks) + "\n")
    if not final:
        v = val_score()
        log.write("0\tnan\t" + "\t".join(f"{v[k]:.6f}" for k in ks) + extra() + "\n")
    stopped, run_loss, step = "max_steps", [], 0
    while step < max_steps:
        model.train()
        opt.zero_grad()
        tot = 0.0
        for _ in range(oc["batch_episodes"]):
            K, ep = sample(rs)
            loss = M.rmse_loss(predict(model, a.arm, ctx, ep.inputs(K)), ep.y_q) / oc["batch_episodes"]
            loss.backward()
            tot += loss.item()
        step += 1
        if not np.isfinite(tot):
            stopped = "non_finite_loss"
            break
        opt.step()
        run_loss.append(tot)
        if step % oc["eval_every"] == 0 or step == max_steps:
            if final:
                log.write(f"{step}\t{np.mean(run_loss):.6f}" + "\tnan" * len(ks) + extra() + "\n")
            else:
                v = val_score()
                log.write(f"{step}\t{np.mean(run_loss):.6f}\t" + "\t".join(f"{v[k]:.6f}" for k in ks) + extra() + "\n")
            log.flush()
            run_loss = []
    log.close()
    res = dict(arm=a.arm, lr=a.lr, seed=a.seed, fold=a.fold, n_parameters=int(n_par), steps_run=step, stopped=stopped, max_steps=max_steps,
               n_sampler_families=len(sample.families), n_val_episodes=len(val_eps), val_families=int(len({e.family for e in val_eps})),
               lam=float(model.lam().item()) if is_ad else None, eta={str(k): float(model.eta(k).item()) for k in ks} if is_ad else None,
               gpu=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu")
    if final:
        ck = os.path.join(out, "model.pt")
        torch.save(dict(state_dict={k: t.cpu() for k, t in model.state_dict().items()}, arm=a.arm, seed=a.seed, lr=a.lr, steps=step), ck)
        modes = ("normal", "zero_residual", "shuffle_residual") if is_ad else ("normal",)
        full = score_table(model, a.arm, ctx, val_eps, ks, modes)
        full.insert(0, "run_id", a.run_id)
        full.to_csv(os.path.join(out, "confirmation_episodes.tsv"), sep="\t", index=False, float_format="%.6g")
        agg, _ = family_mean(full, by=("K", "mode"))
        res.update(confirmation_delta_rmse={f"{r.mode}|{int(r.K)}": float(r.delta_rmse) for r in agg.itertuples()}, checkpoint=ck, checkpoint_sha256=sha256(ck),
                   feature_basis_sha256=sha256(fpath), repeat_basis=bpath, repeat_basis_sha256=sha256(bpath) if bpath else "", anchor_selection_run=ctx.sel.get("run_id", ""))
    dump(run_record(a.run_id, ("cv_" if not final else "final_") + a.arm, t0, **res), os.path.join(out, "result.json"))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
