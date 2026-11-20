"""Output-path sensitivity of the TRAINED FiLM adapters (B01: film + iid, B11: film + cov) on their existing checkpoints.

Same checkpoint, same support labels and drug features; only the global state entering the FiLM gamma changes:
  actual        the episode's own global state
  train_mean    the source-train mean state (support-mean control = source-train family mean -> global block = 0)
  other_family  the global state of a pre-fixed other source-train family (next family in sorted order, its seed-1 episode)
  gamma0        FiLM switched off (h = hd), i.e. gamma = 0
Measured after the support centring (the kernel the GLS solve sees) and on the final correction / prediction.
Episodes: source-train dev episodes (seed 1, query cap 200, leave-family-out f0). The checkpoints were trained on these families,
so the delta-RMSE rows are IN-SAMPLE sensitivity, not out-of-fold performance. Output changes need no label.
This is an input-usage sensitivity: swapped states can form unrealistic combinations; usefulness is not tested here.
"""
import argparse, json, math, os, time
import numpy as np
import pandas as pd
import torch

from src.replicate_calibrated_r8.data import Store, Features
from src.replicate_calibrated_r8.arms import ARMS, Context, load_anchor_selection
from src.replicate_calibrated_r8.suite import load_learned
from src.replicate_calibrated_r8 import methods as M
from .common import orig_man, dump, new_run_dir, run_record, pub_man
from . import stats as S

torch.set_grad_enabled(False)


def embed(model, ft, side, glob, mode):
    hd = model.lin_d(torch.cat([ft["fp_" + side], ft["f0r_" + side]], 1))
    gam = model.gamma(glob)
    if mode != "gamma0":
        hd = hd * (1.0 + gam)
    h = hd + model.lin_w(ft["ctrl_" + side])
    return model.head(h) / math.sqrt(model.out), gam


def correction(model, anc, ft, glob, mode):
    zs, g = embed(model, ft, "s", glob, mode)
    zq, _ = embed(model, ft, "q", glob, mode)
    m = (anc["w"][:, None] * zs).sum(0, keepdim=True)
    zs, zq = zs - m, zq - m
    Kss, Kqs = (zs @ zs.T).double(), (zq @ zs.T).double()
    corr = M.gls_solve(Kss, Kqs, anc["R"].double(), model.lam().double(), anc["Om"]).float() * model.eta(len(zs))
    return Kss, Kqs, corr, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args()
    t0 = time.time()
    out = new_run_dir("film", a.run_id)
    st = Store("cpu")
    fmap = Features.load(os.path.join(os.environ["R8_PROC"], "feature_basis.npz"), "cpu")
    ctx = Context(fmap, st.FP, selection=load_anchor_selection(committed_only=True))
    sel = json.load(open(orig_man("model_selection.json")))["final"]
    eps = st.episodes("train", [1], 200)
    fams = sorted({e.family for e in eps})
    donor = {}
    first = {}
    for e in eps:
        first.setdefault(e.family, e)
    for i, f in enumerate(fams):
        donor[f] = first[fams[(i + 1) % len(fams)]]
    rows = []
    for arm in ("B01", "B11"):
        for seed, entry in sel[arm].items():
            _, model = load_learned(entry, "cpu")
            obs = ARMS[arm]["obs"]
            for ep in eps:
                for K in (8, 16, 32):
                    if K > len(ep.g_s):
                        continue
                    inp = ep.inputs(K)
                    anc = ctx.anchor(inp, obs)
                    ft = ctx.features(inp, anc)
                    dinp = donor[ep.family].inputs(K)
                    g_other = fmap.glob(dinp["x0_s"])
                    states = dict(actual=ft["glob"], train_mean=torch.zeros_like(ft["glob"]), other_family=g_other, gamma0=ft["glob"])
                    ref = None
                    for name, g in states.items():
                        Kss, Kqs, corr, gam = correction(model, anc, ft, g, "gamma0" if name == "gamma0" else "film")
                        pred = anc["fq"] + anc["b"] + corr
                        if name == "actual":
                            ref = (Kss, Kqs, corr, pred)
                            check = model(anc, ft)
                            r0 = dict(arm=arm, model_seed=int(seed), cell=ep.name, family=ep.family, support_seed=ep.seed, K=K, state="actual",
                                      gamma_abs_mean=float(gam.abs().mean()), glob_norm=float(g.norm()), path_matches_model=float((check - pred).abs().max()),
                                      kss_rel_change=0.0, kqs_rel_change=0.0, corr_rel_change=0.0, corr_rms=float(corr.pow(2).mean().sqrt()),
                                      delta_rmse=float(torch.sqrt(((pred - ep.y_q) ** 2).mean(1)).mean()))
                            rows.append(r0)
                            continue
                        rel = lambda x, y: float((x - y).norm() / y.norm().clamp(min=1e-30))
                        rows.append(dict(arm=arm, model_seed=int(seed), cell=ep.name, family=ep.family, support_seed=ep.seed, K=K, state=name,
                                         gamma_abs_mean=float(gam.abs().mean()) if name != "gamma0" else 0.0, glob_norm=float(g.norm()),
                                         path_matches_model=np.nan, kss_rel_change=rel(Kss, ref[0]), kqs_rel_change=rel(Kqs, ref[1]), corr_rel_change=rel(corr, ref[2]),
                                         corr_rms=float(corr.pow(2).mean().sqrt()), delta_rmse=float(torch.sqrt(((pred - ep.y_q) ** 2).mean(1)).mean())))
            print(arm, seed, f"{time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "film_episodes.tsv.gz"), sep="\t", index=False, float_format="%.6g")
    # model seeds averaged per episode, then the family-equal mean; delta-RMSE change vs actual with a family bootstrap
    ep = df.groupby(["arm", "state", "cell", "family", "support_seed", "K"]).mean(numeric_only=True).reset_index()
    tab = []
    for (arm, K, state), g in ep.groupby(["arm", "K", "state"]):
        f = g.groupby(["family", "cell"]).mean(numeric_only=True).groupby(level=0).mean()
        base = ep[(ep.arm == arm) & (ep.K == K) & (ep.state == "actual")].groupby(["family", "cell"]).delta_rmse.mean().groupby(level=0).mean()
        d = (f.delta_rmse - base).dropna().to_numpy()
        r = dict(arm=arm, K=K, state=state, n_families=len(f), gamma_abs_mean=float(f.gamma_abs_mean.mean()), glob_norm=float(f.glob_norm.mean()),
                 kss_rel_change=float(f.kss_rel_change.mean()), kqs_rel_change=float(f.kqs_rel_change.mean()), corr_rel_change=float(f.corr_rel_change.mean()),
                 corr_rms=float(f.corr_rms.mean()), delta_rmse=float(f.delta_rmse.mean()), max_path_mismatch=float(df[(df.arm == arm) & (df.K == K)].path_matches_model.max()))
        if state != "actual":
            s = S.summarise_diff(d, float(base.mean()))
            r.update(delta_rmse_change=s["diff"], lo=s["lo"], hi=s["hi"], n_families_lower=int((d < 0).sum()))
        tab.append(r)
    tab = pd.DataFrame(tab)
    tab["episodes"] = "source-train dev episodes (seed 1, cap 200): IN-SAMPLE for these checkpoints; output changes are label-free"
    tab.to_csv(pub_man("state_path_usage.tsv"), sep="\t", index=False, float_format="%.6g")
    dump(run_record(a.run_id, "film", t0, n_rows=len(df), checkpoints={arm: {s: e["checkpoint_sha256"] for s, e in sel[arm].items()} for arm in ("B01", "B11")}),
         os.path.join(out, "result.json"))
    pd.set_option("display.width", 250)
    print(tab.to_string())


if __name__ == "__main__":
    main()
