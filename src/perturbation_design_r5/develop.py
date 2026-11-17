"""Source-only development: D3 noise model + gate, readout selection (and D1 on dev), GP hyper-parameters, likelihood
temperature, D2 diagnostic, learned-loss training. Nothing here reads meta-validation treated values except the D3
evaluation pairs (replicate noise, not a policy score), and nothing reads HCC treated values."""
import argparse, json, os, pickle, time
from multiprocessing import get_context
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

from .common import protocol, runs, public, dump_json, nproc, commit, sha_file, sha_hex
from .data import Store, make_episode, target_cells, f0_table, allowed_families
from .models import EpisodeKernels, Readout, GP, make_gps, mixture_weights, nu_vector, noise_features, KERNELS, HYPOTHESES
from .oracle import Truth, score
from .replay import run_selection, auc
from .acquisition import Context, learned_loss_features, _fixed_post

ALPHAS = [0.0, 0.5, 1.0]
LAMS = [0.1, 0.3, 1, 3, 10, 30, 100]
SB2 = [0.03, 0.1, 0.3, 1, 3, 10]
S2 = [0.03, 0.1, 0.3, 1, 3]
C0 = [0.25, 0.5, 1, 2, 4]
TEMPS = [1 / 32, 1 / 8, 1 / 2, 1]
OUT = lambda *p: public("manifests", "perturbation_design_r5", "develop", *p)


# ------------------------------------------------------------------ D3
def d3(st):
    b = st.bases
    U, sq = b["z_U"].astype(np.float64), np.sqrt(b["z_lam"].astype(np.float64))
    d = U.shape[1]
    rows = st.rows[(st.rows.role == "source") & (st.rows.n_trt_wells == 1)]
    pairs = []
    for (cell, g), idx in rows.groupby(["cell", "drug_group"], sort=True).indices.items():
        if len(idx) < 2:
            continue
        s = rows.iloc[idx].sort_values("row_hash")
        s = s.drop_duplicates("det_plate")
        if len(s) < 2:
            continue
        pairs.append((cell, g, s.iloc[0], s.iloc[1]))
    meta = []
    for cell, g, a, bb in pairs:
        meta.append(dict(cell=cell, family=a.family, meta_role=a.meta_role, drug_group=g, ia=int(a.store_idx), ib=int(bb.store_idx),
                         sd_a=a.ctl_sd, sd_b=bb.ctl_sd, n_a=a.n_ctl, n_b=bb.n_ctl))
    P = pd.DataFrame(meta)
    wd = st.well_delta
    za = (np.asarray(wd[P.ia.to_numpy()], np.float64) @ U) / sq
    zb = (np.asarray(wd[P.ib.to_numpy()], np.float64) @ U) / sq
    P["h"] = ((za - zb) ** 2).sum(1) / (2 * d)
    # f0 features per pair (meta-train: leave own family out; meta-val: meta-train families)
    f0n = np.full(len(P), np.nan)
    nf = np.full(len(P), np.nan)
    for (role, fam), idx in P.groupby(["meta_role", "family"]).indices.items():
        allowed = allowed_families(st, "dev" if role == "train" else "meta_val", fam)
        names, F0, nc = f0_table(st, allowed)
        pos = pd.Series(np.arange(len(names)), index=names)
        gg = P.drug_group.iloc[idx]
        ok = gg.isin(pos.index).to_numpy()
        j = pos[gg[ok]].to_numpy()
        f0n[idx[ok]] = np.linalg.norm((F0[j].astype(np.float64) @ U) / sq, axis=1) / np.sqrt(d)
        nf[idx[ok]] = nc[j]
    P["f0n"], P["nf0"] = f0n, nf
    P = P[np.isfinite(P.f0n)].reset_index(drop=True)
    med = np.nanmedian(st.rows.ctl_sd)
    for c in ("sd_a", "sd_b"):
        P[c] = P[c].fillna(med)

    def X(side):
        return np.column_stack([np.ones(len(P)), np.log(P[f"sd_{side}"]), np.log(P[f"n_{side}"]), np.log(np.maximum(P.f0n, 1e-6)), np.log1p(P.nf0)])

    Xa, Xb = X("a"), X("b")
    tr = (P.meta_role == "train").to_numpy()
    h = P.h.to_numpy()

    def loss(beta, m):
        mu = 0.5 * (np.exp(Xa[m] @ beta) + np.exp(Xb[m] @ beta))
        return np.mean(h[m] / mu + np.log(mu))

    b0 = np.zeros(5)
    b0[0] = np.log(h[tr].mean())
    fit = minimize(loss, b0, args=(tr,), method="L-BFGS-B")
    beta = fit.x
    P["pred"] = 0.5 * (np.exp(Xa @ beta) + np.exp(Xb @ beta))
    const = h[tr].mean()

    def dev_ratio(m):
        dmod = np.mean(h[m] / P.pred[m] + np.log(P.pred[m]))
        dcon = np.mean(h[m] / const + np.log(const))
        return float(dmod - dcon)

    per = []
    for (role, fam), g in P.groupby(["meta_role", "family"]):
        if len(g) < 20:
            continue
        rho = spearmanr(g.pred, g.h).correlation
        m = (P.family == fam).to_numpy()
        per.append(dict(meta_role=role, family=fam, pairs=len(g), spearman=rho, deviance_minus_const=dev_ratio(m), mean_h=g.h.mean()))
    per = pd.DataFrame(per)
    val = per[per.meta_role == "val"]
    rs = np.random.RandomState(20261117)
    bs = val.spearman.to_numpy()[rs.randint(0, len(val), size=(2000, len(val)))].mean(1)
    gate = dict(val_families=int(len(val)), mean_spearman=float(val.spearman.mean()), lo=float(np.percentile(bs, 2.5)),
                hi=float(np.percentile(bs, 97.5)), threshold=0.10)
    gate["passed"] = bool(gate["mean_spearman"] >= 0.10 and gate["lo"] > 0)
    tr_rho = per[per.meta_role == "train"].spearman.mean()
    res = dict(beta=beta.tolist(), beta_names=["const", "log ctl_sd", "log n_ctl", "log |f0|_z", "log1p n_f0"],
               fit_success=bool(fit.success), pairs_train=int(tr.sum()), pairs_val=int((~tr).sum()),
               const_h_train=float(const), train_family_mean_spearman=float(tr_rho), gate=gate,
               val_deviance_minus_const_family_mean=float(val.deviance_minus_const.mean()))
    per.to_csv(OUT("d3_families.tsv"), sep="\t", index=False, float_format="%.5f")
    dump_json(res, OUT("d3_noise_model.json"))
    print("D3", json.dumps(res, indent=1))
    return res


# ------------------------------------------------------------------ random sequences on dev
_G = {}


def _init():
    _G["st"] = Store()
    _G["pr"] = protocol()


def _rand_job(args):
    cell, fam, seed, mode, hp = args
    st, pr = _G["st"], _G["pr"]
    budgets = pr["episodes"]["budgets_K"]
    ep = make_episode(st, "dev", cell, fam, seed, pr=pr)
    sel = run_selection(st, ep, "random", hp or {}, budgets)
    ek = sel["ek"]
    if mode == "readout":
        tr = Truth(st, ep)
        out = []
        cfgs = [("mean_offset", 0, 1)] + [("main", a, l) for a in ALPHAS for l in LAMS]
        for K in budgets:
            if K > len(sel["S"]):
                continue
            S = np.array(sel["S"][:K])
            for kind, a, l in cfgs:
                ro = Readout(a, l, kind)
                sc, _ = score(ro.predict(ep, ek, S, sel["Y"][:K], ep.query), tr)
                out.append(dict(cell=cell, family=fam, seed=seed, K=K, kind=kind, alpha=a, lam=l, rmse=sc["rmse"]))
        return out
    if mode == "gp":
        S = np.array(sel["S"][:32])
        Z = ek.to_z(ep, S, sel["Y"][:32])
        return dict(cell=cell, fam=fam, seed=seed, S=S, Z=Z, ep=ep, ek=ek)
    if mode == "temp":
        out = []
        nu = nu_vector(ep, ek, hp)
        gps = make_gps(ek, nu, hp, "mixture")
        Sall = np.array(sel["S"])
        Zall = ek.to_z(ep, Sall, sel["Y"])
        for K in range(4, len(Sall) - 3, 4):
            S, Z = Sall[:K], Zall[:K]
            B, ZB = Sall[K:K + 4], Zall[K:K + 4]
            lls = np.array([g.loglik(S, Z) for _, g in gps])
            pred = []
            for _, g in gps:      # log p_h(Z_B | S) = loglik(S+B) - loglik(S)
                pred.append(g.loglik(np.concatenate([S, B]), np.concatenate([Z, ZB])) - g.loglik(S, Z))
            pred = np.array(pred)
            for T in TEMPS:
                lw = T * lls
                w = np.exp(lw - lw.max())
                w /= w.sum()
                m = pred.max()
                out.append(dict(cell=cell, family=fam, seed=seed, K=K, T=T, logpred=float(m + np.log((w * np.exp(pred - m)).sum()))))
        return out
    if mode == "d2":
        tr = Truth(st, ep)
        ro = Readout(hp["readout"]["alpha"], hp["readout"]["lam"], "main")
        rs = np.random.RandomState(int(sha_hex(f"d2|{cell}|{seed}")[:8], 16))
        out = []
        wd = st.well_delta
        for K in (4, 16):
            S = list(sel["S"][:K])
            Y = sel["Y"][:K]
            base, _ = score(ro.predict(ep, ek, np.array(S), Y, ep.query), tr)
            Z = ek.to_z(ep, np.array(S), Y)
            ctx = Context(ep, ek, hp, S, Y, Z, 0, 0)
            cand = ctx.cand
            pick = rs.choice(len(cand), min(64, len(cand)), replace=False)
            nu_h, gh, _, Sig_h = _fixed_post(ctx, hetero=True)
            nu0, g0, _, Sig0 = _fixed_post(ctx, hetero=False)
            pos = {g: i for i, g in enumerate(cand)}
            Ri = np.array([pos[g] for g in ctx.R])
            A = (Sig0[Ri] ** 2).sum(0) / (g0.nu[cand] + np.diag(Sig0)) / len(Ri)
            T = ek.tan
            for p in pick:
                j = int(cand[p])
                yj = np.asarray(wd[int(ep.rep.store_idx.iloc[j])], np.float32)
                sc, _ = score(ro.predict(ep, ek, np.array(S + [j]), np.vstack([Y, yj]), ep.query), tr)
                out.append(dict(cell=cell, family=fam, seed=seed, K=K, group=ep.groups[j], gain=base["rmse"] - sc["rmse"],
                                latent_var=Sig0[p, p], total_var=Sig_h[p, p] + gh.nu[j], nu=gh.nu[j], A_fixed=A[p],
                                tanimoto_dist=1 - T[j, S].max()))
        return out
    if mode == "ll":
        ro = Readout(hp["readout"]["alpha"], hp["readout"]["lam"], "main")
        rs = np.random.RandomState(int(sha_hex(f"ll|{cell}|{seed}")[:8], 16))
        wd = st.well_delta
        X, yv = [], []
        for K in (4, 8, 16):
            S = list(sel["S"][:K])
            Y = sel["Y"][:K]
            Z = ek.to_z(ep, np.array(S), Y)
            ctx = Context(ep, ek, hp, S, Y, Z, 0, 0)
            F = learned_loss_features(ctx)
            pick = rs.choice(len(ctx.cand), min(200, len(ctx.cand)), replace=False)
            js = ctx.cand[pick]
            pred = ro.predict(ep, ek, np.array(S), Y, js)
            true = np.asarray(wd[ep.rep.store_idx.to_numpy()[js]], np.float64)
            X.append(F[pick])
            yv.append(np.log(np.sqrt(((pred - true) ** 2).mean(1))))
        return dict(X=np.concatenate(X), y=np.concatenate(yv), fam=fam)
    raise ValueError(mode)


def _map(jobs):
    with get_context("fork").Pool(nproc(), initializer=_init) as pool:
        return pool.map(_rand_job, jobs, chunksize=1)


def fam_mean(df, by, val):
    return df.groupby(["family"] + by)[val].mean().groupby(by).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="d3,readout,gp,temp,d2,ll")
    a = ap.parse_args()
    steps = a.steps.split(",")
    t0 = time.time()
    st = Store()
    pr = protocol()
    dev = target_cells(st, "dev", pr)
    dump_json(dict(dev=dev, meta_val=target_cells(st, "meta_val", pr)), OUT("roles.json"))
    hp_path = OUT("hyper.json")
    hp = json.load(open(hp_path)) if os.path.exists(hp_path) else {}
    if "d3" in steps:
        r = d3(st)
        hp.update(beta=r["beta"], hetero=r["gate"]["passed"], d3_gate=r["gate"])
    if "readout" in steps:
        res = _map([(c, f, s, "readout", None) for c, f in dev for s in range(1, 11)])
        df = pd.DataFrame([x for r in res for x in r])
        df.to_csv(OUT("readout_grid_dev.tsv.gz"), sep="\t", index=False)
        ep_auc = df.groupby(["family", "seed", "kind", "alpha", "lam"]).apply(lambda g: auc(g.sort_values("K").K, g.sort_values("K").rmse)).rename("auc").reset_index()
        tab = fam_mean(ep_auc, ["kind", "alpha", "lam"], "auc").reset_index()
        tab.to_csv(OUT("readout_selection_dev.tsv"), sep="\t", index=False, float_format="%.5f")
        best = tab[tab.kind == "main"].sort_values(["auc", "lam"]).iloc[0]
        hp["readout"] = dict(alpha=float(best.alpha), lam=float(best.lam), dev_auc=float(best.auc),
                             dev_auc_mean_offset=float(tab[tab.kind == "mean_offset"].auc.iloc[0]))
        d1 = fam_mean(df[(df.kind == "mean_offset") | ((df.alpha == best.alpha) & (df.lam == best.lam))], ["kind", "K"], "rmse").reset_index()
        d1.to_csv(OUT("d1_random_curve_dev.tsv"), sep="\t", index=False, float_format="%.5f")
        print(tab.sort_values("auc").head(8).to_string(), "\n", d1.to_string())
    if "gp" in steps:
        res = _map([(c, f, s, "gp", None) for c, f in dev for s in (1, 2, 3)])
        grid = []

        def total(kname, sb2, s2, nu_fn):
            ll = 0.0
            for r in res:
                g = GP(r["ek"].K(kname), nu_fn(r), sb2, s2)
                ll += g.loglik(r["S"], r["Z"])
            return ll

        for sb2 in SB2:
            for s2 in S2:
                for c0 in C0:
                    grid.append(dict(kernel="combined", sb2=sb2, s2=s2, c0=c0, ll=total("combined", sb2, s2, lambda r: np.full(len(r["ep"].groups), c0))))
        g = pd.DataFrame(grid)
        best = g.sort_values("ll", ascending=False).iloc[0]
        sb2, c0 = float(best.sb2), float(best.c0)
        s2 = {"combined": float(best.s2)}
        for k in ("tanimoto", "f0_cosine"):
            gk = [dict(kernel=k, sb2=sb2, s2=v, c0=c0, ll=total(k, sb2, v, lambda r: np.full(len(r["ep"].groups), c0))) for v in S2]
            grid += gk
            s2[k] = float(max(gk, key=lambda x: x["ll"])["s2"])
        hp.update(sb2=sb2, s2=s2, noise_homo=c0)
        if hp.get("beta") is not None:
            het = []
            for c1 in C0:
                hh = dict(hp, hetero=True, noise_hetero=c1)
                het.append(dict(kernel="combined", c1=c1, ll=total("combined", sb2, s2["combined"], lambda r: nu_vector(r["ep"], r["ek"], hh))))
            hb = max(het, key=lambda x: x["ll"])
            hp["noise_hetero"] = float(hb["c1"])
            hp["hetero_vs_homo_loglik"] = float(hb["ll"] - best.ll)
            grid += het
        pd.DataFrame(grid).to_csv(OUT("gp_grid_dev.tsv"), sep="\t", index=False, float_format="%.4f")
        print("GP", {k: hp[k] for k in ("sb2", "s2", "noise_homo", "noise_hetero") if k in hp})
    if "temp" in steps:
        res = _map([(c, f, s, "temp", hp) for c, f in dev for s in (4, 5, 6)])
        df = pd.DataFrame([x for r in res for x in r])
        tab = fam_mean(df, ["T"], "logpred").reset_index()
        tab.to_csv(OUT("temperature_dev.tsv"), sep="\t", index=False, float_format="%.4f")
        hp["T"] = float(tab.sort_values(["logpred", "T"], ascending=[False, True]).iloc[0]["T"])
        print("T", tab.to_string())
    dump_json(hp, hp_path)
    if "d2" in steps:
        res = _map([(c, f, s, "d2", hp) for c, f in dev for s in (1, 2, 3)])
        df = pd.DataFrame([x for r in res for x in r])
        df.to_csv(OUT("d2_candidates_dev.tsv.gz"), sep="\t", index=False)
        rec = []
        for (fam, seed, K), g in df.groupby(["family", "seed", "K"]):
            for m in ("latent_var", "total_var", "nu", "A_fixed", "tanimoto_dist"):
                rec.append(dict(family=fam, seed=seed, K=K, feature=m, spearman=spearmanr(g[m], g.gain).correlation))
        r = pd.DataFrame(rec)
        tab = r.groupby(["family", "K", "feature"]).spearman.mean().groupby(["K", "feature"]).agg(["mean", "std", "count"]).reset_index()
        gain = df.groupby(["family", "K"]).gain.agg(["mean", "median", lambda x: (x > 0).mean()]).groupby("K").mean()
        tab.to_csv(OUT("d2_spearman_dev.tsv"), sep="\t", index=False, float_format="%.4f")
        gain.to_csv(OUT("d2_gain_dev.tsv"), sep="\t", float_format="%.5f")
        print("D2", tab.to_string(), gain.to_string())
    if "ll" in steps:
        from sklearn.ensemble import GradientBoostingRegressor
        res = _map([(c, f, s, "ll", hp) for c, f in dev for s in (7, 8, 9, 10)])
        X = np.nan_to_num(np.concatenate([r["X"] for r in res]))
        y = np.concatenate([r["y"] for r in res])
        m = GradientBoostingRegressor(random_state=0).fit(X, y)
        path = os.path.join(runs(), "learned_loss_v1.pkl")
        with open(path, "wb") as f:
            pickle.dump(m, f)
        hp["learned_loss_model_path"] = path
        hp["learned_loss_model_sha256"] = sha_file(path)
        hp["learned_loss_train"] = dict(rows=int(len(y)), r2_train=float(m.score(X, y)))
        print("LL", hp["learned_loss_train"])
    hp["commit"] = commit()
    hp["seconds"] = round(time.time() - t0, 1)
    dump_json(hp, hp_path)


if __name__ == "__main__":
    main()
