"""R8-E2-B (conditional; private plan 2026-11-19 section 5.3). Entered because E2-A met entry condition (b) (see
e1_source_decisions.json 'e2b_entry'); its own specification is configs/r8_transfer_v2/e2b_spec.yaml (freeze F3).

  channel  input-CHANNEL sensitivity on the E1 episodes of one outer partition (views G / M1 / M2, contexts as in E1):
           masks full (f* anchor + f0 kernel), no_f0_kernel (anchor kept, kernel chem+ctrl), no_anchor (f* := 0, f0 kernel kept),
           no_both (f* := 0, kernel chem+ctrl); per mask C0_<mask> (kappa_b) and K_<mask> (lambda, eta with eta = 0 allowed),
           selected per mask on the matching inner partition. Query responses were inside the PCA / nuisance fits, so this is an
           input-dependence description, not a test of unseen-drug generalisation.
  sru      source-reference-UNAVAILABLE validation of one outer partition: drug groups (canonical store groups) are split into 3
           pre-fixed drug folds; for drug fold j the source f0 table, the f0 PCA basis, the kernel scales, the ridge map and every
           selection are rebuilt WITHOUT the fold-j groups (inner partition too); queries are fold-j units, supports are other
           units. Arms: fb_C0 / fb_C1chem (f0_q = 0, n0_q = 0 only for the held query drugs), ridge_C0 / ridge_C1chem (source-train
           ECFP ridge prediction as the query anchor), and the more-informed references ref_C0 / ref_C1 (full f0; not an upper bound).
           The eligibility table is written first; scoring is skipped if the pre-fixed eligibility rule fails.
"""
import argparse, json, os, time, zlib
import numpy as np
import pandas as pd
import torch
import yaml

from src.replicate_calibrated_r8.data import Features, Episode
from src.r8_followup_v1 import engine as Eng
from src.r8_followup_v1 import fmethods as F
from src.r8_followup_v1.partitions import Bank, INNER_OF
from . import engine2 as E2
from .e1 import matched_list, ks_of, oq_of
from .common import v1_runs, dump, new_run_dir, run_record, sha256, log

torch.set_grad_enabled(False)
SPEC = "configs/r8_transfer_v2/e2b_spec.yaml"
V1_PAIRS = "p1/p1_v1/d1_plate_pairs.tsv"
MASKS = {"full": (True, "all"), "no_f0_kernel": (True, "chem+ctrl"), "no_anchor": (False, "all"), "no_both": (False, "chem+ctrl")}


def spec():
    return yaml.safe_load(open(SPEC))


def drug_fold(g, n=3):
    return zlib.crc32(("r8v2_drug|" + str(g)).encode()) % n


# ================================================================= channel sensitivity
def mask_parts(ctx, inp, K, base, anchor_on, cfg):
    kf = base["kappa_f"][str(K)]
    fs, fq = F.fstar(inp, kf)
    bl = F.kernel_blocks(ctx.feats(inp, fs, fq))          # kernel features always from the true f* representation
    Kc = E2.combine2(bl, cfg, ctx.scales_k[str(K)])
    if not anchor_on:
        fs, fq = torch.zeros_like(fs), torch.zeros_like(fq)
    E = inp["y_s"] - fs
    me = E.mean(0, keepdim=True)
    w = torch.full((K,), 1.0 / K)
    return fq, E, me, E - me, w, Kc


def channel_select(ctx, base, eps, g, tie, cname, tie_log):
    KB = [Eng.kb(x) for x in g["kappa_b"]]
    LAM, ETA0 = [float(x) for x in g["lambda"]], [0.0] + [float(x) for x in g["eta"]]
    out = {}
    for mname, (an, cfg) in MASKS.items():
        r0, r1 = [], []
        for ep, b, ks in eps:
            for K in ks:
                fq, E, me, R, w, Kc = mask_parts(ctx, Eng.inputs(ep, K), K, base, an, cfg)
                for kbv in KB:
                    r0.append(dict(b, K=K, kappa_b=kbv, delta_rmse=F.rmse(fq + me * F.alpha(K, kbv), ep.y_q)))
        c0, _ = E2.choose2(r0, ["kappa_b"], "kappa_b", tie, f"C0_{mname}", f"{ctx.name}|{cname}", tie_log)
        for ep, b, ks in eps:
            for K in ks:
                fq, E, me, R, w, Kc = mask_parts(ctx, Eng.inputs(ep, K), K, base, an, cfg)
                bb = me * F.alpha(K, c0[str(K)]["kappa_b"])
                I = torch.eye(K, dtype=torch.float64)
                for lam in LAM:
                    cc = F.krr(Kc[0], Kc[1], w, R, lam, I)
                    for eta in ETA0:
                        r1.append(dict(b, K=K, lam=lam, eta=eta, delta_rmse=F.rmse(fq + bb + eta * cc, ep.y_q)))
        k1, _ = E2.choose2(r1, ["lam", "eta"], "lam_eta", tie, f"K_{mname}", f"{ctx.name}|{cname}", tie_log)
        out[mname] = dict(c0=c0, k=k1)
    return out


def channel_arms(ctx, inp, K, base, cc):
    Ks = str(K)
    for mname, (an, cfg) in MASKS.items():
        fq, E, me, R, w, Kc = mask_parts(ctx, inp, K, base, an, cfg)
        yield f"C0_{mname}", fq + me * F.alpha(K, cc[mname]["c0"][Ks]["kappa_b"])
        bb = me * F.alpha(K, cc[mname]["c0"][Ks]["kappa_b"])
        k = cc[mname]["k"][Ks]
        yield f"K_{mname}", fq + bb + k["eta"] * F.krr(Kc[0], Kc[1], w, R, k["lam"], torch.eye(K, dtype=torch.float64))


def stage_channel(a, out):
    t0 = time.time()
    sp, e1s = spec(), yaml.safe_load(open("configs/r8_transfer_v2/run_spec.yaml"))
    g, tie = e1s["grids"], e1s["tie_rule"]
    pairs = pd.read_csv(v1_runs(V1_PAIRS), sep="\t")
    P, IP = a.partition, INNER_OF[a.partition]
    base = json.load(open(os.path.join(a.p0_select, "constants_v2.json")))
    assert base["partition"] == IP
    bank = Bank()
    ictx = E2.Ctx2(bank, IP, mode="v2")
    ictx.fit_nuisance(pairs, base["kappa_f"])
    gen = [(ep, dict(cell=ep.name, family=ep.family, support_seed=ep.seed), ks_of(ep)) for ep in ictx.episodes("val", [1], 200)]
    mat = matched_list(ictx, 2, (1,))
    mb = lambda m: dict(cell=m["cell"], family=m["family"], support_seed=f"{m['plate_query']}|{m['plate_other']}|{m['support_seed']}")
    CC, tl = {}, []
    for cname, eps in (("gen", gen), ("M1", [(e1, mb(m), ks_of(e1, m["kmax"])) for m, e1, e2, pc in mat]), ("M2", [(e2, mb(m), ks_of(e2, m["kmax"])) for m, e1, e2, pc in mat])):
        CC[cname] = channel_select(ictx, base, eps, g, tie, cname, tl)
    dump(dict(partition=IP, constants=CC), os.path.join(out, "channel_constants.json"))
    pd.DataFrame(tl).to_csv(os.path.join(out, "tie_log.tsv"), sep="\t", index=False, float_format="%.17g")
    log("selected", f"{time.time() - t0:.0f}s")
    ctx = E2.Ctx2(bank, P, mode="v2")
    ctx.fit_nuisance(pairs, base["kappa_f"])
    rows = []
    ev = e1s["evaluation"]
    for ep in ctx.episodes("val", ev["general"]["support_seeds"], None):
        b = dict(fold=P, view="G", cell=ep.name, family=ep.family, support_seed=str(ep.seed))
        for K in ks_of(ep):
            for m, p in channel_arms(ctx, Eng.inputs(ep, K), K, base, CC["gen"]):
                rows.append(dict(b, K=K, method=m, **F.score(p.numpy(), ep.y_q.numpy())))
    log("general", f"{time.time() - t0:.0f}s")
    for meta, e1, e2, pc in matched_list(ctx, ev["matched"]["max_pairs"], tuple(ev["matched"]["support_seeds"])):
        b = dict(fold=P, cell=meta["cell"], family=meta["family"], support_seed=f"{meta['plate_query']}|{meta['plate_other']}|{meta['support_seed']}")
        for view, ep, cn in (("M1", e1, "M1"), ("M2", e2, "M2")):
            for K in ks_of(ep, meta["kmax"]):
                for m, p in channel_arms(ctx, Eng.inputs(ep, K), K, base, CC[cn]):
                    rows.append(dict(b, view=view, K=K, method=m, **F.score(p.numpy(), pc["truth"])))
    pd.DataFrame(rows).to_csv(os.path.join(out, "episodes.tsv.gz"), sep="\t", index=False, float_format="%.8g")
    dump(run_record(a.run_id, "e2b_channel", t0, partition=P), os.path.join(out, "result.json"))
    log("done", f"{time.time() - t0:.0f}s")


# ================================================================= source-reference-unavailable validation
def hold_store(bank, name, held):
    """Store of partition `name` whose source f0 table has no row for the held groups (n = 0, no leave-family-out subtraction)."""
    st = bank.store(name)
    h = torch.as_tensor(np.where(held)[0])
    st.S_train[h] = 0.0
    st.n_train[h] = 0.0
    st.S_all[h] = 0.0
    st.n_all[h] = 0.0
    st.fg_index_t[:, h] = -1
    return st


class HCtx(E2.Ctx2):
    def __init__(self, bank, name, held):
        self.name = name
        self.st = hold_store(bank, name, held)
        self.fmap = Features.fit(self.st)                 # f0 PCA basis without the held groups
        self.cov, self.scales, self.ut = {}, {}, None
        self.mode = "v2"
        self.cov_kf, self.scales_k, self.kfK = {}, {}, {}


def ridge_fit(st, alpha):
    ok = (st.n_train >= 1).numpy()
    X = st.FP.numpy()[ok].astype(np.float64)
    Y = (st.S_train.numpy()[ok] / st.n_train.numpy()[ok][:, None]).astype(np.float64)
    xm, ym = X.mean(0), Y.mean(0)
    Xc = X - xm
    Wt = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]), Xc.T @ (Y - ym))
    return dict(W=Wt, xm=xm, ym=ym, n_groups=int(ok.sum()))


def ridge_pred(rm, fp):
    return torch.tensor((fp.astype(np.float64) - rm["xm"]) @ rm["W"] + rm["ym"], dtype=torch.float32)


def held_episodes(st_full, st_held, held, role, seeds, cap, min_q, min_s, max_k=32):
    """Standard E1 episode RNG on the FULL store's eligible units; queries = held-group units of the query part, supports = the
    non-held units of the support order (first max_k). Returns episodes with full-f0 copies for the reference arms."""
    out, elig = [], []
    for ci in st_full.eval_cells(role):
        ui = st_full.eligible_units(ci)
        for seed in seeds:
            p = np.random.RandomState(880000 + 1000 * seed + int(zlib.crc32(st_full.cell_name[ci].encode()) % 100000)).permutation(len(ui))
            nq = int(round(0.2 * len(ui)))
            q, s = ui[p[:nq]], ui[p[nq:]]
            q = q[held[st_full.u_group[q]]]
            if cap:
                q = q[:cap]
            s = s[~held[st_full.u_group[s]]][:max_k]
            ok = len(q) >= min_q and len(s) >= min_s
            elig.append(dict(cell=st_full.cell_name[ci], family=st_full.cell_family[ci], support_seed=seed, n_query=len(q), n_support=len(s), eligible=ok))
            if not ok:
                continue
            ep = st_full.episode_from_units(ci, s, q, seed=seed)            # full f0 (reference contract)
            regime, fam = st_held.regime_of(ci), st_held.cell_family[ci]
            f0s, n0s = st_held.f0(regime, st_full.u_group[s], fam)
            assert torch.equal(n0s, ep.n0_s) and torch.allclose(f0s, ep.f0_s, atol=1e-6), "support f0 must not depend on the held groups"
            eh = Episode(name=ep.name, family=ep.family, seed=seed, g_s=ep.g_s, g_q=ep.g_q, x0_s=ep.x0_s, x0_q=ep.x0_q, f0_s=f0s, f0_q=torch.zeros_like(ep.f0_q),
                         n0_s=n0s, n0_q=torch.zeros_like(ep.n0_q), y_s=ep.y_s, y_q=ep.y_q, o_s=ep.o_s, q_s=ep.q_s)
            ep.o_qs = eh.o_qs = Eng.cross_overlap(st_full, q, s)     # metadata only; no E2-B arm uses it
            out.append((eh, ep))
    return out, elig


def sru_arms(hctx, fctx, eh, ef, K, base, c, rm):
    """Yields (method, prediction). eh: held-contract inputs; ef: full-f0 inputs (reference arms only)."""
    Ks = str(K)
    kf = base["kappa_f"][Ks]
    I = torch.eye(K, dtype=torch.float64)
    w = torch.full((K,), 1.0 / K)
    inp = Eng.inputs(eh, K)
    fs, fq0 = F.fstar(inp, kf)                       # fq0 = 0 (n0_q = 0)
    rq = ridge_pred(rm, hctx.st.FP[torch.as_tensor(eh.g_q)].numpy())
    for anc, fq in (("fb", fq0), ("ridge", rq)):
        E = inp["y_s"] - fs
        me = E.mean(0, keepdim=True)
        b = me * F.alpha(K, c[f"{anc}_C0"][Ks]["kappa_b"])
        yield f"{anc}_C0", fq + b
        bl = F.kernel_blocks(hctx.feats(inp, fs, fq))
        Kc = E2.combine2(bl, "chem+ctrl", hctx.scales_k[Ks])
        k = c[f"{anc}_C1chem"][Ks]
        b2 = me * F.alpha(K, c[f"{anc}_C1chem"][Ks]["kappa_b"])
        yield f"{anc}_C1chem", fq + b2 + k["eta"] * F.krr(Kc[0], Kc[1], w, E - me, k["lam"], I)
    if ef is None:
        return
    inf = Eng.inputs(ef, K)
    fs, fq = F.fstar(inf, kf)
    E = inf["y_s"] - fs
    me = E.mean(0, keepdim=True)
    b = me * F.alpha(K, c["ref_C0"][Ks]["kappa_b"])
    yield "ref_C0", fq + b
    bl = F.kernel_blocks(fctx.feats(inf, fs, fq))
    Kc = E2.combine2(bl, "all", fctx.scales_k[Ks])
    k = c["ref_C1"][Ks]
    yield "ref_C1", fq + b + k["eta"] * F.krr(Kc[0], Kc[1], w, E - me, k["lam"], I)


def sru_select(hctx, fctx, pairs_eps, base, g, tie, rms, tie_log, P):
    """kappa_b, lambda, eta per arm and the ridge alpha (on the ridge_C0 score) from the inner episodes of the same contract."""
    KB = [Eng.kb(x) for x in g["kappa_b"]]
    LAM, ETA0 = [float(x) for x in g["lambda"]], [0.0] + [float(x) for x in g["eta"]]
    # ridge alpha
    ra = []
    for alpha, rm in rms.items():
        for eh, ef in pairs_eps:
            for K in ks_of(eh):
                inp = Eng.inputs(eh, K)
                fs, _ = F.fstar(inp, base["kappa_f"][str(K)])
                rq = ridge_pred(rm, hctx.st.FP[torch.as_tensor(eh.g_q)].numpy())
                me = (inp["y_s"] - fs).mean(0, keepdim=True)
                ra.append(dict(cell=eh.name, family=eh.family, support_seed=eh.seed, K=K, ridge_alpha=alpha, delta_rmse=F.rmse(rq + me * F.alpha(K, 16.0), eh.y_q)))
    df = pd.DataFrame(ra)
    from src.episodic_residual_r2.evalcore import family_mean
    agg, _ = family_mean(df, by=["ridge_alpha"])
    best_alpha = float(agg.sort_values(["delta_rmse", "ridge_alpha"], ascending=[True, False]).iloc[0].ridge_alpha)
    rm = rms[best_alpha]
    c = {}
    for arm in ("fb_C0", "fb_C1chem", "ridge_C0", "ridge_C1chem", "ref_C0", "ref_C1"):
        r0 = []
        for eh, ef in pairs_eps:
            for K in ks_of(eh):
                Ks = str(K)
                if arm.startswith("ref"):
                    inp = Eng.inputs(ef, K)
                    fs, fq = F.fstar(inp, base["kappa_f"][Ks])
                else:
                    inp = Eng.inputs(eh, K)
                    fs, fq = F.fstar(inp, base["kappa_f"][Ks])
                    if arm.startswith("ridge"):
                        fq = ridge_pred(rm, hctx.st.FP[torch.as_tensor(eh.g_q)].numpy())
                me = (inp["y_s"] - fs).mean(0, keepdim=True)
                for kbv in KB:
                    r0.append(dict(cell=eh.name, family=eh.family, support_seed=eh.seed, K=K, kappa_b=kbv, delta_rmse=F.rmse(fq + me * F.alpha(K, kbv), eh.y_q)))
        c[arm], _ = E2.choose2(r0, ["kappa_b"], "kappa_b", tie, f"{arm}_kappa_b", P, tie_log)
    for arm, cfg in (("fb_C1chem", "chem+ctrl"), ("ridge_C1chem", "chem+ctrl"), ("ref_C1", "all")):
        r1 = []
        kb_arm = c[arm if arm != "ref_C1" else "ref_C0"]
        for eh, ef in pairs_eps:
            for K in ks_of(eh):
                Ks = str(K)
                ctx_, src = (fctx, ef) if arm == "ref_C1" else (hctx, eh)
                inp = Eng.inputs(src, K)
                fs, fq = F.fstar(inp, base["kappa_f"][Ks])
                if arm.startswith("ridge"):
                    fq = ridge_pred(rm, hctx.st.FP[torch.as_tensor(eh.g_q)].numpy())
                E = inp["y_s"] - fs
                me = E.mean(0, keepdim=True)
                b = me * F.alpha(K, kb_arm[Ks]["kappa_b"])
                Kc = E2.combine2(F.kernel_blocks(ctx_.feats(inp, fs, fq)), cfg, ctx_.scales_k[Ks])
                w = torch.full((K,), 1.0 / K)
                for lam in LAM:
                    cc = F.krr(Kc[0], Kc[1], w, E - me, lam, torch.eye(K, dtype=torch.float64))
                    for eta in ETA0:
                        r1.append(dict(cell=eh.name, family=eh.family, support_seed=eh.seed, K=K, lam=lam, eta=eta, delta_rmse=F.rmse(fq + b + eta * cc, eh.y_q)))
        k, _ = E2.choose2(r1, ["lam", "eta"], "lam_eta", tie, f"{arm}_lam_eta", P, tie_log)
        for K in k:
            k[K]["kappa_b"] = kb_arm[K]["kappa_b"]
        c[arm] = k
    return c, best_alpha, agg


def stage_sru(a, out):
    t0 = time.time()
    sp, e1s = spec(), yaml.safe_load(open("configs/r8_transfer_v2/run_spec.yaml"))
    g, tie = e1s["grids"], e1s["tie_rule"]
    sr = sp["sru"]
    pairs = pd.read_csv(v1_runs(V1_PAIRS), sep="\t")
    P, IP = a.partition, INNER_OF[a.partition]
    base_i = json.load(open(os.path.join(a.p0_select_inner, "constants_v2.json")))
    assert base_i["partition"] == IP
    bank = Bank()
    groups = bank.groups.drug_group.to_numpy()
    dfold = np.array([drug_fold(x, sr["n_drug_folds"]) for x in groups])
    rows, elig_all, cons = [], [], {}
    for j in range(sr["n_drug_folds"]):
        held = dfold == j
        # ---- inner (selection) with the same availability restriction
        fi = E2.Ctx2(bank, IP, mode="v2")
        fi.fit_nuisance(pairs, base_i["kappa_f"])
        hi = HCtx(bank, IP, held)
        hi.fit_nuisance(pairs, base_i["kappa_f"])
        ieps, ielig = held_episodes(fi.st, hi.st, held, "val", [1], 200, sr["min_query"], sr["min_support"])
        rms = {al: ridge_fit(hi.st, al) for al in sr["ridge_alpha"]}
        c, alpha, agg = sru_select(hi, fi, ieps, base_i, g, tie, rms, [], f"{IP}|drugfold{j}")
        cons[j] = dict(constants=c, ridge_alpha=alpha, ridge_alpha_scores=agg.to_dict("records"), n_inner_episodes=len(ieps),
                       n_inner_families=len({e.family for e, _ in ieps}))
        log(P, "drug fold", j, "selected", alpha, f"{time.time() - t0:.0f}s")
        # ---- outer
        fo = E2.Ctx2(bank, P, mode="v2")
        fo.fit_nuisance(pairs, base_i["kappa_f"])
        ho = HCtx(bank, P, held)
        ho.fit_nuisance(pairs, base_i["kappa_f"])
        rm = ridge_fit(ho.st, alpha)
        oeps, oelig = held_episodes(fo.st, ho.st, held, "val", [1, 2, 3], None, sr["min_query"], sr["min_support"])
        for e in oelig:
            elig_all.append(dict(e, fold=P, drug_fold=j))
        fam_ok = len({e.family for e, _ in oeps if len(e.g_s) >= 16})
        if fam_ok < sr["min_families_per_fold"]:
            log("eligibility rule failed", P, j, fam_ok)
            continue
        for eh, ef in oeps:
            for K in ks_of(eh):
                for m, p in sru_arms(ho, fo, eh, ef, K, base_i, c, rm):
                    rows.append(dict(fold=P, drug_fold=j, cell=eh.name, family=eh.family, support_seed=f"{eh.seed}|d{j}", K=K, method=m, **F.score(p.numpy(), eh.y_q.numpy())))
        log(P, "drug fold", j, "scored", len(oeps), f"{time.time() - t0:.0f}s")
    pd.DataFrame(elig_all).to_csv(os.path.join(out, "eligibility.tsv"), sep="\t", index=False)
    pd.DataFrame(rows).to_csv(os.path.join(out, "episodes.tsv.gz"), sep="\t", index=False, float_format="%.8g")
    dump(dict(partition=P, drug_fold_constants=cons, drug_fold_sizes={int(j): int((dfold == j).sum()) for j in range(sr["n_drug_folds"])}), os.path.join(out, "sru_constants.json"))
    dump(run_record(a.run_id, "e2b_sru", t0, partition=P), os.path.join(out, "result.json"))
    log("done", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["channel", "sru"])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--partition", required=True)
    ap.add_argument("--p0-select", default="")
    ap.add_argument("--p0-select-inner", default="")
    a = ap.parse_args()
    out = new_run_dir("e2b_" + a.stage, a.run_id)
    {"channel": stage_channel, "sru": stage_sru}[a.stage](a, out)
