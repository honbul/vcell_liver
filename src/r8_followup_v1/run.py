"""Driver of the follow-up source analyses.

  p1        recompute the plate-pair table, reproduce the original V0 constants, dev closed-form scores and matched-plate
            (source-train) scores with the original code path and constants; covariance versions of the global partition
  select    nested selection inside one INNER partition (inner0 / inner1 / inner2 / final_inner)
  evaluate  one OUTER partition (outer0 / outer1 / outer2): refit nuisance on outer-train, score the outer-test families on
            general episodes and on matched-plate E1 / E2 episodes (+ replicate positive control); influence, LOSO, n0 strata,
            reference halves, full-row permutation check
No confirmation family and no HCC value is read here.
"""
import argparse, json, os, time, zlib
import numpy as np
import pandas as pd
import torch
import yaml

from src.replicate_calibrated_r8.data import Features
from src.replicate_calibrated_r8 import methods as M
from src.replicate_calibrated_r8 import diag as D
from src.episodic_residual_r2.evalcore import family_mean
from .common import CFG, orig_man, runs, dump, new_run_dir, run_record, sha256
from .partitions import Bank, INNER_OF
from . import engine as Eng
from . import fmethods as F
from .matched import matched_episodes, load_rows

torch.set_grad_enabled(False)


def log(*a):
    print(*a, flush=True)


def load_pairs(path):
    return pd.read_csv(path, sep="\t")


def orig_const(fx, cov_v0):
    """The original constants in the follow-up constant layout (V0 = original covariance)."""
    return dict(kappa_f={K: float(v) for K, v in fx["kappa_f"].items()}, shrunk_kappa_f={K: float(v) for K, v in fx["shrunk_kappa_f"].items()},
                kappa_b=dict(iid={K: Eng.kb(v) for K, v in fx["kappa_b"]["iid"].items()}, V0={K: Eng.kb(v) for K, v in fx["kappa_b"]["cov"].items()}),
                c1=dict(iid=fx["c1"]["iid"], V0=fx["c1"]["cov"]))


# ---------------------------------------------------------------- p1
def p1(a, cfg, out):
    t0 = time.time()
    proc = os.environ["R8_PROC"]
    rows = pd.read_csv(os.path.join(proc, "rows.tsv"), sep="\t")
    plates = pd.read_csv(os.path.join(proc, "plates.tsv"), sep="\t")
    cells = pd.read_csv(os.path.join(proc, "cells.tsv"), sep="\t")
    RD = D.row_delta(rows)
    pairs = D.d1(rows, plates, cells, RD)
    pairs.to_csv(os.path.join(out, "d1_plate_pairs.tsv"), sep="\t", index=False, float_format="%.10g")
    orig = pd.read_csv(os.path.join(os.environ["R8_RUNS"], "diag", "diag_v1", "d1_plate_pairs.tsv"), sep="\t")
    chk = dict(d1_pairs=dict(n_recomputed=len(pairs), n_original=len(orig),
                             max_abs_diff_v_e=float(np.abs(pairs.v_e.to_numpy() - orig.v_e.to_numpy()).max()) if len(orig) == len(pairs) else None,
                             max_abs_diff_v_s=float(np.abs(pairs.v_s.to_numpy() - orig.v_s.to_numpy()).max()) if len(orig) == len(pairs) else None))
    log(chk, f"{time.time() - t0:.0f}s")
    bank = Bank()
    fpath = os.path.join(proc, "feature_basis.npz")
    ctx = Eng.Ctx(bank, "global", fmap=Features.load(fpath, "cpu"))
    refit = Features.fit(ctx.st)
    chk["feature_basis"] = dict(path=fpath, sha256=sha256(fpath), max_abs_diff_refit={k: float((getattr(refit, k) - getattr(ctx.fmap, k)).abs().max()) for k in Features.KEYS})
    fx = Eng.load_fixed(orig_man("anchor_selection.json"))
    v = ctx.fit_cov(pairs, 8.0)
    cm = json.load(open(orig_man("covariance_model.json")))
    chk["covariance_V0"] = dict(original={k: cm[k] for k in ("v_r", "v_e", "v_s")}, recomputed={k: v["V0"][k] for k in ("v_r", "v_e", "v_s")},
                                rel_diff={k: abs(v["V0"][k] - cm[k]) / abs(cm[k]) for k in ("v_r", "v_e", "v_s")},
                                original_v_tot=cm["provenance"]["v_tot_unit_residual"], recomputed_v_tot=v["V0"]["v_tot"], original_n_pairs=cm["provenance"]["n_pairs"],
                                recomputed_n_pairs=v["V0"]["n_pairs"])
    ctx.ut.to_csv(os.path.join(out, "unit_table_global.tsv"), sep="\t", index=False, float_format="%.8g")
    dump(v, os.path.join(out, "cov_versions_global.json"))
    log(json.dumps(chk["covariance_V0"], indent=1))
    # dev closed-form reproduction (original code path: methods.anchor / krr_correction with the original covariance model)
    st, fm = ctx.st, ctx.fmap
    cov0 = fx["covariance_model"]
    recs = []
    const = orig_const(fx, v["V0"])
    for ep in ctx.episodes("train", [1], 200):
        for K in (8, 16, 32):
            if K > len(ep.g_s):
                continue
            inp = Eng.inputs(ep, K)
            Ks = str(K)
            b = dict(cell=ep.name, family=ep.family, support_seed=ep.seed, K=K)
            r = lambda m, p: recs.append(dict(method=m, delta_rmse=F.rmse(p, ep.y_q), **b))
            r("no_change", M.no_change(inp)), r("same_drug_mean", M.same_drug_mean(inp)), r("mean_offset", M.mean_offset(inp))
            r("shrunk_same_drug_mean", M.anchor(inp, fx["shrunk_kappa_f"][Ks], "inf")["fq"])
            for obs, om in (("iid", None), ("cov", cov0)):
                kbv = fx["kappa_b"][obs][Ks]
                anc = M.anchor(inp, fx["kappa_f"][Ks], Eng.kb(kbv) if kbv != "inf" else "inf", om)
                r(f"C0_{obs}", M.c0(anc))
                c = fx["c1"][obs][Ks]
                r(f"C1_{obs}", M.c1(anc, fm.build(inp, st.FP, anc["fs"], anc["fq"]), c["lam"], c["eta"]))
            # the follow-up code path with the original constants and the recomputed V0 must give the same numbers
            for v_ in ("iid", "V0"):
                Om = ctx.omega(inp, v_)
                aa = F.anchor_om(inp, const["kappa_f"][Ks], const["kappa_b"][v_][Ks], Om)
                r(f"fu_C0_{v_}", aa["fq"] + aa["b"])
                c = const["c1"][v_][Ks]
                Kall = F.combine(F.kernel_blocks(ctx.feats(inp, aa["fs"], aa["fq"])), "all")
                r(f"fu_C1_{v_}", aa["fq"] + aa["b"] + c["eta"] * F.krr(Kall[0], Kall[1], aa["w"], aa["R"], c["lam"], Om))
    df = pd.DataFrame(recs)
    agg, _ = family_mean(df)
    ocs = pd.read_csv(orig_man("closed_form_summary.tsv"), sep="\t")
    ocs = ocs[ocs.set == "dev"]
    m = agg.merge(ocs[["K", "method", "delta_rmse"]].rename(columns={"delta_rmse": "original"}), on=["K", "method"], how="left")
    alias = {"fu_C0_iid": "C0_iid", "fu_C0_V0": "C0_cov", "fu_C1_iid": "C1_iid", "fu_C1_V0": "C1_cov"}
    for fu, o in alias.items():
        for K in (8, 16, 32):
            m.loc[(m.method == fu) & (m.K == K), "original"] = float(ocs[(ocs.method == o) & (ocs.K == K)].delta_rmse.iloc[0])
    m["abs_diff"] = (m.delta_rmse - m.original).abs()
    m.to_csv(os.path.join(out, "v0_dev_reproduction.tsv"), sep="\t", index=False, float_format="%.8g")
    chk["dev_closed_form"] = dict(max_abs_diff=float(m.abs_diff.max()), n_families=int(agg.n_families.max()), rows=m.to_dict("records"))
    log(m.to_string())
    # matched-plate (source-train cells, 2 pairs per cell, support seed 1) with the original constants
    X0 = np.load(os.path.join(os.environ["TB_PROC"], "x0_all.npy"), mmap_mode="r")
    mrows, mplates, MRD = load_rows(proc, D.row_delta)
    rec2 = []
    for meta, e1, e2, pc in matched_episodes(st, mrows, mplates, MRD, X0, cell_role="train", max_pairs=2, seeds=(1,)):
        for block, ep in (("in_block", e1), ("other_block", e2)):
            for K in (8, 16, 32):
                if K > meta["kmax"]:
                    continue
                inp = Eng.inputs(ep, K)
                Ks = str(K)
                preds = dict(shrunk_same_drug_mean=M.anchor(inp, fx["shrunk_kappa_f"][Ks], "inf")["fq"])
                for obs, om in (("iid", None), ("cov", cov0)):
                    kbv = fx["kappa_b"][obs][Ks]
                    anc = M.anchor(inp, fx["kappa_f"][Ks], Eng.kb(kbv) if kbv != "inf" else "inf", om)
                    preds[f"C0_{obs}"] = M.c0(anc)
                    c = fx["c1"][obs][Ks]
                    preds[f"C1_{obs}"] = M.c1(anc, fm.build(inp, st.FP, anc["fs"], anc["fq"]), c["lam"], c["eta"])
                for mth, p in preds.items():
                    rec2.append(dict(meta, block=block, K=K, method=mth, delta_rmse=F.rmse(p, ep.y_q)))
    md = pd.DataFrame(rec2)
    epm = md.groupby(["cell", "family", "plate_query", "plate_other", "support_seed", "block", "K", "method"]).delta_rmse.mean().reset_index()
    agg2, _ = family_mean(epm, by=("K", "block", "method"))
    omp = pd.read_csv(orig_man("matched_plate_transfer.tsv"), sep="\t")
    omp = omp[(omp.set == "train_closed_form") & (omp.row == "method_level") & (omp.metric == "delta_rmse")]
    cmp_ = []
    for r_ in agg2.itertuples():
        o = omp[(omp.K == r_.K) & (omp.name == r_.method)]
        if len(o):
            ov = float(o[r_.block].iloc[0])
            cmp_.append(dict(K=r_.K, block=r_.block, method=r_.method, recomputed=r_.delta_rmse, original=ov, abs_diff=abs(r_.delta_rmse - ov), n_families=r_.n_families))
    cmp_ = pd.DataFrame(cmp_)
    cmp_.to_csv(os.path.join(out, "v0_matched_reproduction.tsv"), sep="\t", index=False, float_format="%.8g")
    chk["matched_plate_train"] = dict(max_abs_diff=float(cmp_.abs_diff.max()), n_families=int(agg2.n_families.max()), n_episode_rows=len(md))
    log(cmp_.to_string())
    dump(run_record(a.run_id, "p1", t0, checks=chk), os.path.join(out, "result.json"))


# ---------------------------------------------------------------- select
def select(a, cfg, out):
    t0 = time.time()
    pairs = load_pairs(a.pairs)
    bank = Bank()
    ctx = Eng.Ctx(bank, a.partition)
    const = Eng.select(ctx, pairs, cfg["grids"], log)
    const["describe"] = bank.describe(a.partition)
    const["pairs_sha256"] = sha256(a.pairs)
    dump(const, os.path.join(out, "constants.json"))
    ctx.ut.to_csv(os.path.join(out, "unit_table.tsv"), sep="\t", index=False, float_format="%.8g")
    dump(run_record(a.run_id, "select", t0, partition=a.partition, n_selection_episodes=const["n_selection_episodes"]), os.path.join(out, "result.json"))


# ---------------------------------------------------------------- evaluate
def half_tables(st):
    fams = [f for f, r in zip(st.fam_names, st.fam_role) if r == "train"]
    half = {f: zlib.crc32(("half|" + f).encode()) % 2 for f in fams}
    fg = st.FG.numpy().astype(np.float64)
    out = []
    for h in (0, 1):
        m = np.array([st.fam_role[fi] == "train" and half.get(st.fam_names[fi]) == h for fi in st.fg_fam])
        S = np.zeros((st.G, fg.shape[1]))
        np.add.at(S, st.fg_group[m], fg[m])
        n = np.bincount(st.fg_group[m], minlength=st.G).astype(np.float64)
        out.append((S, n))
    return out, half


def with_reference(inp, S, n, gs, gq):
    out = dict(inp)
    for side, g in (("s", gs), ("q", gq)):
        nn = n[g]
        out["f0_" + side] = torch.tensor(S[g] / np.maximum(nn, 1)[:, None], dtype=torch.float32)
        out["n0_" + side] = torch.tensor(nn, dtype=torch.float32)
    return out


N0_BINS = [(1, 1), (2, 3), (4, 7), (8, 15), (16, 10 ** 6)]


def evaluate(a, cfg, out):
    t0 = time.time()
    pairs = load_pairs(a.pairs)
    const = json.load(open(a.constants))
    assert const["partition"] == INNER_OF[a.partition], "constants must come from the matching inner partition"
    fx = Eng.load_fixed(orig_man("anchor_selection.json"))
    bank = Bank()
    ctx = Eng.Ctx(bank, a.partition)
    v = Eng.fit_outer(ctx, pairs, const)
    dump(dict(cov=v, scales=ctx.scales, describe=bank.describe(a.partition)), os.path.join(out, "outer_nuisance.json"))
    ctx.ut.to_csv(os.path.join(out, "unit_table.tsv"), sep="\t", index=False, float_format="%.8g")
    st = ctx.st
    halves, _ = half_tables(st)
    rows, infl, n0rows, perm = [], [], [], []
    eps = ctx.episodes("val", cfg["episodes"]["evaluation"]["support_seeds"], None)
    log(a.partition, "episodes", len(eps), f"{time.time() - t0:.0f}s")
    for i, ep in enumerate(eps):
        truth = ep.y_q.numpy()
        for K in (8, 16, 32):
            if K > len(ep.g_s):
                continue
            inp = Eng.inputs(ep, K)
            b = dict(fold=a.partition, cell=ep.name, family=ep.family, support_seed=ep.seed, K=K)
            preds = {}
            for m, p, ex in Eng.method_preds(ctx, inp, K, const, fx):
                preds[m] = p
                rows.append(dict(b, method=m, **F.score(p.numpy(), truth), **ex))
            if K != 16:
                continue
            Ks = "16"
            # n0 strata (shrunk mean, C0, C1)
            n0 = inp["n0_q"].numpy()
            for m in ("shrunk_mean", "C0_iid", "C1_iid"):
                e = np.sqrt(((preds[m].numpy() - truth) ** 2).mean(1))
                for lo, hi in N0_BINS:
                    sel = (n0 >= lo) & (n0 <= hi)
                    if sel.any():
                        n0rows.append(dict(b, method=m, n0_bin=f"{lo}-{hi}" if hi < 10 ** 6 else f">={lo}", sum_rmse=float(e[sel].sum()), n=int(sel.sum())))
            # influence and leave-one-support-out for C1-iid
            kf, kbi, c = const["kappa_f"][Ks], const["kappa_b"]["iid"][Ks], const["c1"]["iid"][Ks]
            fs, fq = F.fstar(inp, kf)
            bl = F.kernel_blocks(ctx.feats(inp, fs, fq))
            Kall = F.combine(bl, "all")
            aa = F.anchor_om(inp, kf, kbi, torch.eye(16, dtype=torch.float64))
            A = F.influence(Kall[0], Kall[1], aa["w"], c["lam"], aa["Om"], c["eta"])
            corr_direct = F.krr(Kall[0], Kall[1], aa["w"], aa["R"], c["lam"], aa["Om"]) * c["eta"]
            sv = torch.linalg.svdvals(A)
            absA = A.abs()
            tot = absA.sum(1).clamp(min=1e-30)
            o_qs = inp["o_qs"].double()
            shared = (o_qs > 0).double()
            top3 = torch.topk(bl["chem"][1].double(), 3, dim=1).indices
            infl.append(dict(b, influence_matches_direct=float((A.float() @ aa["E"] - corr_direct).abs().max()),
                             eff_rank_participation=float(sv.pow(2).sum() ** 2 / sv.pow(4).sum().clamp(min=1e-30)), top_support_share=float((absA.max(1).values / tot).mean()),
                             shared_plate_share=float(((absA * shared).sum(1) / tot).mean()), shared_plate_fraction_supports=float(shared.mean()),
                             top3_chem_neighbour_share=float((absA.gather(1, top3).sum(1) / tot).mean()), row_abs_sum_mean=float(tot.mean())))
            full = F.rmse(preds["C1_iid"], ep.y_q)
            dl = []
            for j in range(16):
                keep = np.array([x for x in range(16) if x != j])
                sub = dict(inp)
                tk = torch.as_tensor(keep)
                for k_ in ("x0_s", "f0_s", "n0_s", "y_s", "q_s"):
                    sub[k_] = inp[k_][tk]
                sub["g_s"] = np.asarray(inp["g_s"])[keep]
                sub["o_s"] = inp["o_s"][tk][:, tk]
                sub["o_qs"] = inp["o_qs"][:, tk]
                fs_, fq_ = F.fstar(sub, kf)
                a2 = F.anchor_om(sub, kf, kbi, torch.eye(15, dtype=torch.float64))
                K2 = F.combine(F.kernel_blocks(ctx.feats(sub, fs_, fq_)), "all")
                p2 = a2["fq"] + a2["b"] + c["eta"] * F.krr(K2[0], K2[1], a2["w"], a2["R"], c["lam"], a2["Om"])
                dl.append(F.rmse(p2, ep.y_q) - full)
            infl[-1].update(loso_max_abs_delta=float(np.max(np.abs(dl))), loso_mean_delta=float(np.mean(dl)), loso_max_increase=float(np.max(dl)))
            # reference halves (C0 / C1 with f0 from one pre-fixed half of the training families)
            for h, (S, n) in enumerate(halves):
                hin = with_reference(inp, S, n, np.asarray(inp["g_s"]), np.asarray(inp["g_q"]))
                fs_, fq_ = F.fstar(hin, kf)
                ah = F.anchor_om(hin, kf, kbi, torch.eye(16, dtype=torch.float64))
                Kh = F.combine(F.kernel_blocks(ctx.feats(hin, fs_, fq_)), "all")
                rows.append(dict(b, method=f"C0_iid_refhalf{h}", **F.score((ah["fq"] + ah["b"]).numpy(), truth)))
                rows.append(dict(b, method=f"C1_iid_refhalf{h}", **F.score((ah["fq"] + ah["b"] + c["eta"] * F.krr(Kh[0], Kh[1], ah["w"], ah["R"], c["lam"], ah["Om"])).numpy(), truth)))
            # full-row permutation invariance (prediction must not change)
            pp = np.random.RandomState(5).permutation(16)
            pin = F.permute_inputs(inp, pp)
            pr = {m: p for m, p, _ in Eng.method_preds(ctx, pin, 16, const, fx) if m in ("C0_iid", "C1_iid", "C1_V3", "C0_V3", "J1", "J2", "kn_all")}
            perm.append(dict(b, **{f"maxabs_{m}": float((pr[m] - preds[m]).abs().max()) for m in pr}))
        if i % 20 == 0:
            log(a.partition, i, len(eps), f"{time.time() - t0:.0f}s")
    pd.DataFrame(rows).to_csv(os.path.join(out, "episodes.tsv.gz"), sep="\t", index=False, float_format="%.7g")
    pd.DataFrame(infl).to_csv(os.path.join(out, "influence.tsv"), sep="\t", index=False, float_format="%.6g")
    pd.DataFrame(n0rows).to_csv(os.path.join(out, "n0_strata.tsv"), sep="\t", index=False, float_format="%.7g")
    pd.DataFrame(perm).to_csv(os.path.join(out, "row_permutation_check.tsv"), sep="\t", index=False, float_format="%.3g")
    log("general done", f"{time.time() - t0:.0f}s")
    # matched-plate E1 / E2 on the outer-test families (2 pairs per cell, support seed 1: original train-set rule)
    X0 = np.load(os.path.join(os.environ["TB_PROC"], "x0_all.npy"), mmap_mode="r")
    mrows, mplates, MRD = load_rows(os.environ["R8_PROC"], D.row_delta)
    rec = []
    for meta, e1, e2, pc in matched_episodes(st, mrows, mplates, MRD, X0, cell_role="val", max_pairs=2, seeds=(1,)):
        b = dict(fold=a.partition, **meta)
        truth = pc["truth"]
        for name in ("same_drug_other_plate", "other_drug_other_plate"):
            rec.append(dict(b, block="replicate", K=0, method=f"PC_{name}", ctrl_shift_rms=pc["ctrl_shift_rms"], **F.score(pc[name], truth)))
        for block, ep in (("in_block", e1), ("other_block", e2)):
            for K in (8, 16, 32):
                if K > meta["kmax"]:
                    continue
                inp = Eng.inputs(ep, K)
                for m, p, ex in Eng.method_preds(ctx, inp, K, const, fx):
                    rec.append(dict(b, block=block, K=K, method=m, ctrl_shift_rms=pc["ctrl_shift_rms"], **F.score(p.numpy(), truth), **ex))
    pd.DataFrame(rec).to_csv(os.path.join(out, "matched_episodes.tsv.gz"), sep="\t", index=False, float_format="%.7g")
    dump(run_record(a.run_id, "evaluate", t0, partition=a.partition, constants=a.constants, constants_sha256=sha256(a.constants), n_episodes=len(eps),
                    n_matched_rows=len(rec)), os.path.join(out, "result.json"))
    log("done", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["p1", "select", "evaluate"])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--partition", default="")
    ap.add_argument("--pairs", default="")
    ap.add_argument("--constants", default="")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(CFG))
    out = new_run_dir(a.stage, a.run_id)
    {"p1": p1, "select": select, "evaluate": evaluate}[a.stage](a, cfg, out)
