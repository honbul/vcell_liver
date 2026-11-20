"""R8-P0 corrections (private plan 2026-11-19 section 5.1). CPU closed-form recomputation only; no training, no new selection on
target or confirmation data.

  select   one INNER partition: v1 selection replayed with the full unrounded grids (nuisance 'v1', tie rule 'v1'; must reproduce
           the committed v1 constants exactly) and the corrected selection (nuisance 'v2' = per-K kappa_f, tie rule 'v2').
           Also the section-10 budget kappa_b / lambda / eta grids (v1 rule) for the tie check.               -> P0-B, P0-C
  anchor   one OUTER partition (source OOF) or 'global' (confirmation families, their matched plates, HCC): anchor_matched
           (f* with the kappa_f that C0/C1 actually use), shrunk_mean, C0-iid, C1-iid recomputed with the v1 constants, and a
           replay check against the stored v1 per-episode / per-unit values.                                    -> P0-A
  k8       'global' only: the methods whose nuisance depends on kappa_f (engine2.AFFECTED) at K = 8/16/32 under
           replay (v1 nuisance, v1 constants) / fixed (v2 nuisance, v1 constants) / reselected (v2 nuisance, v2 constants),
           on the confirmation families, their matched plates and HCC (already-exposed panels: correction report only).
  report   tables from the immutable run directories (no prediction recomputed).
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
import torch
import yaml

from src.replicate_calibrated_r8.data import Features
from src.replicate_calibrated_r8 import diag as D
from src.episodic_residual_r2.build_units import rows_to_units
from src.r8_followup_v1 import engine as Eng
from src.r8_followup_v1 import fmethods as F
from src.r8_followup_v1 import stats as S
from src.r8_followup_v1.partitions import Bank, INNER_OF
from src.r8_followup_v1.matched import matched_episodes, load_rows
from src.r8_followup_v1.budget import budget_episodes
from src.r8_followup_v1.final_eval import load_target, hcc_episode, unit_table as hcc_unit_table
from src.r8_followup_v1.common import orig_man
from . import engine2 as E2
from .common import CFG, TARGETS, v1_runs, v1_man, pub_man, dump, new_run_dir, run_record, sha256, log

torch.set_grad_enabled(False)
V1_PAIRS = "p1/p1_v1/d1_plate_pairs.tsv"
V1_SEL = {p: f"select/sel_{p}_v1/constants.json" for p in ("inner0", "inner1", "inner2", "final_inner")}
V1_BSEL = {p: f"branch_select/bsel_{p}_v1/branch_constants.json" for p in ("inner0", "inner1", "inner2", "final_inner")}
V1_EVAL = {p: f"evaluate/ev_{p}_v1" for p in ("outer0", "outer1", "outer2")}
V1_FINAL = "final_eval/final_v1"


def v1_grids_cfg():
    return yaml.safe_load(open("configs/r8_followup_v1/followup.yaml"))["grids"]


def same(a, b, path=""):
    """Exact structural equality of two constant trees (floats compared exactly; 'inf' strings and float inf equal)."""
    diffs = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                diffs.append(f"{path}/{k}: missing")
            else:
                diffs += same(a[k], b[k], f"{path}/{k}")
        return diffs
    fa = Eng.kb(a) if isinstance(a, (str, int, float)) and not isinstance(a, bool) else a
    fb = Eng.kb(b) if isinstance(b, (str, int, float)) and not isinstance(b, bool) else b
    if fa != fb:
        diffs.append(f"{path}: {a} != {b}")
    return diffs


SEL_KEYS = ("kappa_f", "shrunk_kappa_f", "kappa_b", "c1", "kn")


def strip_lamdev(c):
    out = {}
    for k in SEL_KEYS:
        if k in ("c1", "kn"):
            out[k] = {g: {K: {kk: vv for kk, vv in v.items() if kk != "delta_rmse"} for K, v in t.items()} for g, t in c[k].items()}
        else:
            out[k] = c[k]
    return out


def jstrip(j):
    return {m: {K: {k: v for k, v in d.items() if k != "delta_rmse"} for K, d in j[m].items()} for m in ("j1", "j2", "j3") if m in j}


# ---------------------------------------------------------------- select (inner partitions)
def stage_select(a, out):
    t0 = time.time()
    grids = v1_grids_cfg()
    pairs = pd.read_csv(v1_runs(V1_PAIRS), sep="\t")
    bank = Bank()
    P = a.partition
    res, tie_log, glong, keys = {}, [], [], []
    for mode, tie in (("v1", "v1"), ("v2", "v2")):
        ctx = E2.Ctx2(bank, P, mode=mode)
        tl = tie_log if mode == "v1" else []
        const, gt = E2.select2(ctx, pairs, grids, tie=tie, log=log, tie_log=tl)
        jc, gtj = E2.select_j2(ctx, const, grids, tie=tie, log=log, tie_log=tl)
        if mode == "v2":
            for r in tl:
                r["note"] = "v2 run (corrected nuisance, v2 tie rule)"
            tie_log_v2 = tl
        glong += E2.grids_long({**gt, **gtj}, P, mode, tie)
        keys += ctx.key_table()
        dump(const, os.path.join(out, f"constants_{mode}.json"))
        dump(jc, os.path.join(out, f"jconst_{mode}.json"))
        res[mode] = (const, jc)
    # replay check against the committed / stored v1 constants
    v1c = json.load(open(v1_runs(V1_SEL[P])))
    v1j = json.load(open(v1_runs(V1_BSEL[P])))
    chk = dict(partition=P,
               v1_constants_sha256=sha256(v1_runs(V1_SEL[P])), v1_jconst_sha256=sha256(v1_runs(V1_BSEL[P])),
               replay_const_diffs=same(strip_lamdev(res["v1"][0]), strip_lamdev(v1c)),
               replay_j_diffs=same(jstrip(res["v1"][1]), jstrip(v1j)),
               v2_vs_v1_const_diffs=same(strip_lamdev(res["v2"][0]), strip_lamdev(v1c)),
               v2_vs_v1_j_diffs=same(jstrip(res["v2"][1]), jstrip(v1j)))
    # budget (section 10) selection grids, v1 rule (no covariance involved: P0-B does not apply)
    if "budget" in v1j:
        ctx = E2.Ctx2(bank, P, mode="v1")
        const = res["v1"][0]
        rows, plates, RD = load_rows(os.environ["R8_PROC"], D.row_delta)
        X0 = np.load(os.path.join(os.environ["TB_PROC"], "x0_all.npy"), mmap_mode="r")
        KB = [Eng.kb(x) for x in grids["kappa_b"]]
        LAM, ETA = [float(x) for x in grids["lambda"]], [float(x) for x in grids["eta"]]
        arm_k = {"D16": 16, "REP8": 8}
        eps = list(budget_episodes(ctx.st, rows, plates, RD, X0, cell_role="val", seeds=(1,), design=v1j["budget"]["design"]))
        r0 = []
        for meta, ee in eps:
            for arm, (ep, cost) in ee.items():
                K = arm_k[arm]
                inp = Eng.inputs(ep, K)
                b = dict(cell=meta["cell"], family=meta["family"], support_seed=f"{meta['plate_x']}|{meta['plate_y']}|{meta['plate_query']}", arm=arm, K=K)
                fs, fq = F.fstar(inp, const["kappa_f"][str(K)])
                E = inp["y_s"] - fs
                for kbv in KB:
                    r0.append(dict(b, kappa_b=kbv, delta_rmse=F.rmse(fq + E.mean(0, keepdim=True) * F.alpha(K, kbv), ep.y_q)))
        bc, gtb = {}, {}
        for arm in arm_k:
            c, gtb[f"budget_kappa_b_{arm}"] = E2.choose2([r for r in r0 if r["arm"] == arm], ["kappa_b"], "kappa_b", "v1", f"budget_kappa_b_{arm}", P, tie_log)
            bc[arm] = dict(kappa_b=c[str(arm_k[arm])]["kappa_b"])
        r1 = []
        feats_fit = ctx.fmap
        for meta, ee in eps:
            for arm, (ep, cost) in ee.items():
                K = arm_k[arm]
                inp = Eng.inputs(ep, K)
                b = dict(cell=meta["cell"], family=meta["family"], support_seed=f"{meta['plate_x']}|{meta['plate_y']}|{meta['plate_query']}", arm=arm, K=K)
                kf_ = const["kappa_f"][str(K)]
                fs, fq = F.fstar(inp, kf_)
                aa = F.anchor_om(inp, kf_, bc[arm]["kappa_b"], torch.eye(K, dtype=torch.float64))
                Kall = F.combine(F.kernel_blocks(feats_fit.build(inp, ctx.st.FP, fs, fq)), "all")
                for lam in LAM:
                    corr = F.krr(Kall[0], Kall[1], aa["w"], aa["R"], lam, aa["Om"])
                    for eta in ETA:
                        r1.append(dict(b, lam=lam, eta=eta, delta_rmse=F.rmse(aa["fq"] + aa["b"] + eta * corr, ep.y_q)))
        for arm in arm_k:
            c, gtb[f"budget_lam_eta_{arm}"] = E2.choose2([r for r in r1 if r["arm"] == arm], ["lam", "eta"], "lam_eta", "v1", f"budget_lam_eta_{arm}", P, tie_log)
            bc[arm].update(lam=c[str(arm_k[arm])]["lam"], eta=c[str(arm_k[arm])]["eta"])
        glong += E2.grids_long(gtb, P, "v1", "v1")
        chk["budget_replay_diffs"] = same({k: {kk: vv for kk, vv in v.items() if kk in ("kappa_b", "lam", "eta")} for k, v in bc.items()},
                                          {k: {kk: vv for kk, vv in v.items() if kk in ("kappa_b", "lam", "eta")} for k, v in v1j["budget"]["arms"].items()})
    pd.DataFrame(glong).to_csv(os.path.join(out, "grids.tsv.gz"), sep="\t", index=False, float_format="%.17g")
    pd.DataFrame(tie_log).to_csv(os.path.join(out, "tie_log_v1grids.tsv"), sep="\t", index=False, float_format="%.17g")
    pd.DataFrame(tie_log_v2).to_csv(os.path.join(out, "tie_log_v2grids.tsv"), sep="\t", index=False, float_format="%.17g")
    pd.DataFrame(keys).to_csv(os.path.join(out, "nuisance_keys.tsv"), sep="\t", index=False, float_format="%.10g")
    dump(chk, os.path.join(out, "checks.json"))
    log(json.dumps(chk, indent=1))
    dump(run_record(a.run_id, "p0_select", t0, partition=P), os.path.join(out, "result.json"))


# ---------------------------------------------------------------- anchor_matched (P0-A)
def p0a_preds(ctx, inp, K, const):
    Ks = str(K)
    kf = const["kappa_f"][Ks]
    yield "anchor_matched", F.fstar(inp, kf)[1]
    yield "shrunk_mean", F.fstar(inp, const["shrunk_kappa_f"][Ks])[1]
    a = F.anchor_om(inp, kf, const["kappa_b"]["iid"][Ks], torch.eye(K, dtype=torch.float64))
    yield "C0_iid", a["fq"] + a["b"]
    c = const["c1"]["iid"][Ks]
    Kall = F.combine(F.kernel_blocks(ctx.feats(inp, a["fs"], a["fq"])), "all")
    yield "C1_iid", a["fq"] + a["b"] + c["eta"] * F.krr(Kall[0], Kall[1], a["w"], a["R"], c["lam"], a["Om"])


def centred_max_diff(p, q):
    pc, qc = p - p.mean(0, keepdim=True), q - q.mean(0, keepdim=True)
    return float((pc - qc).abs().max())


def score_rows(ctx, ep, K, const, truth, b, rows):
    preds = {}
    for m, p in p0a_preds(ctx, Eng.inputs(ep, K), K, const):
        preds[m] = p
        rows.append(dict(b, K=K, method=m, anchor_kappa_f=const["kappa_f"][str(K)], anchor_contract_id=f"kappa_f={const['kappa_f'][str(K)]:g}|{const['partition']}",
                         selection_partition=const["partition"], **F.score(p.numpy(), truth)))
    rows[-1]["centred_pred_maxdiff_C0_vs_anchor_matched"] = centred_max_diff(preds["C0_iid"], preds["anchor_matched"])
    return preds


def stage_anchor(a, out):
    t0 = time.time()
    P = a.partition
    bank = Bank()
    if P == "global":
        const = json.load(open(v1_runs(V1_SEL["final_inner"])))
        ctx = Eng.Ctx(bank, "global", fmap=Features.load(os.path.join(os.environ["R8_PROC"], "feature_basis.npz"), "cpu"))
        gseeds, mp, mseeds = [1, 2, 3], 6, (1, 2, 3)
        ks = (8, 16, 32)
    else:
        const = json.load(open(v1_runs(V1_SEL[INNER_OF[P]])))
        ctx = Eng.Ctx(bank, P)
        gseeds, mp, mseeds = [1, 2, 3], 2, (1,)
        ks = (8, 16, 32)
    assert const["partition"] == ("final_inner" if P == "global" else INNER_OF[P])
    rows, mrows = [], []
    for ep in ctx.episodes("val", gseeds, None):
        for K in ks:
            if K > len(ep.g_s):
                continue
            score_rows(ctx, ep, K, const, ep.y_q.numpy(), dict(fold=P, set="general", cell=ep.name, family=ep.family, support_seed=ep.seed), rows)
    log("general", len(rows), f"{time.time() - t0:.0f}s")
    X0 = np.load(os.path.join(os.environ["TB_PROC"], "x0_all.npy"), mmap_mode="r")
    mr, mpl, MRD = load_rows(os.environ["R8_PROC"], D.row_delta)
    for meta, e1, e2, pc in matched_episodes(ctx.st, mr, mpl, MRD, X0, cell_role="val", max_pairs=mp, seeds=mseeds):
        for blk, ep in (("in_block", e1), ("other_block", e2)):
            for K in ks:
                if K > meta["kmax"]:
                    continue
                b = dict(fold=P, set="E1" if blk == "in_block" else "E2", **{k: v for k, v in meta.items() if k != "n_query"})
                score_rows(ctx, ep, K, const, pc["truth"], b, mrows)
    log("matched", len(mrows), f"{time.time() - t0:.0f}s")
    pd.DataFrame(rows + mrows).to_csv(os.path.join(out, "episodes.tsv.gz"), sep="\t", index=False, float_format="%.9g")
    units = []
    if P == "global" and a.hcc:
        ofz = json.load(open(orig_man("freeze_manifest.json")))
        for t in TARGETS:
            trows, x0, x1 = load_target(t)
            tab, ud, ux = rows_to_units(trows, x1 - x0, x0)
            for split, lists in ofz["targets"][t].items():
                ep, qn, is_test = hcc_episode(ctx.st, t, split, lists, trows, tab, ud, ux)
                for K in ks:
                    if K > len(ep.g_s):
                        continue
                    for m, p in p0a_preds(ctx, Eng.inputs(ep, K), K, const):
                        units += hcc_unit_table(p.numpy(), ep.y_q.numpy(), qn, is_test, dict(method=m, K=K, split_seed=int(split), target=t,
                                                                                               anchor_kappa_f=const["kappa_f"][str(K)]))
            log(t, f"{time.time() - t0:.0f}s")
        pd.concat(units, ignore_index=True).to_csv(os.path.join(out, "hcc_per_unit.tsv.gz"), sep="\t", index=False, float_format="%.9g")
    # replay check against the stored v1 values (C0_iid, C1_iid, shrunk_mean)
    chk = {}
    ep_df = pd.DataFrame(rows + mrows)
    if P == "global":
        v1g = pd.read_csv(v1_runs(V1_FINAL, "confirmation_episodes.tsv.gz"), sep="\t")
        v1m = pd.read_csv(v1_runs(V1_FINAL, "matched_episodes.tsv.gz"), sep="\t")
    else:
        v1g = pd.read_csv(v1_runs(V1_EVAL[P], "episodes.tsv.gz"), sep="\t")
        v1m = pd.read_csv(v1_runs(V1_EVAL[P], "matched_episodes.tsv.gz"), sep="\t")
    for name, mine, v1, keys in (("general", ep_df[ep_df.set == "general"], v1g, ["cell", "support_seed", "K", "method"]),
                                  ("matched", ep_df[ep_df.set != "general"].assign(block=lambda x: np.where(x.set == "E1", "in_block", "other_block")), v1m,
                                   ["cell", "plate_query", "plate_other", "support_seed", "block", "K", "method"])):
        m = mine.merge(v1[keys + ["delta_rmse", "centred_rmse"]], on=keys, suffixes=("", "_v1"))
        m = m[m.method.isin(["C0_iid", "C1_iid", "shrunk_mean"])]
        chk[name] = dict(n_matched_rows=int(len(m)), n_expected=int(mine.method.isin(["C0_iid", "C1_iid", "shrunk_mean"]).sum()),
                         max_abs_diff_delta_rmse=float((m.delta_rmse - m.delta_rmse_v1).abs().max()), max_abs_diff_centred_rmse=float((m.centred_rmse - m.centred_rmse_v1).abs().max()))
    if units:
        u = pd.concat(units, ignore_index=True)
        v1u = pd.read_csv(v1_runs(V1_FINAL, "hcc_per_unit.tsv.gz"), sep="\t")
        keys = ["target", "split_seed", "K", "method", "query_set", "drug_group"]
        m = u.merge(v1u[keys + ["delta_rmse", "centred_rmse"]], on=keys, suffixes=("", "_v1"))
        chk["hcc"] = dict(n_matched_rows=int(len(m)), max_abs_diff_delta_rmse=float((m.delta_rmse - m.delta_rmse_v1).abs().max()),
                          max_abs_diff_centred_rmse=float((m.centred_rmse - m.centred_rmse_v1).abs().max()))
    chk["max_centred_pred_diff_C0_vs_anchor_matched"] = float(ep_df["centred_pred_maxdiff_C0_vs_anchor_matched"].max())
    dump(chk, os.path.join(out, "replay_checks.json"))
    log(json.dumps(chk, indent=1))
    dump(run_record(a.run_id, "p0_anchor", t0, partition=P, constants=v1_runs(V1_SEL["final_inner" if P == "global" else INNER_OF[P]])), os.path.join(out, "result.json"))


# ---------------------------------------------------------------- K8 consistency impact on the exposed panels (P0-B)
def stage_k8(a, out):
    t0 = time.time()
    pairs = pd.read_csv(v1_runs(V1_PAIRS), sep="\t")
    v1c = json.load(open(v1_runs(V1_SEL["final_inner"])))
    v1j = json.load(open(v1_runs(V1_BSEL["final_inner"])))
    v2c = json.load(open(os.path.join(a.p0_select, "constants_v2.json")))
    v2j = json.load(open(os.path.join(a.p0_select, "jconst_v2.json")))
    assert v2c["partition"] == "final_inner" and v2c["nuisance_mode"] == "v2"
    bank = Bank()
    fmap = Features.load(os.path.join(os.environ["R8_PROC"], "feature_basis.npz"), "cpu")
    cv1 = E2.Ctx2(bank, "global", fmap=fmap, mode="v1")
    cv1.fit_nuisance(pairs, v1c["kappa_f"], extra_kf=(8.0,))
    cv2 = E2.Ctx2(bank, "global", fmap=fmap, mode="v2")
    cv2.fit_nuisance(pairs, v1c["kappa_f"], extra_kf=(8.0,))
    assert v2c["kappa_f"] == v1c["kappa_f"], "reselected kappa_f differs: the fixed-constant variant would need its own nuisance"
    variants = (("replay", cv1, v1c, v1j), ("fixed", cv2, v1c, v1j), ("reselected", cv2, v2c, v2j))
    keys = cv1.key_table() + cv2.key_table()
    rows, units = [], []

    def run(ep, K, truth, b, sink, hcc=None):
        inp = Eng.inputs(ep, K)
        for var, ctx, c, j in variants:
            for m, p in E2.affected_preds(ctx, inp, K, c, j):
                if hcc is None:
                    sink.append(dict(b, K=K, variant=var, method=m, **F.score(p.numpy(), truth)))
                else:
                    qn, is_test, ex = hcc
                    sink += hcc_unit_table(p.numpy(), truth, qn, is_test, dict(ex, method=m, K=K, variant=var))
    for ep in cv1.episodes("val", [1, 2, 3], None):
        for K in (8, 16, 32):
            run(ep, K, ep.y_q.numpy(), dict(set="confirmation", cell=ep.name, family=ep.family, support_seed=ep.seed), rows)
    log("confirmation", f"{time.time() - t0:.0f}s")
    X0 = np.load(os.path.join(os.environ["TB_PROC"], "x0_all.npy"), mmap_mode="r")
    mr, mpl, MRD = load_rows(os.environ["R8_PROC"], D.row_delta)
    for meta, e1, e2, pc in matched_episodes(cv1.st, mr, mpl, MRD, X0, cell_role="val", max_pairs=6, seeds=(1, 2, 3)):
        for blk, ep in (("in_block", e1), ("other_block", e2)):
            for K in (8, 16, 32):
                if K <= meta["kmax"]:
                    run(ep, K, pc["truth"], dict(set="E1" if blk == "in_block" else "E2", block=blk, **{k: v for k, v in meta.items() if k != "n_query"}), rows)
    log("matched", f"{time.time() - t0:.0f}s")
    ofz = json.load(open(orig_man("freeze_manifest.json")))
    for t in TARGETS:
        trows, x0, x1 = load_target(t)
        tab, ud, ux = rows_to_units(trows, x1 - x0, x0)
        for split, lists in ofz["targets"][t].items():
            ep, qn, is_test = hcc_episode(cv1.st, t, split, lists, trows, tab, ud, ux)
            for K in (8, 16, 32):
                if K <= len(ep.g_s):
                    run(ep, K, ep.y_q.numpy(), None, units, hcc=(qn, is_test, dict(split_seed=int(split), target=t)))
        log(t, f"{time.time() - t0:.0f}s")
    df = pd.DataFrame(rows)
    u = pd.concat(units, ignore_index=True)
    df.to_csv(os.path.join(out, "panel_episodes.tsv.gz"), sep="\t", index=False, float_format="%.9g")
    u.to_csv(os.path.join(out, "hcc_per_unit.tsv.gz"), sep="\t", index=False, float_format="%.9g")
    pd.DataFrame(keys).to_csv(os.path.join(out, "nuisance_keys_global.tsv"), sep="\t", index=False, float_format="%.10g")
    # replay check vs stored v1 values
    v1g = pd.read_csv(v1_runs(V1_FINAL, "confirmation_episodes.tsv.gz"), sep="\t")
    v1m = pd.read_csv(v1_runs(V1_FINAL, "matched_episodes.tsv.gz"), sep="\t")
    v1u = pd.read_csv(v1_runs(V1_FINAL, "hcc_per_unit.tsv.gz"), sep="\t")
    rp = df[df.variant == "replay"]
    chk = {}
    m = rp[rp.set == "confirmation"].merge(v1g[["cell", "support_seed", "K", "method", "delta_rmse"]], on=["cell", "support_seed", "K", "method"], suffixes=("", "_v1"))
    chk["confirmation"] = dict(n=int(len(m)), max_abs_diff=float((m.delta_rmse - m.delta_rmse_v1).abs().max()))
    kk = ["cell", "plate_query", "plate_other", "support_seed", "block", "K", "method"]
    m = rp[rp.set != "confirmation"].merge(v1m[kk + ["delta_rmse"]], on=kk, suffixes=("", "_v1"))
    chk["matched"] = dict(n=int(len(m)), max_abs_diff=float((m.delta_rmse - m.delta_rmse_v1).abs().max()))
    kk = ["target", "split_seed", "K", "method", "query_set", "drug_group"]
    m = u[u.variant == "replay"].merge(v1u[kk + ["delta_rmse"]], on=kk, suffixes=("", "_v1"))
    chk["hcc"] = dict(n=int(len(m)), max_abs_diff=float((m.delta_rmse - m.delta_rmse_v1).abs().max()))
    # K16 / K32 must be unchanged between replay and the corrected variants (identical nuisance keys and constants)
    dk = df.assign(plate_query=df.plate_query.fillna(""), plate_other=df.plate_other.fillna(""))
    piv = dk.pivot_table(index=["set", "cell", "family", "support_seed", "plate_query", "plate_other", "K", "method"], columns="variant", values="delta_rmse")
    for K in (8, 16, 32):
        p = piv.xs(K, level="K")
        chk[f"K{K}_max_abs_fixed_minus_replay"] = float((p["fixed"] - p["replay"]).abs().max())
        chk[f"K{K}_max_abs_reselected_minus_replay"] = float((p["reselected"] - p["replay"]).abs().max())
    dump(chk, os.path.join(out, "checks.json"))
    log(json.dumps(chk, indent=1))
    dump(run_record(a.run_id, "p0_k8", t0, p0_select=a.p0_select, v2_constants_sha256=sha256(os.path.join(a.p0_select, "constants_v2.json")),
                    v2_jconst_sha256=sha256(os.path.join(a.p0_select, "jconst_v2.json"))), os.path.join(out, "result.json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["select", "anchor", "k8"])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--partition", default="")
    ap.add_argument("--hcc", action="store_true")
    ap.add_argument("--p0-select", default="")
    a = ap.parse_args()
    out = new_run_dir("p0_" + a.stage, a.run_id)
    {"select": stage_select, "anchor": stage_anchor, "k8": stage_k8}[a.stage](a, out)
