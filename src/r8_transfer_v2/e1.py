"""R8-E1 (environment transfer, private plan 2026-11-19 section 5.2) and R8-E2-A (information cut on the same episodes, 5.3).

Input contracts (configs/r8_transfer_v2/run_spec.yaml, reports/r8_transfer_v2/INPUT_CONTRACTS.md):
  planned            query-support plate overlap o_qS known (real value); query DMSO control x0_q observed
  known_separate     the query is known to be on a plate no support shares (o_qS = 0 is TRUE information); x0_q observed
  missing            o_qS not given; x0_q observed. Two versions:
                       *_zero_overlap_fallback : planned-contract constants with o_qS := 0 (old v1 '*_unknown_env'; it assumes
                                                 no shared environment, it is NOT 'no query environment information')
                       missing_reselected      : constants reselected on source inner with o_qS := 0 in selection too
Selection contexts (source inner-validation families only; constants per context and K):
  gen_planned   general inner episodes, real o_qS          gen_missing   general inner episodes, o_qS := 0
  M1            inner matched-plate, support on the query plate (o_qS = 1)
  M2            inner matched-plate, support on the other X-replicate plate (o_qS = 0)
Arms: anchor_matched (f* with the kappa_f of C0/C1), anchor_shrunk (independently selected kappa_f; reference only), C0, C1,
      J1, C1J1 (f* + b_q(J1) + r(C1); C1 residual constants fixed, a,c selected on the composite score), C1J1_full
      (sensitivity: a, c, lambda, eta jointly; 660 candidates), J2, J3; E2-A: kn_ctrl, kn_chem+ctrl, kn_f0+ctrl, kn_all
      (block-normalised kernels, own lambda / eta with eta = 0 (= C0) in the grid) and the controls kn_all_resshuf1-3 (residual
      rows deranged; f*, b, kernels fixed) and kn_all_f0perm / kn_f0+ctrl_f0perm (support f0 representation permuted inside
      pre-fixed n0 strata; anchor untouched).
Stages: select (inner partition), evaluate (outer partition, outer-test families), report (source decisions), hcc (post-freeze).
"""
import argparse, json, os, subprocess, time, zlib
import numpy as np
import pandas as pd
import torch
import yaml

from src.replicate_calibrated_r8.data import Features
from src.replicate_calibrated_r8 import diag as D
from src.episodic_residual_r2.build_units import rows_to_units
from src.r8_followup_v1 import engine as Eng
from src.r8_followup_v1 import fmethods as F
from src.r8_followup_v1.partitions import Bank, INNER_OF
from src.r8_followup_v1.matched import matched_episodes, load_rows
from src.r8_followup_v1.final_eval import load_target, hcc_episode, unit_table as hcc_unit_table
from src.r8_followup_v1.common import orig_man
from . import engine2 as E2
from .common import CFG, MAN, TARGETS, v1_runs, dump, new_run_dir, run_record, sha256, log

torch.set_grad_enabled(False)
E2A_CFGS = ("ctrl", "chem+ctrl", "f0+ctrl", "all")
N0_BINS = [(1, 1), (2, 3), (4, 7), (8, 15), (16, 10 ** 6)]
CONTEXTS = ("gen_planned", "gen_missing", "M1", "M2")
V1_PAIRS = "p1/p1_v1/d1_plate_pairs.tsv"


def spec():
    return yaml.safe_load(open(CFG))


def oq_of(inp, mode):
    if mode == "zero":
        return torch.zeros_like(inp["o_qs"])
    if mode == "one":
        return torch.ones_like(inp["o_qs"])
    return inp["o_qs"]


def ks_of(ep, kmax=None):
    m = len(ep.g_s) if kmax is None else kmax
    return [K for K in E2.KS if K <= m]


# ---------------------------------------------------------------- selection in one context
def context_select(ctx, base, eps, oq_mode, g, tie, cname, tie_log):
    """eps: list of (episode, base-row dict, K list). Returns constants per arm and K, and the grid tables."""
    KB = [Eng.kb(x) for x in g["kappa_b"]]
    LAM, ETA = [float(x) for x in g["lambda"]], [float(x) for x in g["eta"]]
    ETA0 = [0.0] + ETA
    J1G = F.J1_GRID
    P = f"{ctx.name}|{cname}"
    rows = {k: [] for k in ("c0", "c1", "j1", "c1j1_full", "j2", "j3") + tuple("kn_" + c for c in E2A_CFGS)}
    for ep, b, ks in eps:
        for K in ks:
            inp = Eng.inputs(ep, K)
            fs, fq = F.fstar(inp, base["kappa_f"][str(K)])
            E = inp["y_s"] - fs
            for kbv in KB:
                rows["c0"].append(dict(b, K=K, kappa_b=kbv, delta_rmse=F.rmse(fq + E.mean(0, keepdim=True) * F.alpha(K, kbv), ep.y_q)))
    out, gt = {}, {}
    out["c0"], gt["c0"] = E2.choose2(rows["c0"], ["kappa_b"], "kappa_b", tie, "c0_kappa_b", P, tie_log)
    for ep, b, ks in eps:
        for K in ks:
            Ks = str(K)
            inp = Eng.inputs(ep, K)
            oq = oq_of(inp, oq_mode)
            kf = base["kappa_f"][Ks]
            fs, fq = F.fstar(inp, kf)
            E = inp["y_s"] - fs
            a = F.anchor_om(inp, kf, out["c0"][Ks]["kappa_b"], torch.eye(K, dtype=torch.float64))
            bl = F.kernel_blocks(ctx.feats(inp, fs, fq))
            Kall = F.combine(bl, "all")
            corr = {lam: F.krr(Kall[0], Kall[1], a["w"], a["R"], lam, a["Om"]) for lam in LAM}
            y = ep.y_q
            for lam in LAM:
                for eta in ETA:
                    rows["c1"].append(dict(b, K=K, lam=lam, eta=eta, delta_rmse=F.rmse(a["fq"] + a["b"] + eta * corr[lam], y)))
            me = E.mean(0, keepdim=True)
            oE = (oq.float() @ E) / K
            for a_, c_ in J1G:
                bq = a_ * me + c_ * oE
                rows["j1"].append(dict(b, K=K, a=a_, c=c_, delta_rmse=F.rmse(fq + bq, y)))
                for lam in LAM:
                    for eta in ETA:
                        rows["c1j1_full"].append(dict(b, K=K, a=a_, c=c_, lam=lam, eta=eta, delta_rmse=F.rmse(fq + bq + eta * corr[lam], y)))
            c3 = ctx.cov_k(K)["V3"]
            u, ve, vs = max(c3["v_tot"], 1e-8), c3["v_e"], c3["v_s"]
            Kn = E2.combine2(bl, "all", ctx.scales_k[Ks])
            for tb in g["j2_tau_b_over_u"]:
                for s0 in g["j2_sigma0_over_u"]:
                    for tk in g["j3_tau_k_over_u"]:
                        p = F.joint(fq, E, inp["o_s"], oq, inp["q_s"], tb * u, vs, ve, s0 * u, tau_k=tk * u, Kss=Kn[0], Kqs=Kn[1], jitter=1e-6 * u)
                        r = dict(b, K=K, tau_b_over_u=float(tb), sigma0_over_u=float(s0), tau_k_over_u=float(tk), delta_rmse=F.rmse(p, y))
                        rows["j3"].append(r)
                        if float(tk) == 0.0:
                            rows["j2"].append(r)
            for cfg in E2A_CFGS:
                Kc = E2.combine2(bl, cfg, ctx.scales_k[Ks])
                for lam in LAM:
                    cc = F.krr(Kc[0], Kc[1], a["w"], a["R"], lam, a["Om"])
                    for eta in ETA0:
                        rows["kn_" + cfg].append(dict(b, K=K, lam=lam, eta=eta, delta_rmse=F.rmse(a["fq"] + a["b"] + eta * cc, y)))
    out["c1"], gt["c1"] = E2.choose2(rows["c1"], ["lam", "eta"], "lam_eta", tie, "c1", P, tie_log)
    out["j1"], gt["j1"] = E2.choose2(rows["j1"], ["a", "c"], "j1", tie, "j1", P, tie_log)
    out["c1j1_full"], gt["c1j1_full"] = E2.choose2(rows["c1j1_full"], ["a", "c", "lam", "eta"], "c1j1_full", tie, "c1j1_full", P, tie_log)
    sub = [r for r in rows["c1j1_full"] if r["lam"] == out["c1"][str(r["K"])]["lam"] and r["eta"] == out["c1"][str(r["K"])]["eta"]]
    out["c1j1"], gt["c1j1"] = E2.choose2(sub, ["a", "c"], "c1j1", tie, "c1j1", P, tie_log)
    out["j2"], gt["j2"] = E2.choose2(rows["j2"], ["tau_b_over_u", "sigma0_over_u"], "j2", tie, "j2", P, tie_log)
    out["j3"], gt["j3"] = E2.choose2(rows["j3"], ["tau_b_over_u", "sigma0_over_u", "tau_k_over_u"], "j3", tie, "j3", P, tie_log)
    for cfg in E2A_CFGS:
        out["kn_" + cfg], gt["kn_" + cfg] = E2.choose2(rows["kn_" + cfg], ["lam", "eta"], "lam_eta", tie, "kn_" + cfg, P, tie_log)
    out["n_selection_episodes"] = len(eps)
    out["n_selection_families"] = len({b["family"] for _, b, _ in eps})
    out["o_qs_mode"] = oq_mode
    return out, gt


def matched_list(ctx, max_pairs, seeds, cell_role="val"):
    X0 = np.load(os.path.join(os.environ["TB_PROC"], "x0_all.npy"), mmap_mode="r")
    mr, mpl, MRD = load_rows(os.environ["R8_PROC"], D.row_delta)
    pert = dict(zip(mpl.det_plate, mpl.pert_plate))
    out = []
    for meta, e1, e2, pc in matched_episodes(ctx.st, mr, mpl, MRD, X0, cell_role=cell_role, max_pairs=max_pairs, seeds=seeds):
        meta = dict(meta, pert_plate_query=pert[meta["plate_query"]], pert_plate_other=pert[meta["plate_other"]])
        out.append((meta, e1, e2, pc))
    return out


def stage_select(a, out):
    t0 = time.time()
    sp = spec()
    g, tie = sp["grids"], sp["tie_rule"]
    pairs = pd.read_csv(v1_runs(V1_PAIRS), sep="\t")
    base = json.load(open(os.path.join(a.p0_select, "constants_v2.json")))
    assert base["partition"] == a.partition and base["nuisance_mode"] == "v2" and base["tie_rule"] == "v2"
    bank = Bank()
    ctx = E2.Ctx2(bank, a.partition, mode="v2")
    ctx.fit_nuisance(pairs, base["kappa_f"])
    gen = [(ep, dict(cell=ep.name, family=ep.family, support_seed=ep.seed), ks_of(ep)) for ep in ctx.episodes("val", sp["selection"]["general"]["support_seeds"],
                                                                                                    sp["selection"]["general"]["query_cap"])]
    ms = sp["selection"]["matched"]
    mat = matched_list(ctx, ms["max_pairs"], tuple(ms["support_seeds"]))
    mb = lambda m: dict(cell=m["cell"], family=m["family"], support_seed=f"{m['plate_query']}|{m['plate_other']}|{m['support_seed']}")
    m1 = [(e1, mb(m), ks_of(e1, m["kmax"])) for m, e1, e2, pc in mat]
    m2 = [(e2, mb(m), ks_of(e2, m["kmax"])) for m, e1, e2, pc in mat]
    tie_log, glong, cc = [], [], {}
    for cname, eps, mode in (("gen_planned", gen, "real"), ("gen_missing", gen, "zero"), ("M1", m1, "real"), ("M2", m2, "real")):
        c, gt = context_select(ctx, base, eps, mode, g, tie, cname, tie_log)
        cc[cname] = c
        glong += E2.grids_long(gt, f"{a.partition}|{cname}", "v2", tie)
        log(cname, {k: v for k, v in c.items() if isinstance(v, dict) and "16" in v and k in ("c0", "c1", "j1", "c1j1", "c1j1_full", "j2", "j3", "kn_all")}, f"{time.time() - t0:.0f}s")
    cc["partition"] = a.partition
    cc["base_constants"] = os.path.join(a.p0_select, "constants_v2.json")
    cc["base_constants_sha256"] = sha256(cc["base_constants"])
    cc["nuisance_keys"] = ctx.key_table()
    cc["selection_episode_manifest"] = dict(general=len(gen), matched_pairs_orders=len(mat), matched_families=len({m["family"] for m, *_ in mat}),
                                            pair_kinds=pd.Series([m["pair_kind"] for m, *_ in mat]).value_counts().to_dict() if mat else {})
    dump(cc, os.path.join(out, "context_constants.json"))
    pd.DataFrame(glong).to_csv(os.path.join(out, "grids.tsv.gz"), sep="\t", index=False, float_format="%.17g")
    pd.DataFrame(tie_log).to_csv(os.path.join(out, "tie_log.tsv"), sep="\t", index=False, float_format="%.17g")
    dump(run_record(a.run_id, "e1_select", t0, partition=a.partition), os.path.join(out, "result.json"))


# ---------------------------------------------------------------- arms
def n0_strata_perm(n0, seed):
    """Support permutation inside pre-fixed n0 strata (cyclic derangement of a fixed random order; strata of size 1 stay)."""
    n0 = np.asarray(n0)
    p = np.arange(len(n0))
    rs = np.random.RandomState(seed)
    for lo, hi in N0_BINS:
        idx = np.where((n0 >= lo) & (n0 <= hi))[0]
        if len(idx) > 1:
            o = idx[rs.permutation(len(idx))]
            p[o] = np.roll(o, 1)
    return p


def arms(ctx, inp, K, base, cc, oq, e2a=False, controls=False):
    """Yields (method, prediction). cc: constants of one selection context."""
    Ks = str(K)
    kf = base["kappa_f"][Ks]
    fs, fq = F.fstar(inp, kf)
    E = inp["y_s"] - fs
    yield "anchor_matched", fq
    yield "anchor_shrunk", F.fstar(inp, base["shrunk_kappa_f"][Ks])[1]
    a = F.anchor_om(inp, kf, cc["c0"][Ks]["kappa_b"], torch.eye(K, dtype=torch.float64))
    yield "C0", a["fq"] + a["b"]
    bl = F.kernel_blocks(ctx.feats(inp, fs, fq))
    Kall = F.combine(bl, "all")
    c = cc["c1"][Ks]
    r = c["eta"] * F.krr(Kall[0], Kall[1], a["w"], a["R"], c["lam"], a["Om"])
    yield "C1", a["fq"] + a["b"] + r
    me = E.mean(0, keepdim=True)
    oE = (oq.float() @ E) / K
    j = cc["j1"][Ks]
    yield "J1", fq + j["a"] * me + j["c"] * oE
    j = cc["c1j1"][Ks]
    yield "C1J1", fq + j["a"] * me + j["c"] * oE + r
    j = cc["c1j1_full"][Ks]
    yield "C1J1_full", fq + j["a"] * me + j["c"] * oE + j["eta"] * F.krr(Kall[0], Kall[1], a["w"], a["R"], j["lam"], a["Om"])
    c3 = ctx.cov_k(K)["V3"]
    u, ve, vs = max(c3["v_tot"], 1e-8), c3["v_e"], c3["v_s"]
    Kn = E2.combine2(bl, "all", ctx.scales_k[Ks])
    for name in ("j2", "j3"):
        j = cc[name][Ks]
        yield name.upper(), F.joint(fq, E, inp["o_s"], oq, inp["q_s"], j["tau_b_over_u"] * u, vs, ve, j["sigma0_over_u"] * u,
                                    tau_k=j.get("tau_k_over_u", 0.0) * u, Kss=Kn[0], Kqs=Kn[1], jitter=1e-6 * u)
    if not e2a:
        return
    sc = ctx.scales_k[Ks]
    kern = {}
    for cfg in E2A_CFGS:
        kern[cfg] = E2.combine2(bl, cfg, sc)
        cn = cc["kn_" + cfg][Ks]
        yield f"kn_{cfg}", a["fq"] + a["b"] + cn["eta"] * F.krr(kern[cfg][0], kern[cfg][1], a["w"], a["R"], cn["lam"], a["Om"])
    if not controls:
        return
    cn = cc["kn_all"][Ks]
    for jj in (1, 2, 3):
        p = torch.as_tensor(F.derangement(K, jj))
        yield f"kn_all_resshuf{jj}", a["fq"] + a["b"] + cn["eta"] * F.krr(kern["all"][0], kern["all"][1], a["w"], a["R"][p], cn["lam"], a["Om"])
    p = torch.as_tensor(n0_strata_perm(inp["n0_s"].numpy(), 41000 + K))
    blp = dict(bl)
    blp["f0"] = (bl["f0"][0][p][:, p], bl["f0"][1][:, p])
    for cfg in ("all", "f0+ctrl"):
        Kp = E2.combine2(blp, cfg, sc)
        cn = cc["kn_" + cfg][Ks]
        yield f"kn_{cfg}_f0perm", a["fq"] + a["b"] + cn["eta"] * F.krr(Kp[0], Kp[1], a["w"], a["R"], cn["lam"], a["Om"])


# view -> list of (contract label, context constants, o_qS mode, E2-A arms)
VIEWS = {
    "G": [("planned", "gen_planned", "real", True), ("missing_reselected", "gen_missing", "zero", False), ("missing_zero_overlap_fallback", "gen_planned", "zero", False)],
    "M1": [("planned", "M1", "real", True), ("fixed_gen_constants", "gen_planned", "real", False), ("missing_reselected", "gen_missing", "zero", False),
           ("missing_zero_overlap_fallback", "gen_planned", "zero", False)],
    "M2": [("known_separate", "M2", "real", True), ("fixed_gen_constants", "gen_planned", "real", False), ("missing_reselected", "gen_missing", "zero", False)],
}


def h(x):
    return zlib.crc32(",".join(map(str, np.asarray(x).tolist())).encode())


def eval_episode_rows(ctx, ep, K, base, CC, view, b, truth, rows, perm_rows=None):
    inp = Eng.inputs(ep, K)
    for contract, cname, mode, e2a in VIEWS[view]:
        oq = oq_of(inp, mode)
        for m, p in arms(ctx, inp, K, base, CC[cname], oq, e2a=e2a, controls=e2a):
            rows.append(dict(b, view=view, contract=contract, constants=cname, K=K, method=m, **F.score(p.numpy(), truth)))
        if perm_rows is not None and e2a and K == 16:
            pp = np.random.RandomState(5).permutation(K)
            pin = F.permute_inputs(inp, pp)
            ref = {m: p for m, p in arms(ctx, inp, K, base, CC[cname], oq, e2a=True)}
            new = {m: p for m, p in arms(ctx, pin, K, base, CC[cname], oq_of(pin, mode), e2a=True)}
            perm_rows.append(dict(b, view=view, contract=contract, K=K, **{f"maxabs_{m}": float((new[m] - ref[m]).abs().max()) for m in ref
                                                                          if m != "anchor_shrunk"}))


def stage_evaluate(a, out):
    t0 = time.time()
    sp = spec()
    pairs = pd.read_csv(v1_runs(V1_PAIRS), sep="\t")
    CC = json.load(open(os.path.join(a.e1_select, "context_constants.json")))
    assert CC["partition"] == INNER_OF[a.partition]
    base = json.load(open(CC["base_constants"]))
    assert sha256(CC["base_constants"]) == CC["base_constants_sha256"]
    bank = Bank()
    ctx = E2.Ctx2(bank, a.partition, mode="v2")
    ctx.fit_nuisance(pairs, base["kappa_f"])
    dump(dict(keys=ctx.key_table(), scales=ctx.scales_k, cov={str(k): v for k, v in ctx.cov_kf.items()}), os.path.join(out, "outer_nuisance.json"))
    rows, perm, man = [], [], []
    ev = sp["evaluation"]
    eps = ctx.episodes("val", ev["general"]["support_seeds"], ev["general"]["query_cap"])
    for i, ep in enumerate(eps):
        b = dict(fold=a.partition, set="general", cell=ep.name, family=ep.family, support_seed=str(ep.seed))
        man.append(dict(b, n_support=len(ep.g_s), n_query=len(ep.g_q), support_groups_crc=h(ep.g_s), query_groups_crc=h(ep.g_q)))
        for K in ks_of(ep):
            eval_episode_rows(ctx, ep, K, base, CC, "G", b, ep.y_q.numpy(), rows, perm if i < 6 else None)
        if i % 25 == 0:
            log("general", i, len(eps), f"{time.time() - t0:.0f}s")
    mat = matched_list(ctx, ev["matched"]["max_pairs"], tuple(ev["matched"]["support_seeds"]))
    for i, (meta, e1, e2, pc) in enumerate(mat):
        b = dict(fold=a.partition, set="matched", cell=meta["cell"], family=meta["family"], support_seed=f"{meta['plate_query']}|{meta['plate_other']}|{meta['support_seed']}",
                 plate_query=meta["plate_query"], plate_other=meta["plate_other"], pert_plate_query=meta["pert_plate_query"], pert_plate_other=meta["pert_plate_other"],
                 pair_kind=meta["pair_kind"], ctrl_shift_rms=pc["ctrl_shift_rms"])
        man.append(dict(b, n_support=meta["kmax"], n_query=meta["n_query"], support_groups_crc=h(e1.g_s), query_groups_crc=h(e1.g_q),
                        same_support_groups_E1_E2=bool((np.asarray(e1.g_s) == np.asarray(e2.g_s)).all()), same_truth_E1_E2=bool(torch.equal(e1.y_q, e2.y_q))))
        for K in ks_of(e1, meta["kmax"]):
            eval_episode_rows(ctx, e1, K, base, CC, "M1", b, pc["truth"], rows, perm if i < 4 else None)
            eval_episode_rows(ctx, e2, K, base, CC, "M2", b, pc["truth"], rows, perm if i < 4 else None)
        if i % 25 == 0:
            log("matched", i, len(mat), f"{time.time() - t0:.0f}s")
    pd.DataFrame(rows).to_csv(os.path.join(out, "episodes.tsv.gz"), sep="\t", index=False, float_format="%.8g")
    pd.DataFrame(perm).to_csv(os.path.join(out, "row_permutation_check.tsv"), sep="\t", index=False, float_format="%.3g")
    pd.DataFrame(man).to_csv(os.path.join(out, "episode_manifest.tsv"), sep="\t", index=False)
    dump(run_record(a.run_id, "e1_evaluate", t0, partition=a.partition, e1_select=a.e1_select, context_constants_sha256=sha256(os.path.join(a.e1_select, "context_constants.json")),
                    n_general=len(eps), n_matched=len(mat)), os.path.join(out, "result.json"))
    log("done", f"{time.time() - t0:.0f}s")


# ---------------------------------------------------------------- HCC (post-freeze, one pass)
FREEZE = os.path.join(MAN, "e1_source_freeze.json")


def assert_frozen():
    assert os.environ.get("R8V2_HCC_EVAL") == "1", "HCC is only read by the post-freeze job"
    r = subprocess.run(["git", "show", f"HEAD:{FREEZE}"], capture_output=True)
    assert r.returncode == 0, "source freeze manifest is not committed"
    assert r.stdout == open(FREEZE, "rb").read(), "freeze manifest differs from the committed version"
    fz = json.loads(r.stdout)
    assert subprocess.run(["git", "merge-base", "--is-ancestor", fz["code_commit"], "HEAD"]).returncode == 0
    changed = subprocess.run(["git", "diff", "--stat", fz["code_commit"], "HEAD", "--", "src/r8_transfer_v2", "configs/r8_transfer_v2"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "configs"], capture_output=True, text=True).stdout.strip()
    assert changed == "" and dirty == "", f"code changed after the source freeze:\n{changed}\n{dirty}"
    for k in ("context_constants", "base_constants"):
        assert sha256(fz[k]["path"]) == fz[k]["sha256"], f"{k} changed after the freeze"
    return fz


def stage_hcc(a, out):
    t0 = time.time()
    fz = assert_frozen()
    pairs = pd.read_csv(v1_runs(V1_PAIRS), sep="\t")
    CC = json.load(open(fz["context_constants"]["path"]))
    base = json.load(open(fz["base_constants"]["path"]))
    assert CC["partition"] == "final_inner" == base["partition"]
    bank = Bank()
    ctx = E2.Ctx2(bank, "global", fmap=Features.load(os.path.join(os.environ["R8_PROC"], "feature_basis.npz"), "cpu"), mode="v2")
    ctx.fit_nuisance(pairs, base["kappa_f"])
    dump(dict(keys=ctx.key_table(), scales=ctx.scales_k), os.path.join(out, "global_nuisance.json"))
    ofz = json.load(open(orig_man("freeze_manifest.json")))
    units = []
    for t in TARGETS:
        trows, x0, x1 = load_target(t)
        tab, ud, ux = rows_to_units(trows, x1 - x0, x0)
        for split, lists in ofz["targets"][t].items():
            ep, qn, is_test = hcc_episode(ctx.st, t, split, lists, trows, tab, ud, ux)
            for K in ks_of(ep):
                inp = Eng.inputs(ep, K)
                for contract, cname, mode in (("planned", "gen_planned", "real"), ("missing_reselected", "gen_missing", "zero"),
                                              ("missing_zero_overlap_fallback", "gen_planned", "zero")):
                    for m, p in arms(ctx, inp, K, base, CC[cname], oq_of(inp, mode)):
                        units += hcc_unit_table(p.numpy(), ep.y_q.numpy(), qn, is_test, dict(method=f"{m}|{contract}", K=K, split_seed=int(split), target=t))
            log(t, split, f"{time.time() - t0:.0f}s")
    u = pd.concat(units, ignore_index=True)
    p = os.path.join(out, "hcc_per_unit.tsv.gz")
    u.to_csv(p, sep="\t", index=False, float_format="%.8g")
    dump(run_record(a.run_id, "e1_hcc", t0, freeze=FREEZE, code_commit=fz["code_commit"], hcc_per_unit=p, hcc_per_unit_sha256=sha256(p)), os.path.join(out, "result.json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["select", "evaluate", "hcc"])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--partition", default="")
    ap.add_argument("--p0-select", default="")
    ap.add_argument("--e1-select", default="")
    a = ap.parse_args()
    out = new_run_dir("e1_" + a.stage, a.run_id)
    {"select": stage_select, "evaluate": stage_evaluate, "hcc": stage_hcc}[a.stage](a, out)
