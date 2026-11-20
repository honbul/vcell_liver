"""v2 partition context, source-inner selection with full candidate grids, and the v1 method subset touched by P0-B.

P0-B (plan 5.1): v1 fitted the V3 covariance and the block-normalised kernel scales once with the K = 16 kappa_f and reused them
for every K. v2 keys the covariance by (partition, reference contract, kappa_f) and the kernel scales by
(partition, reference contract, K, kappa_f): every K uses the kappa_f of its own anchor. mode='v1' reproduces the old behaviour
exactly (used for the replay checks and the P0-C grids); mode='v2' is the corrected one.

P0-C (plan 5.1): selection keeps the full unrounded candidate grids. Two tie rules:
  v1  code order of the follow-up v1 (anchor kappa_f asc then kappa_b asc; kappa_b asc; eta asc then lambda desc;
      J1 a asc then c asc; J2 tau_b asc then sigma0 desc; J3 tau_k asc, tau_b asc, sigma0 desc)
  v2  the documented order (config 'ties': weaker correction = larger kappa_b, smaller eta, larger lambda, smaller a + c);
      orders the config does not define (kappa_f, J2/J3 scalars) keep the v1 behaviour.
Only exact equality of the unrounded family-mean score is a tie; near-equal scores are never merged.
No query label is read here except by the scoring calls of the selection loops (inner-validation families only).
"""
import math
import numpy as np
import pandas as pd
import torch

from src.replicate_calibrated_r8.data import Features
from src.episodic_residual_r2.evalcore import family_mean
from src.r8_followup_v1 import engine as Eng
from src.r8_followup_v1 import fmethods as F
from src.r8_followup_v1.covrepair import pair_constants, unit_table, versions, cov_dict

KS = (8, 16, 32)
# block configurations: v1 configurations plus the E2-A cuts (ctrl kept in every kernel arm, plan 5.3)
CONFIGS2 = dict(F.CONFIGS, **{"chem+ctrl": ("chem", "ctrl"), "f0+ctrl": ("ctrl", "f0")})


# ---------------------------------------------------------------- tie rules
RULES = {
    # grid: (v1 order, v2 order); each order is a list of (column, ascending)
    "anchor_iid": ([("kappa_f", True), ("kappa_b", True)], [("kappa_f", True), ("kappa_b", False)]),
    "shrunk": ([("kappa_f", True)], [("kappa_f", True)]),
    "kappa_b": ([("kappa_b", True)], [("kappa_b", False)]),
    "lam_eta": ([("eta", True), ("lam", False)], [("eta", True), ("lam", False)]),
    "j1": ([("a", True), ("c", True)], [("ac", True), ("a", True), ("c", True)]),
    "j2": ([("tau_b_over_u", True), ("sigma0_over_u", False)], [("tau_b_over_u", True), ("sigma0_over_u", False)]),
    "j3": ([("tau_k_over_u", True), ("tau_b_over_u", True), ("sigma0_over_u", False)],
           [("tau_k_over_u", True), ("tau_b_over_u", True), ("sigma0_over_u", False)]),
    "c1j1": ([("a", True), ("c", True)], [("ac", True), ("a", True), ("c", True)]),
    "c1j1_full": ([("eta", True), ("lam", False), ("a", True), ("c", True)], [("eta", True), ("lam", False), ("ac", True), ("a", True), ("c", True)]),
}


def _pick(g, order):
    cols = ["delta_rmse"] + [c for c, _ in order]
    asc = [True] + [a for _, a in order]
    return g.sort_values(cols, ascending=asc).iloc[0]


def _plain(x):
    return x.item() if isinstance(x, (np.floating, np.integer)) else x


def choose2(rows, keys, rule, tie="v2", grid="", partition="", tie_log=None):
    """Family-mean score per candidate (unrounded), selection under tie rule `tie`; logs exact ties and the v1/v2 choices."""
    df = pd.DataFrame(rows)
    agg, _ = family_mean(df, by=["K"] + keys)
    if "a" in keys and "c" in keys:
        agg["ac"] = agg["a"] + agg["c"]
    out = {}
    o1, o2 = RULES[rule]
    for K, g in agg.groupby("K"):
        b1, b2 = _pick(g, o1), _pick(g, o2)
        b = b2 if tie == "v2" else b1
        out[str(int(K))] = {k: _plain(b[k]) for k in keys + ["delta_rmse"]}
        if tie_log is not None:
            s = np.sort(g.delta_rmse.to_numpy())
            best = s[0]
            n_exact = int((g.delta_rmse == best).sum())
            tie_log.append(dict(partition=partition, grid=grid, K=int(K), n_candidates=len(g), best_score=float(best),
                                runner_up_gap=float(s[1] - s[0]) if len(s) > 1 else float("nan"), n_exact_tie_at_best=n_exact,
                                chosen_v1=json_c(b1, keys), chosen_v2=json_c(b2, keys),
                                selection_changed=bool(any(_plain(b1[k]) != _plain(b2[k]) for k in keys)), n_families=int(g.n_families.iloc[0])))
    return out, agg


def json_c(b, keys):
    return ";".join(f"{k}={_plain(b[k])}" for k in keys)


# ---------------------------------------------------------------- context with per-K nuisance
class Ctx2(Eng.Ctx):
    def __init__(self, bank, name, fmap=None, mode="v2"):
        super().__init__(bank, name, fmap)
        assert mode in ("v1", "v2")
        self.mode = mode
        self.cov_kf, self.scales_k, self.kfK = {}, {}, {}

    def fit_nuisance(self, pairs, kappa_f, seeds=(1,), cap=200, extra_kf=()):
        """kappa_f: dict K -> predictor anchor kappa_f. v1 mode: every K uses the K = 16 value (old behaviour)."""
        k16 = float(kappa_f["16"])
        self.kfK = {K: (k16 if self.mode == "v1" else float(kappa_f[K])) for K in kappa_f}
        self.pred_kfK = {K: float(kappa_f[K]) for K in kappa_f}
        train_f = sorted(set(self.st.cell_family[self.st.cell_role == "train"]))
        pc = pair_constants(pairs, train_f)
        self.cov_kf = {}
        for kf in sorted(set(self.kfK.values()) | {k16} | set(float(x) for x in extra_kf)):
            ut = unit_table(self.st, kf)
            self.cov_kf[kf] = versions(ut, pc, kf)
            if kf == k16:
                self.ut = ut
        self.cov = self.cov_kf[k16]
        self.fit_scales_k(seeds, cap)
        return self.cov_kf

    def cov_k(self, K):
        return self.cov_kf[self.kfK[str(K)]]

    def omega_k(self, inp, name, K):
        if name == "iid":
            return torch.eye(len(inp["y_s"]), dtype=torch.float64)
        c = self.cov_k(K)
        if name == "V3s":
            return F.omega_trace_matched(inp, cov_dict(c["V3"]), cov_dict(c["V0"]))
        return F.omega(inp, cov_dict(c[name]))

    def fit_scales_k(self, seeds=(1,), cap=200):
        """As engine.Ctx.fit_scales, with the kappa_f of each K (v2) or of K = 16 (v1); extra E2-A configurations."""
        eps = self.episodes("train", seeds, cap)
        sc = {}
        for K in KS:
            kf = self.kfK[str(K)]
            tr = {b: [] for b in F.BLOCKS}
            blocks_all = []
            for ep in eps:
                if K > len(ep.g_s):
                    continue
                inp = Eng.inputs(ep, K)
                fs, fq = F.fstar(inp, kf)
                bl = F.kernel_blocks(self.feats(inp, fs, fq))
                blocks_all.append(bl)
                for b in F.BLOCKS:
                    tr[b].append(F.centred_trace_over_k(bl[b][0]))
            sblock = {b: float(np.mean(v)) for b, v in tr.items()}
            comb = {}
            for c in CONFIGS2:
                comb[c] = float(np.mean([F.centred_trace_over_k(combine2(bl, c, dict(block=sblock, comb={c: 1.0}))[0]) for bl in blocks_all]))
            sc[str(K)] = dict(block=sblock, comb=comb, n_episodes=len(blocks_all), residual_anchor_kappa_f=kf, predictor_anchor_kappa_f=self.pred_kfK[str(K)],
                              zero_trace_fraction={b: float(np.mean(np.array(v) <= 1e-12)) for b, v in tr.items()})
        self.scales_k = sc
        self.scales = sc
        return sc

    def key_table(self):
        """Identity of the nuisance used at each K (P0-B record)."""
        out = []
        for K in KS:
            Ks = str(K)
            c = self.cov_k(K)["V3"]
            out.append(dict(partition=self.name, mode=self.mode, K=K, predictor_anchor_kappa_f=self.pred_kfK[Ks], residual_anchor_kappa_f=self.kfK[Ks],
                            kappa_f_consistent=bool(self.kfK[Ks] == self.pred_kfK[Ks]), V3_kappa_f=c["kappa_f"], V3_v_tot=c["v_tot"], V3_v_r=c["v_r"],
                            V3_v_e=c["v_e"], V3_v_s=c["v_s"], scale_kappa_f=self.scales_k[Ks]["residual_anchor_kappa_f"],
                            **{f"scale_{b}": v for b, v in self.scales_k[Ks]["block"].items()}))
        return out


def combine2(blocks, config, scales=None):
    """F.combine generalised to CONFIGS2 (identical arithmetic for the v1 configurations)."""
    names = CONFIGS2[config]
    order = [b for b in ("chem", "ctrl", "f0") if b in names]
    if scales is None:
        ss, qs = blocks[order[0]]
        for b in order[1:]:
            ss, qs = ss + blocks[b][0], qs + blocks[b][1]
        return ss, qs
    ss, qs = 0.0, 0.0
    for b in order:
        s = scales["block"][b]
        if s > 0:
            ss, qs = ss + blocks[b][0].double() / s, qs + blocks[b][1].double() / s
        else:
            ss, qs = ss + 0.0 * blocks[b][0].double(), qs + 0.0 * blocks[b][1].double()
    sc = scales["comb"][config]
    return (ss / sc, qs / sc) if sc > 0 else (ss * 0.0, qs * 0.0)


# ---------------------------------------------------------------- v1 selection replayed with full grids (P0-B / P0-C)
def select2(ctx, pairs, grids, tie="v2", log=print, tie_log=None):
    """engine.select with the per-K nuisance of ctx.mode and tie rule `tie`. Returns constants and the unrounded grid tables."""
    eps = ctx.episodes("val", [1], 200)
    KF, KB = [float(x) for x in grids["kappa_f"]], [Eng.kb(x) for x in grids["kappa_b"]]
    LAM, ETA = [float(x) for x in grids["lambda"]], [float(x) for x in grids["eta"]]
    base = lambda ep, K: dict(cell=ep.name, family=ep.family, support_seed=ep.seed, K=K)
    P = ctx.name
    gt = {}
    rows = []
    with torch.no_grad():
        for ep in eps:
            for K in KS:
                if K > len(ep.g_s):
                    continue
                inp = Eng.inputs(ep, K)
                for kf in KF:
                    fs, fq = F.fstar(inp, kf)
                    eb = (inp["y_s"] - fs).mean(0, keepdim=True)
                    for kbv in KB:
                        rows.append(dict(kappa_f=kf, kappa_b=kbv, delta_rmse=F.rmse(fq + eb * F.alpha(K, kbv), ep.y_q), **base(ep, K)))
    anc, gt["anchor_iid"] = choose2(rows, ["kappa_f", "kappa_b"], "anchor_iid", tie, "anchor_iid", P, tie_log)
    shr, gt["shrunk"] = choose2([r for r in rows if math.isinf(r["kappa_b"])], ["kappa_f"], "shrunk", tie, "shrunk", P, tie_log)
    kappa_f = {K: anc[K]["kappa_f"] for K in anc}
    log(f"[{P}/{ctx.mode}/{tie}] anchor {anc}")
    ctx.fit_nuisance(pairs, kappa_f)
    const = dict(partition=P, nuisance_mode=ctx.mode, tie_rule=tie, kappa_f=kappa_f, shrunk_kappa_f={K: shr[K]["kappa_f"] for K in shr},
                 kappa_b=dict(iid={K: anc[K]["kappa_b"] for K in anc}), c1={}, kn={}, dev=dict(anchor_iid=anc))
    rows = []
    with torch.no_grad():
        for ep in eps:
            for K in KS:
                if K > len(ep.g_s):
                    continue
                inp = Eng.inputs(ep, K)
                fs, fq = F.fstar(inp, kappa_f[str(K)])
                E = inp["y_s"] - fs
                for v in ("V0", "V3"):
                    w, keff = F.gls_weights(ctx.omega_k(inp, v, K))
                    eb = (w.float()[:, None] * E).sum(0, keepdim=True)
                    for kbv in KB:
                        rows.append(dict(obs=v, kappa_b=kbv, delta_rmse=F.rmse(fq + eb * F.alpha(keff, kbv), ep.y_q), **base(ep, K)))
    for v in ("V0", "V3"):
        c, gt[f"kappa_b_{v}"] = choose2([r for r in rows if r["obs"] == v], ["kappa_b"], "kappa_b", tie, f"kappa_b_{v}", P, tie_log)
        const["kappa_b"][v] = {K: c[K]["kappa_b"] for K in c}
    rows, rows_kn = [], []
    with torch.no_grad():
        for ep in eps:
            for K in KS:
                if K > len(ep.g_s):
                    continue
                inp = Eng.inputs(ep, K)
                fs, fq = F.fstar(inp, kappa_f[str(K)])
                bl = F.kernel_blocks(ctx.feats(inp, fs, fq))
                for v in ("iid", "V0", "V3"):
                    Om = ctx.omega_k(inp, v, K)
                    a = F.anchor_om(inp, kappa_f[str(K)], const["kappa_b"][v][str(K)], Om)
                    Kss, Kqs = F.combine(bl, "all")
                    for lam in LAM:
                        corr = F.krr(Kss, Kqs, a["w"], a["R"], lam, Om)
                        for eta in ETA:
                            rows.append(dict(obs=v, lam=lam, eta=eta, delta_rmse=F.rmse(a["fq"] + a["b"] + eta * corr, ep.y_q), **base(ep, K)))
                    if v == "iid":
                        for cfg in F.CONFIGS:
                            Kss, Kqs = F.combine(bl, cfg, ctx.scales_k[str(K)])
                            for lam in LAM:
                                corr = F.krr(Kss, Kqs, a["w"], a["R"], lam, Om)
                                for eta in ETA:
                                    rows_kn.append(dict(cfg=cfg, lam=lam, eta=eta, delta_rmse=F.rmse(a["fq"] + a["b"] + eta * corr, ep.y_q), **base(ep, K)))
    for v in ("iid", "V0", "V3"):
        const["c1"][v], gt[f"c1_{v}"] = choose2([r for r in rows if r["obs"] == v], ["lam", "eta"], "lam_eta", tie, f"c1_{v}", P, tie_log)
    for cfg in F.CONFIGS:
        const["kn"][cfg], gt[f"kn_{cfg}"] = choose2([r for r in rows_kn if r["cfg"] == cfg], ["lam", "eta"], "lam_eta", tie, f"kn_{cfg}", P, tie_log)
    const["cov_by_kappa_f"] = {str(k): v for k, v in ctx.cov_kf.items()}
    const["scales"] = ctx.scales_k
    const["nuisance_keys"] = ctx.key_table()
    const["n_selection_episodes"] = len(eps)
    const["n_selection_families"] = int(len({e.family for e in eps}))
    log(f"[{P}/{ctx.mode}/{tie}] kappa_b {const['kappa_b']} c1 {const['c1']}")
    return const, gt


def select_j2(ctx, const, grids, tie="v2", log=print, tie_log=None):
    """engine.select_j (with J3) with the per-K nuisance of ctx.mode: u, v_e, v_s from V3 of the K's kappa_f, kernel scales of K."""
    eps = ctx.episodes("val", [1], 200)
    kappa_f = const["kappa_f"]
    base = lambda ep, K: dict(cell=ep.name, family=ep.family, support_seed=ep.seed, K=K)
    rows1, rows2, rows3 = [], [], []
    tks = [float(x) for x in grids["j3_tau_k_over_u"]]
    scal = {}
    with torch.no_grad():
        for ep in eps:
            for K in KS:
                if K > len(ep.g_s):
                    continue
                c3 = ctx.cov_k(K)["V3"]
                u, ve, vs = max(c3["v_tot"], 1e-8), c3["v_e"], c3["v_s"]
                scal[str(K)] = dict(u=u, v_e=ve, v_s=vs, residual_anchor_kappa_f=ctx.kfK[str(K)])
                inp = Eng.inputs(ep, K)
                fs, fq = F.fstar(inp, kappa_f[str(K)])
                E = inp["y_s"] - fs
                for a_, c_ in F.J1_GRID:
                    rows1.append(dict(a=a_, c=c_, delta_rmse=F.rmse(F.j1(fq, E, inp["o_qs"], a_, c_), ep.y_q), **base(ep, K)))
                Kn = F.combine(F.kernel_blocks(ctx.feats(inp, fs, fq)), "all", ctx.scales_k[str(K)])
                for tb in grids["j2_tau_b_over_u"]:
                    for s0 in grids["j2_sigma0_over_u"]:
                        for tk in tks:
                            p = F.joint(fq, E, inp["o_s"], inp["o_qs"], inp["q_s"], tb * u, vs, ve, s0 * u, tau_k=tk * u, Kss=Kn[0], Kqs=Kn[1], jitter=1e-6 * u)
                            r = dict(tau_b_over_u=float(tb), sigma0_over_u=float(s0), tau_k_over_u=tk, delta_rmse=F.rmse(p, ep.y_q), **base(ep, K))
                            rows3.append(r)
                            if tk == 0.0:
                                rows2.append(r)
    out, gt = {}, {}
    P = ctx.name
    out["j1"], gt["j1"] = choose2(rows1, ["a", "c"], "j1", tie, "j1", P, tie_log)
    out["j2"], gt["j2"] = choose2(rows2, ["tau_b_over_u", "sigma0_over_u"], "j2", tie, "j2", P, tie_log)
    out["j3"], gt["j3"] = choose2(rows3, ["tau_b_over_u", "sigma0_over_u", "tau_k_over_u"], "j3", tie, "j3", P, tie_log)
    out["j_scale_by_K"] = scal
    out["partition"] = P
    log(f"[{P}/{ctx.mode}/{tie}] j1 {out['j1']} j2 {out['j2']} j3 {out['j3']}")
    return out, gt


def grids_long(gt, partition, mode, tie):
    """Long table of every candidate score (unrounded; written with %.17g)."""
    out = []
    for grid, agg in gt.items():
        keys = [c for c in agg.columns if c not in ("K", "delta_rmse", "n_families", "ac")]
        for r in agg.itertuples(index=False):
            d = r._asdict()
            out.append(dict(partition=partition, nuisance_mode=mode, tie_rule=tie, grid=grid, K=int(d["K"]), candidate=";".join(f"{k}={d[k]}" for k in keys),
                            score=float(d["delta_rmse"]), n_families=int(d["n_families"])))
    return out


# ---------------------------------------------------------------- v1 methods touched by P0-B (names as in the v1 tables)
AFFECTED = ("C0_V3", "C1_V3", "P_weights_only_V3", "P_shrink_only_V3", "P_both_V3", "P_weights_only_V3s", "P_shrink_only_V3s", "P_both_V3s",
            "kn_chem", "kn_f0", "kn_ctrl", "kn_chem+f0", "kn_all", "J2", "J3", "J2_unknown_env", "J3_unknown_env")


def affected_preds(ctx, inp, K, const, jc):
    """Same arithmetic as engine.method_preds for the AFFECTED methods, but every nuisance quantity is the one of K (ctx.mode)."""
    Ks = str(K)
    kf = const["kappa_f"][Ks]
    fs, fq = F.fstar(inp, kf)
    E = inp["y_s"] - fs
    bl = F.kernel_blocks(ctx.feats(inp, fs, fq))
    Kall = F.combine(bl, "all")
    Om = ctx.omega_k(inp, "V3", K)
    a = F.anchor_om(inp, kf, const["kappa_b"]["V3"][Ks], Om)
    yield "C0_V3", a["fq"] + a["b"]
    c = const["c1"]["V3"][Ks]
    yield "C1_V3", a["fq"] + a["b"] + c["eta"] * F.krr(Kall[0], Kall[1], a["w"], a["R"], c["lam"], Om)
    for v in ("V3", "V3s"):
        paths, _ = F.c0_paths(E, ctx.omega_k(inp, v, K), const["kappa_b"]["iid"][Ks])
        for pname in ("weights_only", "shrink_only", "both"):
            yield f"P_{pname}_{v}", fq + paths[pname]
    ai = F.anchor_om(inp, kf, const["kappa_b"]["iid"][Ks], torch.eye(K, dtype=torch.float64))
    for cfg in F.CONFIGS:
        Kc = F.combine(bl, cfg, ctx.scales_k[Ks])
        cc = const["kn"][cfg][Ks]
        yield f"kn_{cfg}", ai["fq"] + ai["b"] + cc["eta"] * F.krr(Kc[0], Kc[1], ai["w"], ai["R"], cc["lam"], ai["Om"])
    c3 = ctx.cov_k(K)["V3"]
    u, ve, vs = c3["v_tot"], c3["v_e"], c3["v_s"]
    Kn = F.combine(bl, "all", ctx.scales_k[Ks])
    for name in ("j2", "j3"):
        j = jc[name][Ks]
        tk = j.get("tau_k_over_u", 0.0) * u
        for mode, oq in ((name.upper(), inp["o_qs"]), (name.upper() + "_unknown_env", torch.zeros_like(inp["o_qs"]))):
            yield mode, F.joint(fq, E, inp["o_s"], oq, inp["q_s"], j["tau_b_over_u"] * u, vs, ve, j["sigma0_over_u"] * u, tau_k=tk, Kss=Kn[0], Kqs=Kn[1], jitter=1e-6 * u)
