"""Partition contexts, nested source-only selection and the evaluated method set of the follow-up.

Information boundary of a partition context: every nuisance quantity (features, covariance versions, kernel block scales)
is fitted on the partition's 'train' families; constants come from the matching INNER partition (selection on inner-validation
families with nuisance fitted on inner-train families); 'val' families are scored. Query labels (ep.y_q) are only read by
the scoring calls in this module.
"""
import json, math, zlib
import numpy as np
import pandas as pd
import torch

from src.replicate_calibrated_r8.data import Features
from src.episodic_residual_r2.evalcore import family_mean
from . import fmethods as F
from .covrepair import plate_weights, unit_table, versions, pair_constants, cov_dict

ORIG_FIXED = None      # filled from the committed original anchor_selection.json by load_fixed()


def load_fixed(path):
    """Original R8 constants (fixed-hyper rows). kappa_f is the same under iid and cov."""
    s = json.load(open(path))
    fx = dict(kappa_f={K: s["anchor"]["iid"][K]["kappa_f"] for K in ("8", "16", "32")},
              kappa_b=dict(iid={K: s["anchor"]["iid"][K]["kappa_b"] for K in ("8", "16", "32")}, cov={K: s["anchor"]["cov"][K]["kappa_b"] for K in ("8", "16", "32")}),
              c1=dict(iid={K: dict(lam=s["c1"]["iid"][K]["lam"], eta=s["c1"]["iid"][K]["eta"]) for K in ("8", "16", "32")},
                      cov={K: dict(lam=s["c1"]["cov"][K]["lam"], eta=s["c1"]["cov"][K]["eta"]) for K in ("8", "16", "32")}),
              shrunk_kappa_f={K: s["shrunk_same_drug_mean"][K]["kappa_f"] for K in ("8", "16", "32")}, covariance_model=s["covariance_model"])
    return fx


def kb(x):
    return float("inf") if (isinstance(x, str) and x == "inf") else float(x)


# ---------------------------------------------------------------- episodes with unit indices and query overlap
def plate_matrix(st, units, pos=None):
    ds = [plate_weights(st, u)[0] for u in units]
    if pos is None:
        pos = {p: i for i, p in enumerate(sorted(set(p for d in ds for p in d)))}
    W = np.zeros((len(units), len(pos)))
    for i, d in enumerate(ds):
        for p, x in d.items():
            if p in pos:
                W[i, pos[p]] += x
    return W, pos


def cross_overlap(st, qu, su):
    Ws, pos = plate_matrix(st, su)
    Wq, _ = plate_matrix(st, qu, pos)
    return torch.tensor(Wq @ Ws.T, dtype=torch.float32)


def eval_episode(st, ci, seed, cap=None, min_groups=40, max_k=32):
    """Same rule and RNG as Store.eval_episode, keeping the unit indices and adding the query-support plate overlap."""
    ui = st.eligible_units(ci)
    if len(ui) < min_groups:
        return None
    p = np.random.RandomState(880000 + 1000 * seed + int(zlib.crc32(st.cell_name[ci].encode()) % 100000)).permutation(len(ui))
    nq = int(round(0.2 * len(ui)))
    q, s = ui[p[:nq]], ui[p[nq:]]
    if cap:
        q = q[:cap]
    su = s[:max_k]
    ep = st.episode_from_units(ci, su, q, seed=seed)
    ep.su, ep.qu, ep.cell_idx = su, q, ci
    ep.o_qs = cross_overlap(st, q, su)
    return ep


def inputs(ep, K):
    inp = ep.inputs(K)
    inp["o_qs"] = ep.o_qs[:, :K]
    return inp


# ---------------------------------------------------------------- context
class Ctx:
    def __init__(self, bank, name, fmap=None):
        self.name = name
        self.st = bank.store(name)
        self.fmap = fmap if fmap is not None else Features.fit(self.st)
        self.cov, self.scales, self.ut = {}, {}, None

    def episodes(self, role, seeds, cap):
        out = []
        for ci in self.st.eval_cells(role):
            for s in seeds:
                ep = eval_episode(self.st, ci, s, cap)
                if ep is not None:
                    out.append(ep)
        return out

    def fit_cov(self, pairs, kappa_f, extra_kf=()):
        """V0..V3 on the partition's training families (V3 with the partition's kappa_f; extra V3 variants for other kappa_f)."""
        train_f = sorted(set(self.st.cell_family[self.st.cell_role == "train"]))
        pc = pair_constants(pairs, train_f)
        self.ut = unit_table(self.st, kappa_f)
        v = versions(self.ut, pc, kappa_f)
        for kf in extra_kf:
            if kf != kappa_f:
                v[f"V3_kf{kf:g}"] = dict(versions(unit_table(self.st, kf), pc, kf)["V3"], version=f"V3_kf{kf:g}")
        self.cov = v
        return v

    def obs(self, name):
        return None if name == "iid" else cov_dict(self.cov[name])

    def omega(self, inp, name):
        K = len(inp["y_s"])
        if name == "iid":
            return torch.eye(K, dtype=torch.float64)
        if name.endswith("s"):                                     # 'V3s': V3 matched per episode to the trace of V0
            return F.omega_trace_matched(inp, cov_dict(self.cov[name[:-1]]), cov_dict(self.cov["V0"]))
        return F.omega(inp, cov_dict(self.cov[name]))

    def feats(self, inp, fs, fq):
        return self.fmap.build(inp, self.st.FP, fs, fq)

    def fit_scales(self, kappa_f, seeds=(1,), cap=200):
        """Kernel block scales: mean centred trace / K over the partition's TRAINING-family episodes, per K; then per config."""
        eps = self.episodes("train", seeds, cap)
        sc = {}
        for K in (8, 16, 32):
            tr = {b: [] for b in F.BLOCKS}
            blocks_all = []
            for ep in eps:
                if K > len(ep.g_s):
                    continue
                inp = inputs(ep, K)
                fs, fq = F.fstar(inp, kappa_f)
                bl = F.kernel_blocks(self.feats(inp, fs, fq))
                blocks_all.append(bl)
                for b in F.BLOCKS:
                    tr[b].append(F.centred_trace_over_k(bl[b][0]))
            sblock = {b: float(np.mean(v)) for b, v in tr.items()}
            sb = dict(block=sblock, comb={c: 1.0 for c in F.CONFIGS})
            comb = {}
            for c in F.CONFIGS:
                comb[c] = float(np.mean([F.centred_trace_over_k(F.combine(bl, c, sb)[0]) for bl in blocks_all]))
            sc[str(K)] = dict(block=sblock, comb=comb, n_episodes=len(blocks_all),
                              zero_trace_fraction={b: float(np.mean(np.array(v) <= 1e-12)) for b, v in tr.items()})
        self.scales = sc
        return sc


# ---------------------------------------------------------------- selection on inner-validation episodes
def _choose(rows, keys, sort_cols, ascending):
    df = pd.DataFrame(rows)
    agg, _ = family_mean(df, by=["K"] + keys)
    out = {}
    for K, g in agg.groupby("K"):
        b = g.sort_values(["delta_rmse"] + sort_cols, ascending=[True] + ascending).iloc[0]
        out[str(int(K))] = {k: (b[k] if not isinstance(b[k], (np.floating, np.integer)) else b[k].item()) for k in keys + ["delta_rmse"]}
    return out, agg


def select(ctx, pairs, grids, log=print):
    """Full source-only selection inside an INNER partition. Returns constants and the grid tables."""
    eps = ctx.episodes("val", [1], 200)
    st = ctx.st
    KF, KB = [float(x) for x in grids["kappa_f"]], [kb(x) for x in grids["kappa_b"]]
    LAM, ETA = [float(x) for x in grids["lambda"]], [float(x) for x in grids["eta"]]
    base = lambda ep, K: dict(cell=ep.name, family=ep.family, support_seed=ep.seed, K=K)
    tabs = {}
    # A: kappa_f, kappa_b under iid; kappa_f of the shrunk same-drug mean
    rows = []
    with torch.no_grad():
        for ep in eps:
            for K in (8, 16, 32):
                if K > len(ep.g_s):
                    continue
                inp = inputs(ep, K)
                for kf in KF:
                    fs, fq = F.fstar(inp, kf)
                    eb = (inp["y_s"] - fs).mean(0, keepdim=True)
                    for kbv in KB:
                        rows.append(dict(kappa_f=kf, kappa_b=kbv, delta_rmse=F.rmse(fq + eb * F.alpha(K, kbv), ep.y_q), **base(ep, K)))
    anc, tabs["anchor_iid"] = _choose(rows, ["kappa_f", "kappa_b"], ["kappa_f", "kappa_b"], [True, True])
    shr, _ = _choose([r for r in rows if math.isinf(r["kappa_b"])], ["kappa_f"], ["kappa_f"], [True])
    kf16 = anc["16"]["kappa_f"]
    kappa_f = {K: anc[K]["kappa_f"] for K in anc}
    log(f"[{ctx.name}] anchor {anc}")
    # covariance versions of the inner partition (V3 with the K=16 kappa_f)
    ctx.fit_cov(pairs, kf16)
    ctx.fit_scales(kf16)
    const = dict(partition=ctx.name, kappa_f=kappa_f, shrunk_kappa_f={K: shr[K]["kappa_f"] for K in shr}, kappa_b=dict(iid={K: anc[K]["kappa_b"] for K in anc}),
                 c1={}, kn={}, j1={}, j2={}, dev=dict(anchor_iid=anc))
    # B: kappa_b per observation model (kappa_f fixed)
    rows = []
    with torch.no_grad():
        for ep in eps:
            for K in (8, 16, 32):
                if K > len(ep.g_s):
                    continue
                inp = inputs(ep, K)
                fs, fq = F.fstar(inp, kappa_f[str(K)])
                E = inp["y_s"] - fs
                for v in ("V0", "V3"):
                    w, keff = F.gls_weights(ctx.omega(inp, v))
                    eb = (w.float()[:, None] * E).sum(0, keepdim=True)
                    for kbv in KB:
                        rows.append(dict(obs=v, kappa_b=kbv, delta_rmse=F.rmse(fq + eb * F.alpha(keff, kbv), ep.y_q), **base(ep, K)))
    for v in ("V0", "V3"):
        c, _ = _choose([r for r in rows if r["obs"] == v], ["kappa_b"], ["kappa_b"], [True])
        const["kappa_b"][v] = {K: c[K]["kappa_b"] for K in c}
    # C: C1 lambda / eta per observation model (original kernel), and the normalised kernel configurations (iid)
    rows, rows_kn = [], []
    with torch.no_grad():
        for ep in eps:
            for K in (8, 16, 32):
                if K > len(ep.g_s):
                    continue
                inp = inputs(ep, K)
                fs, fq = F.fstar(inp, kappa_f[str(K)])
                bl = F.kernel_blocks(ctx.feats(inp, fs, fq))
                for v in ("iid", "V0", "V3"):
                    Om = ctx.omega(inp, v)
                    a = F.anchor_om(inp, kappa_f[str(K)], const["kappa_b"][v][str(K)], Om)
                    Kss, Kqs = F.combine(bl, "all")
                    for lam in LAM:
                        corr = F.krr(Kss, Kqs, a["w"], a["R"], lam, Om)
                        for eta in ETA:
                            rows.append(dict(obs=v, lam=lam, eta=eta, delta_rmse=F.rmse(a["fq"] + a["b"] + eta * corr, ep.y_q), **base(ep, K)))
                    if v == "iid":
                        for cfg in F.CONFIGS:
                            Kss, Kqs = F.combine(bl, cfg, ctx.scales[str(K)])
                            for lam in LAM:
                                corr = F.krr(Kss, Kqs, a["w"], a["R"], lam, Om)
                                for eta in ETA:
                                    rows_kn.append(dict(cfg=cfg, lam=lam, eta=eta, delta_rmse=F.rmse(a["fq"] + a["b"] + eta * corr, ep.y_q), **base(ep, K)))
    for v in ("iid", "V0", "V3"):
        c, _ = _choose([r for r in rows if r["obs"] == v], ["lam", "eta"], ["eta", "lam"], [True, False])
        const["c1"][v] = c
    for cfg in F.CONFIGS:
        c, _ = _choose([r for r in rows_kn if r["cfg"] == cfg], ["lam", "eta"], ["eta", "lam"], [True, False])
        const["kn"][cfg] = c
    # D: J1 and J2 (same kappa_f; J2 scale u = V3 v_tot of the partition)
    u = max(ctx.cov["V3"]["v_tot"], 1e-8)
    ve, vs = ctx.cov["V3"]["v_e"], ctx.cov["V3"]["v_s"]
    rows1, rows2 = [], []
    with torch.no_grad():
        for ep in eps:
            for K in (8, 16, 32):
                if K > len(ep.g_s):
                    continue
                inp = inputs(ep, K)
                fs, fq = F.fstar(inp, kappa_f[str(K)])
                E = inp["y_s"] - fs
                for a_, c_ in F.J1_GRID:
                    rows1.append(dict(a=a_, c=c_, ac=a_ + c_, delta_rmse=F.rmse(F.j1(fq, E, inp["o_qs"], a_, c_), ep.y_q), **base(ep, K)))
                for tb in grids["j2_tau_b_over_u"]:
                    for s0 in grids["j2_sigma0_over_u"]:
                        p = F.joint(fq, E, inp["o_s"], inp["o_qs"], inp["q_s"], tb * u, vs, ve, s0 * u, jitter=1e-6 * u)
                        rows2.append(dict(tau_b_over_u=float(tb), sigma0_over_u=float(s0), delta_rmse=F.rmse(p, ep.y_q), **base(ep, K)))
    const["j1"], _ = _choose(rows1, ["a", "c"], ["a", "c"], [True, True])
    const["j2"], _ = _choose(rows2, ["tau_b_over_u", "sigma0_over_u"], ["tau_b_over_u", "sigma0_over_u"], [True, False])
    const["j2_scale"] = dict(u=u, v_e=ve, v_s=vs, jitter_over_u=1e-6)
    const["cov"] = ctx.cov
    const["scales"] = ctx.scales
    const["n_selection_episodes"] = len(eps)
    const["n_selection_families"] = int(len({e.family for e in eps}))
    log(f"[{ctx.name}] kappa_b {const['kappa_b']} c1 {const['c1']} j1 {const['j1']} j2 {const['j2']}")
    return const


# ---------------------------------------------------------------- evaluation method set
def method_preds(ctx, inp, K, const, fx, with_extras=True):
    """Yields (method, prediction, extra-dict). const: retuned constants (inner selection); fx: original constants (fixed-hyper)."""
    Ks = str(K)
    st = ctx.st
    yield "no_change", torch.zeros_like(inp["f0_q"]), {}
    yield "same_drug_mean", inp["f0_q"], {}
    yield "shrunk_mean", F.fstar(inp, const["shrunk_kappa_f"][Ks])[1], {}
    kf = const["kappa_f"][Ks]
    fs, fq = F.fstar(inp, kf)
    E = inp["y_s"] - fs
    bl = F.kernel_blocks(ctx.feats(inp, fs, fq))
    Kall = F.combine(bl, "all")
    # retuned C0 / C1 per observation model
    anc = {}
    for v in ("iid", "V0", "V3"):
        Om = ctx.omega(inp, v)
        a = F.anchor_om(inp, kf, const["kappa_b"][v][Ks], Om)
        anc[v] = a
        yield f"C0_{v}", a["fq"] + a["b"], {"k_eff": a["keff"], "alpha": a["shrink"]}
        c = const["c1"][v][Ks]
        corr = F.krr(Kall[0], Kall[1], a["w"], a["R"], c["lam"], Om)
        ex = {}
        if v == "iid":
            r = c["eta"] * corr
            ex = dict(b_rms=float(a["b"].pow(2).mean().sqrt()), r_rms=float(r.pow(2).mean().sqrt()), r_query_var=float(r.var(0).mean()) if len(r) > 1 else 0.0,
                      fstar_rms=float(a["fq"].pow(2).mean().sqrt()), corr_mean_rms=float(r.mean(0).pow(2).mean().sqrt()))
        yield f"C1_{v}", a["fq"] + a["b"] + c["eta"] * corr, ex
    # four C0 paths (common kappa_b = the retuned iid value) for V0, V3 and V3 trace-matched to V0
    for v in ("V0", "V3", "V3s"):
        paths, d = F.c0_paths(E, ctx.omega(inp, v), const["kappa_b"]["iid"][Ks])
        for pname in ("weights_only", "shrink_only", "both"):
            yield f"P_{pname}_{v}", fq + paths[pname], (dict(d, path_version=v) if pname == "both" else {})
    # fixed-hyper (original constants)
    kf0 = fx["kappa_f"][Ks]
    for v in ("iid", "V0", "V3fx", "V3fxs"):
        on = {"iid": "iid", "V0": "V0", "V3fx": "V3_fx", "V3fxs": "V3_fxs"}[v]
        if on == "V3_fx":
            Om = F.omega(inp, cov_dict(ctx.cov["V3_fx"]))
        elif on == "V3_fxs":
            Om = F.omega_trace_matched(inp, cov_dict(ctx.cov["V3_fx"]), cov_dict(ctx.cov["V0"]))
        else:
            Om = ctx.omega(inp, on)
        ob = "iid" if v == "iid" else "cov"
        a = F.anchor_om(inp, kf0, kb(fx["kappa_b"][ob][Ks]), Om)
        name = {"iid": "iid", "V0": "V0", "V3fx": "V3", "V3fxs": "V3s"}[v]
        yield f"fx_C0_{name}", a["fq"] + a["b"], {"k_eff": a["keff"], "alpha": a["shrink"]}
        fs0, fq0 = a["fs"], a["fq"]
        bl0 = bl if kf0 == kf else F.kernel_blocks(ctx.feats(inp, fs0, fq0))
        K0 = F.combine(bl0, "all")
        c = fx["c1"][ob][Ks]
        yield f"fx_C1_{name}", a["fq"] + a["b"] + c["eta"] * F.krr(K0[0], K0[1], a["w"], a["R"], c["lam"], Om), {}
    if not with_extras:
        return
    # C1 direct residual shuffle (iid; 3 pre-fixed derangements); f*, b, features, queries fixed
    a = anc["iid"]
    c = const["c1"]["iid"][Ks]
    Om = torch.eye(K, dtype=torch.float64)
    for j in (1, 2, 3):
        p = torch.as_tensor(F.derangement(K, j))
        yield f"C1_iid_shuf{j}", a["fq"] + a["b"] + c["eta"] * F.krr(Kall[0], Kall[1], a["w"], a["R"][p], c["lam"], Om), {}
    # kernel components: original scale with the C1-iid constants (sensitivity) and normalised scale with own constants
    for cfg in F.CONFIGS:
        if cfg != "all":
            Kc = F.combine(bl, cfg)
            yield f"kr_{cfg}", a["fq"] + a["b"] + c["eta"] * F.krr(Kc[0], Kc[1], a["w"], a["R"], c["lam"], Om), {}
        Kc = F.combine(bl, cfg, ctx.scales[Ks])
        cc = const["kn"][cfg][Ks]
        yield f"kn_{cfg}", a["fq"] + a["b"] + cc["eta"] * F.krr(Kc[0], Kc[1], a["w"], a["R"], cc["lam"], Om), {}
    # J1 / J2 (query metadata: planned plate overlap) and their unknown-environment mode (o_qS = 0)
    j = const["j1"][Ks]
    yield "J1", F.j1(fq, E, inp["o_qs"], j["a"], j["c"]), {}
    yield "J1_unknown_env", F.j1(fq, E, torch.zeros_like(inp["o_qs"]), j["a"], j["c"]), {}
    u, ve, vs = ctx.cov["V3"]["v_tot"], ctx.cov["V3"]["v_e"], ctx.cov["V3"]["v_s"]
    j = const["j2"][Ks]
    for mode, oq in (("J2", inp["o_qs"]), ("J2_unknown_env", torch.zeros_like(inp["o_qs"]))):
        p = F.joint(fq, E, inp["o_s"], oq, inp["q_s"], j["tau_b_over_u"] * u, vs, ve, j["sigma0_over_u"] * u, jitter=1e-6 * u)
        corr = p - fq
        yield mode, p, (dict(corr_mean_rms=float(corr.mean(0).pow(2).mean().sqrt()), corr_spec_rms=float((corr - corr.mean(0, keepdim=True)).pow(2).mean().sqrt()))
                        if mode == "J2" else {})


def fit_outer(ctx, pairs, const):
    """Refit nuisance on the outer partition's training families with the inner-selected kappa_f (and kappa_f = 8 for fixed rows)."""
    kf = const["kappa_f"]["16"]
    v = ctx.fit_cov(pairs, kf, extra_kf=(8.0,))
    v["V3_fx"] = v.get("V3_kf8", v["V3"])
    ctx.fit_scales(kf)
    return v
