"""R8 E2-B 'separated' correction (v3): the held-out drug is removed from fitting AND from constant selection.

v2 (src/r8_transfer_v2/e2b.py, stage sru) removed the outer query drugs (drug fold j) from the source f0 table, f0 PCA, kernel
scales and ridge map, but its inner selection scored the SAME drug-fold-j groups on inner-validation families, and kappa_f came from
the full-reference P0 selection. v3 contract (configs/r8_unseen_drug_v3/run_spec.yaml; private plan 2026-11-20 section 4.2):

  outer   family fold P (outer0-2, cells.tsv cv_fold) x drug fold j (v2 rule crc32('r8v2_drug|' + group) % 3). Queries = fold-j units
          of the 20 % query part of outer-val cells; held-for-fit set Hc = alias closure of fold j (configs/.../alias_exceptions.tsv).
  inner   partition INNER_OF[P] (inner-train / inner-val families inside the outer-train families); allowed groups A = not Hc,
          split into 3 inner drug folds i (crc32('r8v3_inner_drug|' + group) % 3). Inner fold i: queries = A & fold i on inner-val
          cells (seed 1, cap 200), held-for-fit = Hc | closure(A & fold i), supports exclude that set. Every constant
          (kappa_f, kappa_b, ridge alpha, lambda, eta) is selected on the pooled inner folds.
  fits    f0 table, f0 PCA, kernel scales, ridge map of every context are built from its 'train' families with the held-for-fit
          groups removed (n = 0, no leave-family-out entry). Covariance nuisances are not built (no arm uses them).
Arms (K = 16 only): fb_C0, fb_C1chem, ridge_C0, ridge_C1chem (definitions as v2 e2b.sru_arms; ref arms are not run).
Query labels (ep.y_q) are read only by the inner selection scores and by the outer scoring / prediction dump.
"""
import argparse, hashlib, json, os, time, zlib
import numpy as np
import pandas as pd
import torch
import yaml

from src.replicate_calibrated_r8.data import Features, Episode
from src.episodic_residual_r2.evalcore import family_mean
from src.r8_followup_v1 import engine as Eng
from src.r8_followup_v1 import fmethods as F
from src.r8_followup_v1.partitions import Bank, INNER_OF
from src.r8_transfer_v2 import engine2 as E2
from src.r8_transfer_v2.e2b import drug_fold, ridge_fit, ridge_pred

torch.set_grad_enabled(False)
SPEC = "configs/r8_unseen_drug_v3/run_spec.yaml"
ALIAS = "configs/r8_unseen_drug_v3/alias_exceptions.tsv"
ARMS = ("fb_C0", "fb_C1chem", "ridge_C0", "ridge_C1chem")
K = 16
KS_ = str(K)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def spec():
    return yaml.safe_load(open(SPEC))


def runs(*p):
    return os.path.join(os.environ["R8V3_RUNS"], *p)


def sha256(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def sha_list(x):
    return hashlib.sha256("|".join(sorted(map(str, x))).encode()).hexdigest()[:16]


def dump(obj, path):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(path + ".tmp", path)


def git_head():
    import subprocess
    h = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    d = subprocess.run(["git", "status", "--porcelain", "--", "src", "configs", "experiments"], capture_output=True, text=True).stdout.strip()
    return h, bool(d)


def inner_fold(g, n=3):
    return zlib.crc32(("r8v3_inner_drug|" + str(g)).encode()) % n


# ---------------------------------------------------------------- alias closure (identity exceptions, fixed before scoring)
class Alias:
    def __init__(self, groups, path=ALIAS):
        self.G = len(groups)
        pos = {g: i for i, g in enumerate(groups)}
        self.members = []
        if os.path.exists(path):
            t = pd.read_csv(path, sep="\t")
            for _, s in t.groupby("cluster"):
                idx = [pos[g] for g in s.drug_group if g in pos]
                if len(idx) > 1:
                    self.members.append(np.array(sorted(set(idx))))

    def close(self, mask):
        m = mask.copy()
        for idx in self.members:
            if m[idx].any():
                m[idx] = True
        return m


def build_alias(out):
    """Distinct canonical groups that share the largest-fragment non-isomeric SMILES or an InChIKey first block (union-find)."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    g = pd.read_csv(os.path.join(os.environ["R8_PROC"], "groups.tsv"), sep="\t")
    d = pd.read_csv(os.path.join(os.environ["TB_PROC"], "drugs.tsv"), sep="\t")
    d = d[d.drug_group.isin(set(g.drug_group))]

    def lf(s):
        m = Chem.MolFromSmiles(s) if isinstance(s, str) and s else None
        if m is None:
            return None
        fr = max(Chem.GetMolFrags(m, asMols=True), key=lambda x: x.GetNumHeavyAtoms())
        return Chem.MolToSmiles(fr, isomericSmiles=False)
    keys = []
    for r in d.itertuples():
        k = lf(r.rdkit_smiles)
        if k:
            keys.append(("largest_fragment_smiles", k, r.drug_group, r.pert_id))
        for ik in str(r.inchi_keys).split(";"):
            if ik and ik != "nan":
                keys.append(("inchikey_block1", ik[:14], r.drug_group, r.pert_id))
    kt = pd.DataFrame(keys, columns=["criterion", "key", "drug_group", "pert_id"])
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    hits = []
    for (crit, key), s in kt.groupby(["criterion", "key"]):
        gs = sorted(set(s.drug_group))
        if len(gs) > 1:
            for x in gs[1:]:
                parent[find(x)] = find(gs[0])
            hits.append(dict(criterion=crit, key=key, groups=";".join(gs)))
    rows = []
    allg = sorted({x for h in hits for x in h["groups"].split(";")})
    roots = {x: find(x) for x in allg}
    for x in allg:
        crit = ";".join(sorted({h["criterion"] for h in hits if x in h["groups"].split(";")}))
        rows.append(dict(cluster=roots[x], drug_group=x, criterion=crit, v2_outer_drug_fold=drug_fold(x)))
    t = pd.DataFrame(rows).sort_values(["cluster", "drug_group"])
    t.to_csv(os.path.join(out, "alias_exceptions.tsv"), sep="\t", index=False)
    pd.DataFrame(hits).to_csv(os.path.join(out, "alias_hits.tsv"), sep="\t", index=False)
    log("alias clusters", t.cluster.nunique(), "groups", len(t))


# ---------------------------------------------------------------- held contexts
class HeldCtx(E2.Ctx2):
    """Partition context whose f0 table has no row for the held-for-fit groups (as v2 e2b.hold_store); eligibility and the
    episode permutation are taken from the full store BEFORE removal (metadata only, identical to v2 held_episodes)."""

    def __init__(self, bank, name, held_fit, tag):
        self.name, self.tag = name, tag
        st = bank.store(name)
        self.val_cells = st.eval_cells("val")
        self.elig = {ci: st.eligible_units(ci) for ci in self.val_cells}
        h = torch.as_tensor(np.where(held_fit)[0])
        st.S_train[h] = 0.0
        st.n_train[h] = 0.0
        st.S_all[h] = 0.0
        st.n_all[h] = 0.0
        st.fg_index_t[:, h] = -1
        self.st, self.held_fit = st, held_fit
        self.fmap = Features.fit(st)
        self.cov, self.scales, self.ut, self.mode = {}, {}, None, "v2"
        self.cov_kf, self.scales_k, self.kfK, self.pred_kfK = {}, {}, {}, {}
        self.scale_groups, self.scale_fams = set(), set()

    def episodes(self, role, seeds, cap):
        eps = super().episodes(role, seeds, cap)
        if role == "train":                                  # kernel-scale episodes: record their data scope
            for ep in eps:
                self.scale_groups |= set(map(int, ep.g_s)) | set(map(int, ep.g_q))
                self.scale_fams.add(ep.family)
        return eps

    def fit_scales(self, kappa_f):
        self.kfK = {k: float(kappa_f) for k in ("8", "16", "32")}
        self.pred_kfK = dict(self.kfK)
        self.fit_scales_k((1,), 200)

    def f0_groups(self):
        return set(np.where((self.st.n_train >= 1).numpy())[0].tolist())

    def f0_fams(self):
        st = self.st
        fi = np.where(st.fam_role == "train")[0]
        ok = (st.fg_index_t[torch.as_tensor(fi)] >= 0).numpy().any(1)
        return set(st.fam_names[fi[ok]])

    def fams(self, role):
        return set(self.st.fam_names[self.st.fam_role == role])


def build_eps(ctx, q_mask, s_excl, seeds, cap, min_q, min_s, max_k=32, tag=""):
    """v2 held_episodes rule: E1 RNG over the full-store eligible units; queries = q_mask units of the 20 % query part (cap),
    supports = units of the support order outside s_excl (first max_k). Query f0 = 0, n0 = 0; support f0 from the held store."""
    st = ctx.st
    out, elig = [], []
    for ci in ctx.val_cells:
        ui = ctx.elig[ci]
        for seed in seeds:
            p = np.random.RandomState(880000 + 1000 * seed + int(zlib.crc32(st.cell_name[ci].encode()) % 100000)).permutation(len(ui))
            nq = int(round(0.2 * len(ui)))
            q, s = ui[p[:nq]], ui[p[nq:]]
            q = q[q_mask[st.u_group[q]]]
            if cap:
                q = q[:cap]
            s = s[~s_excl[st.u_group[s]]][:max_k]
            ok = len(q) >= min_q and len(s) >= min_s
            elig.append(dict(level=tag, cell=st.cell_name[ci], family=st.cell_family[ci], support_seed=seed, n_query=len(q), n_support=len(s), eligible=ok))
            if not ok:
                continue
            gs, gq = st.u_group[s], st.u_group[q]
            f0s, n0s = st.f0(st.regime_of(ci), gs, st.cell_family[ci])
            o, qq = st.unit_overlap(s)
            G = st.D.shape[1]
            ep = Episode(name=st.cell_name[ci], family=st.cell_family[ci], seed=seed, g_s=gs, g_q=gq, x0_s=st.X0[torch.as_tensor(s)],
                         x0_q=st.X0[torch.as_tensor(q)], f0_s=f0s, f0_q=torch.zeros(len(q), G), n0_s=n0s, n0_q=torch.zeros(len(q)),
                         y_s=st.D[torch.as_tensor(s)], y_q=st.D[torch.as_tensor(q)], o_s=o, q_s=qq)
            ep.o_qs = torch.zeros(len(q), len(s))            # unused by every v3 arm (Eng.inputs requires the attribute)
            ep.su, ep.qu, ep.cell_idx = s, q, ci
            out.append(ep)
    return out, elig


# ---------------------------------------------------------------- arms
def parts(ctx, ep, kappa_f):
    inp = Eng.inputs(ep, K)
    fs, fq0 = F.fstar(inp, kappa_f)                       # fq0 = 0 because n0_q = 0
    E = inp["y_s"] - fs
    me = E.mean(0, keepdim=True)
    return inp, fs, fq0, E, me


def arm_preds(ctx, ep, c, rm):
    inp, fs, fq0, E, me = parts(ctx, ep, c["kappa_f"])
    rq = ridge_pred(rm, ctx.st.FP[torch.as_tensor(ep.g_q)].numpy())
    w = torch.full((K,), 1.0 / K)
    I = torch.eye(K, dtype=torch.float64)
    for anc, fq in (("fb", fq0), ("ridge", rq)):
        a0, a1 = c[f"{anc}_C0"], c[f"{anc}_C1chem"]
        yield f"{anc}_C0", fq + me * F.alpha(K, a0["kappa_b"])
        Kc = E2.combine2(F.kernel_blocks(ctx.feats(inp, fs, fq)), "chem+ctrl", ctx.scales_k[KS_])
        yield f"{anc}_C1chem", fq + me * F.alpha(K, a1["kappa_b"]) + a1["eta"] * F.krr(Kc[0], Kc[1], w, E - me, a1["lam"], I)


# ---------------------------------------------------------------- selection on the inner contract
def select(ictx, ieps, g, tie, tl, tag):
    """ieps: list of (inner fold i, episode). Returns the constants and the score tables."""
    KF = [float(x) for x in g["kappa_f"]]
    KB = [Eng.kb(x) for x in g["kappa_b"]]
    LAM, ETA0 = [float(x) for x in g["lambda"]], [0.0] + [float(x) for x in g["eta"]]
    ALPHA = [float(x) for x in g["ridge_alpha"]]
    base = lambda i, ep: dict(cell=ep.name, family=ep.family, support_seed=f"{ep.seed}|i{i}", K=K)
    tabs = {}
    # 1. kappa_f and kappa_b jointly on the fb_C0 score
    rows = []
    for i, ep in ieps:
        inp = Eng.inputs(ep, K)
        for kf in KF:
            fs, fq0 = F.fstar(inp, kf)
            me = (inp["y_s"] - fs).mean(0, keepdim=True)
            for kbv in KB:
                rows.append(dict(base(i, ep), kappa_f=kf, kappa_b=kbv, delta_rmse=F.rmse(fq0 + me * F.alpha(K, kbv), ep.y_q)))
    anc, tabs["fb_C0_kappa_f_kappa_b"] = E2.choose2(rows, ["kappa_f", "kappa_b"], "anchor_iid", tie, "fb_C0_kappa_f_kappa_b", tag, tl)
    kf = float(anc[KS_]["kappa_f"])
    c = dict(kappa_f=kf)
    c["fb_C0"] = dict(kappa_b=anc[KS_]["kappa_b"])
    # 2. ridge alpha on the ridge_C0 score with kappa_b = 16 fixed (v2 rule; ties: larger alpha)
    rms = {i: {al: ridge_fit(ictx[i].st, al) for al in ALPHA} for i in ictx}
    rq_cache = {}
    rows = []
    for n, (i, ep) in enumerate(ieps):
        inp, fs, fq0, E, me = parts(ictx[i], ep, kf)
        for al in ALPHA:
            rq = ridge_pred(rms[i][al], ictx[i].st.FP[torch.as_tensor(ep.g_q)].numpy())
            rq_cache[(n, al)] = rq
            rows.append(dict(base(i, ep), ridge_alpha=al, delta_rmse=F.rmse(rq + me * F.alpha(K, 16.0), ep.y_q)))
    agg, _ = family_mean(pd.DataFrame(rows), by=["K", "ridge_alpha"])
    agg = agg.sort_values(["delta_rmse", "ridge_alpha"], ascending=[True, False])
    tabs["ridge_alpha"] = agg
    s_ = agg.delta_rmse.to_numpy()
    tl.append(dict(partition=tag, grid="ridge_alpha", K=K, n_candidates=len(agg), best_score=float(s_[0]), runner_up_gap=float(s_[1] - s_[0]),
                   n_exact_tie_at_best=int((s_ == s_[0]).sum()), chosen_v1="", chosen_v2=f"ridge_alpha={agg.ridge_alpha.iloc[0]}",
                   selection_changed=False, n_families=int(agg.n_families.iloc[0])))
    alpha = float(agg.ridge_alpha.iloc[0])
    c["ridge_alpha"] = alpha
    # 3. kappa_b per anchor (C1chem arms keep the C0 kappa_b of their anchor, as v2)
    rows = []
    for n, (i, ep) in enumerate(ieps):
        inp, fs, fq0, E, me = parts(ictx[i], ep, kf)
        for anc_, fq in (("fb", fq0), ("ridge", rq_cache[(n, alpha)])):
            for kbv in KB:
                rows.append(dict(base(i, ep), anchor=anc_, kappa_b=kbv, delta_rmse=F.rmse(fq + me * F.alpha(K, kbv), ep.y_q)))
    for anc_ in ("fb", "ridge"):
        kb_, tabs[f"{anc_}_C0_kappa_b"] = E2.choose2([r for r in rows if r["anchor"] == anc_], ["kappa_b"], "kappa_b", tie, f"{anc_}_C0_kappa_b", tag, tl)
        c[f"{anc_}_C0"] = dict(kappa_b=kb_[KS_]["kappa_b"])
    assert c["fb_C0"]["kappa_b"] == anc[KS_]["kappa_b"]
    # 4. kernel scales with the selected kappa_f, then lambda, eta (eta = 0 allowed) per C1chem arm
    for i in ictx:
        ictx[i].fit_scales(kf)
    rows = []
    w = torch.full((K,), 1.0 / K)
    I = torch.eye(K, dtype=torch.float64)
    for n, (i, ep) in enumerate(ieps):
        inp, fs, fq0, E, me = parts(ictx[i], ep, kf)
        for anc_, fq in (("fb", fq0), ("ridge", rq_cache[(n, alpha)])):
            b = me * F.alpha(K, c[f"{anc_}_C0"]["kappa_b"])
            Kc = E2.combine2(F.kernel_blocks(ictx[i].feats(inp, fs, fq)), "chem+ctrl", ictx[i].scales_k[KS_])
            for lam in LAM:
                cc = F.krr(Kc[0], Kc[1], w, E - me, lam, I)
                for eta in ETA0:
                    rows.append(dict(base(i, ep), anchor=anc_, lam=lam, eta=eta, delta_rmse=F.rmse(fq + b + eta * cc, ep.y_q)))
    for anc_ in ("fb", "ridge"):
        k1, tabs[f"{anc_}_C1chem_lam_eta"] = E2.choose2([r for r in rows if r["anchor"] == anc_], ["lam", "eta"], "lam_eta", tie, f"{anc_}_C1chem_lam_eta", tag, tl)
        c[f"{anc_}_C1chem"] = dict(kappa_b=c[f"{anc_}_C0"]["kappa_b"], lam=k1[KS_]["lam"], eta=k1[KS_]["eta"])
    c["inner_scale_info"] = {str(i): dict(n_episodes=ictx[i].scales_k[KS_]["n_episodes"], block=ictx[i].scales_k[KS_]["block"],
                                          comb_chem_ctrl=ictx[i].scales_k[KS_]["comb"]["chem+ctrl"]) for i in ictx}
    return c, tabs


# ---------------------------------------------------------------- one outer-family-fold x drug-fold block
def block(a):
    t0 = time.time()
    sp = spec()
    g, tie, ep_ = sp["grids"], sp["tie_rule"], sp["episodes"]
    P, j = a.partition, int(a.drug_fold)
    IP = INNER_OF[P]
    out = runs(a.stage, a.run_id)
    os.makedirs(out, exist_ok=False)
    head, dirty = git_head()
    log("start", a.stage, P, j, head, "dirty" if dirty else "clean")
    bank = Bank()
    G = bank.groups.drug_group.to_numpy()
    al = Alias(G)
    ofold = np.array([drug_fold(x, 3) for x in G])
    ifold = np.array([inner_fold(x, 3) for x in G])
    H = ofold == j
    Hc = al.close(H)
    A = ~Hc
    checks, scope, elig = [], [], []

    def chk(level, name, n, detail=""):
        checks.append(dict(partition=P, drug_fold=j, level=level, check=name, n_violations=int(n), detail=detail))

    def sc(level, item, fams, groups, response_based, note=""):
        scope.append(dict(partition=P, drug_fold=j, level=level, item=item, response_based=response_based, n_families=len(fams),
                          families_sha=sha_list(fams), n_groups=len(groups), groups_sha=sha_list(groups),
                          n_outer_query_groups_inside=len(set(groups) & set(np.where(Hc)[0].tolist())), note=note))
    # ---- inner contexts and episodes
    ictx, ieps = {}, []
    inner_q, inner_s, inner_fit = set(), set(), set()
    for i in range(3):
        I_ = A & (ifold == i)
        held_i = Hc | al.close(I_)
        ctx = HeldCtx(bank, IP, held_i, f"{P}|d{j}|inner{i}")
        eps, el = build_eps(ctx, I_, held_i, ep_["inner"]["support_seeds"], ep_["inner"]["query_cap"], ep_["min_query"], ep_["min_support"], tag=f"inner{i}")
        for e in el:
            elig.append(dict(e, partition=P, drug_fold=j))
        ctx.episodes("train", (1,), 200)                  # records the kernel-scale data scope (refit later inside fit_scales)
        qg = set(int(x) for e in eps for x in e.g_q)
        sg = set(int(x) for e in eps for x in e.g_s)
        fg = ctx.f0_groups()
        lv = f"inner{i}"
        Ic = set(np.where(al.close(I_))[0].tolist())
        chk(lv, "inner_query_groups_in_f0_table_pca_ridge", len(qg & fg))
        chk(lv, "inner_query_alias_closure_in_f0_table_pca_ridge", len(Ic & fg))
        chk(lv, "inner_query_alias_closure_in_kernel_scale_episodes", len(Ic & ctx.scale_groups))
        chk(lv, "inner_query_alias_closure_in_inner_supports", len(Ic & sg))
        chk(lv, "inner_query_groups_outside_inner_fold", len(qg - set(np.where(I_)[0].tolist())))
        chk(lv, "episodes_with_support_query_group_overlap", sum(bool(set(e.g_s) & set(e.g_q)) for e in eps))
        chk(lv, "inner_val_families_in_inner_train_families", len(ctx.fams("val") & ctx.fams("train")))
        chk(lv, "kernel_scale_families_outside_inner_train", len(ctx.scale_fams - ctx.fams("train")))
        chk(lv, "f0_table_families_outside_inner_train", len(ctx.f0_fams() - ctx.fams("train")))
        sc(lv, "f0_table+f0_pca+ridge", ctx.f0_fams(), fg, "yes", "groups with n_train >= 1 after removal (inner-train families)")
        sc(lv, "ctrl_pca+glob_pca", ctx.fams("train"), [], "no", "label-free control profiles x0 of inner-train cells (all units)")
        sc(lv, "kernel_scale_episodes", ctx.scale_fams, ctx.scale_groups, "indirect (f0 repr of non-held support groups)", "inner-train cells, seed 1, cap 200")
        sc(lv, "selection_episodes", {e.family for e in eps}, qg | sg, "yes", f"{len(eps)} episodes; queries {len(qg)} groups")
        inner_q |= qg
        inner_s |= sg
        inner_fit |= fg | ctx.scale_groups
        ictx[i] = ctx
        ieps += [(i, e) for e in eps]
    Hset = set(np.where(Hc)[0].tolist())
    chk("inner_all", "outer_query_alias_closure_in_inner_queries", len(Hset & inner_q))
    chk("inner_all", "outer_query_alias_closure_in_inner_supports", len(Hset & inner_s))
    chk("inner_all", "outer_query_alias_closure_in_inner_fits", len(Hset & inner_fit))
    inner_fams = ictx[0].fams("train") | ictx[0].fams("val")
    # v2 bug record: the groups v2's inner selection scored (fold-j groups on inner-val cells, seed 1, cap 200)
    v2q = set()
    st0 = ictx[0].st
    for ci in ictx[0].val_cells:
        ui = ictx[0].elig[ci]
        p = np.random.RandomState(880000 + 1000 + int(zlib.crc32(st0.cell_name[ci].encode()) % 100000)).permutation(len(ui))
        q = ui[p[:int(round(0.2 * len(ui)))]]
        v2q |= set(int(x) for x in st0.u_group[q[H[st0.u_group[q]]][:200]])
    chk("v2_contract_reference", "v2_inner_selection_query_groups_equal_to_outer_query_groups", len(v2q & set(np.where(H)[0].tolist())),
        "EXPECTED > 0: documents the v2 leak (not a v3 check)")
    log("inner built", len(ieps), f"{time.time() - t0:.0f}s")
    # ---- outer context and episodes
    octx = HeldCtx(bank, P, Hc, f"{P}|d{j}|outer")
    oeps, oel = build_eps(octx, H, Hc, ep_["outer"]["support_seeds"], ep_["outer"]["query_cap"], ep_["min_query"], ep_["min_support"], tag="outer")
    for e in oel:
        elig.append(dict(e, partition=P, drug_fold=j))
    octx.episodes("train", (1,), 200)
    oq = set(int(x) for e in oeps for x in e.g_q)
    os_ = set(int(x) for e in oeps for x in e.g_s)
    ofg = octx.f0_groups()
    chk("outer", "outer_query_groups_outside_drug_fold", len(oq - set(np.where(H)[0].tolist())))
    chk("outer", "outer_query_alias_closure_in_f0_table_pca_ridge", len(Hset & ofg))
    chk("outer", "outer_query_alias_closure_in_kernel_scale_episodes", len(Hset & octx.scale_groups))
    chk("outer", "outer_query_alias_closure_in_outer_supports", len(Hset & os_))
    chk("outer", "episodes_with_support_query_group_overlap", sum(bool(set(e.g_s) & set(e.g_q)) for e in oeps))
    chk("outer", "outer_val_families_in_outer_train_families", len(octx.fams("val") & octx.fams("train")))
    chk("outer", "inner_families_outside_outer_train_families", len(inner_fams - octx.fams("train")))
    chk("outer", "scored_families_in_any_fit_or_selection_family_set", len({e.family for e in oeps} & (octx.f0_fams() | octx.scale_fams | inner_fams)))
    chk("outer", "kernel_scale_families_outside_outer_train", len(octx.scale_fams - octx.fams("train")))
    sc("outer", "f0_table+f0_pca+ridge", octx.f0_fams(), ofg, "yes", "groups with n_train >= 1 after removal (outer-train families)")
    sc("outer", "ctrl_pca+glob_pca", octx.fams("train"), [], "no", "label-free control profiles x0 of outer-train cells (all units)")
    sc("outer", "kernel_scale_episodes", octx.scale_fams, octx.scale_groups, "indirect (f0 repr of non-held support groups)", "outer-train cells, seed 1, cap 200")
    sc("outer", "scored_episodes_supports", {e.family for e in oeps}, os_, "input (support labels)", "target-family support responses, non-held groups")
    sc("outer", "scored_episodes_queries", {e.family for e in oeps}, oq, "scoring only", f"{len(oeps)} episodes")
    fam_ok = len({e.family for e in oeps if len(e.g_s) >= ep_["min_support"]})
    scored = fam_ok >= ep_["min_families_per_block"]
    el = pd.DataFrame([e for e in elig if e["level"] == "outer"])
    blk = dict(partition=P, drug_fold=j, n_episodes=len(el), n_eligible=int(el.eligible.sum()), n_families_eligible=fam_ok,
               mean_query=float(el[el.eligible].n_query.mean()), mean_support=float(el[el.eligible].n_support.mean()), block_scored=bool(scored),
               n_inner_episodes=len(ieps), n_inner_families=len({e.family for _, e in ieps}),
               n_held_groups=int(H.sum()), n_held_for_fit_groups=int(Hc.sum()), n_alias_added=int(Hc.sum() - H.sum()))
    pd.DataFrame(checks).to_csv(os.path.join(out, "overlap_checks.tsv"), sep="\t", index=False)
    pd.DataFrame(scope).to_csv(os.path.join(out, "fit_scope.tsv"), sep="\t", index=False)
    pd.DataFrame(elig).to_csv(os.path.join(out, "eligibility_episodes.tsv"), sep="\t", index=False)
    dump(blk, os.path.join(out, "block.json"))
    bad = [c for c in checks if c["level"] != "v2_contract_reference" and c["n_violations"] != 0]
    assert not bad, f"overlap check failed: {bad}"
    log("checks ok", blk, f"{time.time() - t0:.0f}s")
    rec = dict(run_id=a.run_id, stage=a.stage, partition=P, drug_fold=j, commit=head, dirty_tree=dirty, slurm_job=os.environ.get("SLURM_JOB_ID", ""),
               host=os.uname().nodename)
    if a.stage == "check":
        dump(dict(rec, seconds=round(time.time() - t0, 1)), os.path.join(out, "result.json"))
        return
    # ---- selection (inner contract only)
    tl = []
    c, tabs = select(ictx, ieps, g, tie, tl, f"{P}|d{j}")
    c.update(partition=P, drug_fold=j, inner_partition=IP, n_inner_episodes=len(ieps), n_inner_families=len({e.family for _, e in ieps}),
             selection_scope="inner-val families of " + IP + "; queries = outer-allowed groups of the inner fold; fits without outer query groups")
    dump(c, os.path.join(out, "selected_constants.json"))
    pd.DataFrame(tl).to_csv(os.path.join(out, "tie_log.tsv"), sep="\t", index=False, float_format="%.17g")
    with open(os.path.join(out, "selection_grids.tsv"), "w") as f:
        pd.concat([t.assign(grid=k) for k, t in tabs.items()]).to_csv(f, sep="\t", index=False, float_format="%.17g")
    log("selected", json.dumps({k: v for k, v in c.items() if k != "inner_scale_info"}), f"{time.time() - t0:.0f}s")
    del ictx, ieps
    if a.stage == "pilot" or not scored:
        dump(dict(rec, seconds=round(time.time() - t0, 1), scored=False, reason="pilot (selection only)" if a.stage == "pilot" else "eligibility rule failed"),
             os.path.join(out, "result.json"))
        return
    # ---- outer: refit with the selected constants on the outer-allowed scope, score once
    rm = ridge_fit(octx.st, c["ridge_alpha"])
    octx.fit_scales(c["kappa_f"])
    rows, meta, truth, preds = [], [], [], {m: [] for m in ARMS}
    for n, ep in enumerate(oeps):
        b = dict(partition=P, drug_fold=j, cell=ep.name, family=ep.family, support_seed=f"{ep.seed}|d{j}", K=K)
        yq = ep.y_q.numpy()
        for m, p in arm_preds(octx, ep, c, rm):
            rows.append(dict(b, method=m, **F.score(p.numpy(), yq)))
            preds[m].append(p.numpy().astype(np.float32))
        truth.append(yq.astype(np.float32))
        for u, gq in zip(ep.qu, ep.g_q):
            meta.append(dict(episode=n, partition=P, drug_fold=j, cell=ep.name, family=ep.family, seed=ep.seed, unit_idx=int(u), group_idx=int(gq), drug_group=G[gq]))
    pd.DataFrame(rows).to_csv(os.path.join(out, "episodes.tsv.gz"), sep="\t", index=False, float_format="%.8g")
    pd.DataFrame(meta).to_csv(os.path.join(out, "pred_meta.tsv.gz"), sep="\t", index=False)
    np.savez(os.path.join(out, "preds.npz"), truth=np.concatenate(truth), **{m: np.concatenate(v) for m, v in preds.items()})
    files = {f: dict(sha256=sha256(os.path.join(out, f)), bytes=os.path.getsize(os.path.join(out, f))) for f in ("preds.npz", "pred_meta.tsv.gz", "episodes.tsv.gz")}
    dump(dict(rec, seconds=round(time.time() - t0, 1), scored=True, n_episodes=len(oeps), n_queries=len(meta), files=files), os.path.join(out, "result.json"))
    log("done", f"{time.time() - t0:.0f}s")


def write_splits(out):
    """Metadata only (cells.tsv, groups.tsv, alias table): outer / inner family and drug splits."""
    d = os.environ["R8_PROC"]
    cells = pd.read_csv(os.path.join(d, "cells.tsv"), sep="\t")
    G = pd.read_csv(os.path.join(d, "groups.tsv"), sep="\t").drug_group.to_numpy()
    al = Alias(G)
    from src.r8_followup_v1.partitions import inner_val_family
    fam = cells.drop_duplicates("family")
    ofold = np.array([drug_fold(x, 3) for x in G])
    ifold = np.array([inner_fold(x, 3) for x in G])
    tab = pd.DataFrame(dict(drug_group=G, outer_drug_fold=ofold, inner_drug_fold_raw=ifold))
    outer = dict(rule_family="outer{k}: val = cv_fold k (meta_role train); train = other meta-train families; meta-val families off",
                 rule_drug="crc32('r8v2_drug|' + drug_group) % 3 (v2)", alias_table=ALIAS, alias_sha256=sha256(ALIAS), folds={})
    inner = dict(rule_family="inner{k}: meta-train families with cv_fold != k; inner-val = crc32('r8f_inner|' + family) % 3 == 0",
                 rule_drug="allowed A = not closure(outer fold j); inner fold i = A & crc32('r8v3_inner_drug|' + group) % 3 == i", blocks={})
    for k in range(3):
        tr = fam[(fam.meta_role == "train") & (fam.cv_fold != k)].family
        outer["folds"][f"outer{k}"] = dict(val_families=sorted(fam[(fam.meta_role == "train") & (fam.cv_fold == k)].family),
                                           train_families=sorted(tr))
        iv = sorted(f for f in tr if inner_val_family(f))
        inner["blocks"][f"inner{k}"] = dict(val_families=iv, train_families=sorted(set(tr) - set(iv)))
    for j in range(3):
        H = ofold == j
        Hc = al.close(H)
        outer[f"drug_fold_{j}"] = dict(n_query_groups=int(H.sum()), query_groups_sha=sha_list(G[H]), n_held_for_fit=int(Hc.sum()),
                                       alias_added=sorted(G[Hc & ~H].tolist()))
        A = ~Hc
        tab[f"inner_fold_given_outer_d{j}"] = np.where(A, ifold, -1)
        for i in range(3):
            I_ = A & (ifold == i)
            inner[f"outer_d{j}_inner{i}"] = dict(n_query_groups=int(I_.sum()), query_groups_sha=sha_list(G[I_]),
                                                 n_held_for_fit=int((Hc | al.close(I_)).sum()), overlap_with_outer_held=int((I_ & Hc).sum()))
    dump(outer, os.path.join(out, "outer_splits.json"))
    dump(inner, os.path.join(out, "inner_splits.json"))
    tab.to_csv(os.path.join(out, "drug_folds.tsv.gz"), sep="\t", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["alias", "splits", "check", "pilot", "run"])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--partition", default="")
    ap.add_argument("--drug-fold", default="")
    a = ap.parse_args()
    if a.stage == "alias":
        o = runs("alias", a.run_id)
        os.makedirs(o, exist_ok=False)
        build_alias(o)
    elif a.stage == "splits":
        o = runs("splits", a.run_id)
        os.makedirs(o, exist_ok=False)
        write_splits(o)
    else:
        for P_, j_ in ([(a.partition, a.drug_fold)] if a.drug_fold != "" else [(a.partition, x) for x in (0, 1, 2)]):
            a2 = argparse.Namespace(**vars(a))
            a2.partition, a2.drug_fold = P_, j_
            if a.drug_fold == "":
                a2.run_id = f"{a.run_id}_d{j_}"
            block(a2)
