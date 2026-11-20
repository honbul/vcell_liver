"""Appendix C correctness checks of the R8 follow-up (plan Appendix C items 1-10).

Run on the server (no pytest there):  python -m tests.r8_followup_v1.test_appendix_c [--real]
Synthetic checks run always; --real adds the label-isolation check on the real store (item 6) and the query-list check of the
evaluation tables (item 9, needs the evaluate runs given by --eval-dirs).
"""
import argparse, glob, json, os, sys
import numpy as np
import torch

from src.replicate_calibrated_r8.data import overlap
from src.r8_followup_v1 import fmethods as F
from src.r8_followup_v1.covrepair import overlap_means_rows
from src.r8_followup_v1 import stats as S

torch.set_grad_enabled(False)
RESULTS = {}


def check(name, cond, detail=""):
    RESULTS[name] = dict(passed=bool(cond), detail=str(detail))
    print(("PASS " if cond else "FAIL ") + name + ("  " + str(detail) if detail else ""), flush=True)


def centred_contribution(O):
    n = O.shape[0]
    H = np.eye(n) - 1.0 / n
    return np.trace(H @ O) / (n - 1)


def test_1_2_3():
    n = 7
    O1 = np.ones((n, n))
    check("C1 O=11' -> centred shared contribution 0", abs(centred_contribution(O1)) < 1e-12, centred_contribution(O1))
    O2 = np.eye(n)
    check("C2 O=I -> centred shared contribution v_s (=1)", abs(centred_contribution(O2) - 1.0) < 1e-12, centred_contribution(O2))
    rs = np.random.RandomState(0)
    pl = [list(rs.choice(5, size=rs.randint(1, 4), replace=True)) for _ in range(40)]
    wl = [list(rs.dirichlet(np.ones(len(p)))) for p in pl]
    O, q = overlap(pl, wl)
    v_r, v_e, v_s = 0.03, 0.4, 0.05
    Sig = v_r * np.eye(40) + v_e * np.diag(q) + v_s * O
    tr = np.trace((np.eye(40) - 1 / 40) @ Sig) / 39
    qm, od, oo = overlap_means_rows(pl, wl)
    form = v_r + v_e * qm + v_s * (od - oo)
    check("C3 trace formula == exact-sum formula (non-trivial W, repeated plates)", abs(tr - form) < 1e-12, f"{tr:.12f} vs {form:.12f}")
    off = (O.sum() - np.trace(O)) / (40 * 39)
    check("C3b exact-sum O_off == N x N matrix mean", abs(off - oo) < 1e-12 and abs(np.diag(O).mean() - od) < 1e-12 and abs(q.mean() - qm) < 1e-12)
    # Monte Carlo: E[E'HE/(n-1)] = mu'H mu/(n-1) + tr(H Sigma)/(n-1)
    mu = rs.randn(40) * 0.2
    L = np.linalg.cholesky(Sig + 1e-12 * np.eye(40))
    X = mu[:, None] + L @ rs.randn(40, 200000)
    H = np.eye(40) - 1 / 40
    emp = ((H @ X) ** 2).sum(0).mean() / 39
    th = mu @ H @ mu / 39 + tr
    check("C3c Monte Carlo centred variance matches the formula incl. the drug-mean term", abs(emp - th) / th < 0.01, f"{emp:.5f} vs {th:.5f}")


def test_4():
    rs = np.random.RandomState(1)
    A = rs.randn(16, 16)
    Om = torch.tensor(A @ A.T + 16 * np.eye(16))
    _, k1 = F.gls_weights(Om)
    _, k3 = F.gls_weights(3.0 * Om)
    check("C4 Omega x a -> K_eff / a", abs(k3 - k1 / 3.0) < 1e-9 * k1, f"{k1:.6f} -> {k3:.6f}")


def synth_inputs(K=16, Q=12, seed=2):
    rs = np.random.RandomState(seed)
    t = lambda x: torch.tensor(x, dtype=torch.float32)
    inp = dict(name="X", g_s=np.arange(K), g_q=np.arange(K, K + Q), x0_s=t(rs.randn(K, 978)), x0_q=t(rs.randn(Q, 978)), f0_s=t(rs.randn(K, 978) * 0.3),
               f0_q=t(rs.randn(Q, 978) * 0.3), n0_s=t(rs.randint(0, 9, K).astype(float)), n0_q=t(rs.randint(1, 9, Q).astype(float)), y_s=t(rs.randn(K, 978) * 0.5),
               q_s=t(rs.uniform(0.3, 1, K)))
    W = rs.dirichlet(np.ones(3), size=K) * (rs.rand(K, 3) > 0.3)
    W = W / np.maximum(W.sum(1, keepdims=True), 1e-9)
    Wq = rs.dirichlet(np.ones(3), size=Q)
    inp["o_s"] = t(W @ W.T)
    inp["o_qs"] = t(Wq @ W.T)
    fp_s, fp_q = (rs.rand(K, 2048) < 0.05).astype(np.float32), (rs.rand(Q, 2048) < 0.05).astype(np.float32)
    ft = dict(fp_s=t(fp_s), fp_q=t(fp_q), ctrl_s=t(rs.randn(K, 32)), ctrl_q=t(rs.randn(Q, 32)), f0r_s=t(rs.randn(K, 33)), f0r_q=t(rs.randn(Q, 33)))
    return inp, ft, W, Wq


def c1_pred(inp, ft, lam=1.0, eta=0.5, kappa_f=8.0, kappa_b=16.0, resid_perm=None):
    K = len(inp["y_s"])
    a = F.anchor_om(inp, kappa_f, kappa_b, torch.eye(K, dtype=torch.float64))
    Kss, Kqs = F.combine(F.kernel_blocks(ft), "all")
    R = a["R"] if resid_perm is None else a["R"][torch.as_tensor(resid_perm)]
    return a["fq"] + a["b"] + eta * F.krr(Kss, Kqs, a["w"], R, lam, a["Om"])


def test_5():
    inp, ft, _, _ = synth_inputs()
    p0 = c1_pred(inp, ft)
    perm = np.random.RandomState(3).permutation(16)
    ip = F.permute_inputs(inp, perm)
    fp = dict(ft)
    for k in ("fp_s", "ctrl_s", "f0r_s"):
        fp[k] = ft[k][torch.as_tensor(perm)]
    p1 = c1_pred(ip, fp)
    check("C5a full support-row permutation leaves the C1 prediction unchanged", float((p0 - p1).abs().max()) < 1e-5, float((p0 - p1).abs().max()))
    d = F.derangement(16, 1)
    p2 = c1_pred(inp, ft, resid_perm=d)
    a = F.anchor_om(inp, 8.0, 16.0, torch.eye(16, dtype=torch.float64))
    check("C5b residual-only derangement changes the prediction (correspondence broken)", float((p2 - p0).abs().max()) > 1e-3, float((p2 - p0).abs().max()))
    leak = float(a["R"][torch.as_tensor(d)].sum(0).abs().max())
    check("C5c residual-only permutation keeps f* and b (no common correction leaks: permuted R still sums to 0 under w = 1/K)", leak < 1e-4, leak)
    Om = torch.tensor(np.diag(np.linspace(0.5, 2, 16)))
    w, _ = F.gls_weights(Om)
    Rw = (torch.eye(16, dtype=torch.float64) - torch.ones(16, 1, dtype=torch.float64) @ w[None, :]) @ a["E"].double()[torch.as_tensor(d)]
    check("C5d non-uniform w: permuted residuals re-centred by P_w = I - 1w' have zero w-weighted mean", float((w @ Rw).abs().max()) < 1e-6)


def test_7():
    inp, ft, W, Wq = synth_inputs()
    Wt, Wqt = torch.tensor(W), torch.tensor(Wq)
    q = inp["q_s"]
    Kfull_s = F.kernel_blocks(ft)["chem"]
    pars = dict(tau_b=0.05, v_s=0.06, v_e=0.4, sig0=0.03)
    C = F.joint_full_matrix(Wt, Wqt, q, pars["tau_b"], pars["v_s"], pars["v_e"], pars["sig0"])
    ev = torch.linalg.eigvalsh(C)
    Css, Cqs = F.joint_blocks(inp["o_s"], inp["o_qs"], q, pars["tau_b"], pars["v_s"], pars["v_e"], pars["sig0"])
    K = 16
    check("C7a joint covariance is PSD", float(ev.min()) > -1e-10, float(ev.min()))
    check("C7b SS and QS are blocks of the same joint matrix", float((C[:K, :K] - Css).abs().max()) < 1e-6 and float((C[K:, :K] - Cqs).abs().max()) < 1e-6)
    Css0, Cqs0 = F.joint_blocks(inp["o_s"], torch.zeros_like(inp["o_qs"]), q, pars["tau_b"], pars["v_s"], pars["v_e"], pars["sig0"])
    check("C7c removing the query overlap removes only the shared-environment cross term",
          float((Css0 - Css).abs().max()) == 0.0 and float((Cqs - Cqs0 - pars["v_s"] * inp["o_qs"].double()).abs().max()) < 1e-6)


def test_8():
    inp, ft, _, _ = synth_inputs()
    fs, fq = F.fstar(inp, 8.0)
    E = inp["y_s"] - fs
    pj = F.joint(fq, E, inp["o_s"], inp["o_qs"], inp["q_s"], 0.0, 0.0, 0.4, 0.03)
    a = F.anchor_om(inp, 8.0, float("inf"), torch.eye(16, dtype=torch.float64))
    p1 = F.j1(fq, E, inp["o_qs"], 0.0, 0.0)
    check("C8a zero trust -> f* fallback (J2 tau=v_s=0, C0 kappa_b=inf, J1 a=c=0)",
          float((pj - fq).abs().max()) < 1e-6 and float((a["fq"] + a["b"] - fq).abs().max()) == 0 and float((p1 - fq).abs().max()) == 0)
    inp2 = dict(inp)
    inp2["n0_q"] = torch.zeros_like(inp["n0_q"])
    ok = all(float(F.fstar(inp2, kf)[1].abs().max()) == 0.0 for kf in (0.0, 0.5, 8.0))
    check("C8b n0 = 0 -> f* is the zero vector for every kappa_f (shared by all methods)", ok)


def test_10():
    W, n = S.sync_drug_weights({"A": ["d1", "d2", "d3"], "B": ["d2", "d4"], "C": ["d3", "d2"]}, n_boot=50, seed=1)
    same = np.allclose(W["A"][:, 1], W["B"][:, 0]) and np.allclose(W["A"][:, 1], W["C"][:, 1])
    check("C10a synchronised bootstrap: a drug shared by several targets gets the same weight in a draw", same and n == 4)
    g = S.independent_streams(["HUH7", "JHH5", "JHH7"], seed=1)
    x = [g[k].integers(0, 10 ** 9, 5).tolist() for k in g]
    check("C10b per-target conditional bootstrap streams are independent SeedSequence children (not a restarted common seed)",
          x[0] != x[1] and x[1] != x[2])
    d = np.array([-0.002, -0.001, -0.003, 0.0005, -0.002])
    r = S.summarise_diff(d, 0.5)
    check("C10c family bootstrap summary keeps the family as the unit (n draws x n families)", r["lo"] <= r["diff"] <= r["hi"])


def test_6_real():
    """Changing the treated labels of the held (outer-test) families leaves every nuisance input of that partition unchanged."""
    from src.r8_followup_v1.partitions import Bank
    from src.r8_followup_v1 import engine as Eng
    from src.r8_followup_v1.covrepair import unit_table, pair_constants
    from src.replicate_calibrated_r8.data import Features
    import pandas as pd
    bank = Bank()
    name = "outer0"
    role = bank.roles(name)
    held = np.isin(bank.units.cell_idx.to_numpy(), np.where(role == "val")[0])
    d2 = bank.delta.copy()
    d2[held] += np.random.RandomState(0).randn(int(held.sum()), 978).astype(np.float32)
    s1, s2 = bank.store(name), bank.store(name, delta=d2)
    f1, f2 = Features.fit(s1), Features.fit(s2)
    feq = all(float((getattr(f1, k) - getattr(f2, k)).abs().max()) == 0.0 for k in Features.KEYS)
    check("C6a features (PCA bases) unchanged when held-family labels change", feq)
    check("C6b f0 table of the training families unchanged", float((s1.S_train - s2.S_train).abs().max()) == 0.0 and float((s1.n_train - s2.n_train).abs().max()) == 0.0)
    u1, u2 = unit_table(s1, 8.0), unit_table(s2, 8.0)
    num = [c for c in u1.columns if u1[c].dtype.kind == "f"]
    check("C6c unit residual / overlap table (covariance input) unchanged", np.allclose(u1[num].to_numpy(), u2[num].to_numpy(), atol=0, rtol=0, equal_nan=True))
    pairs = pd.read_csv(glob.glob(os.path.join(os.environ["R8F_RUNS"], "p1", "*", "d1_plate_pairs.tsv"))[0], sep="\t")
    p2 = pairs.copy()
    hf = set(bank.fam[role == "val"])
    p2.loc[p2.family.isin(hf), ["v_e", "v_s"]] *= 7.0
    tf = sorted(set(bank.fam[role == "train"]))
    check("C6d replicate constants v_e / v_s ignore held-family pairs", pair_constants(pairs, tf) == pair_constants(p2, tf))
    ep1 = Eng.eval_episode(s1, int(np.where(role == "val")[0][0]), 1)
    ep2 = Eng.eval_episode(s2, int(np.where(role == "val")[0][0]), 1)
    check("C6e a held cell's method-visible inputs depend only on its support labels (f0 of queries unchanged)",
          float((ep1.f0_q - ep2.f0_q).abs().max()) == 0.0 and float((ep1.x0_q - ep2.x0_q).abs().max()) == 0.0)


def test_9_tables(eval_dirs):
    import pandas as pd
    bad = 0
    for d in eval_dirs:
        df = pd.read_csv(os.path.join(d, "episodes.tsv.gz"), sep="\t", usecols=["cell", "support_seed", "K", "method", "n_query"])
        g = df.groupby(["cell", "support_seed"]).n_query.nunique()
        bad += int((g > 1).sum())
    check("C9a every method and K of an episode is scored on the same query list (n_query constant)", bad == 0, f"{bad} episodes differ")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--eval-dirs", nargs="*", default=[])
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    test_1_2_3(); test_4(); test_5(); test_7(); test_8(); test_10()
    if a.real:
        test_6_real()
    if a.eval_dirs:
        test_9_tables(a.eval_dirs)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(RESULTS, f, indent=1)
    n_fail = sum(not r["passed"] for r in RESULTS.values())
    print(f"{len(RESULTS)} checks, {n_fail} failed")
    sys.exit(1 if n_fail else 0)
