"""Synthetic fixtures. Every check is an assert; the JSON written at the end lists what was checked.

  f0          family-balanced mean and leave-family-out exclusion against brute force
  anchor      Omega = I reproduces the plain mean and K_eff = K; the Cholesky solve equals the explicit inverse; Omega is positive definite
  one_plate   a fully shared plate (Omega = a I + c 11') leaves the centred residual regression equal to Omega = a I
  no_label    no prediction changes when the query labels change
  global_path a common shift of every control profile (a change of the global state only)
              - leaves the within-only model unchanged (the erased path of the predecessor),
              - is erased exactly by 'linear additive input then support centring' and by 'one scalar multiplier then LayerNorm' (the two traps),
              - changes the centred support kernel and the prediction of the per-dimension multiplicative model
"""
import json
import numpy as np
import pandas as pd
import torch

from .common import N_GENES, public, dump
from .data import Store, Features, dummy_context
from . import methods as M
from .arms import ARMS, Context


def synthetic(seed=0, n_groups=120):
    rs = np.random.RandomState(seed)
    fam = ["F0", "F0", "F0", "F1", "F1", "F2", "F3", "F4", "F5", "V0", "V1", "V2"]
    cells = pd.DataFrame(dict(cell_idx=range(12), cell=[f"C{i}" for i in range(12)], family=fam, meta_role=["train"] * 9 + ["val"] * 3))
    groups = pd.DataFrame(dict(group_idx=range(n_groups), drug_group=[f"G{i}" for i in range(n_groups)], rep_drug_idx=range(n_groups)))
    rows = []
    for c in range(12):
        for g in np.sort(rs.choice(n_groups, size=rs.randint(60, 100), replace=False)):
            rows.append((c, g, f"P{c}_{rs.randint(3)}"))
    units = pd.DataFrame(rows, columns=["cell_idx", "group_idx", "plates"])
    units.insert(0, "unit_idx", range(len(units)))
    units["cell"] = units.cell_idx.map(cells.cell)
    delta = rs.randn(len(units), N_GENES).astype(np.float32)
    x0 = (8 + rs.randn(len(units), N_GENES)).astype(np.float32)
    fp = (rs.rand(n_groups, 2048) < 0.03).astype(np.float32)
    return units, groups, cells, delta, x0, fp


def main():
    torch.manual_seed(0)
    units, groups, cells, delta, x0, fp = synthetic()
    st = Store.from_arrays(units, groups, cells, delta, x0, fp)
    checks = {}

    # ---- f0
    fam_cells = {f: g.cell_idx.to_numpy() for f, g in cells.groupby("family")}
    train_fams = [f for f in fam_cells if (cells.meta_role[fam_cells[f]] == "train").all()]

    def brute(allowed_fams, g):
        v = [delta[np.isin(st.u_cell, fam_cells[f]) & (st.u_group == g)].astype(np.float64).mean(0) for f in allowed_fams
             if (np.isin(st.u_cell, fam_cells[f]) & (st.u_group == g)).any()]
        return (np.mean(v, 0) if v else np.zeros(N_GENES)), len(v)
    for fam in train_fams:
        for g in np.random.RandomState(1).choice(st.G, 10, replace=False):
            ref, cnt = brute([f for f in train_fams if f != fam], g)
            f0, n = st.f0("lfo", [g], fam, allow_missing=True)
            assert int(n[0]) == cnt and np.allclose(f0[0].numpy(), ref, atol=2e-4)
    for g in range(10):
        ref, cnt = brute(train_fams, g)
        f0, n = st.f0("train", [g], allow_missing=True)
        assert int(n[0]) == cnt and np.allclose(f0[0].numpy(), ref, atol=2e-4)
        ref, cnt = brute(list(fam_cells), g)
        f0, n = st.f0("hcc", [g], allow_missing=True)
        assert int(n[0]) == cnt and np.allclose(f0[0].numpy(), ref, atol=2e-4)
    # a three-cell family counts once: family-balanced differs from the cell-equal mean
    g = next(g for g in range(st.G) if (np.isin(st.u_cell, fam_cells["F0"]) & (st.u_group == g)).sum() == 3 and st.n_train[g] >= 3)
    cell_equal = delta[np.isin(st.u_cell, np.where(st.cell_role == "train")[0]) & (st.u_group == g)].mean(0)
    assert not np.allclose(st.f0("train", [g])[0][0].numpy(), cell_equal, atol=1e-3)
    checks["f0"] = "family-balanced mean, leave-family-out and HCC regimes equal brute force; multi-cell family counted once"

    # ---- anchor / solver
    fmap = Features.fit(st)
    ep = st.eval_episode(st.eval_cells("val")[0], 1)
    cov = dict(kind="cov", v_r=0.5, v_e=0.3, v_s=0.2)
    sel = dict(anchor=dict(iid={str(k): dict(kappa_f=2.0, kappa_b=8.0) for k in (8, 16, 32)}, cov={str(k): dict(kappa_f=2.0, kappa_b=8.0) for k in (8, 16, 32)}),
               c1=dict(iid={str(k): dict(lam=1.0, eta=1.0) for k in (8, 16, 32)}, cov={str(k): dict(lam=1.0, eta=1.0) for k in (8, 16, 32)}),
               shrunk_same_drug_mean={str(k): dict(kappa_f=2.0) for k in (8, 16, 32)}, covariance_model=cov)
    ctx = Context(fmap, st.FP, sel)
    inp = ep.inputs(16)
    a = M.anchor(inp, 2.0, 8.0, None)
    fs = inp["f0_s"] * (inp["n0_s"] / (inp["n0_s"] + 2.0))[:, None]
    assert abs(a["keff"] - 16) < 1e-9 and torch.allclose(a["ebar"], (inp["y_s"] - fs).mean(0, keepdim=True), atol=1e-5) and abs(a["shrink"] - 16 / 24) < 1e-9
    ac = M.anchor(inp, 2.0, 8.0, cov)
    Om = ac["Om"]
    assert torch.linalg.eigvalsh(Om).min() > M.FLOOR * 0.99 and torch.allclose(Om, Om.T)
    inv = torch.linalg.inv(Om)
    one = torch.ones(16, dtype=torch.float64)
    assert abs(ac["keff"] - float(one @ inv @ one)) < 1e-8
    assert torch.allclose(ac["ebar"][0].double(), (one @ inv @ ac["E"].double()) / (one @ inv @ one), atol=1e-5)
    Kss = torch.randn(16, 16, dtype=torch.float64)
    Kss = Kss @ Kss.T
    Kqs = torch.randn(5, 16, dtype=torch.float64)
    R = torch.randn(16, 7, dtype=torch.float64)
    assert torch.allclose(M.gls_solve(Kss, Kqs, R, 0.7, Om), Kqs @ torch.linalg.inv(Kss + 0.7 * Om) @ R, atol=1e-8)
    assert ac["keff"] < 16.0                        # single-row units sharing plates count for fewer than K independent observations
    checks["anchor"] = "Omega=I gives the plain mean, K_eff=K and shrink K/(K+kappa); Cholesky solve equals the explicit inverse; Omega symmetric positive definite"

    # ---- one fully shared plate: Omega = a I + c 11'  ->  centred regression equals Omega = a I
    a_, c_ = 0.8, 0.5
    Om1 = a_ * torch.eye(16, dtype=torch.float64) + c_ * torch.ones(16, 16, dtype=torch.float64)
    w = torch.full((16,), 1 / 16, dtype=torch.float64)
    Kc, Kqc = M.centre_kernel(Kss, Kqs, w)
    Rc = R - R.mean(0, keepdim=True)
    assert torch.allclose(M.gls_solve(Kc, Kqc, Rc, 0.7, Om1), M.gls_solve(Kc, Kqc, Rc, 0.7, a_ * torch.eye(16, dtype=torch.float64)), atol=1e-7)
    checks["one_plate"] = "with every support on one plate the shared component changes only the mean-correction shrinkage and the ridge scale"

    # ---- no query label reaches a prediction
    models = {arm: M.StateAdapter(state=ARMS[arm]["state"], film_init=0.3) for arm in ARMS}
    cnp = M.GenericCNP()
    import copy
    ep2 = copy.copy(ep)
    ep2.y_q = ep.y_q + 3.0
    with torch.no_grad():
        for K in (8, 16, 32):
            i1, i2 = ep.inputs(K), ep2.inputs(K)
            assert "y_q" not in i1
            for obs in ("iid", "cov"):
                assert torch.equal(ctx.c0(i1, obs), ctx.c0(i2, obs)) and torch.equal(ctx.c1(i1, obs), ctx.c1(i2, obs))
            for arm, m in models.items():
                assert torch.equal(ctx.adapter(m, arm, i1), ctx.adapter(m, arm, i2))
            assert torch.equal(ctx.cnp(cnp, i1), ctx.cnp(cnp, i2))
    checks["no_label"] = "C0, C1, every adapter arm and the CNP are unchanged when the query labels change"

    # ---- global path
    shift = torch.randn(1, N_GENES) * 2.0
    inp_b = dict(inp)
    inp_b["x0_s"], inp_b["x0_q"] = inp["x0_s"] + shift, inp["x0_q"] + shift            # the same vector added to every control profile
    with torch.no_grad():
        anc = ctx.anchor(inp, "iid")
        fa, fb = ctx.features(inp, anc), ctx.features(inp_b, anc)
        assert torch.allclose(fa["ctrl_s"], fb["ctrl_s"], atol=1e-4) and torch.allclose(fa["ctrl_q"], fb["ctrl_q"], atol=1e-4), "within block must not see a common shift"
        assert (fa["glob"] - fb["glob"]).abs().max() > 0.1, "global block must see a common shift"
        d_none = (ctx.adapter(models["B00"], "B00", inp) - ctx.adapter(models["B00"], "B00", inp_b)).abs().max().item()
        assert d_none < 1e-3, "within-only model changed under a common shift"

        def centred_kernel(m, ft):
            z = m.embed(ft, "s")
            z = z - z.mean(0, keepdim=True)
            return z @ z.T
        m = models["B01"]
        dk = (centred_kernel(m, fa) - centred_kernel(m, fb)).abs().max().item()
        dp = (ctx.adapter(m, "B01", inp) - ctx.adapter(m, "B01", inp_b)).abs().max().item()
        assert dk > 1e-3 and dp > 1e-3, "multiplicative global conditioning was erased"
        # zero-initialised gamma starts exactly at the within-only model (same weights)
        m0 = M.StateAdapter(state="film")
        mn = M.StateAdapter(state="none")
        mn.load_state_dict({k: v for k, v in m0.state_dict().items() if not k.startswith("gamma")})
        assert torch.allclose(ctx.adapter(m0, "B01", inp), ctx.adapter(mn, "B00", inp), atol=1e-5)
        # trap 1: linear additive global input followed by support centring is erased exactly
        Wd, Wg = torch.randn(40, 64).double(), torch.randn(16, 64).double()
        d = torch.randn(16, 40).double()
        za, zb = d @ Wd + fa["glob"].double() @ Wg, d @ Wd + fb["glob"].double() @ Wg
        assert torch.allclose(za - za.mean(0), zb - zb.mean(0), atol=1e-6)
        # trap 2: one scalar multiplier for every dimension followed by LayerNorm is erased
        ln = torch.nn.LayerNorm(64)
        h = (d @ Wd).float()
        assert torch.allclose(ln(h * 1.0), ln(h * 1.7), atol=1e-4)
        # dummy context: deterministic per cell, different between cells, independent of the control profiles
        fd_a, fd_b = ctx.features(inp, anc, dummy=True), ctx.features(inp_b, anc, dummy=True)
        assert torch.equal(fd_a["glob"], fd_b["glob"]) and not np.allclose(dummy_context("C1", 16), dummy_context("C2", 16))
    checks["global_path"] = dict(within_only_max_change=d_none, film_centred_kernel_max_change=dk, film_prediction_max_change=dp,
                                 note="common control shift: within-only unchanged; additive-linear+centring and scalar+LayerNorm erased; per-dimension FiLM not erased; "
                                      "zero-initialised FiLM equals the within-only model")
    n_par = {arm: int(sum(p.numel() for p in m.parameters())) for arm, m in models.items()}
    n_par["cnp"] = int(sum(p.numel() for p in cnp.parameters()))
    checks["n_parameters"] = n_par
    dump(checks, public("manifests", "replicate_calibrated_r8", "fixtures.json"))
    print(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()
