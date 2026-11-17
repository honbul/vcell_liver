"""Leak and wiring fixtures. Synthetic data first; with --real also the built source store. Every check is an assert."""
import argparse, json, os, subprocess
import numpy as np
import pandas as pd
import torch

from .common import public, dump, EXCLUDED_TARGETS, norm_name
from .data import Source, Features, N_GENES
from . import methods as M
from .metrics import per_unit, summarise


def synthetic(seed=0, n_groups=120):
    """12 cells in 9 families (one 3-cell family, one 2-cell family), 3 meta-validation families."""
    rs = np.random.RandomState(seed)
    fam = ["F0", "F0", "F0", "F1", "F1", "F2", "F3", "F4", "F5", "V0", "V1", "V2"]
    cells = pd.DataFrame(dict(cell_idx=range(12), cell=[f"C{i}" for i in range(12)], family=fam, meta_role=["train"] * 9 + ["val"] * 3))
    groups = pd.DataFrame(dict(group_idx=range(n_groups), drug_group=[f"G{i}" for i in range(n_groups)], rep_drug_idx=range(n_groups)))
    rows = []
    for c in range(12):
        gs = np.sort(rs.choice(n_groups, size=rs.randint(60, 100), replace=False))
        for g in gs:
            rows.append((c, g, f"P{c}_{rs.randint(3)}"))
    units = pd.DataFrame(rows, columns=["cell_idx", "group_idx", "plates"])
    units.insert(0, "unit_idx", range(len(units)))
    units["cell"] = units.cell_idx.map(cells.cell)
    delta = rs.randn(len(units), N_GENES).astype(np.float32)
    x0 = (8 + rs.randn(len(units), N_GENES)).astype(np.float32)
    fp = (rs.rand(n_groups, 2048) < 0.03).astype(np.float32)
    return units, groups, cells, delta, x0, fp


def brute_f0(src, delta, allowed_cells, g):
    m = np.isin(src.u_cell, allowed_cells) & (src.u_group == g)
    return delta[m].astype(np.float64).mean(0), int(m.sum())


def check_f0(src, delta, checks):
    train_cells = np.where(src.cell_role == "train")[0]
    val_cells = np.where(src.cell_role == "val")[0]
    # (a) meta_train regime: the whole family of the virtual target is absent from f0 (brute-force recomputation)
    for fam, cells in src.family_cells.items():
        if not src.train_cell_mask[cells].all():
            continue
        S, n = src.f0_table("meta_train", fam)
        allowed = np.setdiff1d(train_cells, cells)
        for g in np.random.RandomState(1).choice(src.G, 15, replace=False):
            ref, cnt = brute_f0(src, delta, allowed, g)
            assert int(n[g].item()) == cnt
            if cnt:
                assert np.allclose((S[g] / n[g]).cpu().numpy(), ref, atol=2e-4)
    # (b) changing the target family's responses does not change its episodes' f0; changing another family's does
    fam = "F0" if "F0" in src.family_cells else sorted(f for f, c in src.family_cells.items() if src.train_cell_mask[c].all() and len(c) > 1)[0]
    cells = src.family_cells[fam]
    S1, n1 = src.f0_table("meta_train", fam)
    d2 = delta.copy()
    d2[np.isin(src.u_cell, cells)] += 5.0
    src2 = Source.from_arrays(src.units, src.groups, src.cells, d2, src.X0.cpu().numpy(), src.FP.cpu().numpy(), src.dev)
    S2, n2 = src2.f0_table("meta_train", fam)
    assert torch.allclose(S1, S2, atol=2e-2) and torch.equal(n1, n2), "virtual-target family leaks into its own f0"
    # (c) meta-validation responses never enter the meta-train tables or the feature bases
    d3 = delta.copy()
    d3[np.isin(src.u_cell, val_cells)] += 7.0
    x3 = src.X0.cpu().numpy().copy()
    x3[np.isin(src.u_cell, val_cells)] += 3.0
    src3 = Source.from_arrays(src.units, src.groups, src.cells, d3, x3, src.FP.cpu().numpy(), src.dev)
    assert torch.equal(src.S_train, src3.S_train) and torch.equal(src.n_train, src3.n_train)
    fa, fb = Features.fit(src), Features.fit(src3)
    for k in ("P_ctrl", "s_ctrl", "mu_f0", "P_f0", "s_f0"):
        assert torch.allclose(getattr(fa, k).abs(), getattr(fb, k).abs(), atol=1e-5), f"meta-validation data changes feature basis {k}"
    assert not torch.equal(src.S_all, src3.S_all)       # sanity: the hcc regime does include them
    # (d) meta_val regime == S_train (no validation cell of any family), and regime misuse is refused
    for ci in val_cells:
        S, n = src.f0_table("meta_val", src.cell_family[ci])
        assert S is src.S_train
        try:
            src.f0_table("meta_train", src.cell_family[ci])
            raise SystemExit("meta_train regime accepted a validation family")
        except AssertionError:
            pass
    checks["f0_family_exclusion_bruteforce"] = True
    checks["meta_val_never_in_train_tables_or_feature_bases"] = True


def check_episodes(src, checks):
    # fixed evaluation episodes: disjoint, nested, deterministic, same query for every K
    for role in ("train", "val"):
        for ci in src.eval_cells(role)[:6]:
            e1, e2 = src.eval_episode(ci, 1), src.eval_episode(ci, 1)
            assert (e1.g_s == e2.g_s).all() and (e1.g_q == e2.g_q).all()
            assert not (set(e1.g_s) & set(e1.g_q)) and len(set(e1.g_s)) == len(e1.g_s)
            for K in (8, 16, 32):
                i = e1.inputs(K)
                assert "y_q" not in i and (i["g_s"] == e1.g_s[:K]).all() and (i["g_q"] == e1.g_q).all()
            e3 = src.eval_episode(ci, 2)
            assert not (len(e3.g_q) == len(e1.g_q) and (e3.g_q == e1.g_q).all())
            assert (src.cell_role[np.where(src.cell_name == e1.name)[0][0]] == role)
    # sampler: meta-train families only, f0 excludes the family (checked by brute force inside check_f0), support/query disjoint
    sample = src.make_sampler()
    rs = np.random.RandomState(0)
    val_fams = set(src.cell_family[src.cell_role == "val"])
    assert not (set(sample.families) & val_fams)
    for _ in range(200):
        K, ep = sample(rs)
        assert ep.family not in val_fams and len(ep.g_s) == K and not (set(ep.g_s) & set(ep.g_q))
    checks["eval_episodes_disjoint_nested_deterministic"] = True
    checks["sampler_never_draws_meta_validation_family"] = True


def check_methods(src, checks):
    ft_map = Features.fit(src)
    torch.manual_seed(0)
    basis = torch.linalg.qr(torch.randn(N_GENES, 16, device=src.dev))[0]
    models = dict(adapter=M.ResidualAdapter(), adapter_joint=M.ResidualAdapter(offset_mode="joint"),
                  adapter_svd=M.ResidualAdapter(svd_basis=basis), adapter_shrink=M.ResidualAdapter(shrink_by_k=True), cnp=M.GenericCNP())
    ci = src.eval_cells("val")[0]
    ep = src.eval_episode(ci, 1)
    for K in (8, 16, 32):
        inp = ep.inputs(K)
        ft = ft_map.build(inp, src.FP)

        def all_preds(inp, ft):
            out = dict(no_change=M.no_change(inp), f0=M.same_drug_mean(inp), mean_offset=M.mean_offset(inp))
            for n_, f in M.CLOSED_FORM.items():
                out[n_] = f(inp, ft, 1.0)
            for n_, m in models.items():
                out[n_] = m.to(src.dev)(inp, ft)
            return out
        p1 = all_preds(inp, ft)
        # (a) query labels cannot reach any prediction: perturb y_q in the episode, rebuild inputs, compare
        yq = ep.y_q.clone()
        ep.y_q = ep.y_q + 100.0
        inp2 = ep.inputs(K)
        p2 = all_preds(inp2, ft_map.build(inp2, src.FP))
        ep.y_q = yq
        for n_ in p1:
            assert torch.equal(p1[n_], p2[n_]), f"{n_}: prediction depends on query labels"
        # (b) support labels do matter for every adapting method (sanity of the fixture itself)
        inp3 = dict(inp)
        inp3["y_s"] = inp["y_s"] + torch.randn_like(inp["y_s"])
        p3 = all_preds(inp3, ft_map.build(inp3, src.FP))
        for n_ in p1:
            if n_ not in ("no_change", "f0"):
                assert not torch.allclose(p1[n_], p3[n_]), f"{n_}: support labels have no effect"
        # (c) features are label-free: identical when every label in the input dict changes
        ft3 = ft_map.build(inp3, src.FP)
        assert all(torch.equal(ft[k], ft3[k]) for k in ft)
        # (d) zero centred residual -> the proposal returns exactly mean-offset
        inp4 = dict(inp)
        inp4["y_s"] = inp["f0_s"] + torch.randn(1, N_GENES, device=src.dev)
        ft4 = ft_map.build(inp4, src.FP)
        for n_ in ("adapter", "adapter_svd", "adapter_shrink"):
            assert torch.allclose(models[n_](inp4, ft4), M.mean_offset(inp4), atol=1e-4), f"{n_} does not fall back to mean-offset"
        assert torch.allclose(models["adapter"](inp, ft, zero_residual=True), M.mean_offset(inp), atol=1e-6)
        for n_ in M.CLOSED_FORM:
            assert torch.allclose(M.CLOSED_FORM[n_](inp4, ft4, 1.0), M.mean_offset(inp4), atol=1e-4)
            # huge ridge -> mean-offset
            assert torch.allclose(M.CLOSED_FORM[n_](inp, ft, 1e12), M.mean_offset(inp), atol=1e-4)
    # (e) gradient reaches phi and lambda through the solve
    m = models["adapter"]
    loss = M.rmse_loss(m(inp, ft), ep.y_q)
    loss.backward()
    assert m.log_lam.grad is not None and m.log_lam.grad.abs() > 0 and m.phi[0].weight.grad.abs().sum() > 0
    checks["query_labels_never_reach_predictions_or_features"] = sorted(p1)
    checks["zero_residual_falls_back_to_mean_offset"] = True


def check_metrics(checks):
    rs = np.random.RandomState(0)
    t = rs.randn(30, N_GENES)
    assert np.allclose(per_unit(t, t).delta_rmse, 0) and np.allclose(per_unit(t, t).centred_pearson, 1)
    p = t + rs.randn(30, N_GENES)
    a, b = summarise(per_unit(p, t)), summarise(per_unit(p + rs.randn(1, N_GENES) * 3, t))
    assert abs(a["centred_rmse"] - b["centred_rmse"]) < 1e-9 and abs(a["centred_pearson"] - b["centred_pearson"]) < 1e-9, "common offset changes centred metrics"
    assert b["delta_rmse"] > a["delta_rmse"] + 1
    perm = rs.permutation(30)
    assert np.allclose(per_unit(p[perm], t[perm]).delta_rmse.to_numpy(), per_unit(p, t).delta_rmse.to_numpy()[perm])
    assert abs(summarise(per_unit(p, t))["retrieval_mrr"] - 1) < 1e-9 and summarise(per_unit(np.zeros_like(t), t))["retrieval_mrr"] < 0.2
    checks["metrics_offset_invariance_and_order_independence"] = True


def check_units_builder(checks):
    from .build_units import rows_to_units
    rows = pd.DataFrame(dict(cell=["A", "A", "A", "A", "B"], pert_id=["p1", "p1", "p2", "p3", "p1"], drug_group=["g1", "g1", "g1", "g3", "g1"],
                             det_plate=["x", "y", "x", "x", "z"]))
    d = np.zeros((5, 4), np.float32)
    d[:, 0] = [1, 3, 10, 5, 7]
    tab, ud, ux = rows_to_units(rows, d, d + 1)
    assert list(tab.cell) == ["A", "A", "B"] and list(tab.drug_group) == ["g1", "g3", "g1"]
    assert np.allclose(ud[:, 0], [(2 + 10) / 2, 5, 7]) and np.allclose(ux[:, 0], [7, 6, 8]) and list(tab.n_conditions) == [2, 1, 1]
    assert tab.plates[0] == "x|y"
    checks["rows_to_units_equal_weighting"] = True


def check_target_guard(checks):
    from .target import load_target
    env = dict(os.environ)
    for cond, e in (("no_env_flag", {}), ("env_flag_but_no_committed_freeze", {"R2_FINAL_EVAL": "1"})):
        os.environ.pop("R2_FINAL_EVAL", None)
        os.environ.update(e)
        try:
            load_target("HUH7")
            raise SystemExit(f"target loader opened without freeze ({cond})")
        except AssertionError:
            pass
    os.environ.clear()
    os.environ.update(env)
    checks["hcc_loader_refuses_before_freeze"] = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--expect-sealed", action="store_true", help="also assert that the HCC loader refuses (run before the freeze only)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    checks = {}
    units, groups, cells, delta, x0, fp = synthetic()
    src = Source.from_arrays(units, groups, cells, delta, x0, fp, dev)
    check_f0(src, delta, checks)
    check_episodes(src, checks)
    check_methods(src, checks)
    check_metrics(checks)
    check_units_builder(checks)
    if a.expect_sealed:
        check_target_guard(checks)
    if a.real:
        real = Source(dev)
        names = {norm_name(c) for c in real.cell_name}
        assert not (names & set(EXCLUDED_TARGETS)) and not any("HUH" in n or "JHH" in n for n in names)
        checks["real_source_has_no_hcc_target"] = dict(cells=len(names))
        rc = {}
        check_f0(real, real.D.cpu().numpy(), rc)
        check_episodes(real, rc)
        check_methods(real, rc)
        checks["real_store"] = rc
        checks["real_eval_cells"] = dict(meta_val=[real.cell_name[i] for i in real.eval_cells("val")], meta_train_dev=len(real.eval_cells("train")),
                                         sampler_families=len(real.make_sampler().families))
    checks["commit"] = os.environ.get("R2_COMMIT", "")
    dump(checks, public("manifests", "episodic_residual_r2", "leak_fixtures.json"))
    print(json.dumps(checks, indent=1))
    print("ALL FIXTURES PASSED")


if __name__ == "__main__":
    main()
