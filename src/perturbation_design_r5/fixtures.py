"""Leak and numeric fixtures (source dev episodes and toy matrices only)."""
import json
import numpy as np

from .common import protocol, public, dump_json
from .data import Store, make_episode, target_cells
from .models import GP
from .oracle import Oracle
from .acquisition import _greedy_vr
from .replay import run_selection, load_hyper, HYPER
import os

POLS = ["random", "diversity", "latent_variance", "fixed_vr", "proposed_nam", "total_variance", "pdbal_rna"]


def main():
    st = Store()
    pr = protocol()
    hp = load_hyper() if os.path.exists(HYPER) else None
    budgets = pr["episodes"]["budgets_K"]
    (cell, fam) = target_cells(st, "dev", pr)[0]
    ep = make_episode(st, "dev", cell, fam, 1, pr=pr)
    res = {"episode": dict(cell=cell, family=fam, seed=1, n=len(ep.groups), pool=len(ep.pool), query=len(ep.query))}
    true = {int(i): np.asarray(st.well_delta[int(ep.rep.store_idx.iloc[int(i)])], np.float32) for i in ep.pool}

    # 1. oracle: one row per reveal, no repeat, no query reveal
    o = Oracle(st, ep)
    v = o.reveal(ep.pool[0])
    ok1 = v.shape == (st.well_delta.shape[1],) and np.allclose(v, true[int(ep.pool[0])])
    try:
        o.reveal(ep.pool[0]); ok2 = False
    except ValueError:
        ok2 = True
    try:
        o.reveal(ep.query[0]); ok3 = False
    except PermissionError:
        ok3 = True
    res["oracle_one_row"] = bool(ok1 and o.cost()["trt_wells"] == int(ep.rep.n_trt_wells.iloc[int(ep.pool[0])]))
    res["oracle_no_repeat"] = ok2
    res["oracle_no_query"] = ok3

    if hp is not None:
        for pol in POLS:
            if pol == "pdbal_rna" and not hp.get("noise_hetero") and hp.get("hetero"):
                continue
            a = run_selection(st, ep, pol, hp, budgets, override=true)
            # 2. query-swap invariance: change every query unit's stored truth -> selection unchanged
            saved = st.unit_delta
            st.unit_delta = np.array(saved)
            qi = st.units.set_index(["cell", "drug_group"]).loc[[(cell, g) for g in ep.groups[ep.query]], "unit_idx"].to_numpy()
            st.unit_delta[qi] = st.unit_delta[np.random.RandomState(0).permutation(qi)] + 5.0
            b = run_selection(st, ep, pol, hp, budgets, override=true)
            st.unit_delta = saved
            # 3. unrevealed-label permutation: values revealed after the 3rd batch are permuted among themselves
            cut = 4 + 4 * 3
            later = a["S"][cut:]
            ov = dict(true)
            perm = np.random.RandomState(1).permutation(len(later))
            for x, y in zip(later, np.array(later)[perm]):
                ov[x] = true[int(y)] * 1.7 - 0.3
            c = run_selection(st, ep, pol, hp, budgets, override=ov)
            res[f"query_swap_invariant_{pol}"] = a["S"] == b["S"]
            res[f"unrevealed_permutation_prefix_{pol}"] = a["S"][:cut + 4] == c["S"][:cut + 4]

    # 4. synthetic: two candidates with identical prior information, one with 10x noise -> the quiet one is chosen
    rs = np.random.RandomState(0)
    n = 6
    X = rs.randn(n, 3)
    K = X @ X.T / 3 + 0.5
    K = np.vstack([K, K[2:3]])
    K = np.hstack([K, K[:, 2:3]])               # group 6 duplicates group 2
    nu = np.full(n + 1, 0.5)
    nu[2] = 5.0
    g = GP(K, nu, 0.3, 1.0)
    _, Sig = g.posterior([0], np.zeros((1, 2)), np.arange(1, n + 1))
    cand = np.arange(1, n + 1)
    pick = _greedy_vr(Sig, g.nu, cand, cand, 1)[0]
    pos = {c: i for i, c in enumerate(cand)}
    A = lambda j: (Sig[:, pos[j]] ** 2).sum() / (g.nu[j] + Sig[pos[j], pos[j]])
    res["synthetic_quiet_duplicate_preferred"] = bool(A(6) > A(2))
    res["synthetic_first_pick"] = int(pick)

    # 5. GP posterior equals the direct inverse and variances are non-negative
    S = [0, 1, 3]
    Z = rs.randn(3, 4)
    P = [2, 4, 5, 6]
    mu, Sg = g.posterior(S, Z, P)
    C = g.C
    A_ = C[np.ix_(S, S)] + np.diag(g.nu[S])
    mu2 = C[np.ix_(P, S)] @ np.linalg.inv(A_) @ Z
    S2 = C[np.ix_(P, P)] - C[np.ix_(P, S)] @ np.linalg.inv(A_) @ C[np.ix_(S, P)]
    res["gp_matches_direct_inverse"] = bool(np.allclose(mu, mu2) and np.allclose(Sg, S2))
    res["gp_variance_nonnegative"] = bool((np.diag(Sg) >= -1e-10).all())
    res["all_passed"] = all(v for k, v in res.items() if isinstance(v, bool))
    dump_json(res, public("manifests", "perturbation_design_r5", "fixtures.json"))
    print(json.dumps(res, indent=1, default=str))
    assert res["all_passed"], "fixture failed"


if __name__ == "__main__":
    main()
