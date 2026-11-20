"""Integrity fixtures of the follow-up (plan Appendix C items 3, 5, 6, 7, 8) plus two contract checks of the D3 arms.
Each function raises on failure and returns a small dict of what was checked. `python -m src.r7_followup_v1.fixtures`
runs all of them and writes manifests/r7_followup_v1/fixtures.json. No test delta is opened."""
import json, os
import numpy as np


# ---------------------------------------------------------------- C3 retrieval ties
def fx_retrieval():
    from .metrics import rank_counts, tie_corrected, retrieval_by_pool, chance
    N = 5
    H5 = sum(1 / i for i in range(1, N + 1))
    r = tie_corrected(*rank_counts(np.zeros((N, N))))                       # every candidate tied
    assert np.allclose(r["top1_tie_expected"], 1 / N) and np.allclose(r["top5_tie_expected"], 1.0) and np.allclose(r["mrr_tie_expected"], H5 / N)
    assert np.allclose(r["legacy_midrank_mrr"], 1 / 3) and np.allclose(r["legacy_unique_top1"], 0)
    assert abs(chance(N)["mrr_tie_expected"] - H5 / N) < 1e-12 and chance(N)["top1_tie_expected"] == 1 / N
    r = tie_corrected(*rank_counts(np.eye(N)))                              # perfect
    assert np.allclose(r["top1_tie_expected"], 1) and np.allclose(r["mrr_tie_expected"], 1) and np.allclose(r["legacy_midrank_mrr"], 1)
    r = tie_corrected(*rank_counts(1 - np.eye(N)))                          # reversed: correct candidate is last
    assert np.allclose(r["top1_tie_expected"], 0) and np.allclose(r["mrr_tie_expected"], 1 / N) and np.allclose(r["top5_tie_expected"], 1)
    sim = np.array([[0.2, 0.9, 0.2, 0.1]])                                  # a = 1, t = 2 -> ranks 2 or 3
    r = tie_corrected(*rank_counts(sim, correct=[0]), ks=(1, 2, 5))
    assert np.allclose(r["top1_tie_expected"], 0) and np.allclose(r["top2_tie_expected"], 0.5) and np.allclose(r["mrr_tie_expected"], (1 / 2 + 1 / 3) / 2)
    assert np.allclose(r["legacy_midrank_mrr"], 1 / 2.5)
    rng = np.random.RandomState(0)
    true = rng.normal(size=(7, 20))
    z = retrieval_by_pool(np.zeros((7, 20)), true, np.zeros(7, int))        # zero-norm centred prediction
    H7 = sum(1 / i for i in range(1, 8))
    assert np.allclose(z["mrr_tie_expected"], H7 / 7) and np.allclose(z["top1_tie_expected"], 1 / 7) and (z["n_tied"] == 7).all()
    c = retrieval_by_pool(np.ones((7, 20)) * 3.0, true, np.zeros(7, int))   # the same vector for every query: also zero after centring
    assert np.allclose(c["mrr_tie_expected"], H7 / 7)
    p = retrieval_by_pool(true + 0.0, true, np.zeros(7, int))
    assert np.allclose(p["mrr_tie_expected"], 1) and np.allclose(p["legacy_unique_top1"], 1)
    return dict(all_tie_mrr=H5 / N, all_tie_legacy_midrank=1 / 3, zero_norm_is_all_tie=True)


# ---------------------------------------------------------------- C7 information paths
def _synthetic_graph():
    """984 nodes: 0..5 off-panel, 6..983 the 978 landmarks (the condition encoder of the original class expects 978).
    1 -> 6 (one hop); 2 -> 3 -> 7 (two hops); 4 -> 5 -> 2 (node 4 needs four hops: unreachable); 0 isolated."""
    ei = np.array([[1, 2, 3, 4, 5], [6, 3, 7, 5, 2]])
    n = 984; lm = np.arange(6, 984)
    rng = np.random.RandomState(1)
    meas = np.zeros(n, np.float32); meas[lm] = 1
    return dict(edge_index=ei, edge_rel=rng.randint(0, 6, ei.shape[1]), edge_feat=rng.rand(ei.shape[1], 11).astype(np.float32), measured=meas, lm_idx=lm), n


def _changed_landmarks(corr, G, node, n_lm, torch):
    fp = torch.randn(1, 2048); x0 = torch.randn(1, n_lm); status = torch.tensor([[1.0, 1.0, 0.3, 0.5]])
    empty = (torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long), torch.zeros(0, 9))
    seed = (torch.zeros(1, dtype=torch.long), torch.tensor([node]), torch.ones(1, 9))
    with torch.no_grad():
        r1, _ = corr(fp, x0, status, seed); r0, _ = corr(fp, x0, status, empty)     # q / status identical in both calls
    return set(np.where((r1 - r0).abs().numpy()[0] > 0)[0].tolist())


def fx_paths(real_graph=True):
    import torch
    from src.evidence_gated_r7.models import GraphCorrection
    from .kgraph import node_reach, load_kg, reach_class
    torch.manual_seed(0)
    G, n = _synthetic_graph()
    corr = GraphCorrection(G, gated=True)
    torch.nn.init.normal_(corr.out.weight, std=1.0)
    R = node_reach(G["edge_index"], n, G["lm_idx"])
    expect = {8: {2}, 1: {0}, 2: {1}, 3: {1}, 4: set(), 0: set(), 5: set(), 6: {0}}     # node -> changed landmark positions
    got = {}
    for node, exp in expect.items():
        ch = _changed_landmarks(corr, G, node, 978, torch)
        pred = set(np.where(R["reach_mat"][node].toarray()[0])[0].tolist())
        assert ch == exp == pred, (node, ch, exp, pred)
        got[node] = sorted(ch)
    out = dict(synthetic={str(k): v for k, v in got.items()}, synthetic_minhop={str(k): int(R["minhop"][k]) for k in expect})
    assert [int(R["minhop"][k]) for k in (8, 1, 2, 4)] == [0, 1, 2, -1]
    if real_graph:
        KG = load_kg(os.environ["R7_KG"])
        Gr = dict(edge_index=KG["edge_index"], edge_rel=KG["edge_rel"], edge_feat=KG["efeat"], measured=KG["measured"].astype(np.float32), lm_idx=KG["lm_idx"])
        corr = GraphCorrection(Gr, gated=True)
        torch.nn.init.normal_(corr.out.weight, std=1.0)
        R = node_reach(KG["edge_index"], KG["n"], KG["lm_idx"])
        rc = reach_class(R); tn = np.unique(KG["seeds"].node)
        rng = np.random.RandomState(2)
        exact, n_checked = 0, 0
        for cls in (0, 1, 2):
            for node in rng.choice(tn[rc[tn] == cls], 6, replace=False):
                ch = _changed_landmarks(corr, Gr, int(node), 978, torch)
                pred = set(np.where(R["reach_mat"][node].toarray()[0])[0].tolist())
                assert ch <= pred, "a seed changed a landmark outside the predicted 2-hop set"
                if cls == 2:
                    assert not ch, "an unreachable seed changed the output with q/status fixed"
                exact += int(ch == pred); n_checked += 1
        out.update(real_graph_nodes_checked=n_checked, real_graph_changed_set_equals_predicted=exact, unreachable_seed_changes_output=False)
    return out


# ---------------------------------------------------------------- C8 target difference; K0 = backbone
def fx_target_difference():
    import torch
    from .models import FollowupModel
    torch.manual_seed(0)
    G, n = _synthetic_graph()
    lm = len(G["lm_idx"])
    m = FollowupModel("TD", G)
    torch.nn.init.normal_(m.corr.out.weight, std=1.0); torch.nn.init.normal_(m.corr.out.bias, std=1.0)
    B = 3
    fp = torch.randn(B, 2048); x0 = torch.randn(B, lm); status = torch.rand(B, 4)
    empty = (torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long), torch.zeros(0, 9))
    with torch.no_grad():
        r_empty, _ = m.correction(fp, x0, status, None, empty)
        plain, _ = m.corr(fp, x0, status, empty)
        r_unreach, _ = m.correction(fp, x0, status, None, (torch.tensor([0]), torch.tensor([4]), torch.ones(1, 9)))
        r_reach, _ = m.correction(fp, x0, status, None, (torch.tensor([0]), torch.tensor([2]), torch.ones(1, 9)))
    assert float(r_empty.abs().max()) == 0.0, "target-difference correction is not exactly zero for an empty seed list"
    assert float(plain.abs().max()) > 0, "the plain graph correction is expected to be non-zero without seeds (generic path)"
    assert float(r_unreach.abs().max()) == 0.0 and float(r_reach[0].abs().max()) > 0 and float(r_reach[1:].abs().max()) == 0.0
    return dict(empty_seed_max_abs=0.0, plain_graph_empty_seed_max_abs=float(plain.abs().max()), unreachable_seed_max_abs=0.0)


# ---------------------------------------------------------------- C5 fold leak, C6 null mapping, arm contract
def fx_fold_and_null():
    import pandas as pd, torch
    from src.evidence_gated_r7.data import Store, Knowledge
    from .common import MAN, sha256, N_FOLDS, NULL_SEEDS
    from .train import Context, FoldStats, train_targets, Bank, aug_rates
    from .shuffle import load_mapping
    S = Store(); K = Knowledge(len(S.groups))
    out = {}
    # C5: nothing fitted on responses changes when the held and inner-validation deltas are replaced by noise
    keep = S._open.copy()
    oi = S.units.open_idx.to_numpy()
    for k in range(N_FOLDS):
        C = Context(S, f"fold{k}")
        st1 = FoldStats(S, C.train); Y1 = train_targets(S, C.train, torch.device("cpu")).numpy().copy()
        rng = np.random.RandomState(k)
        other = np.concatenate([C.sel, C.held, S.val])
        S._open[oi[other]] = rng.normal(size=(len(other), 978)).astype(np.float32)
        st2 = FoldStats(S, C.train); Y2 = train_targets(S, C.train, torch.device("cpu")).numpy()
        assert np.array_equal(st1.cell_mean, st2.cell_mean) and np.array_equal(st1.x0_mu, st2.x0_mu) and np.array_equal(st1.x0_sd, st2.x0_sd)
        assert np.array_equal(Y1, Y2)
        nz = np.where(np.abs(Y1).sum(1) > 0)[0]
        assert set(nz) <= set(C.train.tolist()), "a regression target outside the inner-train units"
        S._open[:] = keep
        g = S.units.drug_group.to_numpy()
        assert not set(g[C.train]) & set(g[C.held]) and not set(g[C.train]) & set(g[C.sel]) and not set(g[C.sel]) & set(g[C.held])
        assert (S.role[C.held] == "train").all() and (S.role[C.train] == "train").all()
        out[f"fold{k}"] = dict(train_units=int(len(C.train)), sel_units=int(len(C.sel)), held_units=int(len(C.held)), cells_without_train_units=int(len(st1.cells_without_train)))
    held_all = np.concatenate([Context(S, f"fold{k}").held for k in range(N_FOLDS)])
    assert len(set(held_all.tolist())) == len(held_all) == len(S.train), "every original-train unit is held exactly once"
    out["held_once"] = True
    # C6: null mapping preserves count / status and the changed count of the manifest
    man = json.load(open(os.path.join(MAN, "target_shuffle_manifest.json")))
    real = Bank(K, "real")
    for key, mm in man["mappings"].items():
        assert sha256(mm["path"]) == mm["sha256"], key
        t = pd.read_csv(mm["path"], sep="\t")
        assert int((t.node_real != t.node_null).sum()) == mm["n_changed"] and len(t) == mm["n_stubs"]
        null = Bank(K, "null", load_mapping(mm["path"]))
        n_changed = 0
        for g, (nodes, f, tids) in real.rows.items():
            nn_, fn, tn = null.rows[g]
            assert len(nn_) == len(nodes) and np.array_equal(f, fn) and tids == tn
            assert np.array_equal(Bank.status(nodes, f), Bank.status(nn_, fn)) and np.array_equal(Bank.meta(nodes, f), Bank.meta(nn_, fn))
            n_changed += int((nn_ != nodes).sum())
        assert n_changed == mm["n_changed"]
        used = t[t.split != "unused"]
        for sp_, d in used.groupby("split"):
            assert d.node_real.value_counts().sort_index().equals(d.node_null.value_counts().sort_index())
    out["null_mappings_checked"] = len(man["mappings"])
    # arm contract: the same augmentation draws for real / null / no-seed stubs
    key = "fold0_null%d" % NULL_SEEDS[0]
    banks = [Bank(K, "real"), Bank(K, "null", load_mapping(man["mappings"][key]["path"])), Bank(K, "noseed")]
    C = Context(S, "fold0"); p = aug_rates(K, S.group_idx[C.train])
    rngs = [np.random.RandomState(1042) for _ in banks]
    r = np.random.RandomState(42); dev = torch.device("cpu")
    n_empty_g0 = 0
    for _ in range(40):
        gi = S.group_idx[r.choice(C.train, 64, replace=False)]
        res = [b.batch(gi, dev, rg, p) for b, rg in zip(banks, rngs)]
        for x in res[1:]:
            assert torch.equal(res[0][1], x[1]) and torch.equal(res[0][2], x[2]) and torch.equal(res[0][3], x[3])
        assert torch.equal(res[0][0][0], res[1][0][0]) and torch.equal(res[0][0][2], res[1][0][2])     # same stubs kept, same features
        assert len(res[2][0][0]) == 0 and float(res[2][3].sum()) > 0                                    # G0: no seed, m_available kept
        n_empty_g0 += 1
    out["augmentation_draws_identical_real_null_noseed_batches"] = n_empty_g0
    out["aug_rates_fold0"] = p
    return out


def main():
    from .common import pman, dump
    res = dict(C3_retrieval=fx_retrieval(), C7_paths=fx_paths(), C8_target_difference=fx_target_difference(), C5_C6_fold_null_contract=fx_fold_and_null(),
               commit=os.environ.get("R7F_COMMIT"), job=os.environ.get("SLURM_JOB_ID"), all_passed=True)
    dump(res, pman("fixtures.json"))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
