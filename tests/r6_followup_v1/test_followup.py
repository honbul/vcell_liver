"""Follow-up integrity fixtures (plan appendix C, items 1-4). Run: python -m unittest tests.r6_followup_v1.test_followup

    1  train-only means     a held-fold drug never enters an anchor; a training anchor excludes its own drug
    2  contrast             adding one pool vector to every prediction leaves the within-pool error unchanged;
                            subtracting the same b from truth and prediction leaves the ordinary RMSE unchanged
    3  retrieval            all tied, perfect, reversed, zero-norm: expected values of appendix B
    4  views                arms share barcodes and thinning draws; fit / eval / reference roles are kept
"""
from __future__ import annotations

import unittest

import numpy as np
import torch

from src.depth_robust_r6 import data as D
from src.r6_followup_v1 import metrics as M
from src.r6_followup_v1.baselines import MeanBaseline
from src.r6_followup_v1.scoring import Scorer
from tests.depth_robust_r6.test_r6 import toy


def toy_eval(seed=0, n_pool=6, per=5, G=40):
    rng = np.random.default_rng(seed)
    pool = np.repeat(np.arange(n_pool), per)
    drug = np.tile(np.arange(per), n_pool) + 100 * (pool % 2)        # a drug appears in several pools
    T = rng.normal(size=(n_pool * per, G)).astype(np.float32)
    P = (T + rng.normal(size=T.shape)).astype(np.float32)
    return T, P, pool, drug


class TrainOnlyMeans(unittest.TestCase):                             # fixture 1
    def setUp(self):
        self.d = toy()
        d = self.d
        self.fit, self.held = [0, 1, 3], [2, 4]
        self.Lf, self.Df = d.conditions(self.fit)
        self.pf = d.pool_id[d.plate[self.Lf, self.Df], self.Lf]
        self.Lq, self.Dq = d.conditions(self.held)
        self.pq = d.pool_id[d.plate[self.Lq, self.Dq], self.Lq]

    def test_held_drug_never_enters_an_anchor(self):
        d = self.d
        mb = MeanBaseline(d.Y, self.Lf, self.Df, self.pf, 2, 4)
        base = {k: mb.predict(self.Lq, self.Dq, self.pq, k)[0] for k in (0.0, 8.0, float("inf"))}
        Y2 = d.Y.copy(); Y2[:, self.held] += 100.0                   # change every held response
        mb2 = MeanBaseline(Y2, self.Lf, self.Df, self.pf, 2, 4)
        for k, b in base.items():
            self.assertTrue(np.array_equal(b, mb2.predict(self.Lq, self.Dq, self.pq, k)[0]))
        with self.assertRaises(AssertionError):                      # a fitting drug as a plain query is refused
            mb.predict(self.Lf, self.Df, self.pf, 0.0)

    def test_training_anchor_excludes_its_own_drug(self):
        d = self.d
        mb = MeanBaseline(d.Y, self.Lf, self.Df, self.pf, 2, 4)
        a0 = mb.predict(self.Lf, self.Df, self.pf, 2.0, exclude_self=True)[0]
        for i in range(len(self.Lf)):                                 # change one condition's own response
            Y2 = d.Y.copy(); Y2[self.Lf[i], self.Df[i]] += 50.0
            a1 = MeanBaseline(Y2, self.Lf, self.Df, self.pf, 2, 4).predict(self.Lf, self.Df, self.pf, 2.0, exclude_self=True)[0]
            self.assertTrue(np.allclose(a0[i], a1[i], atol=1e-4))
        # and it equals the brute-force leave-one-out pool mean at kappa = 0
        a = mb.predict(self.Lf, self.Df, self.pf, 0.0, exclude_self=True)[0]
        for i in range(len(self.Lf)):
            others = (self.pf == self.pf[i]) & (np.arange(len(self.Lf)) != i)
            if others.any():
                self.assertTrue(np.allclose(a[i], d.Y[self.Lf[others], self.Df[others]].mean(0), atol=1e-5))

    def test_source_folds_are_inside_train_and_disjoint(self):
        from src.r6_followup_v1 import common as C
        d = self.d
        tr = set(C.train_drugs(d))
        held_all = []
        for k in C.FOLDS:
            fit, held = C.source_split(d, k)
            self.assertFalse(set(fit) & set(held))
            self.assertTrue(set(fit) | set(held) == tr)
            ifit, idev = C.source_split(d, k, inner=True)
            self.assertFalse(set(ifit) & set(idev)); self.assertFalse((set(ifit) | set(idev)) & set(held))
            held_all += held
        self.assertEqual(sorted(held_all), sorted(tr))
        self.assertFalse(tr & set(C.test_drugs(d)))


class Contrast(unittest.TestCase):                                   # fixture 2
    def test_pool_shift_invariance_and_same_b(self):
        T, P, pool, drug = toy_eval()
        groups = M.groups_of(np.zeros(len(pool), int), pool)
        w0 = M.within_point(M.grams(P.astype(np.float64) - T, groups), groups, len(T))
        shift = np.random.default_rng(1).normal(size=(6, T.shape[1])).astype(np.float32) * 3
        P2 = P + shift[pool]                                          # one vector per pool added to every prediction
        w1 = M.within_point(M.grams(P2.astype(np.float64) - T, groups), groups, len(T))
        self.assertTrue(np.allclose(w0, w1, atol=1e-5))
        self.assertFalse(np.allclose(M.cond_rmse(P, T), M.cond_rmse(P2, T)))   # the total error does change
        # brute force: centre each side by its own pool mean
        Pc, Tc = M.centre_by(P, groups), M.centre_by(T, groups)
        self.assertTrue(np.allclose(w0, np.sqrt(((Pc - Tc) ** 2).mean(1)), atol=1e-6))
        # subtracting the SAME b from truth and prediction is the ordinary RMSE (not a specificity metric)
        b = shift[pool]
        self.assertTrue(np.allclose(M.cond_rmse(P - b, (T - b).astype(np.float64)), M.cond_rmse(P, T.astype(np.float64)), atol=1e-6))
        # a prediction that is constant within the pool has centred prediction 0: l_i = RMS of centred truth
        const = np.repeat(shift, 5, 0)
        wc = M.within_point(M.grams(const.astype(np.float64) - T, groups), groups, len(T))
        self.assertTrue(np.allclose(wc, np.sqrt((Tc ** 2).mean(1)), atol=1e-5))

    def test_pair_identity_and_weighted_bootstrap(self):
        T, P, pool, drug = toy_eval(3)
        E = P.astype(np.float64) - T
        groups = M.groups_of(np.zeros(len(pool), int), pool)
        gr = M.grams(E, groups)
        for K, idx in zip(gr, groups):
            e = E[idx]; m = len(idx)
            pairs = [((e[i] - e[j]) ** 2).mean() for i in range(m) for j in range(i + 1, m)]
            self.assertAlmostEqual(M.pair_contrast_from_gram(K), float(np.sqrt(np.mean(pairs))), places=8)
        du, dinv = np.unique(drug, return_inverse=True)
        Mb = M.multiplicities(len(du), 50, 7)
        Lacc, V = M.within_boot([gr], groups, dinv, Mb)
        for b in (0, 13, 49):                                          # brute force with replicated rows
            w = Mb[b][dinv].astype(float)
            for idx in groups:
                ww = w[idx]
                if (ww > 0).sum() < 2:
                    self.assertTrue((V[b, idx] == 0).all()); continue
                eb = (E[idx] * ww[:, None]).sum(0) / ww.sum()
                ref = np.sqrt(((E[idx] - eb) ** 2).mean(1))
                self.assertTrue(np.allclose(Lacc[b, idx], ref, atol=1e-5))
        # a group with fewer than two distinct drugs contributes nothing
        one = M.groups_of(np.zeros(3, int), np.array([0, 0, 1]))
        self.assertTrue(np.isnan(M.within_point(M.grams(E[:3], one), one, 3)[2]))

    def test_scorer_matches_plain_definitions(self):
        T, P, pool, drug = toy_eval(5)
        line = pool // 2
        sc = Scorer(T, line, drug, pool, None, None, n_boot=200)
        sc.add("m", [P, P + 0.1]); sc.add("z", [np.zeros_like(P)])
        s = sc.summary("m")
        ref = np.mean([M.drug_macro(M.cond_rmse(x, T.astype(np.float64)), drug) for x in (P, P + 0.1)])
        self.assertAlmostEqual(s["total_rmse"], ref, places=6)
        pr = sc.pair("m", "z")
        self.assertAlmostEqual(pr["total_diff"], s["total_rmse"] - sc.summary("z")["total_rmse"], places=8)
        self.assertTrue(pr["within_lo"] <= pr["within_diff"] <= pr["within_hi"] or pr["within_lo"] <= pr["within_hi"])


class Retrieval(unittest.TestCase):                                  # fixture 3
    def setUp(self):
        rng = np.random.default_rng(0)
        self.N = 8
        self.Tc = rng.normal(size=(self.N, 30))
        self.g = [np.arange(self.N)]

    def score(self, Pc, k=5):
        return M.retrieval_scores(*M.retrieval_counts(Pc, self.Tc, self.g), k=k)

    def test_all_tied_and_zero_norm(self):
        N = self.N
        HN = sum(1.0 / j for j in range(1, N + 1))
        for Pc in (np.zeros((N, 30)), np.ones((N, 30)) * 3.0):       # zero vector; constant vector (zero norm after centring)
            s = self.score(Pc)
            self.assertTrue(np.allclose(s["top1"], 1.0 / N)); self.assertTrue(np.allclose(s["topk"], 5.0 / N))
            self.assertTrue(np.allclose(s["mrr"], HN / N))
            self.assertTrue(np.allclose(s["chance_top1"], 1.0 / N)); self.assertTrue(np.allclose(s["chance_mrr"], HN / N))
            self.assertTrue(np.allclose(s["legacy_rank_score"], 0.5)); self.assertTrue((s["legacy_unique_top1"] == 0).all())
        self.assertFalse(np.isclose(HN / N, 1.0 / ((N + 1) / 2)))    # expected MRR is not 1 / mean rank

    def test_perfect(self):
        s = self.score(self.Tc.copy())
        for q in ("top1", "topk", "mrr", "legacy_rank_score", "legacy_unique_top1"):
            self.assertTrue(np.allclose(s[q], 1.0), q)

    def test_reversed(self):
        N = self.N
        # orthonormal candidates; query i correlates with candidate j by a strictly decreasing score that puts i last
        Q, _ = np.linalg.qr(np.random.default_rng(2).normal(size=(30, N)))
        C_ = Q.T - Q.T.mean(1, keepdims=True)
        Qo, _ = np.linalg.qr(C_.T); cand = Qo.T                       # rows orthonormal and mean-free
        P = np.zeros((N, 30))
        for i in range(N):
            w = np.arange(N, 0, -1).astype(float); w = np.roll(w, i + 1)   # candidate i gets the smallest weight
            P[i] = w @ cand
        s = M.retrieval_scores(*M.retrieval_counts(P, cand, [np.arange(N)]), k=5)
        self.assertTrue(np.allclose(s["top1"], 0.0)); self.assertTrue(np.allclose(s["topk"], 0.0))
        self.assertTrue(np.allclose(s["mrr"], 1.0 / N)); self.assertTrue(np.allclose(s["legacy_rank_score"], 0.0))

    def test_partial_tie_formula(self):
        a, t, N = np.array([1]), np.array([3]), np.array([10])       # ranks 2, 3, 4 equally likely
        s = M.retrieval_scores(a, t, N, k=3)
        self.assertAlmostEqual(float(s["top1"][0]), 0.0); self.assertAlmostEqual(float(s["topk"][0]), 2.0 / 3.0)
        self.assertAlmostEqual(float(s["mrr"][0]), (1 / 2 + 1 / 3 + 1 / 4) / 3)
        s = M.retrieval_scores(np.array([0]), np.array([4]), N, k=3)
        self.assertAlmostEqual(float(s["top1"][0]), 0.25); self.assertAlmostEqual(float(s["topk"][0]), 0.75)


class Views(unittest.TestCase):                                      # fixture 4
    def test_view_cache_roles_and_determinism(self):
        from src.r6_followup_v1.views import build_views
        d = toy()
        v1 = build_views(d, seed=42, reps=2); v2 = build_views(d, seed=42, reps=2); v3 = build_views(d, seed=43, reps=2)
        for k in ("cells", "cnt", "crest", "pool", "p"):
            self.assertTrue(np.array_equal(v1[k], v2[k]), k)          # same seed -> the same barcodes and the same draw
        self.assertFalse(np.array_equal(v1["cells"], v3["cells"]))
        for i in range(len(v1["pool"])):
            r = d.pools.iloc[int(v1["pool"][i])]
            c = v1["cells"][i]
            self.assertTrue(((c >= r.fit_start) & (c < r.fit_end)).all())      # fit_input cells of the view's own pool only
            self.assertEqual(len(set(c.tolist())), D.EVAL_N)
            full = d.cnt[c]
            if v1["p"][i] == 1.0:
                self.assertTrue(np.array_equal(v1["cnt"][i], full))
            else:
                self.assertTrue((v1["cnt"][i] <= full).all())
        eval_rows = set(np.concatenate([np.arange(r.eval_start, r.eval_end) for r in d.pools.itertuples()]).tolist())
        self.assertFalse(set(v1["cells"].ravel().tolist()) & eval_rows)   # evaluation input cells never enter a training view
        self.assertEqual(sorted(set(np.round(v1["p"], 3).tolist())), [0.25, 0.5, 1.0])

    def test_arms_draw_identical_batches(self):
        from src.r6_followup_v1.train_fixed import FixedSampler, build_model
        from src.r6_followup_v1.views import build_views
        d = toy()
        v = build_views(d, seed=42, reps=2)
        fit = [0, 1, 3]
        hp = dict(d=16, n_tok=16, n_blocks=1, head_mode="additive", drug_shrink=10.0)
        seqs = []
        for arm in ("S", "SD", "M"):
            torch.manual_seed(42)
            model = build_model(d, arm, hp)                              # consumes the global RNG differently per arm
            g = torch.Generator(); g.manual_seed(42)
            s = FixedSampler(d, fit, "cpu", g, n_views_per_pool=6)
            seqs.append([tuple(x.flatten().tolist() for x in s.draw(3, 4)) for _ in range(5)])
        self.assertEqual(seqs[0], seqs[1]); self.assertEqual(seqs[1], seqs[2])   # the same views, drugs and order for S, SD and M

    def test_sampler_view_belongs_to_the_conditions_pool(self):
        from src.r6_followup_v1.train_fixed import FixedSampler
        from src.r6_followup_v1.views import build_views
        d = toy()
        v = build_views(d, seed=1, reps=2)
        g = torch.Generator(); g.manual_seed(0)
        s = FixedSampler(d, [0, 1, 3], "cpu", g, n_views_per_pool=6)
        for _ in range(20):
            vi, L, Dg = s.draw(4, 3)
            self.assertTrue(set(Dg.flatten().tolist()) <= {0, 1, 3})
            pid = torch.as_tensor(d.pool_id)[torch.as_tensor(d.plate)[L, Dg], L]
            self.assertTrue((torch.as_tensor(v["pool"]).long()[vi][:, None] == pid).all())

    def test_state_only_arm_ignores_the_drug(self):
        from src.r6_followup_v1.train_fixed import build_model
        d = toy()
        hp = dict(d=16, n_tok=16, n_blocks=1, head_mode="additive", drug_shrink=10.0)
        torch.manual_seed(0)
        for arm, same in (("S", True), ("SD", False), ("M", False)):
            m = build_model(d, arm, hp).eval()
            for q in m.parameters():                                  # undo the zero-initialised last layer
                torch.nn.init.normal_(q, std=0.3)
            c = torch.as_tensor(d.cnt[:128]).float().reshape(2, 64, 60); r = torch.as_tensor(d.crest[:128]).float().reshape(2, 64)
            from src.depth_robust_r6.count_view import set_features
            z = m.state(set_features(c, r))[0]
            fp = torch.as_tensor(d.fp[:3])[None].expand(2, 3, 32)
            out = m.delta(z, fp)
            self.assertEqual(bool(torch.allclose(out[:, 0], out[:, 1], atol=1e-6)), same, arm)
            self.assertEqual(tuple(out.shape), (2, 3, 60))


if __name__ == "__main__":
    unittest.main()
