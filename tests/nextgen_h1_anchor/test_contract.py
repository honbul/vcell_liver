import unittest

import numpy as np

from vc_build.evaluate import rank_metrics
from nextgen_h1_anchor.metrics import Retrieval, batch_stats, center2, sample_summaries
from nextgen_h1_anchor.models import Fit, config, mean_scale, select_anchors
from nextgen_h1_anchor.outer import derange
from nextgen_h1_anchor.score import Task, bootstrap


class H1Contract(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(123)

    def test_expected_ties_and_observed_mask_match_registered_score(self):
        y = self.rng.integers(0, 6, size=(25, 32)).astype(float)
        pred = self.rng.integers(0, 6, size=y.shape).astype(float)
        y[self.rng.random(y.shape) < .15] = np.nan
        observed = Retrieval(y)(pred, full=True)
        expected = []
        for actual, forecast in zip(y, pred):
            measured = np.isfinite(actual)
            score = rank_metrics(actual[measured], forecast[measured])
            expected.append([score[k] for k in ("ndcg_at_5", "top5_overlap", "spearman")])
        np.testing.assert_allclose(observed, expected, atol=1e-12)

    def test_masked_center_and_bootstrap_shape(self):
        y = self.rng.normal(size=(18, 40))
        y[0, :2] = np.nan
        p = self.rng.normal(size=y.shape)
        residual = center2(y[None])[0]
        np.testing.assert_allclose(np.nansum(residual, axis=0), 0, atol=1e-10)
        np.testing.assert_allclose(np.nansum(residual, axis=1), 0, atol=1e-10)
        values = batch_stats(y[None], p[None], sample_summaries(y, p)[None])
        self.assertEqual(values.shape, (1, 9))
        self.assertTrue(np.isfinite(values).all())
        a = Task("GDSC2-0", y, {}, np.array(["a", "b", "c"]), np.array([]), None)
        b = Task("CTRPv2-0", y, {}, np.array(["a", "b", "d", "e"]), np.array([]), None)
        bootstrap([a, b])
        for repeated in ("a", "b"):
            j, k = list(a.group).index(repeated), list(b.group).index(repeated)
            np.testing.assert_array_equal((a.drawn_indices == j).sum(1), (b.drawn_indices == k).sum(1))

    def test_nested_anchors_and_blind_single_batch_replay(self):
        y = self.rng.normal(size=(40, 32))
        withheld = self.rng.normal(size=(8, 32))
        m, s = mean_scale(y)
        a = select_anchors(y, np.eye(y.shape[1]), "logdet")
        self.assertEqual(len(set(a)), 8)
        q = np.setdiff1d(np.arange(y.shape[1]), a)
        self.assertEqual(len(q), y.shape[1] - 8)
        self.assertFalse(np.intersect1d(q, a).size)
        model = Fit(y, m, s, a)
        recipe = config("cov", estimator="oas", ridge=.3)
        x = withheld[:, a[:5]]
        truth_changed = withheld.copy()
        truth_changed[:, q] = 9999999
        expected = model.predict(x, 5, recipe)[:, q]
        np.testing.assert_allclose(expected, model.predict(truth_changed[:, a[:5]], 5, recipe)[:, q])
        np.testing.assert_allclose(expected, np.concatenate([model.predict(row[None, :], 5, recipe)[:, q] for row in x]))
        with self.assertRaises(ValueError):
            model.predict(x[:, :-1], 5, recipe)
        shuffled = derange(len(x), seed=123)
        self.assertEqual(set(shuffled), set(range(len(x))))
        self.assertTrue(np.all(shuffled != np.arange(len(x))))


if __name__ == "__main__":
    unittest.main()
