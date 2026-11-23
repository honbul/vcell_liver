from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from vc_build.evaluate import rank_metrics
from nextgen_h1_anchor.metrics import Retrieval, batch_stats, center2, sample_summaries
from nextgen_h1_anchor.models import AFFINE_SHIFT_PENALTY, Fit, config, mean_scale, select_anchors
from nextgen_h1_anchor.outer import derange
from nextgen_h1_anchor.score import Task, bootstrap, claim_score
from nextgen_h1_anchor.tune import baseline_grid


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

    def test_affine_shrinks_shift_and_keeps_scale_positive(self):
        y = self.rng.normal(size=(40, 20))
        m, s = mean_scale(y)
        fit = Fit(y, m, s, np.arange(8))
        for k in (0, 3, 5, 8):
            obs = m[:k] + 2.
            fixed = fit.predict(obs[None, :], k, baseline_grid(False)["B3"][0])[0]
            legacy = fit.predict(obs[None, :], k, config("affine", base="mean"))[0]
            np.testing.assert_allclose(fixed, m + 2 * k / (k + AFFINE_SHIFT_PENALTY))
            np.testing.assert_allclose(legacy, m + (2. if k else 0.))
        # A deliberately inverted anchor response still yields a positive slope.
        fixed = fit.predict((-1000 * m[:8])[None, :], 8, baseline_grid(False)["B3"][0])[0]
        np.testing.assert_array_equal(np.argsort(fixed), np.argsort(m))

    def test_additive_interaction_is_zero_on_a_sparse_mask(self):
        y = self.rng.normal(size=(3, 19, 40))
        y[self.rng.random(y.shape) < .2] = np.nan
        p = self.rng.normal(size=(3, 19, 1)) + self.rng.normal(size=(3, 1, 40))
        sample = np.stack([sample_summaries(a, b) for a, b in zip(y, p)])
        values = batch_stats(y, p, sample)
        np.testing.assert_array_equal(values[:, 5:7], 0.)
        # Nondegenerate interactions still survive scaling.
        for scale in (.0001, 10000.):
            np.testing.assert_allclose(center2(y * scale) / scale, center2(y), atol=1e-12)

    def test_second_score_claim_is_rejected_under_a_different_output(self):
        with TemporaryDirectory() as path:
            base = Path(path)
            (base / "receipt.json").write_text("{}")
            (base / "forecast_receipt.json").write_text("{}")
            claim = claim_score(base, base, base / "original")
            self.assertTrue(claim.exists())
            with self.assertRaises(FileExistsError):
                claim_score(base, base, base / "second")
            self.assertFalse((base / "second").exists())

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
