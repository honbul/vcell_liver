from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from nextgen_h1_anchor.metrics import batch_stats, sample_summaries
from nextgen_h1_anchor.score import Task
from nextgen_h1_v2.anchors import extend_reserve, greedy, restricted_order
from nextgen_h1_v2.boost import region
from nextgen_h1_v2.data import TASKS, partitions, training_context
from nextgen_h1_v2.neural import SetProcess
from nextgen_h1_v2.score import fit_metrics, verify_prelabel
from nextgen_h1_v2.search import extrapolate, fit_model, seed_grid, template


class H1V2Contract(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(112)
        n, d = 33, 31
        self.data = {
            "y": self.rng.normal(size=(n, 4)) @ self.rng.normal(size=(4, d)),
            "x": np.empty((n, 0)), "groups": np.array([f"u{i}" for i in range(n)]),
            "tissue": np.array(["Liver"] * n), "prior_names": np.array([], dtype=str),
        }
        fingerprint = self.rng.normal(size=(d, 24))
        gram = (fingerprint @ fingerprint.T) / 24 + np.eye(d) * .5
        sim = np.exp(-np.square(fingerprint[:, None, :] - fingerprint[None, :, :]).mean(-1))
        self.drug = (gram, sim, sim, sim, fingerprint)
        a, b = next(partitions(self.data["groups"], "test"))
        self.train, self.val = a, b
        self.ctx = training_context("LIMORE_to_LICOB", self.data, a, b, self.drug)
        dev = {k: v[a] if k in {"x", "y", "groups", "tissue"} else v for k, v in self.data.items()}
        self.nested = [training_context("LIMORE_to_LICOB", dev, f, v, self.drug)
                       for f, v in partitions(dev["groups"], "nested")]
        self.a8 = np.arange(8)
        self.a12 = extend_reserve(self.ctx, self.a8, 4)

    def test_supervised_order_cannot_open_outer_or_move_query(self):
        q = region(self.a12, 5, self.ctx.train.shape[1])
        picked = greedy(self.nested, self.a8, q, dual=False, joint=True)
        self.assertEqual(set(picked), set(self.a8))
        self.assertEqual(set(restricted_order(self.ctx, self.a8, "logdet")), set(self.a8))
        self.assertEqual(len(region(self.a12, 12, self.ctx.train.shape[1])), len(q) - 4)
        for p in self.nested:
            self.assertFalse(set(p.groups) & set(self.ctx.valid_groups))
            self.assertFalse(set(p.valid_groups) & set(self.ctx.valid_groups))

    def test_four_family_smoke_on_every_budget(self):
        examples = [template("A") | {"method": "product", "ridge": .003},
                    template("B") | {"rounds": 3}, template("C") | {"epochs": 2, "hidden": 24},
                    template("D") | {"rounds": 3}]
        for cfg in examples:
            fitted = fit_model(self.ctx, self.nested, self.a12, False, cfg, "cpu", {})
            for k in (0, 1, 3, 5, 8, 12):
                predictions = fitted.predict(self.ctx.valid[:, self.a12[:k]], k)
                q = region(self.a12, k, self.ctx.train.shape[1])
                self.assertEqual(predictions.shape, self.ctx.valid.shape)
                self.assertTrue(np.isfinite(predictions[:, q]).all())

    def test_set_encoder_is_invariant_to_anchor_order(self):
        net = SetProcess(24, 24, 4, .2).eval()
        d = torch.as_tensor(self.drug[-1], dtype=torch.float32)
        p = torch.randn(6, 16)
        y = torch.randn(6, 5, 2)
        m = torch.tensor([[True, False, True, True, True]] * len(y))
        ids = torch.tensor([0, 1, 2, 3, 4])
        perm = torch.tensor([3, 0, 1, 4, 2])
        with torch.no_grad():
            one = net(d, p, y, m, ids, True)
            two = net(d, p, y[:, perm], m[:, perm], ids[perm], True)
        torch.testing.assert_close(one, two)

    def test_bounded_weights_and_ridge_expansion(self):
        grid = seed_grid("A", True)
        winning = template("A") | {"ridge": .003, "width": 16., "rna_weight": .95}
        candidates, edges = extrapolate(winning, grid, True)
        self.assertIn(.0015, {c["ridge"] for c in candidates})
        self.assertIn(32., {c["width"] for c in candidates})
        self.assertIn(1., {c["rna_weight"] for c in candidates})
        self.assertTrue(any(e["parameter"] == "ridge" for e in edges))

    def test_fast_metrics_preserve_old_scoring_math(self):
        y = self.rng.normal(size=(19, 25))
        y[self.rng.random(y.shape) < .12] = np.nan
        p = self.rng.normal(size=y.shape)
        t = Task("LICOB-0", y, {"model/5": p}, np.arange(len(y)), np.array(["Liver"] * len(y)), None)
        t.drawn_indices = np.vstack((np.arange(len(y)), self.rng.integers(0, len(y), size=(5, len(y)))))
        _, metrics, _, _ = fit_metrics(t)
        indices = t.drawn_indices
        expected = batch_stats(y[indices], p[indices], sample_summaries(y, p)[indices])
        np.testing.assert_allclose(metrics["model/5"], expected, equal_nan=True)

    def test_score_prelabel_checks_coverage_and_both_seals(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            frozen, forecast, prelabel = (root / field for field in ("frozen", "forecast", "prelabel"))
            frozen.write_bytes(b"models")
            forecast.write_bytes(b"predictions")
            stamp = {"holdout_outcomes_opened": False, "tasks": dict.fromkeys(TASKS),
                     "sealed_models": sha256(frozen.read_bytes()).hexdigest(),
                     "sealed_forecasts": sha256(forecast.read_bytes()).hexdigest()}
            prelabel.write_text(json.dumps(stamp))
            self.assertEqual(verify_prelabel(prelabel, frozen, forecast), sha256(prelabel.read_bytes()).hexdigest())
            changed = [dict(stamp, holdout_outcomes_opened=True), dict(stamp, sealed_models="bad"),
                       dict(stamp, sealed_forecasts="bad"), dict(stamp, tasks={})]
            for receipt in changed:
                prelabel.write_text(json.dumps(receipt))
                with self.assertRaises(ValueError):
                    verify_prelabel(prelabel, frozen, forecast)


if __name__ == "__main__":
    unittest.main()
