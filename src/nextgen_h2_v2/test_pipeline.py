"""Small reproducibility contracts for PSD, exact gating, and both ranking learners."""
import argparse
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from nextgen_h2.kernels import normalized_gram, psd_sum
from .kernel import complete_fallback, kernel_function
from .nonlinear import Candidate, FilmAttention, fit_boost, listnet
from .portable import view_cross
from vc_build.io import publish_json

METRICS = {}


class Contracts(unittest.TestCase):
    def test_psd_all_kernels_match_independent_cross(self):
        torch.manual_seed(731)
        axis = torch.nn.functional.normalize(torch.randn(11, 8, dtype=torch.float64), dim=1)
        base = (axis @ axis.T).pow(2) + .2 * torch.eye(len(axis), dtype=torch.float64) + .1
        observed = torch.as_tensor([True, False, True, False, True, True, False, True, False, True, True])
        cut = 7
        for shape in ("cos", "cos2", "cos3", "rbf_half", "rbf", "rbf_wide"):
            gram = kernel_function(axis, shape)
            sum_psd = psd_sum(base, [(gram, observed)], [3.])
            norm = normalized_gram(sum_psd)
            xgram, _ = view_cross(axis[cut:], axis[:cut], shape)
            cross = base[cut:, :cut] + 3. * (observed[cut:, None] * xgram * observed[None, :cut])
            diagonal = sum_psd.diag().sqrt()
            cross = cross / diagonal[cut:, None] / diagonal[:cut][None, :]
            fit = psd_sum(base[:cut, :cut], [(gram[:cut, :cut], observed[:cut])],
                          [3.], diagonal_normalize=True)
            eigmin = min(torch.linalg.eigvalsh(x).min().item() for x in (sum_psd, norm))
            diff = max(float((cross - norm[cut:, :cut]).abs().max()),
                       float((fit - norm[:cut, :cut]).abs().max()))
            self.assertGreaterEqual(eigmin, 0.)
            torch.testing.assert_close(norm, norm.T, rtol=0., atol=1e-12)
            self.assertLess(diff, 1e-12)
            METRICS[shape] = {"minimum_eigenvalue": eigmin, "max_partition_diff": diff,
                              "symmetry_max_abs": float((norm - norm.T).abs().max())}

    def test_exact_fallback(self):
        base = torch.as_tensor([[.3, .7, .2], [.9, .8, .1]], dtype=torch.float64)
        after = base + 1
        cells = torch.tensor([False, True])
        drugs = torch.tensor([True, False, True])
        together = complete_fallback(after, base, cells, drugs)
        self.assertEqual(float((together[~cells] - base[~cells]).abs().max()), 0.)
        self.assertEqual(float((together[:, ~drugs] - base[:, ~drugs]).abs().max()), 0.)

    def test_nonlinear_losses_and_isolated_lightgbm(self):
        rng = np.random.default_rng(9)
        features = rng.standard_normal((1600, 12), dtype=np.float32)
        aligned = 2 * features[:, 0] + .1 * features[:, 1] ** 2
        panel = pd.DataFrame({"cell_id": np.repeat(np.arange(40), 40).astype(str),
                              "source_id": "S1", "aligned": aligned, "oof": .2 * features[:, 0]})
        held = np.arange(len(panel)) % 9 == 0
        for obj in ("rank", "huber", "l2"):
            model, curve = fit_boost(panel, features, held, Candidate("gbdt", obj, 15, 3, 0., trees=8))
            self.assertEqual(model.backend, "lightgbm")
            self.assertTrue(np.isfinite(model.predict(features[:50])).all())
            self.assertTrue(len(curve))
        seq = FilmAttention([4, 3, 2], [5, 4, 3], 6, 32, 3, .1)
        c = [torch.randn(14, i) for i in (4, 3, 2)]
        d = [torch.randn(14, i) for i in (5, 4, 3)]
        y = seq(c, d, torch.randn(14, 6))
        sample = listnet(y, torch.randn_like(y), torch.arange(14) // 2)
        sample.backward()
        self.assertTrue(float(sample) > 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Contracts)
    run = unittest.TextTestRunner(verbosity=2).run(suite)
    if not run.wasSuccessful():
        raise SystemExit(1)
    publish_json(args.out, {"tests": run.testsRun, "psd": METRICS})
