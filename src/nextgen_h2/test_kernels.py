"""Explicit small matrix comparisons for drug kernels and residual fallback."""
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

from .kernels import guarded, normalized_gram, psd_sum, rbf, shuffled, stress_mask, view_geometry
from vc_build.structured_kernel import solve_residuals


class KernelInvariants(unittest.TestCase):
    evidence = {}
    def test_psd_and_cross_block(self):
        torch.manual_seed(9)
        chem = torch.randn(7, 5, dtype=torch.float64)
        target = torch.randn(7, 3, dtype=torch.float64)
        v1 = torch.randn(7, 4, dtype=torch.float64)
        v2 = torch.randn(7, 6, dtype=torch.float64)
        a = torch.tensor([True, False, True, False, True, True, False])
        b = torch.tensor([False, True, False, False, True, False, True])
        base = rbf(chem) + .4 * (target @ target.T)
        k1, k2 = v1 @ v1.T, rbf(v2)
        unnormalized = psd_sum(base, [(k1, a), (k2, b)], [.2, .5])
        combined = psd_sum(base, [(k1, a), (k2, b)], [.2, .5], diagonal_normalize=True)
        for gram in (unnormalized, combined):
            torch.testing.assert_close(gram, gram.T, atol=1e-12, rtol=0)
            self.assertGreaterEqual(float(torch.linalg.eigvalsh(gram).min()), 0)
        cut = 4
        fitted = psd_sum(base[:cut, :cut], [(k1[:cut, :cut], a[:cut]), (k2[:cut, :cut], b[:cut])],
                         [.2, .5], diagonal_normalize=True)
        torch.testing.assert_close(fitted, combined[:cut, :cut], rtol=0., atol=1e-12)
        cross = (base[cut:, :cut]
                 + .2 * (a[cut:, None] * k1[cut:, :cut] * a[None, :cut])
                 + .5 * (b[cut:, None] * k2[cut:, :cut] * b[None, :cut]))
        rebuilt = cross / unnormalized.diagonal()[cut:, None].sqrt() / unnormalized.diagonal()[None, :cut].sqrt()
        torch.testing.assert_close(rebuilt, combined[cut:, :cut], rtol=0., atol=1e-12)
        self.evidence.update({
            "unnormalized_minimum_eigenvalue": float(torch.linalg.eigvalsh(unnormalized).min()),
            "normalized_minimum_eigenvalue": float(torch.linalg.eigvalsh(combined).min()),
            "normalized_max_asymmetry": float((combined - combined.T).abs().max()),
            "train_max_abs_diff": float((fitted - combined[:cut, :cut]).abs().max()),
            "cross_max_abs_diff": float((rebuilt - combined[cut:, :cut]).abs().max()),
        })
        torch.testing.assert_close(normalized_gram(unnormalized), combined)

    def test_solver_against_dense_prediction(self):
        cell_x = torch.tensor([[1., .2], [.2, 1.]], dtype=torch.float64)
        drug_x = torch.tensor([[1., .5, .2], [.5, 1., .3], [.2, .3, 1.]], dtype=torch.float64)
        weights = torch.tensor([[1., 0., 2.], [.3, 1., 1.]], dtype=torch.float64)
        y = torch.tensor([[1., 2., 3.], [.3, 1.2, -.7]], dtype=torch.float64)
        s = weights.sqrt().ravel()
        expected = torch.linalg.solve(torch.eye(6) * 4 + s[:, None] * torch.kron(cell_x, drug_x) * s[None, :],
                                      s * y.ravel()).reshape(y.shape) * weights.sqrt()
        coefficients, _, _ = solve_residuals(cell_x, drug_x, weights, y, 4., tolerance=1e-10)
        torch.testing.assert_close(cell_x @ coefficients @ drug_x, cell_x @ expected @ drug_x, atol=1e-8, rtol=0)

    def test_finite_geometry_with_large_cached_features(self):
        rng = np.random.default_rng(43)
        raw = rng.normal(size=(15, 12))
        raw[:, 2] *= 1e19
        raw[2, 3] = np.nan
        raw[-1] *= 100
        primary = view_geometry(raw, 12, "white", 8, "rbf", "cpu")
        induced = view_geometry(raw[:12], 12, "white", 8, "rbf", "cpu")
        self.assertTrue(torch.isfinite(primary.gram).all())
        torch.testing.assert_close(primary.gram[:12, :12], induced.gram, atol=1e-6, rtol=0)

    def test_gating_mask_and_permutation(self):
        base = torch.arange(12, dtype=torch.float64).reshape(2, 6) / 7
        extra = 2. * base + 1.
        mask = torch.tensor([True, False, True, True, True, False])
        value = guarded(base, extra, mask, additive=True)
        self.assertTrue(torch.equal(value[:, ~mask], base[:, ~mask]))
        torch.testing.assert_close(value[:, mask], base[:, mask] + extra[:, mask])
        profiles = torch.arange(6, dtype=torch.float64).view(-1, 1)
        perm = shuffled(profiles, mask, 3, 3057).ravel().numpy()
        np.testing.assert_array_equal(np.sort(perm[:3]), np.arange(3.))
        np.testing.assert_array_equal(np.sort(perm[3:]), np.arange(3., 6.))
        self.assertEqual(perm[1], 1.)
        self.assertEqual(perm[5], 5.)
        dropped = stress_mask(mask, 3, .5, 3)
        self.assertTrue(torch.equal(dropped[:3], mask[:3]))
        self.assertEqual(int(dropped[3:].sum()), 1)


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) == 2 else None
    test = unittest.main(argv=[sys.argv[0]], exit=False)
    if not test.result.wasSuccessful():
        raise SystemExit(1)
    if path:
        with path.open("x") as output:
            json.dump({"tests": test.result.testsRun, **KernelInvariants.evidence}, output, indent=2)
