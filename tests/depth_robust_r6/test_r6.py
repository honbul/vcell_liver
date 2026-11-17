"""R6 contract tests (run: python -m unittest tests.depth_robust_r6.test_r6).

    p = 1 identity                     thinning at p = 1 returns the counts unchanged
    library accounting                 thinned library = thinned panel sum + thinned rest; never above the original
    barcode duplication                no input cell holds two roles; roles are barcode-disjoint
    teacher stop-gradient              the consistency terms give the teacher no gradient
    no access to held-out treated RNA  model code never opens treated-cell arrays; the sampler never
                                       yields a non-fitting drug; final training never predicts
    condition-order invariance         predictions do not depend on the order of conditions or of cells in a set
"""
from __future__ import annotations

import csv
import gzip
import re
import unittest
from pathlib import Path

import numpy as np
import torch

from src.depth_robust_r6 import data as D
from src.depth_robust_r6 import loss as LS
from src.depth_robust_r6.count_view import library, lognorm, set_features, thin_np, thin_torch
from src.depth_robust_r6.encoder import ResponseModel

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "depth_robust_r6"


def toy(G=60, n_pool=4, n_lines=2, n_drugs=9, seed=0) -> D.R6Data:
    import pandas as pd
    rng = np.random.default_rng(seed)
    per = D.EVAL_N + 80
    cnt = rng.poisson(0.6, size=(n_pool * per, G)).astype(np.int16)
    crest = rng.poisson(300, size=n_pool * per).astype(np.int32)
    pools = pd.DataFrame(dict(pool_id=range(n_pool), plate_index=[0, 0, 1, 1], line=[0, 1, 0, 1],
                              eval_start=[i * per for i in range(n_pool)], eval_end=[i * per + D.EVAL_N for i in range(n_pool)],
                              fit_start=[i * per + D.EVAL_N for i in range(n_pool)], fit_end=[(i + 1) * per for i in range(n_pool)],
                              n_fit_input=80, n_reference=200, n_total=per + 200))
    pid = np.array([[0, 1], [2, 3]])
    Y = rng.normal(size=(n_lines, n_drugs, G)).astype(np.float32)
    valid = np.ones((n_lines, n_drugs), bool); valid[1, 0] = False
    plate = np.array([[d % 2 for d in range(n_drugs)]] * n_lines)
    splits = pd.DataFrame(dict(group_index=range(n_drugs), group_smiles=[f"C{'C' * i}" for i in range(n_drugs)],
                               split=["train"] * 5 + ["val"] * 2 + ["test"] * 2))
    ev = {}
    rows = np.concatenate([np.arange(r.eval_start, r.eval_end) for r in pools.itertuples()])
    for p in D.PS[1:]:
        for s in D.THIN_SEEDS:
            ev[(p, s)] = thin_np(cnt[rows], crest[rows], p, np.random.default_rng([s, int(p * 100)]))
    return D.R6Data(cnt=cnt, crest=crest, pools=pools, pool_id=pid, Y=Y, valid=valid, plate=plate,
                    fp=(rng.random((n_drugs, 32)) < 0.3).astype(np.float32), splits=splits,
                    fold=np.array([0, 1, 2, 0, 2, -1, -1, -1, -1]), lines=None, x_mean=np.zeros(G, np.float32),
                    x_sd=np.ones(G, np.float32), tiebreak=np.linspace(0.49, 0, G).astype(np.float32), eval_views=ev, root=Path("."))


class Thinning(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.c = rng.poisson(1.0, size=(50, 40)).astype(np.int16)
        self.r = rng.poisson(500, size=50).astype(np.int32)

    def test_p1_identity_numpy(self):
        c, r = thin_np(self.c, self.r, 1.0, np.random.default_rng(0))
        self.assertTrue((c == self.c).all() and (r == self.r).all())

    def test_p1_identity_torch(self):
        c, r = thin_torch(torch.as_tensor(self.c), torch.as_tensor(self.r), 1.0)
        self.assertTrue((c.numpy() == self.c).all() and (r.numpy() == self.r).all())
        p = torch.tensor([1.0] * 25 + [0.25] * 25)
        c, r = thin_torch(torch.as_tensor(self.c), torch.as_tensor(self.r), p)
        self.assertTrue((c[:25].numpy() == self.c[:25]).all() and (r[:25].numpy() == self.r[:25]).all())
        self.assertLess(float(c[25:].sum()), float(self.c[25:].sum()))

    def test_library_accounting(self):
        for p in (0.5, 0.25):
            c, r = thin_np(self.c, self.r, p, np.random.default_rng(3))
            self.assertTrue((c <= self.c).all() and (r <= self.r).all() and (c >= 0).all())
            lib = library(c, r)
            self.assertTrue((lib == c.sum(1) + r).all())                     # the thinned total, not p * original
            self.assertTrue((lib <= library(self.c, self.r)).all())
            self.assertAlmostEqual(float(lib.sum() / library(self.c, self.r).sum()), p, delta=0.02)
            x = lognorm(c.astype(float), lib.astype(float))
            self.assertTrue(np.allclose(np.expm1(x).sum(1) + 1e4 * r / np.maximum(lib, 1), 1e4, atol=1e-6))
        ct, rt = thin_torch(torch.as_tensor(self.c), torch.as_tensor(self.r), 0.5)
        self.assertTrue(torch.equal(ct, ct.round()) and (ct <= torch.as_tensor(self.c)).all())
        f = set_features(ct[None], rt[None])
        self.assertTrue(torch.allclose(f["lib"], ct.sum(-1)[None] + rt[None]))

    def test_same_normalisation_at_every_p(self):
        c = np.array([[4, 0, 2]], dtype=np.int16); r = np.array([4], dtype=np.int32)
        self.assertTrue(np.allclose(lognorm(c.astype(float), np.array([10.0])), np.log1p(1e4 * c / 10.0)))


class Barcodes(unittest.TestCase):
    def test_roles_disjoint(self):
        f = ROOT / "manifests" / "depth_robust_r6" / "input_cells.tsv.gz"
        if not f.exists():
            self.skipTest("contract manifest not built yet")
        seen, roles = set(), {}
        with gzip.open(f, "rt") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                self.assertNotIn(row["barcode"], seen)                        # one role per barcode
                seen.add(row["barcode"])
                roles.setdefault((row["pool_id"], row["role"]), 0)
                roles[(row["pool_id"], row["role"])] += 1
        for (pool, role), n in roles.items():
            if role == "eval_input":
                self.assertEqual(n, D.EVAL_N)
            else:
                self.assertGreaterEqual(n, D.EVAL_N)

    def test_role_rule(self):
        self.assertIsNone(D.pool_roles(2 * D.EVAL_N + D.REF_MIN - 1))
        for n in (256, 300, 448, 5000):
            ne, nf, nr = D.pool_roles(n)
            self.assertEqual(ne + nf + nr, n)
            self.assertTrue(ne == D.EVAL_N and nf >= D.EVAL_N and nr >= D.REF_MIN and nf <= D.FIT_MAX)


class TeacherStopGradient(unittest.TestCase):
    def test_no_gradient_into_teacher(self):
        torch.manual_seed(0)
        d = toy()
        kw = dict(n_tok=16, n_blocks=1, d=16, x_mean=d.x_mean, x_sd=d.x_sd, tiebreak=d.tiebreak)
        student = ResponseModel(60, 32, obs_input=True, latent_proj=True, **kw)
        teacher = ResponseModel(60, 32, obs_input=True, **kw)                 # requires_grad left ON on purpose
        c = torch.as_tensor(d.cnt[:128]).float().reshape(2, 64, 60); r = torch.as_tensor(d.crest[:128]).float().reshape(2, 64)
        fp = torch.as_tensor(d.fp[:3])[None].expand(2, 3, 32)
        ct, rt = thin_torch(c, r, 0.25)
        ps, zs, _ = student(set_features(ct, rt), fp)
        pt, zt, _ = teacher(set_features(c, r), fp)
        (LS.response_consistency(ps, pt) + LS.latent_consistency(student.proj(zs), zt)).backward()
        self.assertTrue(all(q.grad is None or float(q.grad.abs().sum()) == 0.0 for q in teacher.parameters()))
        self.assertTrue(any(q.grad is not None and float(q.grad.abs().sum()) > 0 for q in student.parameters()))

    def test_supervised_term_always_present(self):
        parts = dict(supervised=torch.tensor(2.0), response=torch.tensor(3.0))
        self.assertEqual(float(LS.total(parts, dict(supervised=1.0, response=0.5))), 3.5)
        src = (SRC / "train.py").read_text()
        self.assertIn("LS.total(parts, dict(supervised=1.0, ", src)      # weight 1 on the measured target, always


class NoHeldOutTreatedAccess(unittest.TestCase):
    def test_model_code_never_opens_treated_cells(self):
        for name in ["train.py", "encoder.py", "loss.py", "count_view.py", "data.py", "ridge.py", "diagnostics.py"]:
            src = (SRC / name).read_text()
            for bad in ["tahoe_cells", "h5ad\"", "R3_CACHE", "X.npy", "test_pred", "test_truth"]:
                self.assertNotIn(bad, src, f"{name} references {bad}")

    def test_final_training_never_predicts(self):
        src = (SRC / "train.py").read_text()
        calls = [m.start() for m in re.finditer(r"predict\(model, d, eval_drugs", src)]
        self.assertEqual(len(calls), 2)
        for c in calls:
            guard = src.rfind("if a.setting == \"dev\"", 0, c)
            self.assertTrue(guard >= 0 and c - guard < 400, "prediction of evaluation drugs must sit under the dev guard")

    def test_sampler_yields_fitting_drugs_and_fit_cells_only(self):
        from src.depth_robust_r6.train import Sampler
        d = toy()
        fit, ev = d.split_spec("dev")
        self.assertFalse(set(fit) & set(ev))
        fin_fit, fin_ev = d.split_spec("final")
        self.assertFalse(set(fin_fit) & set(fin_ev))
        self.assertTrue(set(fit) | set(ev) == set(fin_fit))                    # development stays inside TRAIN drugs
        g = torch.Generator(); g.manual_seed(0)
        s = Sampler(d, fit, "cpu", g)
        eval_cells = set(np.concatenate([np.arange(r.eval_start, r.eval_end) for r in d.pools.itertuples()]).tolist())
        for _ in range(20):
            cells, L, Dg = s.draw(3, 4)
            self.assertTrue(set(Dg.flatten().tolist()) <= set(fit))
            self.assertFalse(set(cells.flatten().tolist()) & eval_cells)
            self.assertTrue(all(len(set(row.tolist())) == D.EVAL_N for row in cells))   # N distinct cells
            self.assertTrue(bool(torch.as_tensor(d.valid)[L, Dg].all()))
            pid = torch.as_tensor(d.pool_id)[torch.as_tensor(d.plate)[L, Dg], L]
            fs = torch.as_tensor(d.pools.fit_start.to_numpy())[pid[:, 0]]; fe = torch.as_tensor(d.pools.fit_end.to_numpy())[pid[:, 0]]
            self.assertTrue(bool(((cells >= fs[:, None]) & (cells < fe[:, None])).all()))  # the condition's own pool


class OrderInvariance(unittest.TestCase):
    def test_condition_and_cell_order(self):
        from src.depth_robust_r6.train import predict
        torch.manual_seed(0)
        d = toy()
        m = ResponseModel(60, 32, obs_input=True, n_tok=16, n_blocks=1, d=16, x_mean=d.x_mean, x_sd=d.x_sd, tiebreak=d.tiebreak)
        drugs = [0, 1, 2, 3, 4]
        L1, D1, p1 = predict(m, d, drugs, "cpu", 1.0, views=[(1.0, 0), (0.25, 0)], chunk=3)
        L2, D2, p2 = predict(m, d, drugs[::-1], "cpu", 1.0, views=[(1.0, 0), (0.25, 0)], chunk=7)
        k1 = {(l, g): i for i, (l, g) in enumerate(zip(L1, D1))}
        idx = [k1[(l, g)] for l, g in zip(L2, D2)]
        for v in p1:
            self.assertTrue(np.allclose(p1[v][idx], p2[v], atol=1e-5))
        m.eval()
        c = torch.as_tensor(d.cnt[:64]).float()[None]; r = torch.as_tensor(d.crest[:64]).float()[None]
        perm = torch.randperm(64)
        z1 = m.state(set_features(c, r))[0]; z2 = m.state(set_features(c[:, perm], r[:, perm]))[0]
        self.assertTrue(torch.allclose(z1, z2, atol=1e-4))


class AdditiveHead(unittest.TestCase):
    def test_drug_part_starts_at_zero_and_state_part_ignores_drug(self):
        torch.manual_seed(0)
        d = toy()
        m = ResponseModel(60, 32, obs_input=False, n_tok=16, n_blocks=1, d=16, head_mode="additive",
                          x_mean=d.x_mean, x_sd=d.x_sd, tiebreak=d.tiebreak).eval()
        c = torch.as_tensor(d.cnt[:128]).float().reshape(2, 64, -1); r = torch.as_tensor(d.crest[:128]).float().reshape(2, 64)
        z, _ = m.state(set_features(c, r))
        fp = torch.as_tensor(d.fp[:6]).reshape(2, 3, -1)
        g, h = m.delta_parts(z, fp)
        self.assertEqual(float(h.abs().max()), 0.0)                  # drug-specific part is zero at initialisation
        self.assertEqual(tuple(g.shape), (2, 1, 60))                 # one state-only response per control set
        self.assertTrue(torch.allclose(m.delta(z, fp), g.expand(-1, 3, -1)))
        j = ResponseModel(60, 32, n_tok=16, n_blocks=1, d=16, x_mean=d.x_mean, x_sd=d.x_sd, tiebreak=d.tiebreak)
        self.assertIsNone(j.state_head)                              # the original head is unchanged by default


if __name__ == "__main__":
    unittest.main()
