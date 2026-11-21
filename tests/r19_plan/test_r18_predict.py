"""Synthetic R18 predictor tests without authoritative expression or GPU models."""

from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


@unittest.skipUnless(find_spec("h5py"), "requires the G1 R18 input environment")
class R18PredictTests(unittest.TestCase):
    def setUp(self):
        from src.r11_chemical_probe.common import sha256, write_json
        from src.r19_plan import r18_predict

        self.work = TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        self.root = Path(self.work.name)
        self.sha256, self.write_json = sha256, write_json
        self.adapter = r18_predict

    def fixture(self, case="C1"):
        from src.r15_lincs.design import fold_id
        from src.r19_plan.r18_access_plan import TARGET_FIELDS

        self.slice_id = f"{case}/A/f0-A"
        self.query_parent = next(p for p in ("C" * k for k in range(1, 30))
                                 if fold_id(p, "R15-outer", 3) == 0)
        candidates = [p for p in ("C" * k for k in range(1, 30)) if fold_id(p, "R15-outer", 3) != 0]
        self.fit_parent = [p for p in candidates if fold_id(p, "R15-inner-0", 5) != 0][:2]
        self.select_parent = next(p for p in candidates if fold_id(p, "R15-inner-0", 5) == 0)
        query_line = "HEPG2" if case == "C1" else "HUH7"
        labels = pd.DataFrame([
            ("GSE92742", "t1", self.fit_parent[0], "HEPG2", 6, 3., "P1", "A01", "inner_fit"),
            ("GSE92742", "t2", self.select_parent, "HEPG2", 6, 3., "P2", "A02", "select"),
            ("GSE92742", "t3", self.fit_parent[1], "HEPG2", 6, 3., "P2", "A01", "inner_fit"),
        ], columns=[*TARGET_FIELDS, "inner_phase"])
        query = pd.DataFrame([
            ("GSE92742", "q1", self.query_parent, query_line, 6, 3., "P3", "A01"),
            ("GSE92742", "q2", self.query_parent, query_line, 6, 3., "P3", "A02"),
        ], columns=TARGET_FIELDS)
        controls = pd.DataFrame([
            ("GSE92742", "a1", "P1", "C01", "A"), ("GSE92742", "a2", "P2", "C01", "A"),
            ("GSE92742", "a3", "P3", "C01", "A"), ("GSE92742", "b2", "P2", "C02", "B"),
        ], columns=["study", "inst_id", "rna_plate", "rna_well", "role"])
        empty = pd.DataFrame(columns=["study", "inst_id"])
        plan = {
            "inner_and_outer_labels": labels, "query_features": query, "fit_controls": controls.iloc[:3],
            "select_controls": controls.iloc[3:], "B_donor_inputs": empty, "B_donor_controls": empty,
        }
        arrays = {
            "inner_and_outer_labels": [[1, 3], [4, 6], [8, 10]],
            "fit_controls": [[0, 1], [1, 2], [2, 3]],
            "select_controls": [[1, 3]],
            "B_donor_inputs": np.empty((0, 2)), "B_donor_controls": np.empty((0, 2)),
        }
        self.bundle = self.root / "bundle"
        self.bundle.mkdir()
        self.plan_root = self.root / "locked" / "predict-plan"
        (self.plan_root / self.slice_id).mkdir(parents=True)
        registered = {}
        for name, rows in plan.items():
            filename = f"{name}.tsv"
            path = self.bundle / filename
            rows.to_csv(path, sep="\t", index=False)
            (self.plan_root / self.slice_id / filename).write_bytes(path.read_bytes())
            registered[filename] = self.sha256(path)
        genes = self.bundle / "landmarks.tsv"
        genes.write_text("pr_gene_id\tpr_gene_symbol\n1\tA\n6\tB\n")
        expression = {name: np.asarray(values, dtype=np.float32) for name, values in arrays.items()}
        np.savez(self.bundle / "expression.npz", **expression)
        self.access = self.root / "access-ref.json"
        self.d0 = self.root / "d0-ref.json"
        self.protocol = self.root / "protocol.md"
        self.protocol.write_text("Synthetic confirmation fixture\n")
        self.write_json(self.access, dict(metadata_only=True, expression_read=False,
            files={self.slice_id: {"predict": registered, "score": {"unavailable.tsv": "unavailable"}}}))
        self.write_json(self.d0, dict(landmarks=2, gene_axis_sha256={"GEO": self.sha256(genes)}))
        self.write_json(self.bundle / "receipt.json", dict(
            slice=self.slice_id, access_sha256=self.sha256(self.access), predict_plan_sha256=registered,
            arrays={name: list(value.shape) for name, value in expression.items()},
            files={file.name: self.sha256(file) for file in self.bundle.iterdir()},
        ))
        return self.adapter.PredictionBank(self.bundle, self.access, self.d0, self.slice_id)

    def test_prediction_access_guards_and_C3_position(self):
        bank = self.fixture("C3_6h_HEPG2_only")
        inner, selection, _, query = bank.split()
        np.testing.assert_array_equal(bank.response(inner, "A"), [[1, 2], [7, 8]])
        np.testing.assert_array_equal(bank.response(selection, "A"), [[3, 4]])
        np.testing.assert_array_equal(bank.response(selection, "B"), [[3, 3]])
        with self.assertRaisesRegex(ValueError, "query treatment response"):
            bank.response(query, "A")
        with self.assertRaisesRegex(ValueError, "score-role"):
            bank.control(query, "C")
        with self.assertRaisesRegex(ValueError, "only use the registered validation"):
            bank.response(inner, "B")
        np.testing.assert_array_equal(bank.control(query, "A"), [[2, 3], [2, 3]])
        observed, _, held_position = self.adapter.cohort_position(bank, inner, query, "A")
        np.testing.assert_array_equal(observed, [[1, 2], [7, 8]])
        np.testing.assert_array_equal(held_position, np.zeros((len(query), 2)))
        np.testing.assert_array_equal(bank.transfer(query), np.zeros((len(query), 2)))
        self.assertEqual(bank.chemistry(query).shape, (len(query), 2048))
        with (self.bundle / "query_features.tsv").open("a") as changed:
            changed.write("\n")
        with self.assertRaisesRegex(ValueError, "changed after registration"):
            self.adapter.PredictionBank(self.bundle, self.access, self.d0, self.slice_id)

    def test_B_donor_aggregates_halves_lines_and_releases_independent_of_order(self):
        names = pd.DataFrame([
            ("GSE70138", "P1", "p", 24, "A549"), ("GSE70138", "P1", "p", 24, "A549"),
            ("GSE70138", "P2", "p", 24, "MCF7"), ("CMap2020", "P1", "p", 24, "A549"),
        ], columns=["study", "plate", "parent", "hours", "cell_id"])
        controls = pd.DataFrame([
            ("GSE70138", "P1", "1"), ("GSE70138", "P1", "1"), ("GSE70138", "P1", "2"),
            ("GSE70138", "P2", "1"), ("GSE70138", "P2", "2"),
            ("CMap2020", "P1", "1"), ("CMap2020", "P1", "2"),
        ], columns=["study", "plate", "half"])
        values = np.asarray([[4], [8], [10], [13]], dtype=np.float32)
        vehicles = np.asarray([[0], [0], [4], [4], [2], [3], [5]], dtype=np.float32)
        # Effective release responses are (4 + 7) / 2 and 9. Equal release weighting gives 7.25.
        expected = np.array([7.25])
        output = self.adapter.donor_means(names, values, controls, vehicles)
        np.testing.assert_array_equal(output[("p", 24)], expected)
        rev = self.adapter.donor_means(names.iloc[::-1], values[::-1], controls.iloc[::-1], vehicles[::-1])
        np.testing.assert_array_equal(rev[("p", 24)], expected)

    def test_shared_fit_dispatch_and_merge_have_old_positions_and_exact_axes(self):
        from src.r15_lincs.forecast import position
        from src.r19_plan.r18_score_contract import inspect_forecast

        bank = self.fixture()
        inner, validation, _, _ = bank.split()
        original = position(bank, inner, validation, "A")
        adapted = self.adapter.cohort_position(bank, inner, validation, "A")
        for before, after in zip(original, adapted, strict=True):
            np.testing.assert_array_equal(before, after)

        args = SimpleNamespace(parts=self.root / "parts", out=self.root / "forecasts", protocol=self.protocol)
        r17 = self.adapter.r17

        def inner_known(args, bank, fit, select, observed, truth, anchor, model, work):
            self.assertTrue(set(fit.parent).isdisjoint(select.parent))
            np.testing.assert_array_equal(truth, [[3, 3]])
            return (1,), np.ones_like(truth), (1,)

        def fit_known(args, bank, training, query, labels, truth, anchor, model, work, parameters):
            self.assertIsNone(truth)
            self.assertEqual(parameters, (1,))
            self.assertTrue(set(training.parent).isdisjoint(query.parent))
            return np.full_like(anchor, .5)

        with (patch.object(self.adapter, "asset_fingerprint", return_value={}),
              patch.object(r17, "inner_known", side_effect=inner_known),
              patch.object(r17, "fit_known", side_effect=fit_known),
              patch.object(r17, "choose_proposed", return_value=((('tanimoto', 1.), .5), .2)),
              patch.object(bank, "response", wraps=bank.response) as reads):
            for model in self.adapter.MODELS:
                args.model = model
                self.adapter.predict(args, bank)
            self.assertTrue(all(self.query_parent not in call.args[0].parent.to_numpy()
                                and call.args[1] in "AB" for call in reads.call_args_list))
        self.adapter.merge(args, bank)
        result = args.out / self.slice_id
        digest = inspect_forecast(result, self.slice_id, bank.predict_plan_sha256, self.plan_root,
                                  self.sha256(self.protocol), self.sha256(self.bundle / "landmarks.tsv"), 2)
        self.assertEqual(digest, self.sha256(result / "receipt.json"))
        self.assertEqual((result / "rows.tsv").read_bytes(), (self.bundle / "query_features.tsv").read_bytes())
        self.assertFalse((self.root / "locked" / "score-plan").exists())
        with self.assertRaises(FileExistsError):
            self.adapter.merge(args, bank)
