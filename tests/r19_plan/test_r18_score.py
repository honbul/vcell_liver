"""Verify the separate R18 scoring path on synthetic expression only."""

from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


@unittest.skipUnless(find_spec("h5py"), "requires the G1 R18 input environment")
class R18ScoreTests(unittest.TestCase):
    def setUp(self):
        from src.r19_plan import r18_score

        self.evaluator = r18_score

    def test_score_rejects_bad_freeze_before_any_extraction(self):
        module = self.evaluator
        args = SimpleNamespace(forecasts=None, access=None, access_ref=None, protocol=None,
                               d0_ref=None, freeze_file=None, expected_sha256="incorrect")
        with (patch.object(module, "forecast_manifest", return_value={}),
              patch.object(module, "check_freeze", side_effect=ValueError("changed freeze")),
              patch.object(module, "load_observed") as observed):
            with self.assertRaisesRegex(ValueError, "changed freeze"):
                module.score(args)
        observed.assert_not_called()

    def test_vehicle_means_keep_study_role_and_well_order(self):
        rows = pd.DataFrame({"study": ["A", "B", "A"], "rna_plate": ["R", "R", "R"]})
        controls = rows.assign(role="C")
        values = np.array([[0., 2.], [4., 3.], [2., 4.]])
        means = self.evaluator.controls_for(rows, controls, values, "C")
        np.testing.assert_array_equal(means, [[1., 3.], [4., 3.], [1., 3.]])
        with self.assertRaisesRegex(ValueError, "role changed"):
            self.evaluator.controls_for(rows, controls, values, "A")
        with self.assertRaisesRegex(ValueError, "Missing registered"):
            self.evaluator.controls_for(rows.assign(rna_plate="S"), controls, values, "C")

    def test_repeat_and_contribution_diagnostics_use_same_support(self):
        from src.r19_plan.r18_score_contract import PRIMARY, frozen_arms, summarize_case

        module = self.evaluator
        rows = pd.DataFrame({
            "study": ["S"] * 5, "parent": ["p", "p", "p", "r", "r"], "cell_id": ["HEPG2"] * 5,
            "hours": [24] * 5, "dose": ["0.1", "0.1", "0.2", "0.1", "0.1"],
            "rna_plate": ["P", "Q", "R", "P", "Q"], "rna_well": ["A1", "B1", "A1", "A1", "A1"],
        })
        truth = np.arange(1, 11).reshape(5, 2).astype(float)
        predictions = {name: np.zeros_like(truth) for name in frozen_arms("A")}
        predictions[PRIMARY] = truth
        from collections import defaultdict

        measures, repeats = {}, {}
        behavior = defaultdict(lambda: np.zeros((2, 3)))
        module.accumulate(measures, repeats, behavior, "C2", "A", rows,
                          (truth, 1.5 * truth, truth), predictions)
        self.assertEqual(set(measures), {("C2:0.1", "A"), ("C2:0.2", "A")})
        repeat_points, _ = module.repeat_stats(repeats)
        repeat = {row["mode"]: row for row in repeat_points if row["arm"] == "repeat"}
        self.assertEqual(repeat["other_plate"]["parents"], 2)
        self.assertEqual(repeat["other_plate_and_position"]["parents"], 1)
        self.assertFalse(any(row["scope"] == "C2:0.2" for row in repeat_points))
        _, compared = summarize_case(measures["C2:0.1", "A"], "C2:0.1", "A")
        contributions, leave_out = module.contribution_tables(measures, compared)
        target = [row for row in contributions if row["arm"] == PRIMARY and row["reference"] == "B0"]
        self.assertEqual({(row["parent"], row["metric"]) for row in target},
                         {(p, m) for p in ("p", "r") for m in ("rmse", "r2_pooled")})
        self.assertTrue(all(row["remaining"] == 0 and np.isnan(row["difference"]) for row in leave_out))

    def test_history_checks_each_frozen_report_receipt(self):
        reports = Path(__file__).resolve().parents[2] / "reports"
        latest = [{"scope": "C1", "scenario": "A", "arm": "blend_FiLM", "reference": "B0"}]
        history = self.evaluator.score_history(reports, latest)
        self.assertEqual(set(history.version), {"R15", "R16", "R17", "R18"})
        self.assertEqual(set(history.loc[history.version.eq("R18"), "scope"]), {"C1"})
