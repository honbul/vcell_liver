"""Check score-only metadata never joins the R18 predictor's extraction list."""

from importlib.util import find_spec
from types import SimpleNamespace
import unittest

import pandas as pd


@unittest.skipUnless(find_spec("h5py"), "requires the G1 Level 3 input environment")
class R18AccessPlanTests(unittest.TestCase):
    def setUp(self):
        from src.r19_plan.r18_access_plan import slice_plan

        self.plan = slice_plan
        self.rotation = SimpleNamespace(case="C1", scenario="B", fit="A", select="B", score="C")
        rows = pd.DataFrame([
            dict(study="L", inst_id=well, parent=parent, cell_id="HEPG2", hours="24.0",
                 dose="10.0", rna_plate=f"plate{well}", rna_well="D01")
            for well, parent in zip("xyz", "pqr", strict=True)
        ])
        self.block = pd.concat([
            rows.iloc[[0]].assign(phase="inner_fit"), rows.iloc[[1]].assign(phase="select"),
            rows.iloc[:2].assign(phase="outer_fit"), rows.iloc[[2]].assign(phase="score"),
        ], ignore_index=True)
        self.controls = pd.DataFrame([
            dict(study="L", rna_plate=f"plate{well}", inst_id=f"{well}{role}", role=role)
            for well in "xyz" for role in "ABC"
        ])
        self.donors = pd.DataFrame([
            dict(study="L", inst_id=f"donor{parent}", parent=parent, cohort="C1",
                 hours="24.0", plate="donorPlate")
            for parent in "pqr"
        ])
        self.donor_controls = pd.DataFrame([
            dict(study="L", inst_id=f"donorControl{half}", plate="donorPlate", half=half)
            for half in ("1", "2")
        ])
        self.indices = pd.concat([
            rows[["study", "inst_id"]].assign(use="target_treatment"),
            self.controls[["study", "inst_id"]].assign(use="target_role_DMSO"),
            self.donors[["study", "inst_id"]].assign(use="B_donor"),
            self.donor_controls[["study", "inst_id"]].assign(use="B_donor_DMSO"),
        ], ignore_index=True).assign(matrix="combined", level3_row_index=range(17))

    def build(self):
        return self.plan(self.block, self.rotation, self.controls, self.donors,
                         self.donor_controls, self.indices)

    def test_registered_B_query_has_features_and_donor_but_no_evaluation_expression(self):
        fit, score = self.build()
        self.assertEqual(set(fit["query_features"].inst_id), {"z"})
        self.assertNotIn("level3_row_index", fit["query_features"])
        self.assertEqual(set(fit["B_donor_inputs"].inst_id), {"donorp", "donorq", "donorr"})
        self.assertEqual(set(fit["inner_and_outer_labels"].inst_id), {"x", "y"})
        self.assertEqual(set(fit["fit_controls"].inst_id), {"xA", "yA", "zA"})
        self.assertEqual(set(fit["select_controls"].inst_id), {"yB"})
        self.assertEqual(set(score["score_labels"].inst_id), {"z"})
        self.assertEqual(set(score["score_controls"].inst_id), {"zC"})
        self.rotation.scenario = "A"
        no_donor, _ = self.build()
        self.assertTrue(no_donor["B_donor_inputs"].empty and no_donor["B_donor_controls"].empty)

    def test_rejects_held_parent_in_outer_fit(self):
        self.block.loc[self.block.inst_id.eq("x"), "parent"] = "r"
        with self.assertRaisesRegex(ValueError, "scored parent"):
            self.build()

    def test_rejects_score_DMSO_alias_in_fit_controls(self):
        self.controls.loc[self.controls.inst_id.eq("zA"), "inst_id"] = "zC"
        with self.assertRaisesRegex(ValueError, "score-only physical well"):
            self.build()
