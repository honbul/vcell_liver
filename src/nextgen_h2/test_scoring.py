"""Exercise paired scoring and LICOB donor registration using synthetic outcomes."""
import unittest

import numpy as np
import pandas as pd

from .score import adult_licob, prereg_contrasts, report_scope, summary
from .metrics import DRUG, stack


class ScoreSmoke(unittest.TestCase):
    def test_same_support_metrics_and_exact_fallback(self):
        panel = pd.MultiIndex.from_product([list("abcdef"), list("ABCDEF")],
                                           names=["cell_id", "compound_id"]).to_frame(index=False)
        panel["lineage_group"] = panel.cell_id
        panel["source_id"] = "synthetic"
        panel["truth"] = np.random.default_rng(17).normal(size=len(panel))
        panel["C1"] = panel.truth
        panel["C0"] = -panel.truth
        mask = panel.compound_id.isin(list("ABC"))
        panel["structure_only"] = panel.C1
        panel["champion"] = panel.C1
        panel["champion_mask"] = panel.C1
        panel["champion_shuffle"] = panel.C0
        panel["champion_observed"] = mask
        for view in ("painting", "lincs"):
            panel[view + "_observed"] = mask
            for comparator in ("C2", "C3", "C4"):
                panel[view + "/" + comparator] = panel.C1 if comparator == "C4" else panel.C0
        clusters = pd.Series(dict(zip(list("ABCDEF"), list("AACCFF"))))
        drug, cell = stack(report_scope(panel, clusters, 0, "synthetic", interaction=True))
        effects = prereg_contrasts(drug, cell)
        observed = effects.query("support == 'lincs_observed' and control == 'lincs/C2' and metric == 'spearman'")
        np.testing.assert_allclose(observed.delta, [2.])
        fallback = effects.query("support == 'structure_only'")
        np.testing.assert_array_equal(fallback.delta, 0.)
        self.assertFalse(summary(drug, DRUG).empty)

    def test_adult_repeat_aggregation_keeps_support(self):
        rows = pd.DataFrame({"row_id": ["r1", "r2", "r3"], "cell_id": ["x1", "x2", "x3"],
                             "lineage_group": ["donor", "donor", "minor"], "compound_id": ["A"] * 3,
                             "source_id": ["LICOB"] * 3, "endpoint_id": ["AUC"] * 3,
                             "truth": [.1, .3, .2], "champion_observed": [True] * 3})
        manifest = pd.DataFrame({"row_id": rows.row_id, "donor_id": rows.lineage_group,
                                 "adult_age_confirmed": [True, True, False], "in_source": [True] * 3,
                                 "remaining_unspecified_stereo": [0] * 3})
        group = adult_licob(rows, manifest)
        self.assertEqual(group.cell_id.to_list(), ["donor"])
        self.assertTrue(group.champion_observed.iloc[0])
        self.assertAlmostEqual(group.truth.iloc[0], .2)


if __name__ == "__main__":
    unittest.main()
