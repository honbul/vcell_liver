"""Data-boundary tests for the research-only VCB state head."""

from unittest import TestCase

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.vc_build.state_data import exact_structure, unique_wells
from src.vc_build.state_eval import bootstrap_gain, grouped_squares, partitions
from src.vc_build.state_model import StateModel


class StateBoundaries(TestCase):
    def test_stereoisomers_share_split_group_but_keep_exact_identity(self):
        left, left_group = exact_structure("C[C@H](O)C(=O)O")
        right, right_group = exact_structure("C[C@@H](O)C(=O)O")
        self.assertTrue(left and right and left != right)
        self.assertEqual(left_group, right_group)
        self.assertEqual(exact_structure("CCO.CCO"), ("", ""))

    def test_physical_rna_well_cannot_have_two_ids(self):
        repeated = pd.DataFrame({"rna_plate": ["A", "A"], "rna_well": ["A01", "A01"], "inst_id": ["one", "two"]})
        with self.assertRaisesRegex(ValueError, "physical well"):
            unique_wells(repeated)

    def test_cell_cold_query_plans_drop_nonshared_drugs_and_never_share_plates(self):
        frame = pd.DataFrame([
            ("JHH5", "CVCL1", "p1", "cc"), ("JHH7", "CVCL2", "p2", "cc"),
            ("JHH7", "CVCL2", "p2", "jhh7_only"), ("A549", "CVCL3", "p3", "cc"),
        ], columns=["cell_id", "lineage_group", "rna_plate", "compound_id"])
        fit, query = partitions(frame, "C-both")
        self.assertEqual(set(fit.cell_id), {"A549"})
        self.assertEqual(set(query.cell_id), {"JHH5", "JHH7"})
        self.assertEqual(query.compound_id.tolist(), ["cc", "cc"])
        frame.loc[frame.cell_id.eq("JHH7"), "rna_plate"] = "p3"
        with self.assertRaisesRegex(ValueError, "boundaries"):
            partitions(frame, "C-both")

    def test_zero_vehicle_contrast_projects_to_zero_even_with_nonzero_fit_mean(self):
        y = np.array([[1., 2.], [3., 2.], [2., 3.]])
        pca = PCA(n_components=2).fit(y)
        model = StateModel(np.array([4790, 5594]), pca, np.ones(2), StandardScaler(), pca, pca,
                           pd.DataFrame(), np.empty((0, 0)), Ridge(), Ridge(), Ridge())
        self.assertTrue(np.any(pca.transform(np.zeros((1, 2)))))
        np.testing.assert_array_equal(model.project(np.zeros((1, 2))), np.zeros((1, 2)))

    def test_grouped_confidence_interval_scores_macro_relative_effect(self):
        square = grouped_squares(np.array([[3., 2.], [1., 4.], [1., 5.]]), ["a", "a", "b"])
        np.testing.assert_allclose(bootstrap_gain(square * 0.81, square), 0.10)
