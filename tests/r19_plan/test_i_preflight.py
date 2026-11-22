"""Exercise the I-track metadata lock with no expression files available."""

from importlib.util import find_spec
import unittest
from zlib import crc32

import pandas as pd


@unittest.skipUnless(find_spec("h5py"), "requires the LINCS metadata environment")
class IPreflightTests(unittest.TestCase):
    def setUp(self):
        from src.r15_lincs.design import fold_id
        from src.r19_plan import i_preflight

        self.pre = i_preflight
        parents = ["C" * i for i in range(1, 31)]
        self.pop = pd.DataFrame([
            dict(inst_id=f"treated-{i}", sig_id=f"sig-{i}", parent=parent, cell_id="HEPG2",
                 hours="6.0", dose="10.0", rna_plate="P", rna_well=f"D{i:02}", det_plate="",
                 det_well="", outer_fold=fold_id(parent, "R15-outer", 3))
            for i, parent in enumerate(parents, 1)
        ])
        self.cache = self.pop.assign(index=list(reversed(range(len(parents)))), quality="True", prior_exact="False")
        ids = sorted(("DMSO-1", "DMSO-2", "DMSO-3"), key=lambda key: (crc32(key.encode()), key))
        self.controls = pd.DataFrame([
            dict(inst_id=key, rna_plate="P", rna_well=f"A{i:02}", role=role, study="GSE92742")
            for i, (key, role) in enumerate(zip(ids, "ABC", strict=True), 1)
        ])

    def test_nine_disjoint_rotations_have_no_held_cache_index_in_predictor(self):
        target = self.pre.register_targets(self.pop, self.cache)
        actual = self.pre.register_control_indices(self.controls, pd.DataFrame({"rna_plate": ["P"]}))
        plans = list(self.pre.split_plans(target, actual))
        self.assertEqual(len(plans), 9)
        for support, predictor, scorer in plans:
            query, hidden = predictor["query_features.tsv"], scorer["score_labels.tsv"]
            labels = predictor["inner_and_outer_labels.tsv"]
            self.assertNotIn("cache_index", query.columns)
            self.assertEqual(list(query.inst_id), list(hidden.inst_id))
            self.assertEqual(set(query.scaffold_status), {"acyclic"})
            self.assertFalse(set(hidden.parent) & set(labels.parent))
            self.assertFalse(set(hidden.cache_index) & set(labels.cache_index))
            controls = [predictor["fit_controls.tsv"], predictor["select_controls.tsv"], scorer["score_controls.tsv"]]
            self.assertEqual([x.cache_role_index.iloc[0] for x in controls],
                             ["ABC".index(support[f"{r}_role"]) for r in ("fit", "select", "score")])
            self.assertEqual(len({x.inst_id.iloc[0] for x in controls}), 3)
            self.assertEqual(support["score_acyclic_scaffold_parents"], support["score_parents"])

    def test_blocks_normalized_det_alias_and_a_donor_control_id(self):
        blank = self.pop.iloc[:0]
        shifted = self.pop.iloc[[0]].assign(inst_id="other", sig_id="other", rna_plate="", rna_well="",
                                            det_plate="p ", det_well="d1")
        with self.assertRaisesRegex(ValueError, "R18 score-only wells.+physical coordinate"):
            self.pre.verify_domain(self.pop, self.controls, blank, blank, shifted, blank)
        donated = self.controls.iloc[[2]].assign(inst_id=lambda rows: rows.inst_id.str.lower() + " ")
        with self.assertRaisesRegex(ValueError, "R18 B donor normalization.+inst_id"):
            self.pre.verify_domain(self.pop, self.controls, blank, blank, blank, donated)
        self.pre.verify_domain(self.pop, self.controls, blank, blank, blank, blank)

    def test_blocks_cache_mismatches_and_changed_parent_fold(self):
        with self.assertRaisesRegex(ValueError, "unique cache index"):
            self.pre.register_targets(self.pop, self.cache.assign(index=0))
        with self.assertRaisesRegex(ValueError, "parent folds"):
            bad = self.pop.assign(outer_fold=lambda rows: (rows.outer_fold + 1) % 3)
            self.pre.register_targets(bad, self.cache)

    def test_scaffold_status_uses_only_reference_fit_and_keeps_acyclic_separate(self):
        rows = pd.DataFrame({"parent": ["CC", "benzyl", "furan"]})
        fit = pd.DataFrame({"parent": ["benzene", "ethane"]})
        scaffold = {"CC": "", "ethane": "", "benzyl": "c1ccccc1", "benzene": "c1ccccc1", "furan": "c1ccoc1"}
        status = self.pre.annotate_scaffolds(rows, fit, scaffold)
        self.assertEqual(list(status.scaffold_status), ["acyclic", "seen", "unseen"])
