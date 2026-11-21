"""Use synthetic Level 3 files to check exactly which wells reach a predictor."""

import hashlib
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


@unittest.skipUnless(find_spec("h5py"), "requires the G1 Level 3 input environment")
class R18InputBundleTests(unittest.TestCase):
    def setUp(self):
        import h5py

        from src.r11_chemical_probe.common import sha256, write_json
        from src.r19_plan import r18_input_bundle as module

        self.h5py, self.module = h5py, module
        self.sha256, self.write_json = sha256, write_json
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = root = Path(directory.name)
        self.study = "GSE92742"
        self.case = "C1/B/f0-A"
        self.args = SimpleNamespace(access=root / "access", access_reference=root / "access-reference.json",
                                    slice=self.case, index=root / "index", index_reference=root / "index-ref",
                                    landmarks=root / "landmarks.tsv", out=root / "out")
        self.plan = self.args.access / "predict-plan" / self.case
        self.plan.mkdir(parents=True)
        self.source_ids = [
            "fit_i", "target_i", "score_control", "fit_q", "fit_v", "target_v", "select_v",
            "donor_i", "query", "donor_q", "donor_v", "donor_ctl_2", "donor_ctl_1",
        ]
        self.positions = {well: i for i, well in enumerate(self.source_ids)}
        self.raw = self.root / "synthetic.gctx"
        self.values = np.arange(len(self.source_ids) * 3, dtype=np.float32).reshape(-1, 3)
        self.values[[self.positions[name] for name in ("score_control", "query")]] = np.nan
        with self.h5py.File(self.raw, "w") as archive:
            archive.create_dataset(module.SAMPLE_IDS, data=np.array(self.source_ids, dtype="S32"))
            archive.create_dataset(module.GENE_IDS, data=[b"2", b"4", b"1"])
            archive.create_dataset(module.EXPRESSION, data=self.values)
        self.args.landmarks.write_text("pr_gene_id\tpr_gene_symbol\n1\tA\n2\tB\n")
        index = self.args.index / self.study
        index.mkdir(parents=True)
        axis = index / "level3_landmarks.tsv"
        axis.write_text("gene_id\tlevel3_gene_index\tmatrix\n1\t2\tcombined\n2\t0\tcombined\n")
        self.write_json(index / "receipt.json", dict(
            study=self.study, lock_sha256="synthetic", metadata_only=True, expression_read=False,
            landmarks=2, landmark_axis_sha256=self.sha256(self.args.landmarks),
            files={axis.name: self.sha256(axis)},
            matrix_records={"combined": dict(
                path=str(self.raw), bytes=self.raw.stat().st_size,
                indexed_wells=len(self.source_ids), indexed_genes=3,
                gene_axis_sha256=hashlib.sha256(b"2\n4\n1").hexdigest(),
            )},
        ))
        reference_folder = self.args.index_reference / self.study
        reference_folder.mkdir(parents=True)
        (reference_folder / "receipt.json").write_bytes((index / "receipt.json").read_bytes())

        def target(well, parent, plate, coord="B01"):
            return dict(study=self.study, inst_id=well, parent=parent, cell_id="HEPG2", hours="24.0",
                        dose="10.0", rna_plate=plate, rna_well=coord)

        def control(well, rna_plate, coord, **attrs):
            return dict(study=self.study, inst_id=well, rna_plate=rna_plate, rna_well=coord, **attrs)

        def donor(well, parent, coord):
            return dict(target(well, parent, "external", coord), plate="external", cohort="C1",
                        cell_id="A549", CONFIRM_R18_B_source_donor_allowed="True")

        entries = [
            ("inner_and_outer_labels", [
                dict(target("target_v", "q", "p2"), inner_phase="select"),
                dict(target("target_i", "p", "p1"), inner_phase="inner_fit"),
            ]),
            ("fit_controls", [
                control("fit_q", "p3", "A01", role="A"),
                control("fit_i", "p1", "A01", role="A"),
                control("fit_v", "p2", "A01", role="A"),
            ]),
            ("select_controls", [control("select_v", "p2", "A02", role="B")]),
            ("B_donor_inputs", [donor("donor_i", "p", "B01"), donor("donor_v", "q", "B02"),
                                donor("donor_q", "r", "B03")]),
            ("B_donor_controls", [control("donor_ctl_1", "external", "A01", plate="external", half="1"),
                                  control("donor_ctl_2", "external", "A02", plate="external", half="2")]),
            ("query_features", [target("query", "r", "p3")]),
        ]
        self.tables = {}
        for name, records in entries:
            frame = pd.DataFrame(records)
            if name != "query_features":
                frame = frame.assign(matrix="combined", level3_row_index=frame.inst_id.map(self.positions))
            self.tables[name] = frame
        self.write_plan()

    def write_plan(self):
        for name, data in self.tables.items():
            data.to_csv(self.plan / f"{name}.tsv", sep="\t", index=False)
        plan_hashes = {path.name: self.sha256(path) for path in self.plan.iterdir()}
        self.args.access_reference.unlink(missing_ok=True)
        self.write_json(self.args.access_reference, dict(
            metadata_only=True, expression_read=False, lock_sha256="synthetic",
            files={self.args.slice: {"predict": plan_hashes, "score": {"unavailable": "synthetic"}}},
        ))
        (self.args.access / "receipt.json").write_bytes(self.args.access_reference.read_bytes())

    def test_extracts_only_allowed_rows_and_restores_well_and_gene_order(self):
        reads = []
        original = self.h5py.Dataset.__getitem__

        def get(dataset, request):
            if dataset.name == "/" + self.module.EXPRESSION:
                selected_rows, _ = request
                reads.extend(selected_rows.tolist())
            return original(dataset, request)

        with patch.object(self.h5py.Dataset, "__getitem__", get):
            self.module.run(self.args)
        self.assertFalse((self.args.access / "score-plan").exists())
        self.assertEqual(
            self.sha256(self.plan / "query_features.tsv"), self.sha256(self.args.out / "query_features.tsv")
        )
        expected_read = []
        with np.load(self.args.out / "expression.npz", allow_pickle=False) as arrays:
            self.assertEqual(set(arrays.files), set(self.module.EXPRESSION_TABLES))
            for name in self.module.EXPRESSION_TABLES:
                positions = self.tables[name].level3_row_index.to_numpy()
                expected_read.extend(positions)
                np.testing.assert_array_equal(arrays[name], self.values[np.ix_(positions, [2, 0])])
        self.assertCountEqual(reads, expected_read)

    def test_rejects_altered_plan_before_opening_matrix(self):
        path = self.plan / "fit_controls.tsv"
        path.write_bytes(path.read_bytes() + b"\n")
        with patch.object(self.h5py, "File") as raw:
            with self.assertRaisesRegex(ValueError, "Changed registered R18 input"):
                self.module.run(self.args)
        raw.assert_not_called()
        self.assertFalse(self.args.out.exists())

    def test_rejects_wrong_role_even_with_matching_file_hashes(self):
        self.tables["fit_controls"]["role"] = "C"
        self.write_plan()
        with patch.object(self.h5py, "File") as raw:
            with self.assertRaisesRegex(ValueError, "unexpected DMSO role"):
                self.module.run(self.args)
        raw.assert_not_called()

    def test_rejects_pinned_index_that_points_at_the_wrong_well(self):
        self.tables["fit_controls"].loc[0, "level3_row_index"] = self.positions["score_control"]
        self.write_plan()
        original = self.h5py.Dataset.__getitem__

        def get(dataset, request):
            self.assertNotEqual(dataset.name, "/" + self.module.EXPRESSION)
            return original(dataset, request)

        with patch.object(self.h5py.Dataset, "__getitem__", get):
            with self.assertRaisesRegex(ValueError, "row ID changed"):
                self.module.run(self.args)
        self.assertFalse(self.args.out.exists())

    def test_scenario_a_keeps_source_arrays_empty(self):
        self.args.slice = "C1/A/f0-A"
        self.plan = self.args.access / "predict-plan" / self.args.slice
        self.plan.mkdir(parents=True)
        for name in ("B_donor_inputs", "B_donor_controls"):
            self.tables[name] = self.tables[name].iloc[:0]
        self.write_plan()
        self.module.run(self.args)
        with np.load(self.args.out / "expression.npz", allow_pickle=False) as arrays:
            self.assertEqual(arrays["B_donor_inputs"].shape, (0, 2))
            self.assertEqual(arrays["B_donor_controls"].shape, (0, 2))

    def test_separate_evaluator_uses_held_treatment_and_registered_control_roles(self):
        from src.r11_chemical_probe.common import read_json
        from src.r19_plan.r18_score import load_observed

        score = self.args.access / "score-plan" / self.args.slice
        score.mkdir(parents=True)
        query = self.tables["query_features"]
        query.assign(matrix="combined", level3_row_index=self.positions["query"]).to_csv(
            score / "score_labels.tsv", sep="\t", index=False
        )
        pd.DataFrame([dict(study=self.study, inst_id="score_control", rna_plate="p3", rna_well="A03",
                           role="C", matrix="combined", level3_row_index=self.positions["score_control"])]).to_csv(
            score / "score_controls.tsv", sep="\t", index=False
        )
        self.args.access_ref = self.args.access_reference
        registered = read_json(self.args.access_ref)["files"][self.args.slice]
        registered["score"] = {item.name: self.sha256(item) for item in score.iterdir()}
        observed = self.values
        observed[self.positions["query"]] = [30., 10., 20.]
        observed[self.positions["score_control"]] = [8., 6., 4.]
        with self.h5py.File(self.raw, "r+") as archive:
            archive[self.module.EXPRESSION][:] = observed

        reads = []
        original = self.h5py.Dataset.__getitem__

        def get(dataset, request):
            if dataset.name == "/" + self.module.EXPRESSION:
                rows, _ = request
                reads.extend(rows.tolist())
            return original(dataset, request)

        with patch.object(self.h5py.Dataset, "__getitem__", get):
            truth, donor, direction, _ = load_observed(self.args, self.args.slice, registered, query)
        np.testing.assert_array_equal(truth, [[16., 22.]])
        np.testing.assert_array_equal(donor, [[9., 21.]])
        np.testing.assert_array_equal(direction, [[-7., -1.]])
        self.assertCountEqual(reads, [self.positions[key] for key in ("query", "fit_q", "score_control")])
