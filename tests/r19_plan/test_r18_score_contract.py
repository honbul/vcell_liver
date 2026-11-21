"""Exercise all R18 freeze slices and the original joint gate on synthetic inputs."""

from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd


@unittest.skipUnless(find_spec("h5py"), "requires the G1 R18 input environment")
class R18ScoreContractTests(unittest.TestCase):
    def setUp(self):
        from src.r19_plan import r18_score_contract as contract

        self.contract = contract

    def test_frozen_forecasts_verify_without_score_assets_and_fail_closed_on_tampering(self):
        from src.r11_chemical_probe.common import read_json, sha256, write_json

        module = self.contract
        with TemporaryDirectory() as directory:
            root = Path(directory)
            access = root / "allowlists"
            predictions = root / "forecasts"
            protocol = root / "protocol.md"
            protocol.write_text("Synthetic R18 test\n")
            axis = "pr_gene_id\tpr_gene_symbol\n1\tA\n6\tB\n"
            default = root / "landmarks.tsv"
            default.write_text(axis)
            d0 = root / "d0.json"
            write_json(d0, {"landmarks": 2, "gene_axis_sha256": {"test": sha256(default)}})
            registry = {}
            for slice_id in module.all_slices():
                _, scenario, _, fit = module.check_slice_id(slice_id)
                registered = access / "predict-plan" / slice_id
                registered.mkdir(parents=True)
                for filename in module.PREDICT_PLAN_FILES:
                    (registered / filename).write_text("study\tinst_id\tparent\nX\tw1\tparent1\n")
                files = {a.name: sha256(a) for a in registered.iterdir()}
                registry[slice_id] = {"predict": files, "score": {"score_labels.tsv": "unavailable"}}

                bundle = predictions / slice_id
                bundle.mkdir(parents=True)
                (bundle / "rows.tsv").write_bytes((registered / "query_features.tsv").read_bytes())
                (bundle / "landmarks.tsv").write_text(axis)
                (bundle / "selection.tsv").write_text("fit\tselect\tscore\n" + "\t".join(module.role_names(fit)) + "\n")
                np.savez(bundle / "forecast.npz", **{name: np.zeros((1, 2))
                                                    for name in module.frozen_arms(scenario)})
                write_json(bundle / "receipt.json", dict(slice=slice_id, protocol_sha256=sha256(protocol),
                    predict_plan_sha256=files, files={a.name: sha256(a) for a in bundle.iterdir()}))
            reference = root / "plan.json"
            write_json(reference, {"files": registry})
            (access / "receipt.json").write_bytes(reference.read_bytes())

            frozen = root / "freeze.json"
            manifest = module.forecast_manifest(predictions, access, reference, protocol, d0)
            self.assertEqual(len(manifest["prediction_receipts"]), 108)
            self.assertFalse((access / "score-plan").exists())
            write_json(frozen, manifest)
            digest = sha256(frozen)
            module.check_freeze(frozen, manifest, digest)

            example = predictions / "C1/A/f0-A"
            stamp = example / "receipt.json"
            current = read_json(stamp)
            (example / "selection.tsv").write_text("fit\tselect\tscore\nA\tC\tB\n")
            current["files"]["selection.tsv"] = sha256(example / "selection.tsv")
            stamp.unlink()
            write_json(stamp, current)
            with self.assertRaisesRegex(ValueError, "changed control roles"):
                module.forecast_manifest(predictions, access, reference, protocol, d0)

            with frozen.open("a") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "freeze changed"):
                module.check_freeze(frozen, manifest, digest)

    def test_C1_labels_and_strict_zero_boundary(self):
        module = self.contract
        pairs = [
            dict(scope="C1", scenario=scenario, arm=module.PRIMARY, reference=reference, metric=metric,
                 eligible=True, low99=low, high99=high)
            for scenario in "AB" for reference in module.reference_arms(scenario)
            for metric, low, high in (("rmse", -0.3, -0.1), ("r2_pooled", 0.01, 0.04))
        ]
        self.assertEqual(module.confirmation_label(pairs), "확인됨")
        pairs[0]["high99"] = 0
        self.assertEqual(module.confirmation_label(pairs), "전이 조건에서만 확인됨")
        pairs[-1]["low99"] = 0
        self.assertEqual(module.confirmation_label(pairs), "확인되지 않음")
        with self.assertRaisesRegex(ValueError, "Incomplete or duplicated"):
            module.confirmation_label(pairs[:-1])

    def test_statistics_keep_technical_strata_and_parent_grouping(self):
        from src.r16_lincs.score import moment

        module = self.contract
        rows = pd.DataFrame({"parent": ["p", "q", "q"]})
        truth = np.array([[1., 2.], [2., 3.], [1., 4.]])
        measures = {}
        for arm in module.frozen_arms("A") - set(module.ANCHORS):
            pred = truth if arm == module.PRIMARY else np.zeros_like(truth)
            measures[arm] = [moment(rows, truth, pred), moment(rows, truth, pred)]
        points, pairs = module.summarize_case(measures, "C1", "A")
        self.assertTrue(all(p["parents"] == 2 and p["rows"] == 6 for p in points))
        self.assertTrue(all(p["passes"] for p in pairs if p["arm"] == module.PRIMARY))
        self.assertFalse(any(p["arm"] == p["reference"] for p in pairs))
        for technical in ("C2:0.04", "C3_6h_HEPG2_only"):
            points, pairs = module.summarize_case(measures, technical, "A")
            self.assertTrue(all(not p["eligible"] for p in points))
            self.assertTrue(all(not p["passes"] for p in pairs))
            self.assertTrue(all(np.isfinite((p["low99"], p["high99"])).all() for p in pairs))
