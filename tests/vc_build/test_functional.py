import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from vc_build.functional import (
    DomainError, FunctionalModel, chemistry, fit_arm, fit_base, predict_arm,
)
from vc_build.io import verify_forecast
from vc_build.splits import assignments, chemical_groups
from vc_build.train import PREDICT_INPUTS, run


class FunctionalTest(unittest.TestCase):
    def setUp(self):
        self.cells = [f"cell-{number}" for number in range(10)]
        self.rna = pd.DataFrame(
            np.random.default_rng(1).normal(size=(10, 25)),
            index=self.cells, columns=[f"gene-{number}" for number in range(25)],
        )
        chemicals = ("CC", "CCC", "CCCC", "CCCO", "CCCCO", "CCCN", "CCCCN", "CCCl")
        self.rows = pd.DataFrame([
            {
                "row_id": f"{cell}:{index}", "cell_id": cell,
                "lineage_group": f"lineage-{position // 2}",
                "compound_id": f"drug-{index}", "connectivity_group": f"group-{index}",
                "scaffold_group": "", "isomeric_smiles": structure,
                "endpoint_value": position / 10 + index,
            }
            for position, cell in enumerate(self.cells) for index, structure in enumerate(chemicals)
        ])
        self.rows["chemical_group"] = chemical_groups(self.rows)

    def test_structure_identity_keeps_stereoisomers(self):
        right, left = "C[C@H](F)O", "C[C@@H](F)O"
        self.assertNotEqual(chemistry(right)[0], chemistry(left)[0])
        frame = pd.DataFrame({
            "compound_id": ["a", "b", "c"], "connectivity_group": ["1", "1", "2"],
            "scaffold_group": ["", "scaffold", "scaffold"],
        })
        self.assertEqual(chemical_groups(frame).nunique(), 1)

    def test_crossed_holdouts(self):
        manifest = assignments(self.rows, "CD", 3, 10)
        self.assertIn("embargo", manifest.role.unique())
        for _, fold in manifest.groupby("fold"):
            fit, test = fold[fold.role == "fit"], fold[fold.role == "withheld"]
            self.assertFalse(set(fit.lineage_group) & set(test.lineage_group))
            self.assertFalse(set(fit.chemical_group) & set(test.chemical_group))

    def test_fit_preprocessing_independent_of_query_and_gene_order(self):
        fit = self.rows.iloc[:64]
        expected = fit_base(fit, self.rna)
        altered = self.rna.copy()
        altered.iloc[-2:] *= 1e6
        actual = fit_base(fit, altered)
        np.testing.assert_array_equal(expected["rna"]["pca"].components_, actual["rna"]["pca"].components_)
        bundle = fit_arm(expected, fit, self.rna, "structure_ridge", 10)
        model = FunctionalModel(bundle, {})
        query = self.rna.iloc[-1]
        predictions = [
            model.predict(query.to_dict(), "CC", "drug-0"),
            model.predict(query.iloc[::-1].to_dict(), compound_id="drug-0"),
            predict_arm(bundle, self.rows.iloc[[-8]][PREDICT_INPUTS], self.rna)[0],
        ]
        np.testing.assert_allclose(predictions, predictions[0])
        with self.assertRaisesRegex(DomainError, "insufficient_gene_coverage"):
            model.predict(query.iloc[:7].to_dict(), compound_id="drug-0")
        with self.assertRaisesRegex(DomainError, "compound_id_structure_mismatch"):
            model.predict(query.to_dict(), "CCC", "drug-0")

    def test_smoke_freeze_no_outcome_and_no_overwrite(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            self.rows.to_parquet(tmp / "rows.parquet", index=False)
            self.rna.to_parquet(tmp / "rna.parquet")
            config = root / "configs/vc_build_v1/contract.yaml"
            run(config, tmp, tmp / "run", ["C"], smoke=True)
            result, _ = verify_forecast(tmp / "run/predictions.parquet")
            self.assertNotIn("endpoint_value", result)
            self.assertTrue((tmp / "run/v0-smoke/receipt.json").is_file())
            with self.assertRaises(FileExistsError):
                run(config, tmp, tmp / "run", ["C"], smoke=True)


if __name__ == "__main__":
    unittest.main()
