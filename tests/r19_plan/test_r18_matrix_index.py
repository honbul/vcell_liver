"""Exercise the Level 3 resolver with no expression dataset available."""

from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


@unittest.skipUnless(find_spec("h5py"), "requires the G1 Level 3 input environment")
class MatrixMetadataTests(unittest.TestCase):
    def test_resolver_uses_only_metadata_and_preserves_registered_order(self):
        import h5py
        import pandas as pd

        from src.r19_plan.r18_matrix_index import GENE_IDS, SAMPLE_IDS, metadata_index

        with TemporaryDirectory() as directory:
            path = Path(directory) / "metadata_only.gctx"
            with h5py.File(path, "w") as handle:
                handle.create_dataset(SAMPLE_IDS, data=[b"x", b"y", b"z"])
                handle.create_dataset(GENE_IDS, data=[b"7", b"11", b"5"])
            rows, genes, _ = metadata_index(path, pd.DataFrame({"inst_id": ["z", "x"]}), [5, 7])
            self.assertEqual(rows.level3_row_index.to_list(), [2, 0])
            self.assertEqual(genes.level3_gene_index.to_list(), [2, 0])
            for well, gene in [("absent", 5), ("x", 999)]:
                with self.assertRaisesRegex(ValueError, "Unmatched registered IDs"):
                    metadata_index(path, pd.DataFrame({"inst_id": [well]}), [gene])
