"""Extract exact-parent human single-protein ChEMBL targets from the read-only release."""
import argparse
import sqlite3
from collections import defaultdict
from contextlib import closing
from pathlib import Path

import pandas as pd
from rdkit import RDLogger

from vc_build.function_data import canonical_chemistry
from vc_build.io import publish_json, sha256

from .assets import COLD, KEY, LIMORE, STORAGE

DB = STORAGE / "vcell_liver_evidence_gated_r7/assets/external/chembl_37/chembl_37/chembl_37_sqlite/chembl_37.db"

TARGETS = """
WITH eligible(tid) AS (
  SELECT DISTINCT a.tid
  FROM activities AS x INDEXED BY fk_act_molregno JOIN assays AS a ON a.assay_id = x.assay_id
  WHERE x.molregno = ? AND a.assay_type = 'B' AND a.confidence_score = 9
    AND x.pchembl_value >= 6 AND x.standard_relation = '='
    AND COALESCE(x.data_validity_comment, '') IN ('', 'Manually validated')
  UNION
  SELECT tid FROM drug_mechanism WHERE molregno = ? AND tid IS NOT NULL
)
SELECT DISTINCT cs.accession, cc.component_synonym
FROM eligible AS e JOIN target_dictionary AS t ON t.tid = e.tid
JOIN target_components AS tc ON tc.tid = t.tid
JOIN component_sequences AS cs ON cs.component_id = tc.component_id
JOIN component_synonyms AS cc ON cc.component_id = tc.component_id
WHERE t.organism = 'Homo sapiens' AND t.target_type = 'SINGLE PROTEIN'
  AND cc.syn_type IN ('GENE_SYMBOL','GENE_SYMBOL_OTHER')
"""


def register(assets, out):
    gdsc = pd.read_parquet(assets / "gdsc_query.parquet", columns=[KEY])
    keys = sorted(set(pd.concat([gdsc[KEY], *(pd.read_parquet(path, columns=[KEY])[KEY] for path in [
        COLD / "train.parquet", COLD / "query.parquet", LIMORE / "test_inputs/rows.parquet"
    ])])))
    genes = set(pd.read_parquet(assets / "pathway_weights.parquet").index)
    import pyarrow.parquet as pq
    genes |= set(pq.ParquetFile(COLD / "rna.parquet").schema.names)
    targets, accession, matched = defaultdict(set), defaultdict(set), set()
    RDLogger.DisableLog("rdApp.warning")
    with closing(sqlite3.connect(DB.as_uri() + "?mode=ro&immutable=1", uri=True)) as db:
        db.execute("PRAGMA query_only=ON")
        if not db.execute("SELECT 1 FROM version WHERE name='ChEMBL_37'").fetchone():
            raise ValueError("Unexpected ChEMBL version")
        for family in sorted({x.split("-")[0] for x in keys}):
            molecules = db.execute(
                "SELECT molregno, canonical_smiles FROM compound_structures "
                "WHERE standard_inchi_key >= ? AND standard_inchi_key < ?",
                (family + "-", family + "."))
            for mol, text in molecules:
                try:
                    key = canonical_chemistry(text)[1]
                except (ValueError, RuntimeError, TypeError):
                    continue
                if key not in keys:
                    continue
                matched.add(key)
                for protein, gene in db.execute(TARGETS, (mol, mol)):
                    accession[key].add(protein)
                    if gene in genes:
                        targets[key].add(gene)
    out.mkdir(parents=True, exist_ok=False)
    payload = {key: sorted(value) for key, value in sorted(targets.items())}
    publish_json(out / "targets.json", payload)
    publish_json(out / "receipt.json", {
        "release": "ChEMBL_37", "database": str(DB), "database_bytes": DB.stat().st_size,
        "requested": len(keys), "structure_matches": len(matched), "human_single_protein_matches": len(accession),
        "gene_symbol_matches": len(payload), "target_symbols": len(set().union(*targets.values())),
        "targets_sha256": sha256(out / "targets.json"),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    register(args.assets, args.out)
