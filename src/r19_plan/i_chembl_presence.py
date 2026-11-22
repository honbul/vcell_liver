"""Check ChEMBL quantitative parent availability without materializing all assay rows."""

import argparse
from pathlib import Path
import sqlite3

import pandas as pd

from src.r11_chemical_probe.common import sha256, write_json
from src.r15_lincs.data import table
from src.r19_plan.i_chembl import QUERY

# The quantity, provenance and validity predicates are identical to the extraction.
PROBE = """
SELECT 1
FROM activities AS x INDEXED BY fk_act_molregno
JOIN assays AS a ON a.assay_id = x.assay_id
LEFT JOIN source AS s ON s.src_id = x.src_id
WHERE
""" + QUERY.split("WHERE", 1)[1].split("ORDER BY", 1)[0] + " LIMIT 1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("database", "identities", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--chunk", type=int, required=True)
    parser.add_argument("--chunks", type=int, required=True)
    args = parser.parse_args()
    identities = table(args.identities)
    groups = identities.groupby("parent", sort=True).molregno.agg(list)
    groups = groups.iloc[args.chunk::args.chunks]
    db = sqlite3.connect(args.database.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    try:
        db.execute("PRAGMA query_only=ON")
        db.execute("PRAGMA cache_size=-131072")
        result = []
        for parent, ids in groups.items():
            probe = PROBE.format(placeholders=",".join("?" for _ in ids))
            present = db.execute(probe, [int(item) for item in ids]).fetchone() is not None
            result.append(dict(parent=parent, has_quantitative=present))
    finally:
        db.close()
    args.out.mkdir(exist_ok=False, parents=True)
    pd.DataFrame(result).to_csv(args.out / "presences.tsv", sep="\t", index=False)
    write_json(args.out / "receipt.json", dict(chunk=args.chunk, chunks=args.chunks,
        checked_parents=len(result), identities_sha256=sha256(args.identities),
        output_sha256=sha256(args.out / "presences.tsv"), contents="exact availability, no assay materialization"))


if __name__ == "__main__":
    main()
