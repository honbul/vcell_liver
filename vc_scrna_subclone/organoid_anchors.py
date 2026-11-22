"""Freeze leave-one-donor-out GSE182593 fitting labels independently of prediction."""
import argparse
import json
from pathlib import Path
import tempfile

import pandas as pd

from .cohorts import file_record
from .singlecell_inputs import files_receipt
from .source_panels import frozen


def register(sources, output):
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    source_receipt = sources / "receipt.json"
    registered = json.loads(source_receipt.read_text())
    keys = pd.read_parquet(frozen(sources, "organoids/query.parquet", registered))
    labels = pd.read_parquet(frozen(sources, "organoids/labels.parquet", registered))
    evidence = keys.merge(labels, on="id", validate="1:1")
    if len(evidence) != 24 or len(set(evidence.donor)) != 4:
        raise ValueError("Expected RNA and phenotype of each registered organoid replicate")
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as directory:
        staged = Path(directory)
        for fold, donor in enumerate(sorted(evidence.donor.unique())):
            fit = evidence[evidence.donor.ne(donor)]
            if len(fit) != 18 or fit.donor.nunique() != 3:
                raise ValueError("Invalid leave-one-organoid-donor-out fold")
            fit.to_parquet(staged / f"anchors-{fold}.parquet", index=False)
        (staged / "receipt.json").write_text(json.dumps({
            "sources": file_record(source_receipt), "held_out_donors": sorted(evidence.donor.unique()),
            "files": files_receipt(staged),
        }, indent=2, sort_keys=True) + "\n")
        staged.rename(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    register(a.sources, a.output)


if __name__ == "__main__":
    main()
