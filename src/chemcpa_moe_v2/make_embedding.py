"""Compute the drug embedding chemCPA expects, with chemCPA's own code.

`get_chemical_representation` looks each canonical SMILES up in a parquet of RDKit 2D descriptors.
This builds that parquet for the Tahoe drug groups by calling the repo's own
`embeddings/rdkit/embedding_rdkit.py`, so the descriptor set, the variance filter and the
normalisation are the authors' and not a re-implementation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chemcpa-src", required=True)
    ap.add_argument("--adata", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.chemcpa_src)
    import embeddings.rdkit.embedding_rdkit as er
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    er.compute_rdkit_embeddings(a.adata, output_path=a.out, smiles_key="SMILES",
                                skip_variance_filter=False)
    import pandas as pd
    df = pd.read_parquet(a.out)
    print(f"embedding: {df.shape[0]} SMILES x {df.shape[1]} dims -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
