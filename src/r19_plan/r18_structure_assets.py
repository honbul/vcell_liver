"""Build the unchanged R16 structure representation on all registered R18 parents."""

import argparse
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r15_lincs.data import table
from src.r16_lincs.confirmation import embed
from src.r19_plan.r18_preflight import checked_tables


def run(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    checked_tables(args.lock, args.lock_reference)
    asset_reference = read_json(args.asset_reference)
    if (asset_reference["lock_sha256"] != sha256(args.lock_reference)
            or asset_reference["landmark_sha256"] != sha256(args.landmarks)
            or asset_reference["checkpoint"] != read_json(args.encoder_reference)["checkpoint_files"]):
        raise ValueError("The R18 structure build differs from its audited input or encoder")
    parents = np.array(sorted(set(table(args.lock / "split_rows.tsv").parent)), dtype=str)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".r18-structure-", dir=args.out.parent) as tmp:
        root = Path(tmp)
        cache = root / "metadata-input"
        cache.mkdir()
        pd.DataFrame(dict(parent=parents)).to_csv(cache / "chemicals.tsv", sep="\t", index=False)
        copyfile(args.landmarks, cache / "landmarks.tsv")
        write_json(cache / "receipt.json", dict(
            lock_sha256=sha256(args.lock_reference),
            files={path.name: sha256(path) for path in cache.iterdir()},
        ))
        stage = root / "assets"
        embed(SimpleNamespace(cache=cache, out=stage, checkpoint=args.checkpoint,
                              reference=args.encoder_reference, gene_map=args.gene_map))
        path = stage / "features" / "chemberta.npz"
        with np.load(path, allow_pickle=False) as arrays:
            values = arrays["embedding"]
            if (not np.array_equal(arrays["parents"], parents)
                    or values.ndim != 2 or len(values) != len(parents) or not np.isfinite(values).all()):
                raise ValueError("R18 structure order, coverage, or values differ from the registered parent axis")
            dimension = values.shape[1]
        gene_hash = sha256(stage / "metadata" / "genes.tsv")
        if {source["landmark_map"] for source in asset_reference["existing_assets"].values()} != {gene_hash}:
            raise ValueError("The new R18 comparator gene map differs from R15/R16")
        write_json(stage / "receipt.json", dict(
            code_sha256=sha256(__file__), encoder_code_sha256=sha256(embed.__code__.co_filename),
            lock_sha256=sha256(args.lock_reference), asset_audit_sha256=sha256(args.asset_reference),
            encoder_pin_sha256=sha256(args.encoder_reference), source_gene_map_sha256=sha256(args.gene_map),
            parents=len(parents), dimension=dimension, r18_expression_read=False, fitting=False,
            files={str(file.relative_to(stage)): sha256(file) for file in sorted(stage.rglob("*")) if file.is_file()},
        ))
        stage.rename(args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("lock", "lock-reference", "landmarks", "asset-reference",
                 "checkpoint", "encoder-reference", "gene-map", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    run(parser.parse_args())
