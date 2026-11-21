"""Audit existing structure assets against locked R18 parent and landmark metadata."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r11_chemical_probe.dataset import ENTITY
from src.r15_lincs.data import table
from src.r19_plan.r18_preflight import checked_tables


def inventory(folder, receipt, landmarks):
    features = folder / "features"
    archive = features / "chemberta.npz"
    expected = sha256(receipt)
    registered = read_json(receipt)
    if sha256(features / "receipt.json") != expected or sha256(archive) != registered["chemberta_sha256"]:
        raise ValueError(f"Existing encoder artifact changed: {archive}")
    with np.load(archive, allow_pickle=False) as outputs:
        parents = pd.Index(outputs["parents"])
    entities = table(features / "entities.tsv")
    if (not parents.is_unique or entities[ENTITY].duplicated().any()
            or not entities[ENTITY].eq(entities.smiles).all() or set(entities[ENTITY]) != set(parents)):
        raise ValueError(f"Existing structures do not correspond to the embedding parent axis: {folder}")
    gene_path = folder / "metadata" / "genes.tsv"
    gene_hash = sha256(gene_path)
    if gene_hash != registered["metadata_sha256"]:
        raise ValueError(f"Existing gene map changed: {gene_path}")
    genes = table(gene_path)
    if genes.token_id.duplicated().any():
        raise ValueError(f"Existing landmark map repeats a token: {folder}")
    mapping = genes.set_index("token_id").ensembl_id.reindex(landmarks)
    return set(parents), {
        "path": str(folder), "features": registered["chemberta_sha256"],
        "feature_receipt": expected, "landmark_map": gene_hash,
        "mapped_landmarks": int((mapping.notna() & mapping.ne("")).sum()),
    }


def run(args):
    checked_tables(args.lock, args.lock_reference)
    gene_hash = sha256(args.landmarks)
    r17 = read_json(args.d0_reference)["gene_axis_sha256"]
    if set(r17.values()) != {gene_hash}:
        raise ValueError("R18 landmark order differs from the registered R17 axis")
    landmarks = table(args.landmarks).pr_gene_id
    r15, r15_info = inventory(args.r15, args.r15_reference, landmarks)
    r16, r16_info = inventory(args.r16, args.r16_reference, landmarks)
    checkpoint = read_json(args.encoder_reference)["checkpoint_files"]
    if any(sha256(args.checkpoint / path) != digest for path, digest in checkpoint.items()):
        raise ValueError("The available ChemBERTa encoder differs from the historical pin")

    rows = table(args.lock / "split_rows.tsv")
    support = []
    for (case, scenario), block in rows.groupby(["case", "scenario"], sort=True):
        parents = set(block.parent)
        support.append(dict(case=case, scenario=scenario, parents=len(parents),
                            r15=len(parents & r15), r16_confirm=len(parents & r16),
                            either=len(parents & (r15 | r16)), missing=len(parents - (r15 | r16))))
    all_parents = set(rows.parent)
    support.append(dict(case="all", scenario="all", parents=len(all_parents), r15=len(all_parents & r15),
                        r16_confirm=len(all_parents & r16), either=len(all_parents & (r15 | r16)),
                        missing=len(all_parents - (r15 | r16))))
    args.out.mkdir(exist_ok=False, parents=True)
    pd.DataFrame(support).to_csv(args.out / "support.tsv", sep="\t", index=False)
    write_json(args.out / "receipt.json", dict(
        code_sha256=sha256(__file__), lock_sha256=sha256(args.lock_reference), metadata_only=True,
        r18_expression_read=False, fitting=False, checkpoint=checkpoint,
        existing_assets={"R15": dict(r15_info, parents=len(r15)), "R16_confirm": dict(r16_info, parents=len(r16))},
        landmark_sha256=gene_hash, support_sha256=sha256(args.out / "support.tsv"),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("lock", "lock-reference", "landmarks", "d0-reference", "r15", "r15-reference", "r16", "r16-reference",
                 "encoder-reference", "checkpoint", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    run(parser.parse_args())
