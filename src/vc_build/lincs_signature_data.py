"""Register exact-parent non-liver LINCS Level5 profiles and measured landmarks."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .io import publish_json, sha256

CELLS = ("A375", "A549", "HT29", "MCF7", "PC3", "VCAP")
PREFIX = "GSE92742_Broad_LINCS_"
METADATA = ("pert_info", "sig_info", "sig_metrics", "gene_info")


def read_metadata(root, name, columns):
    return pd.read_csv(root / f"{PREFIX}{name}.txt", sep="\t", usecols=columns, low_memory=False)


def register_design(args):
    from rdkit import RDLogger

    from .cross_source_data import unambiguous_parent
    from .expanded_panel_cv import RECEIPTS
    from .v0plus_cv import training_inputs
    from .yang_hcc_external import require_hash

    require_hash(args.core / "receipt.json", RECEIPTS["core"])
    train, _, _ = training_inputs(args.core)
    registered = sorted(train.compound_id.unique())
    if len(registered) != 68:
        raise ValueError("Unexpected corrected parent panel")
    info = read_metadata(args.root, "pert_info", ["pert_id", "pert_type", "canonical_smiles"])
    structures = info[info.pert_type.eq("trt_cp") & info.canonical_smiles.notna() & info.canonical_smiles.ne("-666")]
    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")
    try:
        parents = {value: unambiguous_parent(value) for value in structures.canonical_smiles.unique()}
    finally:
        RDLogger.EnableLog("rdApp.warning")
        RDLogger.EnableLog("rdApp.error")
    mapped = structures.assign(compound_id=structures.canonical_smiles.map(
        {value: resolved[1] if resolved else None for value, resolved in parents.items()},
    ))[["pert_id", "compound_id"]].drop_duplicates()
    mapped = mapped[~mapped.pert_id.duplicated(keep=False) & mapped.compound_id.isin(registered)]

    sig_cols = ["sig_id", "pert_id", "cell_id", "pert_type", "pert_time", "pert_time_unit",
                "pert_dose", "pert_dose_unit"]
    sig = read_metadata(args.root, "sig_info", sig_cols)
    concentration = pd.to_numeric(sig.pert_dose, errors="coerce")
    time = pd.to_numeric(sig.pert_time, errors="coerce")
    units = sig.pert_dose_unit.str.replace("µ", "u").str.replace("μ", "u").str.lower()
    good = (sig.pert_type.eq("trt_cp") & sig.cell_id.isin(CELLS) & sig.pert_time_unit.eq("h")
            & time.isin((6, 24)) & units.eq("um") & concentration.between(.5, 20))
    sig = sig.loc[good].assign(time_h=time[good].astype(int), dose_um=concentration[good])
    sig = sig.merge(mapped, on="pert_id", validate="many_to_one")
    qc = read_metadata(args.root, "sig_metrics", ["sig_id", "distil_nsample", "distil_cc_q75"])
    sig = sig.merge(qc, on="sig_id", validate="one_to_one")
    accepted = sig[sig.distil_nsample.ge(2) & sig.distil_cc_q75.ge(.15)].copy()
    coverage = accepted.groupby("compound_id").cell_id.nunique()
    accepted = accepted[accepted.compound_id.isin(coverage[coverage.ge(2)].index)]
    if len(accepted) == 0 or not set(accepted.cell_id) <= set(CELLS):
        raise ValueError("No eligible non-liver LINCS signatures")
    columns = ["sig_id", "compound_id", "cell_id", "time_h", "dose_um", "pert_id", "distil_nsample", "distil_cc_q75"]
    args.out.mkdir(parents=True, exist_ok=False)
    design = args.out / "selected.tsv"
    accepted[columns].sort_values("sig_id").to_csv(design, sep="\t", index=False)
    publish_json(args.out / "design.json", {
        "source": "GSE92742", "core_sha256": RECEIPTS["core"], "parents": registered,
        "matched_pert_ids": len(mapped), "structurally_mapped_parents": mapped.compound_id.nunique(),
        "profile_count_pre_quality_filter": len(sig), "consensus_eligible_profile_count": len(accepted),
        "supported": accepted.groupby("compound_id").cell_id.nunique().sort_index().to_dict(),
        "profile_count_by_cell": accepted.cell_id.value_counts().sort_index().to_dict(),
        "design_sha256": sha256(design),
        "metadata_sha256": {name: sha256(args.root / f"{PREFIX}{name}.txt") for name in METADATA},
    })


def decode(dataset, key):
    return pd.Index(dataset[:].astype(str), name=key)


def extract(args):
    import h5py

    design_path = args.design / "selected.tsv"
    design = json.loads((args.design / "design.json").read_text())
    sig = pd.read_csv(design_path, sep="\t")
    if (sha256(design_path) != design["design_sha256"] or not set(sig.cell_id) <= set(CELLS)
            or not sig.sig_id.is_unique or set(sig.compound_id) != set(design["supported"])):
        raise ValueError("Unexpected non-liver signature roster")
    genes = read_metadata(args.root, "gene_info", ["pr_gene_id", "pr_gene_symbol", "pr_is_lm"])
    gene_path = args.root / f"{PREFIX}gene_info.txt"
    if sha256(gene_path) != design["metadata_sha256"]["gene_info"]:
        raise ValueError("GSE92742 gene file differs from registration")
    genes = genes[genes.pr_is_lm.eq(1)].copy()
    genes.pr_gene_id = genes.pr_gene_id.astype(str)
    if len(genes) != 978 or not genes.pr_gene_symbol.is_unique:
        raise ValueError("Unexpected directly measured L1000 gene panel")

    [matrix_path] = args.root.glob("*Level5_COMPZ.MODZ*.gctx")
    with h5py.File(matrix_path, "r") as file:
        genes_idx = decode(file["0/META/ROW/id"], "gene_id").get_indexer(genes.pr_gene_id)
        profile_ids = decode(file["0/META/COL/id"], "sig_id")
        sig = sig.assign(matrix_row=profile_ids.get_indexer(sig.sig_id)).sort_values("matrix_row")
        if sig.matrix_row.min() < 0 or genes_idx.min() < 0 or not set(sig.cell_id) <= set(CELLS):
            raise ValueError("Registered exact signature does not exist in Level5")
        gctx = file["0/DATA/0/matrix"]
        if gctx.shape != (len(profile_ids), 12328):
            raise ValueError("Unexpected public GCTX orientation")
        rows = sig.matrix_row.to_numpy()
        measured = np.concatenate([gctx[rows[first:first + 192], :][:, genes_idx]
                                   for first in range(0, len(rows), 192)])
    if not np.isfinite(measured).all():
        raise ValueError("Observed landmarks contain nonfinite values")
    symbols = genes.pr_gene_symbol.tolist()
    labels = ["compound_id", "cell_id", "time_h", "dose_um"]
    values = pd.DataFrame(measured, columns=symbols)
    condition = values.groupby([sig[name].to_numpy() for name in labels]).median()
    by_time = condition.groupby(level=[0, 1, 2]).median()
    by_cell = by_time.groupby(level=[0, 1]).mean()
    compound = by_cell.groupby(level=0).mean().rename_axis("compound_id")
    if set(compound.index) != set(design["supported"]):
        raise ValueError("A registered compound was lost in Level5")
    args.out.mkdir(exist_ok=False, parents=True)
    profile = args.out / "signatures.parquet"
    compound.to_parquet(profile)
    publish_json(args.out / "receipt.json", design | {
        "design_receipt_sha256": sha256(args.design / "design.json"), "level5_file": matrix_path.name,
        "level5_bytes": matrix_path.stat().st_size,
        "selected_landmark_values_sha256": hashlib.sha256(measured.tobytes()).hexdigest(),
        "landmark_genes": len(symbols), "consensus_parents": len(compound),
        "consensus_sha256": sha256(profile),
        "profile_cells": sorted(sig.cell_id.unique()), "profile_count_by_time": sig.time_h.value_counts().to_dict(),
        "distinct_contexts": len(condition),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("design", "extract"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--core", type=Path)
    parser.add_argument("--design", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    (register_design if args.stage == "design" else extract)(args)


if __name__ == "__main__":
    main()
