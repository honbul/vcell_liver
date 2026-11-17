"""Index the prior study's real single-cell sample for population-level work.

Reads (read-only) the AnnData written by the chemCPA study and the v2 condition table, and writes
into this study's own cache:

    X.npy        the same 1,256,568 real cells, float32 log1p(CP10k) on the 2,000-gene panel,
                 re-ordered so that each (well, cell line) group is one contiguous slice and the
                 cells inside a group are ordered by blake2b(barcode)
    cells.npz    per-cell group id, library size and hash rank u in [0, 1)
    wells.tsv    one row per (well, cell line) group: plate, drug group, role, cell count, slice

Ordering cells by a hash of the barcode makes "the first N cells of a group" a subsample that
depends on nothing but the barcode, and makes even/odd positions a fixed split-half.

No cell is created, duplicated or perturbed here. Roles are the prior study's drug-group split:
chemCPA's "test" is that study's validation, "ood" its held-out test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd


def bhash(s: str) -> float:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big") / 2.0 ** 64


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--v2-data", required=True)
    ap.add_argument("--manifest-dir", default="manifests/chemcpa_moe_v2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--wells-out", default="manifests/anchored_distribution_r3/wells.tsv")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    M = Path(a.manifest_dir)

    obs = ad.read_h5ad(a.h5ad, backed="r").obs.copy()
    lines = pd.read_csv(M / "lines.tsv", sep="\t")
    plates = pd.read_csv(M / "plates.tsv", sep="\t")
    splits = pd.read_csv(M / "splits.tsv", sep="\t")
    line_idx = dict(zip(lines.cell_line_id, lines.line_index))
    plate_idx = dict(zip(plates.plate, plates.plate_index))
    grp_idx = dict(zip(splits.group_smiles.astype(str), splits.group_index))
    grp_split = dict(zip(splits.group_index, splits.split))
    valid = np.load(Path(a.v2_data) / "valid.npy")

    is_ctrl = obs.control.to_numpy() == 1
    drug = np.where(is_ctrl, -1, obs.SMILES.astype(str).map(lambda s: grp_idx.get(s, -2)).to_numpy())
    assert (drug != -2).all(), "treated cell with a structure group missing from splits.tsv"
    u = np.array([bhash(b) for b in obs.barcode.astype(str)])
    key = obs["sample"].astype(str).to_numpy() + "|" + obs.cell_line_id.astype(str).to_numpy()
    codes, uniq = pd.factorize(key, sort=True)
    order = np.lexsort((u, codes))                     # by group, then by barcode hash
    codes_s = codes[order]
    starts = np.r_[0, np.flatnonzero(np.diff(codes_s)) + 1]
    ends = np.r_[starts[1:], len(codes_s)]

    o = obs.iloc[order]
    first = o.iloc[starts]
    w = pd.DataFrame(dict(
        gid=np.arange(len(starts)),
        sample=first["sample"].astype(str).to_numpy(),
        cell_line_id=first.cell_line_id.astype(str).to_numpy(),
        cell_name=first.cell_type.astype(str).to_numpy(),
        plate=first.plate.astype(str).to_numpy(),
        drug=drug[order][starts],
        start=starts, end=ends, n_cells=ends - starts))
    w["line"] = w.cell_line_id.map(line_idx).astype(int)
    w["plate_index"] = w.plate.map(plate_idx).astype(int)
    w["kind"] = np.where(w.drug < 0, "ctrl", "treated")
    w["role"] = [("ctrl" if d < 0 else grp_split[d]) for d in w.drug]
    w["valid"] = [bool(valid[l, d]) if d >= 0 else True for l, d in zip(w.line, w.drug)]
    # a group must be homogeneous in everything recorded per well
    for col in ["plate", "cell_line_id"]:
        nun = o.groupby(codes_s, observed=True)[col].nunique()
        assert (nun == 1).all(), col
    assert (pd.Series(drug[order]).groupby(codes_s).nunique() == 1).all()

    with h5py.File(a.h5ad, "r") as f:
        X = f["X"][...]
    assert X.shape[0] == len(obs) and X.dtype == np.float32
    assert float(X.min()) >= 0.0
    Xs = X[order]
    np.save(out / "X.npy", Xs)
    np.savez(out / "cells.npz", gid=codes_s.astype(np.int32), u=u[order],
             library_size=o.library_size.to_numpy().astype(np.float32))
    Path(a.wells_out).parent.mkdir(parents=True, exist_ok=True)
    w.to_csv(a.wells_out, sep="\t", index=False)
    w.to_csv(out / "wells.tsv", sep="\t", index=False)

    rec = dict(
        n_cells=int(len(Xs)), n_genes=int(Xs.shape[1]), n_groups=int(len(w)),
        n_ctrl_groups=int((w.kind == "ctrl").sum()),
        groups_by_role={k: int(v) for k, v in w.role.value_counts().items()},
        valid_treated_groups_by_role={k: int(v) for k, v in w[(w.kind == "treated") & w.valid].role.value_counts().items()},
        cells_per_valid_treated_group=w[(w.kind == "treated") & w.valid].n_cells.describe().to_dict(),
        sha256_X_reordered=hashlib.sha256(np.ascontiguousarray(Xs).tobytes()).hexdigest(),
        sha256_barcodes_in_order=hashlib.sha256("\n".join(o.barcode.astype(str)).encode()).hexdigest(),
        source_h5ad=a.h5ad, order="lexsort(group, blake2b(barcode))")
    (out / "cache_record.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1), flush=True)


if __name__ == "__main__":
    main()
