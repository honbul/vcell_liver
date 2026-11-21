"""Read the original GEO metadata and extract selected L1000 landmark rows."""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from src.r13_optimization.lincs_transfer import clean, structural_keys


STUDIES = ("GSE92742", "GSE70138")
LIVER = ("HEPG2", "HUH7", "PHH")
EXCLUDE_LINES = {"HEPG2", "HEPG2C3A", "C3A", "HUH7", "HUH751", "PHH", "JHH5", "JHH7", "HEP3B",
                 "HEP3B2", "PLC", "PLCPRF5", "SNU398", "SNU423", "SNU449", "SKHEP1", "HLE", "HLF", "HEPARG"}


def asset(root, study, part):
    tail = ".txt" if study == "GSE92742" else "_*.txt.gz"
    matches = sorted((Path(root) / study).glob(f"*_{part}{tail}"))
    if part == "pert_info" and len(matches) > 1:
        matches = [path for path in matches if "03-06" in path.name]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {study} {part}: {matches}")
    return matches[0]


def table(path, **kwargs):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, **kwargs)


def measures(value, unit, scales):
    codes = unit.str.lower().str.replace("µ", "u", regex=False).str.replace("μ", "u", regex=False)
    return pd.to_numeric(value, errors="coerce") * codes.map(scales)


def numbers(values, scales):
    parts = values.str.extract(r"^\s*([-+\d.eE]+)\s*([^\s]+)\s*$")
    return measures(parts[0], parts[1], scales)


DOSE = {"um": 1., "mm": 1000., "nm": .001}
TIME = {"h": 1., "hr": 1., "d": 24., "m": 1/60}


def signatures(root, study):
    rows = table(asset(root, study, "sig_info"))
    rows["dose"] = numbers(rows.pert_idose, DOSE)
    rows["hours"] = numbers(rows.pert_itime, TIME)
    rows["cell_id"] = rows.cell_id.str.upper().str.strip()
    return rows


def instances(root, study):
    rows = table(asset(root, study, "inst_info"))
    if "rna_plate" not in rows:
        rows = rows.rename(columns={"det_plate": "rna_plate", "det_well": "rna_well"})
    rows["dose"] = measures(rows.pert_dose, rows.pert_dose_unit, DOSE)
    rows["hours"] = measures(rows.pert_time, rows.pert_time_unit, TIME)
    rows["rna_well"] = rows.rna_well.str.strip().str.upper().str.replace(r"^([A-Z])([0-9])$", r"\g<1>0\2", regex=True)
    rows["cell_id"] = rows.cell_id.str.upper().str.strip()
    return rows


def chemicals(root, study):
    source = table(asset(root, study, "pert_info"))
    source = source.loc[source.pert_type.eq("trt_cp")].copy()
    source = source.sort_values("pert_id", kind="stable").drop_duplicates("pert_id")
    resolved = []
    for r in source.itertuples(index=False):
        _, parent = structural_keys(clean(r.canonical_smiles))
        resolved.append(parent)
    source["structure"] = resolved
    source["parent"] = source.structure.where(source.structure.ne(""), "UNRESOLVED:" + source.pert_id)
    return source.reset_index(drop=True)


def genes(root, study):
    raw = table(asset(root, study, "gene_info"))
    keep = raw.loc[raw.pr_is_lm.eq("1"), ["pr_gene_id", "pr_gene_symbol"]].copy()
    keep["pr_gene_id"] = keep.pr_gene_id.astype(int)
    if len(keep) != 978 or not keep.pr_gene_id.is_unique:
        raise ValueError(f"The {study} landmark axis differs from the expected 978")
    return keep.sort_values("pr_gene_id").reset_index(drop=True)


def landmark_rows(path, ids, landmarks):
    """Read only selected matrix slabs, preserving supplied row and gene order."""
    with h5py.File(path, "r") as archive:
        indices = pd.Index(archive["0/META/COL/id"][:].astype(str)).get_indexer(ids)
        columns = pd.Index(archive["0/META/ROW/id"][:].astype(int)).get_indexer(landmarks)
        if (indices < 0).any() or (columns < 0).any():
            raise ValueError(f"Unmatched sample or landmark in {path}")
        matrix = archive["0/DATA/0/matrix"]
        order = np.argsort(indices)
        indices = indices[order]
        breaks = np.r_[0, np.flatnonzero(np.diff(indices // 512)) + 1, len(indices)]
        out = np.empty((len(ids), len(columns)), dtype=np.float32)
        for start, end in zip(breaks[:-1], breaks[1:]):
            subset = indices[start:end]
            values = matrix[subset[0]:subset[-1]+1]
            out[order[start:end]] = values[np.ix_(subset-subset[0], columns)]
    if not np.isfinite(out).all():
        raise ValueError(f"Nonfinite expression in {path}")
    return out
