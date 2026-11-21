"""Guard development inputs against the R18 reservation and validate forecast axes."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def validate_forecast(predicted, observed, prediction_genes, reference_genes):
    genes = tuple(reference_genes)
    if predicted.ndim != 2 or predicted.shape != observed.shape or predicted.shape[1] != len(genes):
        raise ValueError("Invalid response shape")
    if len(set(genes)) != len(genes) or tuple(prediction_genes) != genes:
        raise ValueError("Invalid gene order")
    if not np.isfinite(predicted).all() or not np.isfinite(observed).all():
        raise ValueError("Nonfinite response")


def id_keys(rows, column):
    return set(rows[column].astype(str).str.strip().str.upper()) - {""}


def coord_keys(rows, plate, well):
    positions = rows[[plate, well]].astype(str).apply(lambda col: col.str.strip().str.upper())
    positions[well] = positions[well].str.replace(r"^([A-Z]+)([0-9])$", r"\g<1>0\2", regex=True)
    present = positions.ne("").all(axis=1)
    return set(positions.loc[present].itertuples(index=False, name=None))


def assert_dev_allowed(rows, reserved, donor_exceptions):
    """Require constituent-level assay provenance before using DEV fit/select/pretraining rows."""
    required = {"sig_id", "inst_id", "rna_plate", "rna_well", "det_plate", "det_well"}
    for name, frame in [("candidate", rows), ("manifest", reserved), ("B exceptions", donor_exceptions)]:
        if not required.issubset(frame.columns) or frame[list(required)].isna().any().any():
            raise ValueError(f"Incomplete {name} assay provenance")
    if rows.inst_id.astype(str).str.strip().eq("").any():
        raise ValueError("A DEV assay record lacks a constituent well ID")
    blocked = pd.concat((reserved, donor_exceptions), ignore_index=True)
    for column in ("sig_id", "inst_id"):
        if id_keys(rows, column) & id_keys(blocked, column):
            raise ValueError(f"Reserved R18 assay input in DEV ({column})")
    allowed_positions = coord_keys(rows, "rna_plate", "rna_well") | coord_keys(rows, "det_plate", "det_well")
    held_positions = coord_keys(blocked, "rna_plate", "rna_well") | coord_keys(blocked, "det_plate", "det_well")
    if allowed_positions & held_positions:
        raise ValueError("Reserved R18 assay input in DEV (physical coordinate)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "reserved", "donor-exceptions"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    assert_dev_allowed(*(pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
                         for path in (args.inputs, args.reserved, args.donor_exceptions)))
