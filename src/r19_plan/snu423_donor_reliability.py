"""Audit independent plate/position support in the pinned SNU-423 donors."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import sha256, write_json
from src.r11_chemical_probe.dataset import GROUP
from src.r19_plan.r18_preflight import checked_tables
from src.r19_plan.i_preflight import read_pinned


CELL = ["parent", "cell_id"]


def closest_donors(rows):
    distance = rows.actual_dose_uM.astype(float) - 5.
    if not distance.between(0, 15).all():
        raise ValueError("The original SNU-423 donor dose range changed")
    minimum = distance.groupby([rows.parent, rows.cell_id]).transform("min")
    return rows.loc[distance.eq(minimum)].reset_index(drop=True)


def pair_masks(wells):
    dose, plate, position = [wells[col].to_numpy() for col in ("actual_dose_uM", "rna_plate", "rna_well")]
    identical_dose = (dose[:, None] == dose[None, :]) & np.triu(np.ones((len(wells), len(wells)), dtype=bool), k=1)
    different_plate, different_position = (x[:, None] != x[None, :] for x in (plate, position))
    return dict(independent=identical_dose & different_plate & different_position,
                repeat_position=identical_dose & different_plate & ~different_position,
                repeat_plate=identical_dose & ~different_plate & different_position)


def coverage(source, parents):
    cells = []
    for key, subset in source.groupby(CELL, sort=True):
        counts = {name: int(mask.sum()) for name, mask in pair_masks(subset).items()}
        cells.append(dict(zip(CELL, key, strict=True), wells=len(subset), plates=subset.rna_plate.nunique(), **counts))
    groups = pd.DataFrame(cells)
    table = (groups.assign(independent_context=groups.independent.gt(0), repeat_position_context=groups.repeat_position.gt(0),
                           repeat_plate_context=groups.repeat_plate.gt(0))
             .groupby("parent", sort=True).agg(contexts=("cell_id", "size"), independent=("independent_context", "sum"),
                   repeat_position=("repeat_position_context", "sum"), repeat_plate=("repeat_plate_context", "sum")))
    table = parents.merge(table, on="parent", validate="one_to_one", how="left")
    if table.independent.isna().any() or len(table) != 100:
        raise ValueError("The support scope is not the original 100 donor parents")
    return table, groups


def audit(args):
    receipt = checked_tables(args.audit, args.pin)
    if not receipt["metadata_only"] or receipt["expression_read"]:
        raise ValueError("Expected the locked donor metadata before expression extraction")
    parents = read_pinned(args.audit, "parent_support.tsv", receipt["files"]["parent_support.tsv"])
    parents = parents.loc[parents.measured_5_to_20uM.eq("True"), [GROUP, "parent", "fold"]]
    source = closest_donors(read_pinned(args.audit, "admitted_donors.tsv", receipt["files"]["admitted_donors.tsv"]))
    if source.duplicated(["rna_plate", "rna_well"]).any() or source.hours.ne("24").any():
        raise ValueError("The source donor well registry is invalid")
    table, groups = coverage(source, parents)
    rows = []
    for fold in ("all", "0", "1", "2"):
        part = table if fold == "all" else table.loc[table.fold.eq(fold)]
        chosen = groups.loc[groups.parent.isin(part.parent)]
        rows.append(dict(fold=fold, parent_total=len(part), independent_parents=int(part.independent.gt(0).sum()),
                         multiple_independent_contexts=int(part.independent.ge(2).sum()),
                         same_position_parents=int(part.repeat_position.gt(0).sum()),
                         same_plate_parents=int(part.repeat_plate.gt(0).sum()),
                         source_contexts=len(chosen), independent_contexts=int(chosen.independent.gt(0).sum()),
                         independent_pairs=int(chosen.independent.sum()), same_position_pairs=int(chosen.repeat_position.sum())))
    results = pd.DataFrame(rows)
    gate = (rows[0]["independent_parents"] >= 50 and rows[0]["multiple_independent_contexts"] >= 20
            and min(r["independent_parents"] for r in rows[1:]) >= 10)
    args.out.mkdir(exist_ok=False)
    results.to_csv(args.out / "support.tsv", sep="\t", index=False)
    table.to_csv(args.out / "parent_support.tsv", sep="\t", index=False)
    write_json(args.out / "receipt.json", dict(protocol_sha256=sha256(args.protocol), source_sha256=sha256(args.pin),
        code_sha256=sha256(__file__), gate_pass=gate, metadata_only=True, expression_read=False,
        files={p.name: sha256(p) for p in args.out.iterdir()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("audit", "pin", "protocol", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    audit(parser.parse_args())
