"""Generate only synthetic assay input to exercise the locked intake and scorer."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r19_plan.hcc_off_position_pilot_score import SCALE, frozen_arrays, registered_layout


def generate(args):
    freeze = read_json(args.forecast_pin)
    layout, slots = registered_layout(args.audit, freeze["pilot_sha256"])
    genes, _ = frozen_arrays(args, freeze)
    layouts = layout["plate_slot"].drop_duplicates()
    physical = {x: f"mock-plate-{i:02d}" for i, x in enumerate(layouts, 1)}
    blocks = {x: f"mock-culture-{i:02d}" for i, x in enumerate(layouts, 1)}
    treatments = layout.kind.eq("treated")
    parent = layout.parent_slot.map(slots.set_index("parent_slot").parent).fillna("")
    wells = layout[["plate_slot", "well"]].assign(specimen_id=[f"mock-specimen-{i:04d}" for i in range(1, len(layout)+1)],
        actual_cell=layout.cell, plate_id=layout.plate_slot.map(physical), culture_id=layout.culture_block_slot.map(blocks),
        canonical_parent=parent, material_lot_id=np.where(treatments, "mock-lot-" + layout.parent_slot, "mock-DMSO"),
        actual_uM=np.where(treatments, "10", ""), actual_hours="24", qc_status="PASS", qc_reason="")
    first_plate = layout.plate_slot.iloc[0]
    inactive = [*layout.loc[layout.plate_slot.eq(first_plate) & layout.vehicle_role.eq("A")].index[:2],
                layout.loc[layout.plate_slot.ne(first_plate) & treatments].index[0]]
    wells.loc[inactive, ["qc_status", "qc_reason"]] = ["FAIL", "synthetic exclusion"]
    cultures = layout[["cell", "plate_slot"]].drop_duplicates().assign(plate_id=lambda x:x.plate_slot.map(physical),
        culture_id=lambda x:x.plate_slot.map(blocks), independent_start_id=lambda x:"mock-start-" + x.plate_slot,
        batch_id=lambda x:"mock-batch-" + x.plate_slot.str[-2:], independence_record="Synthetic fixture only").drop(columns="plate_slot")
    args.out.mkdir(exist_ok=False, parents=True)
    wells.to_csv(args.out / "wells.tsv", sep="\t", index=False)
    cultures.to_csv(args.out / "cultures.tsv", sep="\t", index=False)
    (args.out / "qc.md").write_text("Synthetic input generated for software testing. No laboratory records or results.\n")
    valid = wells.loc[wells.qc_status.eq("PASS")]
    rng = np.random.default_rng(191199)
    values = rng.normal(5, .8, size=(len(valid), len(genes))).astype("float32")
    np.savez_compressed(args.out / "measurements.npz", values=values[::-1, ::-1],
                         specimen_id=valid.specimen_id.to_numpy(dtype=str)[::-1], genes=genes[::-1])
    write_json(args.out / "assay.json", dict(mode="synthetic", study_id="off-position-scorer-smoke", platform="Synthetic",
        target_scale=SCALE, verified_by="Synthetic fixture only", measurements_sha256=sha256(args.out / "measurements.npz"),
        wells_sha256=sha256(args.out / "wells.tsv"), cultures_sha256=sha256(args.out / "cultures.tsv"),
        qc_record_sha256=sha256(args.out / "qc.md")))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for field in ("audit", "forecast", "forecast-pin", "out"):
        cli.add_argument("--" + field, type=Path, required=True)
    generate(cli.parse_args())
