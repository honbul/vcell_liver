"""Generate and verify the independent adult HCC pilot's randomized well map."""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.r11_chemical_probe.common import sha256, write_json


SEED = 191196
LINES = ("SNU-423", "HUH7")
ROLES = ("A", "B", "C")
BLOCKS = 8
SLOTS = [f"P{i:03d}" for i in range(1, 97)]


def well_space():
    rows = [dict(well=f"{letter}{j+1:02d}", edge=i in (0, 7) or j in (0, 11), quadrant=2*(i//4)+j//6)
            for i, letter in enumerate("ABCDEFGH") for j in range(12)]
    return pd.DataFrame(rows).set_index("well")


def controls(wells, block, rng):
    draft = []
    free = wells.copy()
    for role_i, role in enumerate(ROLES):
        for quadrant in range(4):
            edge = (block + role_i) % 4 == quadrant
            select = free.index[free.quadrant.eq(quadrant) & free.edge.eq(edge)]
            well = str(rng.choice(select))
            draft.append(dict(well=well, kind="vehicle", parent_slot="", vehicle_role=role, repeat=""))
            free = free.drop(index=well)
    return draft, free


def assign_wells(parents, free, used, rng):
    draft = []
    for edge in (True, False):
        rows = [row for row in parents if row[2] == edge]
        available = free.index[free.edge.eq(edge)]
        cost = rng.random((len(rows), len(available)))
        for i, (parent, _, _) in enumerate(rows):
            cost[i, available.isin(used[parent])] = np.inf
        idx, assigned = linear_sum_assignment(cost)
        if len(assigned) != len(rows) or not np.isfinite(cost[idx, assigned]).all():
            raise ValueError("No eligible non-repeated coordinate for a pilot parent")
        for i, j in zip(idx, assigned, strict=True):
            parent, repeat, _ = rows[i]
            site = available[j]
            used[parent].add(site)
            draft.append(dict(well=site, kind="treated", parent_slot=parent, vehicle_role="", repeat=repeat))
    return draft


def repeat_blocks(rng):
    triplets = sorted(combinations(range(BLOCKS), 3))
    orbits = {}
    for triplet in triplets:
        orbit = [tuple(sorted((block+shift) % BLOCKS for block in triplet)) for shift in range(BLOCKS)]
        orbits[min(orbit)] = sorted(orbit)
    extra = rng.permutation(len(orbits))[:5]
    groups = list(orbits.values())
    schedule = np.array([*triplets, *(triplet for i in extra for triplet in groups[i])])
    rng.shuffle(schedule)
    plate_order = rng.permutation(BLOCKS)
    return plate_order[schedule]


def layout():
    rng = np.random.default_rng(SEED)
    wells = well_space()
    used = {parent: set() for parent in SLOTS}
    tables = []
    for line in LINES:
        schedule = repeat_blocks(rng)
        edge_slots = np.repeat(np.arange(BLOCKS), 12)
        edge_options = (schedule[:, :, None] == edge_slots[None, None, :]).any(axis=1)
        edge_cost = np.where(edge_options, rng.random(edge_options.shape), np.inf)
        rows, chosen = linear_sum_assignment(edge_cost)
        parent_order = rng.permutation(SLOTS)
        assigned = [[] for _ in range(BLOCKS)]
        for i, chosen_edge in zip(rows, edge_slots[chosen], strict=True):
            for repeat, block in enumerate(rng.permutation(schedule[i]), 1):
                assigned[block].append((parent_order[i], repeat, block == chosen_edge))
        for block, parents in enumerate(assigned, 1):
            vehicle, free = controls(wells, block, rng)
            treated = assign_wells(parents, free, used, rng)
            plate = f"{line}-B{block:02d}"
            table = pd.DataFrame([*vehicle, *treated]).join(wells, on="well")
            dose = np.where(table.kind.eq("treated"), 5., np.nan)
            tables.append(table.assign(cell=line, culture_block_slot=plate, plate_slot=plate, hours=24, nominal_uM=dose))
    columns = ["cell", "culture_block_slot", "plate_slot", "well", "edge", "quadrant", "kind", "parent_slot",
               "vehicle_role", "repeat", "hours", "nominal_uM"]
    return pd.concat(tables, ignore_index=True)[columns].sort_values(["cell", "plate_slot", "well"], ignore_index=True)


def verify(frame):
    if len(frame) != 768 or frame.duplicated(["plate_slot", "well"]).any() or not frame.well.isin(well_space().index).all():
        raise ValueError("The pilot does not have unique physical coordinates")
    treatment = frame.loc[frame.kind.eq("treated")]
    vehicles = frame.loc[frame.kind.eq("vehicle")]
    by_parent = treatment.groupby(["cell", "parent_slot"])
    by_plate = treatment.groupby("plate_slot")
    role = vehicles.groupby(["plate_slot", "vehicle_role"])
    variety = by_parent.culture_block_slot.agg(lambda slots: tuple(sorted(slots))).groupby(level="cell").nunique()
    if (set(treatment.parent_slot) != set(SLOTS) or len(by_parent) != 192 or not by_parent.size().eq(3).all()
            or not by_parent.culture_block_slot.nunique().eq(3).all() or not by_parent.edge.sum().eq(1).all()
            or treatment.duplicated(["parent_slot", "well"]).any() or not by_parent.repeat.nunique().eq(3).all()
            or not variety.eq(56).all() or not treatment.nominal_uM.eq(5.).all() or vehicles.nominal_uM.notna().any()
            or len(by_plate) != 16 or not by_plate.size().eq(36).all() or not by_plate.edge.sum().eq(12).all()
            or len(role) != 48 or set(vehicles.vehicle_role) != set(ROLES) or not role.size().eq(4).all()
            or not role.quadrant.nunique().eq(4).all() or not role.edge.sum().eq(1).all()):
        raise ValueError("The adult HCC pilot lost a repeat, a position balance, or an independent vehicle role")
    return by_plate.agg(cell=("cell", "first"), treatments=("parent_slot", "size"), treatment_edge=("edge", "sum")).join(
        vehicles.groupby("plate_slot").agg(vehicles=("well", "size"), vehicle_edge=("edge", "sum"))).reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    map_ = layout()
    checks = verify(map_)
    args.out.mkdir(parents=True, exist_ok=False)
    map_.to_csv(args.out/"plate_layout.tsv", sep="\t", index=False, na_rep="NA")
    checks.to_csv(args.out/"support.tsv", sep="\t", index=False)
    write_json(args.out/"receipt.json", dict(seed=SEED, cell_lines=list(LINES), parent_slots=len(SLOTS), data_measured=False,
        protocol_sha256=sha256(args.protocol), code_sha256=sha256(__file__),
        files={p.name: sha256(p) for p in args.out.iterdir()}))


if __name__ == "__main__":
    main()
