"""Build a prospective HCC layout that avoids the existing donor positions."""

import argparse
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r19_plan.hcc_10um_donor_pilot import inventory
from src.r19_plan.hcc_donor_panel_size import choices
from src.r19_plan.hcc_lincs_24h_access import CONTEXT
from src.r19_plan.hcc_lincs_24h_support import SCOPE
from src.r19_plan.hcc_pilot_chemistry import PHASES
from src.r19_plan.hcc_pilot_layout import BLOCKS, LINES, ROLES, controls, repeat_blocks, well_space
from src.r19_plan.i_preflight import read_pinned


SLOTS = tuple(f"O{i:03d}" for i in range(1, 97))
FIRST_SEED, SEED_ATTEMPTS, DONOR_COUNT = 191197, 16, 8


def source_panels(treated, frame, n):
    wells = well_space()
    enough = frame.loc[frame.cells.ge(n) & frame.plates.ge(n), "parent"]
    panels, descriptions = [], []
    for parent, block in treated.loc[treated.parent.isin(enough)].groupby("parent", sort=True):
        names = sorted(block.cell_id.unique())
        picked = [names[i] for i in choices(parent, names, 0)[:n]]
        panel = block.loc[block.cell_id.isin(picked)]
        excluded = set(panel.rna_well) & set(wells.index)
        available = wells.loc[~wells.index.isin(excluded)]
        edge = int(available.edge.sum())
        distinct_plates = int(panel.rna_plate.nunique())
        descriptions.append(dict(parent=parent, blocked_wells=len(excluded), eligible_edge=edge,
            eligible_inner=len(available)-edge, panel_cells=len(picked), panel_plates=distinct_plates,
            panel_wells=len(panel), position_possible=bool(edge >= 2 and len(available)-edge >= 4 and distinct_plates >= n)))
        panels.append(panel)
    records = pd.DataFrame(descriptions, columns=["parent", "blocked_wells", "eligible_edge", "eligible_inner",
        "panel_cells", "panel_plates", "panel_wells", "position_possible"])
    return records, pd.concat(panels, ignore_index=True) if panels else treated.iloc[:0]


def draft_layout(forbidden, seed):
    rng = np.random.default_rng(seed)
    wells = well_space()
    used = {slot: set(forbidden[slot]) for slot in SLOTS}
    frames = []
    for cell in LINES:
        schedule = repeat_blocks(rng)
        edge_slots = np.repeat(np.arange(BLOCKS), 12)
        possible = (schedule[:, :, None] == edge_slots[None, None, :]).any(axis=1)
        rows, assign = linear_sum_assignment(np.where(possible, rng.random(possible.shape), np.inf))
        slots = rng.permutation(SLOTS)
        allocated = [[] for _ in range(BLOCKS)]
        for i, edge_block in zip(rows, edge_slots[assign], strict=True):
            for replicate, block in enumerate(rng.permutation(schedule[i]), 1):
                allocated[block].append((slots[i], replicate, block == edge_block))
        for block, treatment in enumerate(allocated, 1):
            vehicle, free = controls(wells, block, rng)
            selected = []
            for is_edge in (True, False):
                entries = [row for row in treatment if row[2] == is_edge]
                possible_wells = free.index[free.edge.eq(is_edge)]
                cost = rng.random((len(entries), len(possible_wells)))
                for i, (slot, _, _) in enumerate(entries):
                    cost[i, possible_wells.isin(used[slot])] = np.inf
                ii, jj = linear_sum_assignment(cost)
                if len(ii) != len(entries) or not np.isfinite(cost[ii, jj]).all():
                    raise ValueError("No donor-disjoint well assignment at this seed")
                for i, j in zip(ii, jj, strict=True):
                    slot, replicate, _ = entries[i]
                    site = possible_wells[j]
                    used[slot].add(site)
                    selected.append(dict(well=site, kind="treated", parent_slot=slot, vehicle_role="", repeat=replicate))
            plate = f"{cell}-OFF-B{block:02d}"
            table = pd.DataFrame([*vehicle, *selected]).join(wells, on="well")
            frames.append(table.assign(cell=cell, culture_block_slot=plate, plate_slot=plate, hours=24,
                                       nominal_uM=np.where(table.kind.eq("treated"), 10., np.nan)))
    columns = ["cell", "culture_block_slot", "plate_slot", "well", "edge", "quadrant", "kind", "parent_slot", "vehicle_role",
               "repeat", "hours", "nominal_uM"]
    return pd.concat(frames, ignore_index=True)[columns].sort_values(["cell", "plate_slot", "well"], ignore_index=True)


def verify_layout(layout, forbidden):
    treatments = layout.loc[layout.kind.eq("treated")]
    vehicles = layout.loc[layout.kind.eq("vehicle")]
    for slot, group in treatments.groupby("parent_slot"):
        if set(group.well) & set(forbidden[slot]):
            raise ValueError("A target well overlaps a registered donor index")
    per_parent = treatments.groupby(["cell", "parent_slot"])
    per_plate = treatments.groupby("plate_slot")
    role = vehicles.groupby(["plate_slot", "vehicle_role"])
    triplets = per_parent.culture_block_slot.agg(lambda s: tuple(sorted(s)))
    if (len(layout) != 768 or layout.duplicated(["plate_slot", "well"]).any() or not layout.well.isin(well_space().index).all()
        or set(treatments.parent_slot) != set(SLOTS) or len(per_parent) != 192 or not per_parent.size().eq(3).all()
        or not per_parent.edge.sum().eq(1).all() or not per_parent.repeat.nunique().eq(3).all()
        or not per_parent.culture_block_slot.nunique().eq(3).all() or not triplets.groupby(level="cell").nunique().eq(56).all()
        or treatments.duplicated(["parent_slot", "well"]).any() or not treatments.nominal_uM.eq(10).all()
        or not layout.hours.eq(24).all() or vehicles.nominal_uM.notna().any() or set(layout.cell) != set(LINES)
        or len(per_plate) != 16 or not per_plate.size().eq(36).all() or not per_plate.edge.sum().eq(12).all()
        or len(role) != 48 or set(vehicles.vehicle_role) != set(ROLES) or not role.size().eq(4).all()
        or not role.quadrant.nunique().eq(4).all() or not role.edge.sum().eq(1).all()):
        raise ValueError("The prospective pilot lost a repeat, coordinate, or disjoint vehicle role")
    return per_plate.agg(cell=("cell", "first"), treatments=("parent_slot", "size"), treatment_edge=("edge", "sum")).join(
        vehicles.groupby("plate_slot").agg(vehicles=("well", "size"), vehicle_edge=("edge", "sum"))).reset_index()


def run(args):
    low_pin = args.reports / "r19_plan/data/jhh_low_dose/audit/receipt.json"
    if sha256(args.jhh_low / "receipt.json") != sha256(low_pin):
        raise ValueError("The newer adult target-exposure audit differs from its pin")
    low_rows = read_pinned(args.jhh_low, "targets.tsv", read_json(low_pin)["files"]["targets.tsv"])
    scope = inventory(args, set(low_rows.parent))
    allowed = scope.frame.loc[~scope.frame.prior_adult & ~scope.frame.reserved_chemistry].copy()
    support, all_panels = [], {}
    for n in (8, 16):
        descriptions, donor = source_panels(scope.treated, allowed, n)
        candidates = allowed.merge(descriptions, on="parent", how="inner", validate="one_to_one")
        all_panels[n] = (candidates, donor)
        for label, kept in (("registered_phases", candidates.loc[candidates.phase.isin(PHASES)]), ("all_phases", candidates)):
            viable = kept.loc[kept.position_possible]
            support.append(dict(phase_scope=label, minimum_cells=n, supported_parents=len(kept),
                supported_scaffolds=kept.scaffold_unit.nunique(), with_position_options=len(viable),
                distinct_position_eligible_scaffolds=viable.scaffold_unit.nunique()))
    candidates, roster = all_panels[DONOR_COUNT]
    choose = candidates.loc[candidates.phase.isin(PHASES) & candidates.position_possible].copy()
    choose["order"] = choose.parent.map(lambda p: hashlib.sha256(f"HCC-10uM-20261122|{p}".encode()).hexdigest())
    selected = choose.sort_values("order").drop_duplicates("scaffold_unit").head(len(SLOTS))
    selected = selected.assign(parent_slot=SLOTS[:len(selected)])
    donors = roster.merge(selected[["parent", "parent_slot"]], on="parent", validate="many_to_one")
    eligible_controls = scope.dmsos.merge(donors[["cell_id", "rna_plate"]].drop_duplicates(), on=CONTEXT[1:], validate="many_to_one")
    forbidden = {slot: set(rows.rna_well) & set(well_space().index) for slot, rows in donors.groupby("parent_slot")}
    layout, accepted_seed, checks = None, None, None
    if len(selected) == len(SLOTS):
        for seed in range(FIRST_SEED, FIRST_SEED+SEED_ATTEMPTS):
            try:
                result = draft_layout(forbidden, seed)
            except ValueError:
                continue
            checks = verify_layout(result, forbidden)
            layout, accepted_seed = result, seed
            break
    if args.out.exists():
        raise FileExistsError(args.out)
    with TemporaryDirectory(dir=args.out.parent, prefix=".off-position-pilot-") as temporary:
        stage = Path(temporary)
        selected[["parent_slot", "parent", "brd_id", "drug", "phase", "isomeric_parent", "scaffold_unit", "panel_cells",
                  "panel_plates", "panel_wells", "blocked_wells", "eligible_edge", "eligible_inner"]].to_csv(stage / "slots.tsv", sep="\t", index=False)
        pd.DataFrame(support).to_csv(stage / "support.tsv", sep="\t", index=False)
        donors[["parent_slot", "parent", *SCOPE, "actual_dose_uM"]].to_csv(stage / "donor_wells.tsv", sep="\t", index=False)
        eligible_controls[SCOPE].to_csv(stage / "donor_controls.tsv", sep="\t", index=False)
        pd.DataFrame([(slot, well) for slot, sites in sorted(forbidden.items()) for well in sorted(sites)],
                    columns=["parent_slot", "forbidden_well"]).to_csv(stage / "forbidden_positions.tsv", sep="\t", index=False)
        if layout is not None:
            layout.to_csv(stage / "plate_layout.tsv", sep="\t", index=False, na_rep="NA")
            checks.to_csv(stage / "plate_checks.tsv", sep="\t", index=False)
        write_json(stage / "receipt.json", dict(metadata_only=True, expression_read=False, measured=False, minimum_sources=8,
            independent_finishing_required=True, selected=len(selected), actual_treatment_slots=0 if layout is None else 576,
            actual_vehicle_slots=0 if layout is None else 192, planned_lines=list(LINES), seed=accepted_seed,
            files={path.name: sha256(path) for path in stage.iterdir()}, protocol_sha256=sha256(args.protocol),
            code_sha256=sha256(__file__), prior_5uM_sha256=sha256(scope.pilot_pin), prior_JHH10_sha256=sha256(scope.jhh_pin),
            prior_JHH_low_sha256=sha256(low_pin), prior_16_scope_sha256=sha256(args.reports / "r19_plan/data/hcc_10um_donor_pilot/audit/receipt.json"),
            source_sha256={name: sha256(path) for name,path in scope.files.items()},
            reserve_access_sha256=sha256(args.r18_access / "receipt.json")))
        stage.rename(args.out)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for field in ("base", "reports", "old", "jhh-audit", "jhh-low", "r18-registered", "r18-lock", "r18-access", "r18-index", "protocol", "out"):
        cli.add_argument("--" + field, type=Path, required=True)
    run(cli.parse_args())
