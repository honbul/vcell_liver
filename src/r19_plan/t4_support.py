"""Audit time-paired LINCS constituents using measured dose and protected-well metadata."""

import argparse
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd

from src.r11_chemical_probe.common import sha256, write_json
from src.r15_lincs.data import STUDIES, asset
from src.r15_lincs.design import fold_id
from src.r17_lincs.inventory import ROOT, ids, in_footprint, register_footprint
from src.r19_plan.audit_r18 import geo_metadata, membership, prepare
from src.r19_plan.i_preflight import provenance, read_r18_domains

TIME_PAIR = (6, 24)
SOURCE_KEY = ["cell_id", "parent", "actual_dose_uM"]
PLATE_KEY = ["cell_id", "plate"]


def measured_dose(path, supported):
    """Return exact measured µM keys, avoiding accidental matches after dose rounding."""
    registry = pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False,
        usecols=["inst_id", "pert_dose", "pert_dose_unit"], chunksize=100_000,
    )
    parts = [frame.loc[frame.inst_id.isin(supported)] for frame in registry]
    result = pd.concat(parts, ignore_index=True).drop_duplicates()
    if result.inst_id.duplicated().any():
        raise ValueError("An instance has contradictory measured dose metadata")
    scale = {"um": Decimal(1), "nm": Decimal(".001"), "mm": Decimal(1000)}
    units = result.pert_dose_unit.str.strip().str.lower().str.replace("[µμ]", "u", regex=True)
    decimal_keys = list(zip(result.pert_dose.str.strip(), units))

    def key(value, unit):
        if unit not in scale:
            return ""
        try:
            dose = Decimal(value) * scale[unit]
        except InvalidOperation:
            return ""
        return format(dose.normalize(), "f") if dose.is_finite() and dose > 0 else ""

    converted = {pair: key(*pair) for pair in set(decimal_keys)}
    return result[["inst_id"]].assign(actual_dose_uM=[converted[pair] for pair in decimal_keys])


def eligible_links(inst, sig, protected, measured):
    physical = register_footprint(protected)
    blocked_sigs = set(ids(protected.sig_id)) - {""}
    before = sig.loc[sig.pert_type.eq("trt_cp") & sig.qc & sig.parent.ne("") & sig.hours.isin(TIME_PAIR)]
    if inst.inst_id.duplicated().any():
        raise ValueError("LINCS instance IDs are not unique")
    candidate = membership(before)
    meta = ["inst_id", "cell_id", "pert_id", "hours", "plate", "well", "rna_plate", "rna_well",
            "det_plate", "det_well"]
    physical_meta = inst[meta].rename(columns={"hours": "measured_hours"})
    known = candidate.merge(physical_meta, on=["inst_id", "cell_id", "pert_id"], how="left", validate="many_to_one")
    coordinate = known.plate.fillna("").ne("") & known.well.fillna("").str.fullmatch("[A-Z]+[0-9]{2}")
    found = known.loc[coordinate]
    reserved_members = found.loc[in_footprint(found, physical), "sig_id"]
    reserved_ids = known.loc[ids(known.inst_id).isin(physical[0]), "sig_id"]
    reserved = set(reserved_members) | set(reserved_ids) | set(before.loc[ids(before.sig_id).isin(blocked_sigs), "sig_id"])
    unknown = set(known.loc[~coordinate, "sig_id"])
    allowed = found.loc[~found.sig_id.isin(reserved | unknown) & found.hours.eq(found.measured_hours)]
    allowed = allowed.merge(measured, on="inst_id", validate="many_to_one")
    allowed = allowed.loc[allowed.actual_dose_uM.ne("")]

    active = inst.plate.ne("") & inst.well.str.fullmatch("[A-Z]+[0-9]{2}")
    controls = inst.loc[
        active & inst.pert_type.eq("ctl_vehicle") & (ids(inst.pert_id).eq("DMSO") | ids(inst.pert_iname).eq("DMSO"))
    ]
    controls = controls.loc[~in_footprint(controls, physical)].drop_duplicates(["plate", "well"])
    sizes = controls.groupby(PLATE_KEY).size().rename("usable_controls")
    allowed = allowed.merge(sizes, on=PLATE_KEY, how="left", validate="many_to_one")
    allowed = allowed.loc[allowed.usable_controls.ge(3)].copy()
    coordinates = ["plate", "well"]
    conditions = allowed.drop_duplicates([*coordinates, *SOURCE_KEY, "hours"])
    conflicting = conditions.loc[conditions.duplicated(coordinates, keep=False), coordinates].drop_duplicates()
    ambiguous = pd.MultiIndex.from_frame(allowed[coordinates]).isin(pd.MultiIndex.from_frame(conflicting))
    allowed = allowed.loc[~ambiguous]
    allowed = allowed.sort_values(["sig_id", "inst_id"]).drop_duplicates(["plate", "well"])
    assert not in_footprint(allowed, physical).any()
    assert not ids(allowed.sig_id).isin(blocked_sigs).any()
    return allowed, dict(qc_candidate_sigs=len(before), protected_sigs=len(reserved),
                         unlinked_constituent_sigs=len(unknown), retained_wells=len(allowed),
                         ambiguous_coordinates=len(conflicting), ambiguous_links=int(ambiguous.sum()))


def matched_pairs(rows):
    grouped = rows.groupby([*SOURCE_KEY, "hours"], sort=True).agg(
        wells=("inst_id", "size"), plates=("plate", lambda p: frozenset(p)),
    )
    sides = [grouped.loc[grouped.index.get_level_values("hours") == hour]
             .droplevel("hours").add_suffix(f"_{hour}h") for hour in TIME_PAIR]
    paired = sides[0].join(sides[1], how="inner").reset_index()
    paired["cross_time_plate_disjoint"] = [not bool(a & b) for a, b in zip(paired.plates_6h, paired.plates_24h)]
    paired["both_times_2plates"] = paired.plates_6h.map(len).ge(2) & paired.plates_24h.map(len).ge(2)
    paired["plate_count_6h"] = paired.plates_6h.map(len)
    paired["plate_count_24h"] = paired.plates_24h.map(len)
    paired["fold"] = paired.parent.map(lambda parent: fold_id(parent, "R15-outer", 3))
    return paired


def summarize(study, axis, paired):
    for portion, scope in [("ALL", paired), *paired.groupby("cell_id", sort=True)]:
        for section in ("ALL", 0, 1, 2):
            rows = scope if section == "ALL" else scope.loc[scope.fold.eq(section)]
            independent = rows.loc[rows.cross_time_plate_disjoint]
            repeat = independent.loc[independent.both_times_2plates]
            yield dict(
                study=study, coordinate_axis=axis, cell=portion, outer_fold=section,
                parents=rows.parent.nunique(), parent_cell_dose_pairs=len(rows),
                independent_parents=independent.parent.nunique(), independent_pairs=len(independent),
                independent_repeated_parents=repeat.parent.nunique(), independent_repeated_pairs=len(repeat),
                wells_6h=int(independent.wells_6h.sum()), wells_24h=int(independent.wells_24h.sum()),
                plates_6h=len(set().union(*independent.plates_6h)),
                plates_24h=len(set().union(*independent.plates_24h)),
            )


def run(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    reserved, exceptions, score, donors, _ = read_r18_domains(args)
    protected = pd.concat([provenance(rows) for rows in (reserved, exceptions, score, donors)], ignore_index=True)
    prior = args.base / "vcell_liver_r15_lincs/v1"
    stats, support, dose_support, sources = {}, [], [], {}
    for study in STUDIES:
        inst, sig = geo_metadata(study, prior)
        axis = "RNA" if {"rna_plate", "rna_well"} <= set(inst) else "DET_ONLY_UNVERIFIED_RNA"
        inst = prepare(inst)
        exact = measured_dose(asset(ROOT, study, "inst_info"),
                              set(inst.loc[inst.pert_type.eq("trt_cp") & inst.hours.isin(TIME_PAIR), "inst_id"]))
        rows, stats[study] = eligible_links(inst, sig, protected, exact)
        paired = matched_pairs(rows)
        support.extend(summarize(study, axis, paired))
        for (cell, dose), block in paired.groupby(["cell_id", "actual_dose_uM"], sort=True):
            dose_support.append(dict(study=study, coordinate_axis=axis, cell=cell, actual_dose_uM=dose,
                                     parents=block.parent.nunique(),
                                     independent_parents=block.loc[block.cross_time_plate_disjoint, "parent"].nunique()))
        files = [asset(ROOT, study, part) for part in ("inst_info", "sig_info", "sig_metrics")]
        files.append(prior / "audit" / study / "identities.tsv")
        sources[study] = {str(path): sha256(path) for path in files}
    args.out.mkdir(parents=True)
    pd.DataFrame(support).to_csv(args.out / "support.tsv", sep="\t", index=False)
    pd.DataFrame(dose_support).to_csv(args.out / "dose_support.tsv", sep="\t", index=False)
    write_json(args.out / "receipt.json", dict(
        source_metadata_sha256=sources, code_sha256=sha256(__file__),
        reservation_access_sha256=sha256(args.r18_access / "receipt.json"),
        reservation_registry_sha256=sha256(args.r18_registered / "receipt.json"),
        metadata_only=True, expression_read=False, fitted=False, overlaps_verified_zero=True,
        diagnostics=stats, files={path.name: sha256(path) for path in sorted(args.out.iterdir())},
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "reports", "r18-registered", "r18-lock", "r18-access", "r18-index", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    run(parser.parse_args())
