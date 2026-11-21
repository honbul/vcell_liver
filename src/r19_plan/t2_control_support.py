"""Audit fit-role physical DMSO support for diagonal shift without expression."""

import argparse
from pathlib import Path
from zlib import crc32

import numpy as np
import pandas as pd

from src.goal_tracks.t1_score import STRATUM_LINES
from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r15_lincs.data import DOSE, TIME, asset, measures
from src.r19_plan.contracts import coord_keys, id_keys
from src.r19_plan.i_preflight import provenance, read_pinned, read_r18_domains


MIN_WELLS, MIN_PLATES = 12, 3
FORMAL = "exclude_protected_score_and_donor"
STRICT = "also_exclude_all_R18_control_roles"
PHYSICAL = ["rna_plate", "rna_well"]
LIVER = {STRATUM_LINES[name][0]: name for name in ("HepG2-C3A", "SNU-423")}


def pinned_receipt(root, reference=None):
    path = root / "receipt.json"
    digest = sha256(path)
    if reference is not None and digest != sha256(reference):
        raise ValueError(f"Metadata receipt changed: {root}")
    return read_json(path), digest


def tahoe_support(stage, reference):
    stamp, digest = pinned_receipt(stage)
    if digest != read_json(reference)["stage_sha256"]:
        raise ValueError("Tahoe cache differs from the archived T2 freeze")
    pool = ["plate_id", "cell_line_id", "dose_uM", "time_h"]
    support = []
    for fold in range(3):
        folder = f"f{fold}"
        name = f"{folder}/score_rows.tsv"
        held = read_pinned(stage, name, stamp["files"][name])
        for role in "AB":
            name = f"{folder}/{role}_rows.tsv"
            controls = read_pinned(stage, name, stamp["files"][name])
            keys = ["plate_id", "cell_line_id"]
            for minimum_cells in (20, 200):
                rows = held.loc[held.dose_uM.astype(float).eq(5.) & held.n_cells.astype(int).ge(minimum_cells)]
                rows = rows.loc[rows.groupby(pool).compound_group_id.transform("nunique").ge(3)]
                for accession, line in LIVER.items():
                    expected = rows.loc[rows.cell_line_id.eq(accession), [*keys, f"control_{role}"]].drop_duplicates()
                    use = controls.merge(expected, on=keys, how="right", validate="one_to_one")
                    if not use.physical_well_id.eq(use[f"control_{role}"]).all():
                        raise ValueError("Tahoe target-fit wells do not match the locked evaluation pools")
                    wells, plates = use.physical_well_id.nunique(), use.plate_id.nunique()
                    support.append(dict(
                        fold=fold, fit_role=role, cell_line=line, cellosaurus_id=accession, dose_uM=5., hours=24,
                        treated_minimum_cells=minimum_cells, target_fit_wells=wells,
                        target_fit_plates=plates, control_cell_min=int(use.n_cells.astype(int).min()),
                        control_cell_max=int(use.n_cells.astype(int).max()),
                        passes_support=wells >= MIN_WELLS and plates >= MIN_PLATES,
                        matched_variance_scale=False,
                    ))
    return support, digest


def footprint(frames):
    all_rows = pd.concat([provenance(frame) for frame in frames], ignore_index=True)
    coordinates = coord_keys(all_rows, *PHYSICAL) | coord_keys(all_rows, "det_plate", "det_well")
    return id_keys(all_rows, "inst_id"), id_keys(all_rows, "sig_id"), coordinates


def available(rows, blocked):
    instances, signatures, coordinates = blocked
    cols = [*PHYSICAL, "det_plate", "det_well"]
    normalized = rows.reindex(columns=cols, fill_value="").apply(lambda col: col.str.strip().str.upper())
    keep = ~rows.inst_id.str.strip().str.upper().isin(instances)
    if "sig_id" in rows:
        keep &= ~rows.sig_id.str.strip().str.upper().isin(signatures)
    for plate, well in (PHYSICAL, ("det_plate", "det_well")):
        pos = normalized[well].str.replace(r"^([A-Z]+)([0-9])$", r"\g<1>0\2", regex=True)
        keep &= ~pd.MultiIndex.from_arrays([normalized[plate], pos]).isin(coordinates)
    return keep


def load_instances(path, ids, lines):
    pieces = []
    columns = {"inst_id", "cell_id", *PHYSICAL, "det_plate", "det_well", "pert_id", "pert_iname",
               "pert_type", "pert_dose", "pert_dose_unit", "pert_time", "pert_time_unit"}
    for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                             usecols=lambda col: col in columns, chunksize=200_000):
        vehicle = chunk.pert_type.eq("ctl_vehicle") & chunk.cell_id.str.upper().isin(lines)
        vehicle &= chunk.pert_iname.str.upper().eq("DMSO") | chunk.pert_id.str.upper().eq("DMSO")
        pieces.append(chunk.loc[vehicle | chunk.inst_id.isin(ids)].copy())
    rows = pd.concat(pieces, ignore_index=True)
    if rows.inst_id.duplicated().any():
        raise ValueError("Duplicate GEO constituent ID on the audit axis")
    coordinate_type = "RNA" if set(PHYSICAL) <= set(rows) else "DETECTION_ALIAS_ONLY"
    for name in [*PHYSICAL, "det_plate", "det_well"]:
        rows[name] = rows[name].str.strip().str.upper() if name in rows else ""
    for name in ("rna_well", "det_well"):
        rows[name] = rows[name].str.replace(r"^([A-Z]+)([0-9])$", r"\g<1>0\2", regex=True)
    cols = PHYSICAL if coordinate_type == "RNA" else ["det_plate", "det_well"]
    rows["audit_plate"], rows["audit_well"] = (rows[col] for col in cols)
    rows["actual_muM"] = measures(rows.pert_dose, rows.pert_dose_unit, DOSE)
    rows["actual_hours"] = measures(rows.pert_time, rows.pert_time_unit, TIME)
    rows["cell_id"] = rows.cell_id.str.strip().str.upper()
    return rows, coordinate_type


def domain_support(study, targets, controls, coordinate_type, policy):
    result = []
    qualified = controls.groupby("audit_plate").role.nunique()
    target = targets.loc[targets.audit_plate.map(qualified).eq(3)]
    for label, dose in (("5-20", target.actual_muM.between(5., 20.)),
                         ("10", np.isclose(target.actual_muM, 10., rtol=0., atol=1e-9))):
        for (cell, hours), rows in target.loc[dose].groupby(["cell_id", "actual_hours"], sort=True):
            used = controls.loc[controls.audit_plate.isin(rows.audit_plate)]
            if not used.cell_id.eq(cell).all() or not used.actual_hours.eq(hours).all():
                raise ValueError("LINCS plate mixes cell/time control contexts")
            for role in "ABC":
                fit = used.loc[used.role.eq(role)]
                wells, plates = len(fit.drop_duplicates(["audit_plate", "audit_well"])), fit.audit_plate.nunique()
                result.append(dict(
                    study=study, cell_line=cell, hours=hours, actual_dose_uM=label, exclusion_policy=policy,
                    coordinate_type=coordinate_type, treated_wells=len(rows), treated_plates=rows.audit_plate.nunique(),
                    parent_groups=rows.parent.nunique(), fit_role=role, target_fit_wells=wells, target_fit_plates=plates,
                    passes_numerical_support=wells >= MIN_WELLS and plates >= MIN_PLATES,
                    verified_physical_RNA=coordinate_type == "RNA",
                ))
    return result


def lincs_support(args):
    reserved, donors, score, donor_controls, all_registered_controls = read_r18_domains(args)
    sets = (reserved, donors, score, donor_controls)
    formal = footprint(sets)
    strict = footprint((*sets, all_registered_controls))
    results, lineage = [], {}
    for study in ("GSE92742", "GSE70138"):
        cache = args.r15 / "cache" / study
        reference = args.reports / "r15_lincs" / "data" / "cache" / study / "receipt.json"
        pin, digest = pinned_receipt(cache, reference)
        cached = read_pinned(cache, "target_rows.tsv", pin["files"]["target_rows.tsv"])
        cached = cached.loc[cached.quality.eq("True") & cached.prior_exact.eq("False")].copy()
        path = asset(args.lincs, study, "inst_info")
        raw_digest = sha256(path)
        pinned_source = read_json(args.r15 / "audit" / study / "receipt.json")["inputs"]["inst_info"]["sha256"]
        if raw_digest != pinned_source:
            raise ValueError(f"The registered instance metadata changed: {study}")
        inst, coord = load_instances(path, set(cached.inst_id), set(cached.cell_id))
        names = ["inst_id", "cell_id", *PHYSICAL, "det_plate", "det_well", "audit_plate", "audit_well",
                 "actual_muM", "actual_hours"]
        expected = cached[["inst_id", "parent", "sig_id", "cell_id", "hours", *PHYSICAL]]
        targets = expected.merge(inst[names], on=["inst_id", "cell_id"], how="left", validate="one_to_one",
                                 suffixes=("_cache", ""))
        cache_well = targets.rna_well_cache.str.upper().str.replace(r"^([A-Z]+)([0-9])$", r"\g<1>0\2", regex=True)
        if (targets.audit_plate.isna().any() or not targets.actual_hours.eq(targets.hours.astype(float)).all()
                or not targets.audit_plate.eq(targets.rna_plate_cache.str.upper()).all()
                or not targets.audit_well.eq(cache_well).all()):
            raise ValueError(f"Original GEO metadata and cached target axes differ: {study}")
        targets = targets.loc[targets.actual_muM.between(5., 20.)].copy()
        controls = inst.loc[inst.pert_type.eq("ctl_vehicle") & inst.audit_plate.isin(targets.audit_plate)
                            & inst.audit_well.str.fullmatch(r"[A-Z]+\d{2}")].copy()
        controls = controls.sort_values("inst_id").drop_duplicates(["audit_plate", "audit_well"])
        controls["crc"] = controls.inst_id.map(lambda value: crc32(value.encode()))
        controls = controls.sort_values(["audit_plate", "crc", "inst_id"])
        controls["role"] = controls.groupby("audit_plate").cumcount().mod(3).map(dict(enumerate("ABC")))
        eligible_target = targets.loc[available(targets, formal)]
        for policy, exclusion in ((FORMAL, formal), (STRICT, strict)):
            active = controls.loc[available(controls, exclusion)]
            results.extend(domain_support(study, eligible_target, active, coord, policy))
        lineage[study] = dict(cache_receipt_sha256=digest, original_instances_sha256=raw_digest,
                              input_treated_wells=len(targets), protected_treated_removed=len(targets)-len(eligible_target),
                              input_DMSO_wells=len(controls),
                              protected_DMSO_removed=int((~available(controls, formal)).sum()),
                              all_R18_roles_DMSO_removed=int((~available(controls, strict)).sum()))
    return results, lineage


def run(args):
    tahoe, tahoe_pin = tahoe_support(args.tahoe, args.reports / "goal_tracks/data/known_vc/d5/freeze.json")
    domains, lineage = lincs_support(args)
    args.out.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(tahoe).to_csv(args.out / "tahoe_fit_controls.tsv", sep="\t", index=False)
    pd.DataFrame(domains).to_csv(args.out / "lincs_domains.tsv", sep="\t", index=False)
    write_json(args.out / "receipt.json", dict(
        metadata_only=True, expression_read=False, fitting=False, scoring=False, code_sha256=sha256(__file__),
        minimum_wells=MIN_WELLS, minimum_plates=MIN_PLATES, tahoe_stage_sha256=tahoe_pin,
        r18_access_sha256=sha256(args.r18_access / "receipt.json"), lincs=lineage,
        files={path.name: sha256(path) for path in sorted(args.out.iterdir())},
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reports", "tahoe", "r15", "lincs", "r18-registered", "r18-lock", "r18-access", "r18-index", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    run(parser.parse_args())
