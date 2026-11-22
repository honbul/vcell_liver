"""Audit exact 6/24-hour adult HCC pairs without opening expression files."""

import argparse
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd

from src.r11_chemical_probe.common import sha256, write_json
from src.r15_lincs.data import STUDIES, asset, table
from src.r16_lincs.audit import INST, SIG, add_units
from src.r17_lincs.inventory import CMAP, ROOT, ids, in_footprint, normal, register_footprint
from src.r19_plan.audit_r18 import attach_parent, geo_metadata, prepare
from src.r19_plan.i_audit import structure
from src.r19_plan.i_preflight import provenance, read_r18_domains
from src.r19_plan.t4_support import TIME_PAIR, eligible_links, matched_pairs, measured_dose


CELLS = ("HUH7", "JHH5", "JHH7", "SNU423", "SNU387", "PLCPRF5")


def candidate(rows, field):
    names = normal(rows[field])
    return rows.loc[names.isin(CELLS)].assign(**{field: names.loc[names.isin(CELLS)]})


def cmap(base):
    raw = base / "vcell_liver_transigen_adaptation_b1/assets/raw/cmap2020"
    registry = ROOT / "LINCS2020"
    inst_path = raw / "instinfo_beta.txt"
    sig_path = registry / "siginfo_beta.txt"
    compound_path = registry / "compoundinfo_beta.txt"
    sig = candidate(table(sig_path, usecols=SIG), "cell_iname")
    sig = add_units(sig).rename(columns={"cell_iname": "cell_id", "distil_ids": "members"})
    sig = sig.loc[sig.pert_type.eq("trt_cp")]
    identity = table(compound_path, usecols=["pert_id", "canonical_smiles"])
    identity = identity.loc[identity.pert_id.isin(sig.pert_id)].drop_duplicates()
    identity["parent"] = identity.canonical_smiles.map(structure)
    reg = identity[["pert_id", "parent"]].drop_duplicates()
    reg = reg.loc[~reg.pert_id.duplicated(keep=False)]
    sig = attach_parent(sig, reg)
    sig["qc"] = (pd.to_numeric(sig.nsample, errors="coerce").ge(2)
                 & pd.to_numeric(sig.cc_q75, errors="coerce").ge(.2) & sig.qc_pass.eq("1"))
    inst = pd.concat([candidate(frame, "cell_iname") for frame in pd.read_csv(
        inst_path, sep="\t", dtype=str, keep_default_na=False, usecols=INST, chunksize=160_000)], ignore_index=True)
    inst = add_units(inst).rename(columns={"sample_id": "inst_id", "cell_iname": "cell_id", "cmap_name": "pert_iname"})
    inst["qc_well"] = inst.qc_pass.eq("1")
    return prepare(inst), sig, [inst_path, sig_path, compound_path], raw


def cmap_measured(inst):
    units = inst.pert_dose_unit.str.strip().str.lower().str.replace("[µμ]", "u", regex=True)
    scale = {"um": Decimal(1), "nm": Decimal(".001"), "mm": Decimal(1000)}

    def key(value, unit):
        if unit not in scale:
            return ""
        try:
            dose = Decimal(value) * scale[unit]
        except InvalidOperation:
            return ""
        return format(dose.normalize(), "f") if dose.is_finite() and dose > 0 else ""

    return inst[["inst_id"]].assign(actual_dose_uM=[key(a, b) for a, b in zip(inst.pert_dose, units, strict=True)])


def stage_rows(study, axis, stage, rows):
    pairs = matched_pairs(rows)
    for cell in CELLS:
        selected = rows.loc[rows.cell_id.eq(cell)]
        matched = pairs.loc[pairs.cell_id.eq(cell)]
        disjoint = matched.loc[matched.cross_time_plate_disjoint]
        repeated = disjoint.loc[disjoint.both_times_2plates]
        counts = repeated.groupby("fold").parent.nunique().reindex(range(3), fill_value=0)
        yield dict(study=study, axis=axis, stage=stage, cell=cell,
                   wells6=int(selected.hours.eq(6).sum()), wells24=int(selected.hours.eq(24).sum()),
                   exact_parents=matched.parent.nunique(), exact_conditions=len(matched),
                   time_disjoint_parents=disjoint.parent.nunique(), repeated_parents=repeated.parent.nunique(),
                   repeated_conditions=len(repeated), repeated_fold0=int(counts[0]), repeated_fold1=int(counts[1]),
                   repeated_fold2=int(counts[2]), support_gate=axis == "RNA" and repeated.parent.nunique() >= 100
                   and counts.min() >= 20)


def signature_support(study, sig):
    for cell in CELLS:
        group = sig.loc[sig.cell_id.eq(cell) & sig.pert_type.eq("trt_cp")]
        a, b = (group.loc[group.hours.eq(hour)] for hour in TIME_PAIR)
        matched = (set(a.parent) & set(b.parent)) - {""}
        qc = (set(a.loc[a.qc, "parent"]) & set(b.loc[b.qc, "parent"])) - {""}
        yield dict(study=study, cell=cell, signatures6=len(a), signatures24=len(b), qc_signatures6=int(a.qc.sum()),
                   qc_signatures24=int(b.qc.sum()), shared_parent_any_dose=len(matched), qc_shared_parent_any_dose=len(qc))


def plate_support(study, axis, before, inst, blocked):
    physical = register_footprint(blocked)
    blocked_sigs = set(ids(blocked.sig_id)) - {""}
    controls = inst.loc[inst.pert_type.eq("ctl_vehicle") & inst.plate.ne("")
                        & inst.well.str.fullmatch(r"[A-Z]+[0-9]{2}")
                        & (ids(inst.pert_id).eq("DMSO") | ids(inst.pert_iname).eq("DMSO"))].copy()
    controls["blocked"] = in_footprint(controls, physical)
    controls = controls.groupby(["cell_id", "plate", "well"], as_index=False).blocked.any()
    before = before.assign(blocked=in_footprint(before, physical) | ids(before.sig_id).isin(blocked_sigs))
    for (cell, plate), rows in before.groupby(["cell_id", "plate"]):
        dmsos = controls.loc[controls.cell_id.eq(cell) & controls.plate.eq(plate)]
        yield dict(study=study, axis=axis, cell=cell, plate=plate, wells=len(rows),
                   direct_treatment_or_signature_excluded=int(rows.blocked.sum()),
                   controls_before=len(dmsos), controls_after=int((~dmsos.blocked).sum()))


def run(args):
    protected_sources = read_r18_domains(args)[:4]
    blocked = pd.concat([provenance(f) for f in protected_sources], ignore_index=True)
    prior = args.base / "vcell_liver_r15_lincs/v1"
    support, registry, hashes, release, plates = [], [], {}, [], []
    for study in STUDIES:
        inst, sig = geo_metadata(study, prior)
        axis = "RNA" if {"rna_plate", "rna_well"} <= set(inst) else "DET_ONLY_UNVERIFIED_RNA"
        inst = prepare(candidate(inst, "cell_id"))
        sig = candidate(sig, "cell_id")
        registry.extend(signature_support(study, sig))
        times = inst.pert_type.eq("trt_cp") & inst.hours.isin(TIME_PAIR)
        info = asset(ROOT, study, "inst_info")
        dose = measured_dose(info, set(inst.loc[times, "inst_id"]))
        release.extend((provenance(inst), provenance(sig)))
        for stage, exclusions in (("before_reservation", blocked.iloc[:0]), ("r18_excluded", blocked)):
            rows, _ = eligible_links(inst, sig, exclusions, dose)
            support.extend(stage_rows(study, axis, stage, rows))
            if stage == "before_reservation":
                plates.extend(plate_support(study, axis, rows, inst, blocked))
        paths = [info, asset(ROOT, study, "sig_info"), asset(ROOT, study, "sig_metrics"),
                 prior / "audit" / study / "identities.tsv"]
        hashes[study] = {str(path): sha256(path) for path in paths}
    inst, sig, paths, cmap_root = cmap(args.base)
    hashes[CMAP] = {str(path): sha256(path) for path in paths}
    registry.extend(signature_support(CMAP, sig))
    values = cmap_measured(inst)
    inst = inst.loc[inst.qc_well]
    for stage, exclusions in (("before_reservation", blocked.iloc[:0]), ("r18_excluded", blocked),
                              ("geo_aliases_only_excluded", pd.concat(release, ignore_index=True)),
                              ("r18_and_geo_aliases_excluded", pd.concat((blocked, *release), ignore_index=True))):
        rows, _ = eligible_links(inst, sig, exclusions, values)
        support.extend(stage_rows(CMAP, "RNA", stage, rows))
        if stage == "before_reservation":
            plates.extend(plate_support(CMAP, "RNA", rows, inst, blocked))
    args.out.mkdir(exist_ok=False)
    pd.DataFrame(registry).to_csv(args.out / "signature_support.tsv", sep="\t", index=False)
    pd.DataFrame(support).to_csv(args.out / "physical_support.tsv", sep="\t", index=False)
    pd.DataFrame(plates).to_csv(args.out / "plate_retention.tsv", sep="\t", index=False)
    matrices = {study: sorted(str(p) for p in (ROOT / study).glob("*Level3*.gctx")) for study in STUDIES}
    matrices[CMAP] = sorted(str(p) for p in cmap_root.glob("level3_beta*.gctx"))
    write_json(args.out / "receipt.json", dict(metadata_only=True, expression_read=False, matrix_opened=False,
        protocol_sha256=sha256(args.protocol), code_sha256=sha256(__file__),
        reservation_registry_sha256=sha256(args.r18_registered / "receipt.json"),
        reservation_access_sha256=sha256(args.r18_access / "receipt.json"),
        source_metadata_sha256=hashes, level3_files=matrices,
        files={p.name: sha256(p) for p in args.out.iterdir()}))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for field in ("base", "reports", "r18-registered", "r18-lock", "r18-access", "r18-index", "protocol", "out"):
        cli.add_argument("--" + field, type=Path, required=True)
    run(cli.parse_args())
