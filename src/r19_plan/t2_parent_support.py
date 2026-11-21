"""Count source parent support under physical-domain and parent-fold exclusion."""

import argparse
from pathlib import Path
from zlib import crc32

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r15_lincs.data import asset
from src.r15_lincs.design import fold_id
from src.r19_plan.i_preflight import read_pinned, read_r18_domains
from src.r19_plan.t2_control_support import (
    FORMAL, STRICT, available, footprint, load_instances, pinned_receipt,
)

STUDY = "GSE92742"
DOMAIN = ["cell_id", "actual_hours"]


def domains(args, previous):
    control_pin, control_hash = pinned_receipt(args.control_support)
    if control_pin["r18_access_sha256"] != sha256(args.r18_access / "receipt.json"):
        raise ValueError("T2 parent support differs from the first protected R18 input lock")
    ref = args.reports / "r15_lincs" / "data" / "cache" / STUDY / "receipt.json"
    cache = args.r15 / "cache" / STUDY
    pin, digest = pinned_receipt(cache, ref)
    if digest != control_pin["lincs"][STUDY]["cache_receipt_sha256"]:
        raise ValueError("T2 parent support uses a different quality row cache")
    targets = read_pinned(cache, "target_rows.tsv", pin["files"]["target_rows.tsv"])
    targets = targets.loc[targets.quality.eq("True") & targets.prior_exact.eq("False")]
    source = asset(args.lincs, STUDY, "inst_info")
    if sha256(source) != control_pin["lincs"][STUDY]["original_instances_sha256"]:
        raise ValueError("T2 parent support uses different original GEO metadata")
    inst, kind = load_instances(source, set(targets.inst_id), set(targets.cell_id))
    if kind != "RNA":
        raise ValueError("Parent-domain eligibility must use original physical RNA coordinates")
    names = ["inst_id", "cell_id", "rna_plate", "rna_well", "det_plate", "det_well",
             "audit_plate", "audit_well", "actual_muM", "actual_hours"]
    targets = targets[["inst_id", "parent", "sig_id", "cell_id"]].merge(
        inst[names], on=["inst_id", "cell_id"], how="left", validate="one_to_one",
    )
    if targets.audit_plate.isna().any():
        raise ValueError("A T2 candidate did not match its physical GEO well metadata")
    targets = targets.loc[np.isclose(targets.actual_muM, 10., rtol=0., atol=1e-9)].copy()
    controls = inst.loc[inst.pert_type.eq("ctl_vehicle") & inst.audit_plate.isin(targets.audit_plate)
                        & inst.audit_well.str.fullmatch(r"[A-Z]+\d{2}")].copy()
    controls = controls.sort_values("inst_id").drop_duplicates(["audit_plate", "audit_well"])
    controls["crc"] = controls.inst_id.map(lambda value: crc32(value.encode()))
    controls = controls.sort_values(["audit_plate", "crc", "inst_id"])
    controls["role"] = controls.groupby("audit_plate").cumcount().mod(3).map(dict(enumerate("ABC")))
    reserved, donors, score, donor_controls, all_controls = read_r18_domains(args)
    held = (reserved, donors, score, donor_controls)
    targets = targets.loc[available(targets, footprint(held))].copy()
    targets["fold"] = targets.parent.map(lambda value: fold_id(value, "R15-outer", 3))
    scopes = {}
    for policy, guard in ((FORMAL, held), (STRICT, (*held, all_controls))):
        selected = controls.loc[available(controls, footprint(guard))]
        actual = targets.loc[targets.audit_plate.map(selected.groupby("audit_plate").role.nunique()).eq(3)]
        split = dict(tuple(actual.groupby(DOMAIN, sort=True)))
        expected = previous.loc[previous.exclusion_policy.eq(policy)]
        expected_pairs = {(row.cell_line, float(row.hours)) for row in expected.itertuples(index=False)}
        if set(split) != expected_pairs:
            raise ValueError(f"Physical 10 µM domains diverge from T2 control audit: {policy}")
        for (line, hours), rows in split.items():
            members = selected.loc[selected.audit_plate.isin(rows.audit_plate)]
            check = expected.loc[expected.cell_line.eq(line) & expected.hours.astype(float).eq(hours)]
            for row in check.itertuples(index=False):
                givens = (int(row.parent_groups), int(row.treated_wells), int(row.treated_plates),
                          int(row.target_fit_wells))
                actual_counts = (rows.parent.nunique(), len(rows), rows.audit_plate.nunique(),
                                 int(members.role.eq(row.fit_role).sum()))
                if actual_counts != givens:
                    raise ValueError("10 µM parent-domain row or role support differs from T2 control audit")
        scopes[policy] = split
    return scopes, control_hash


def fit_support(scopes):
    result = []
    for policy, split in scopes.items():
        for (cell, hours), query_domain in split.items():
            for fold in ("full", 0, 1, 2):
                query = query_domain if fold == "full" else query_domain.loc[query_domain.fold.eq(fold)]
                blocked = set(query.parent)
                for (source_cell, source_hours), source in split.items():
                    if (source_cell, source_hours) == (cell, hours):
                        continue
                    keep = ~source.parent.isin(blocked) if fold == "full" else ~source.fold.eq(fold)
                    fit = source.loc[keep]
                    if set(fit.parent) & blocked or set(fit.audit_plate) & set(query.audit_plate):
                        raise ValueError("The T2 pseudo-target split shares query parent or plate")
                    result.append(dict(
                        study=STUDY, dose_uM=10., exclusion_policy=policy, held_cell=cell,
                        held_hours=hours, parent_fold=fold, query_parents=len(blocked),
                        query_treated_wells=len(query), query_plates=query.audit_plate.nunique(),
                        source_cell=source_cell, source_hours=source_hours,
                        source_parents_before=source.parent.nunique(), fit_parents=fit.parent.nunique(),
                        fit_treated_wells=len(fit), fit_plates=fit.audit_plate.nunique(),
                    ))
    return result


def run(args):
    control_pin, _ = pinned_receipt(args.control_support)
    previous = read_pinned(args.control_support, "lincs_domains.tsv", control_pin["files"]["lincs_domains.tsv"])
    previous = previous.loc[previous.study.eq(STUDY) & previous.actual_dose_uM.eq("10")
                            & previous.verified_physical_RNA.eq("True")]
    scope, previous_hash = domains(args, previous)
    args.out.mkdir(parents=True, exist_ok=False)
    file = args.out / "leave_domain_out.tsv"
    pd.DataFrame(fit_support(scope)).to_csv(file, sep="\t", index=False)
    write_json(args.out / "receipt.json", dict(
        metadata_only=True, expression_read=False, fitting=False, scoring=False,
        control_audit_sha256=previous_hash, code_sha256=sha256(__file__),
        r18_access_sha256=sha256(args.r18_access / "receipt.json"),
        full_policy="exclude every observed held-domain parent globally",
        fold_policy="exclude entire fixed R15 parent fold globally",
        files={file.name: sha256(file)},
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reports", "r15", "lincs", "control-support", "r18-registered", "r18-lock", "r18-access",
                 "r18-index", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    run(parser.parse_args())
