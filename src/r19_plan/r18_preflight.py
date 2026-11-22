"""Bind the audited R18 row and DMSO axes to the existing R17 splits using metadata only."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from zlib import crc32

import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r15_lincs.data import table
from src.r15_lincs.design import fold_id
from src.r17_lincs.inventory import in_footprint, register_footprint
from src.r19_plan.audit_r18 import PHYSICAL, combine, geo_metadata, prepare, roles

ROLES = "ABC"
GEO = "GSE92742"
PHYSICAL_RNA = ["rna_plate", "rna_well"]
RESULT_FIELDS = ["study", "inst_id", "sig_id", "parent", "cell_id", "hours", "dose", "dose_well", *PHYSICAL_RNA]


@dataclass(frozen=True)
class Case:
    name: str
    scenario: str
    training: pd.DataFrame
    query: pd.DataFrame


def checked_tables(root, reference):
    receipt = root / "receipt.json"
    if sha256(receipt) != sha256(reference):
        raise ValueError("R18 registered receipt differs from the pinned copy")
    fields = read_json(receipt)
    for name, digest in fields["files"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"Changed R18 metadata artifact: {name}")
    return fields


def check_controls(controls):
    if controls.duplicated(PHYSICAL_RNA).any() or controls.inst_id.duplicated().any():
        raise ValueError("Multiple assignments for one physical DMSO well")
    if controls[PHYSICAL_RNA].eq("").any().any() or controls.inst_id.eq("").any():
        raise ValueError("A registered DMSO well is missing its physical coordinates or ID")
    sorted_controls = controls.assign(crc=controls.inst_id.map(lambda value: crc32(value.encode())))
    sorted_controls = sorted_controls.sort_values(["study", "rna_plate", "crc", "inst_id"])
    grouped = sorted_controls.groupby(["study", "rna_plate"])
    expected = grouped.cumcount().mod(3).map(dict(enumerate(ROLES)))
    if not sorted_controls.role.eq(expected).all() or grouped.role.nunique().ne(3).any():
        raise ValueError("The physical plate does not reproduce the R17 CRC32 role assignment")


def target_rows(exposure):
    rows = exposure.loc[exposure.r18_A_eligible.eq("True")].copy()
    rows[PHYSICAL_RNA] = rows[["verified_rna_plate", "verified_rna_well"]].to_numpy()
    if rows.duplicated(PHYSICAL_RNA).any() or rows.duplicated(["study", "inst_id"]).any():
        raise ValueError("One physical confirmatory well has multiple eligible rows")
    if rows[["parent", "sig_id", "inst_id", *PHYSICAL_RNA]].eq("").any().any():
        raise ValueError("R18 eligible rows lack canonical parent or well provenance")
    for key in ("hours", "dose", "dose_well"):
        rows[key] = pd.to_numeric(rows[key], errors="raise")
    conditions = {
        "C1": rows.study.eq("CMap2020") & rows.cell_id.eq("HEPG2") & rows.hours.eq(24) & rows.dose.between(5, 20)
            & rows.R16.eq("True") & rows.geo_duplicate.eq("False"),
        "C2": rows.study.isin(["CMap2020", "GSE70138"]) & rows.cell_id.eq("HEPG2") & rows.hours.eq(24)
            & rows.dose.between(.04, 3.33) & rows.all_prior_exact.eq("False") & rows.geo_duplicate.eq("False"),
        "C3": rows.study.eq(GEO) & rows.cell_id.eq("HUH7") & rows.hours.isin([6, 24]) & rows.dose.eq(3)
            & rows.all_prior_exact.eq("False"),
    }
    if (set(rows.cohort) != set(conditions) or not rows[["qc", "qc_well"]].eq("True").all().all()
        or any(not condition.loc[rows.cohort.eq(name)].all() for name, condition in conditions.items())):
        raise ValueError("An eligible R18 row violates the registered cohort or QC definition")
    return rows


def c3_metadata(base, target, reserved, donors):
    raw, sig = geo_metadata(GEO, base / "vcell_liver_r15_lincs/v1")
    inst = prepare(raw)
    controls, sizes = roles(inst, GEO, geo=None)
    train = sig.loc[sig.pert_type.eq("trt_cp") & sig.cell_id.eq("HEPG2") & sig.hours.isin([6, 24])
                    & sig.dose.eq(3) & sig.qc & sig.parent.ne("")]
    train = combine(train, inst, exact_dose=False)
    bands = target.loc[target.cohort.eq("C3")].groupby("hours").dose_well.agg(["min", "max"])
    train = train.join(bands, on="hours")
    train = train.loc[train.dose_well.between(train["min"], train["max"]) & train.plate.map(sizes).ge(3)]
    if train.groupby(PHYSICAL).parent.nunique().gt(1).any():
        raise ValueError("A C3 HEPG2 physical well has conflicting canonical parents")
    train = train.drop_duplicates(PHYSICAL).assign(study=GEO)
    if in_footprint(train, register_footprint(reserved)).any():
        raise ValueError("C3 HEPG2 fitting support overlaps a reserved C1–C3 target well")
    donor_keys = pd.MultiIndex.from_frame(donors.loc[donors.cohort.eq("C3"), ["hours", "parent"]].drop_duplicates())
    train["r18_B_eligible"] = pd.MultiIndex.from_frame(train[["hours", "parent"]]).isin(donor_keys)
    controls = controls.loc[controls.rna_plate.isin(train.rna_plate)].assign(study=GEO)
    return train, controls


def split(case, fold):
    train = case.training
    query = case.query
    outer = train.loc[train.parent.map(lambda parent: fold_id(parent, "R15-outer", 3)).ne(fold)]
    score = query.loc[query.parent.map(lambda parent: fold_id(parent, "R15-outer", 3)).eq(fold)]
    mask = outer.parent.map(lambda parent: fold_id(parent, f"R15-inner-{fold}", 5)).eq(0)
    inner = outer.loc[~mask]
    select = outer.loc[mask]
    if not len(inner) or not len(select) or not len(score):
        raise ValueError(f"Empty fit/select/score split for {case.name} {case.scenario} {fold}")
    if set(outer.parent) & set(score.parent) or set(inner.parent) & set(select.parent):
        raise ValueError("An outer or inner held parent crossed release, dose or line")
    return {"inner_fit": inner, "select": select, "outer_fit": outer, "score": score}


def cases(target, train):
    for scenario in "AB":
        eligible = target.loc[target[f"r18_{scenario}_eligible"].eq("True")]
        c3_train = train if scenario == "A" else train.loc[train.r18_B_eligible]
        for cohort in ("C1", "C2"):
            cohort_rows = eligible.loc[eligible.cohort.eq(cohort)]
            yield Case(cohort, scenario, cohort_rows, cohort_rows)
        for hours in (6, 24):
            hepg2 = c3_train.loc[c3_train.hours.eq(hours)]
            huh7 = eligible.loc[eligible.cohort.eq("C3") & eligible.hours.eq(hours)]
            yield Case(f"C3_{hours}h_HEPG2_only", scenario, hepg2, huh7)
            yield Case(f"C3_{hours}h_HUH7_added", scenario, pd.concat([hepg2, huh7]), huh7)


def verify_donors(target, reserved, donors):
    if not donors.CONFIRM_R18_B_source_donor_allowed.eq("True").all():
        raise ValueError("A donor is missing its B input-only authorization")
    if in_footprint(donors, register_footprint(reserved)).any():
        raise ValueError("A B donor overlaps a reserved target physical measurement")
    expected_cells = set(target.cell_id) | {"PHH", "JHH5", "JHH7", "HEP3B", "HEPARG"}
    allowed = {
        "C1": donors.study.eq("CMap2020") & donors.hours.eq(24) & donors.dose.between(5, 20),
        "C2": donors.study.isin(["CMap2020", "GSE70138"]) & donors.hours.eq(24) & donors.dose.between(.04, 3.33),
        "C3": donors.study.eq(GEO) & donors.hours.isin([6, 24]),
    }
    if (donors.cell_id.str.upper().isin(expected_cells).any()
        or any(not valid.loc[donors.cohort.eq(name)].all() for name, valid in allowed.items())):
        raise ValueError("A B source donor violates its registered cell, time or dose scope")
    c3 = donors.loc[donors.cohort.eq("C3")]
    bands = target.loc[target.cohort.eq("C3")].groupby("hours").dose_well.agg(["min", "max"])
    measured = pd.to_numeric(c3.dose_well, errors="raise")
    if not measured.between(c3.hours.map(bands["min"]), c3.hours.map(bands["max"])).all():
        raise ValueError("A C3 donor falls outside the observed actual-dose range")
    keys = ["cohort", "study", "hours", "parent"]
    expected = pd.MultiIndex.from_frame(target[keys]).isin(pd.MultiIndex.from_frame(donors[keys].drop_duplicates()))
    if not target.r18_B_eligible.eq("True").eq(expected).all():
        raise ValueError("B target support differs from the donor registry")


def protected_score_axis(rows, controls):
    key = ["study", "rna_plate"]
    controls_at = pd.MultiIndex.from_frame(controls[key].drop_duplicates())
    if not pd.MultiIndex.from_frame(rows[key]).isin(controls_at).all():
        raise ValueError("An eligible well lacks registered DMSO roles")
    if pd.MultiIndex.from_frame(rows[PHYSICAL_RNA]).isin(pd.MultiIndex.from_frame(controls[PHYSICAL_RNA])).any():
        raise ValueError("A treated well also appears as a physical DMSO control")


def run(registered, reference, base, out):
    checked_tables(registered, reference)
    target = target_rows(table(registered / "exposure_manifest.tsv"))
    reserved = table(registered / "r18_reserved_manifest.tsv")
    donors = table(registered / "r18_B_donor_exceptions.tsv")
    donors[["hours", "dose"]] = donors[["hours", "dose"]].apply(pd.to_numeric)
    verify_donors(target, reserved, donors)

    controls = table(registered / "r18_control_assignments.tsv")
    hepg2, extra_controls = c3_metadata(base, target, reserved, donors)
    controls = pd.concat([controls, extra_controls[controls.columns]], ignore_index=True)
    check_controls(controls)
    protected_score_axis(pd.concat([target, hepg2]), controls)

    case_support, row_blocks, rotations = [], [], []
    observed_targets = table(registered / "support.tsv").query("dose_uM == 'all'")
    for record in observed_targets.itertuples(index=False):
        group = target.loc[target.cohort.eq(record.cohort) & target.study.eq(record.study)
                           & target.hours.eq(float(record.hours))]
        for scenario in "AB":
            rows = group.loc[group[f"r18_{scenario}_eligible"].eq("True")]
            if len(rows) != int(getattr(record, f"RNA_{scenario}_wells")) or (
                rows.parent.nunique() != int(getattr(record, f"RNA_{scenario}_parents"))
            ):
                raise ValueError("Registered target support changed at preflight")
    known_c3 = table(registered / "c3_hepg2_fold_support.tsv")
    for case in cases(target, hepg2):
        for fold in range(3):
            groups = split(case, fold)
            if case.name.endswith("HEPG2_only"):
                hours = str(float(case.query.hours.iloc[0]))
                logged = known_c3.loc[known_c3.hours.eq(hours) & known_c3.design.eq("nominal_3uM")
                                     & known_c3.scenario.eq(case.scenario) & known_c3.fold.eq(str(fold))]
                for split_name, phase in (("inner_fit", "HEPG2_inner_fit"), ("select", "HEPG2_select"), ("score", "HUH7_score")):
                    entry, = logged.loc[logged.phase.eq(phase)].itertuples(index=False)
                    rows = groups[split_name]
                    if len(rows) != int(entry.wells) or rows.parent.nunique() != int(entry.parents):
                        raise ValueError("C3 HEPG2-only fold differs from the audited nominal-dose design")
            for phase, rows in groups.items():
                keys = dict(case=case.name, scenario=case.scenario, fold=fold, phase=phase)
                case_support.append(dict(keys, parents=rows.parent.nunique(), physical_wells=len(rows),
                                         plates=rows.rna_plate.nunique()))
                row_blocks.append(rows[RESULT_FIELDS].assign(**keys))
            for i, fit in enumerate(ROLES):
                rotations.append(dict(case=case.name, scenario=case.scenario, fold=fold,
                                      fit=fit, select=ROLES[(i+1) % 3], score=ROLES[(i+2) % 3]))

    out.mkdir(exist_ok=False, parents=True)
    pd.DataFrame(case_support).to_csv(out / "support.tsv", sep="\t", index=False)
    pd.concat(row_blocks, ignore_index=True).to_csv(out / "split_rows.tsv", sep="\t", index=False)
    pd.DataFrame(rotations).to_csv(out / "rotations.tsv", sep="\t", index=False)
    controls.to_csv(out / "control_assignments.tsv", sep="\t", index=False)
    write_json(out / "receipt.json", dict(registered_support_sha256=sha256(reference), code_sha256=sha256(__file__),
        metadata_only=True, treatment_or_control_expression_read=False, fit=False, score=False, rotation_folds=len(rotations),
        fingerprint=dict(cohort_cases=len(case_support) // 12, unique_target_wells=len(target), distinct_DMSO=len(controls)),
        files={path.name: sha256(path) for path in sorted(out.iterdir())}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("registered", "reference", "base", "out"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    run(**vars(parser.parse_args()))
