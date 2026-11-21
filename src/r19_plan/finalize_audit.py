"""Promote verified RNA coordinates and write the R18 support-only receipt."""

import argparse
from pathlib import Path
import shutil

import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r15_lincs.data import table
from src.r15_lincs.design import fold_id
from src.r17_lincs.inventory import in_footprint, register_footprint
from src.r19_plan.contracts import assert_dev_allowed

GEO = "GSE70138"


def write(frame, path):
    frame.to_csv(path, sep="\t", index=False)


def promote_rows(exposure, crosswalk, RNA_controls):
    # In the GEO release without RNA coordinates, use unambiguous CMap crosswalks only.
    rna_plate, rna_well = "verified_rna_plate", "verified_rna_well"
    exposure[rna_plate], exposure[rna_well] = exposure.rna_plate, exposure.rna_well
    provisional = exposure.study.eq(GEO)
    for original, verified in [("rna_plate", rna_plate), ("rna_well", rna_well)]:
        exposure.loc[provisional, verified] = exposure.loc[provisional, "inst_id"].map(
            crosswalk.set_index("inst_id")[original]).fillna("")
    verified = ~provisional | exposure[rna_plate].isin(RNA_controls.rna_plate)
    verified &= exposure[rna_well].ne("")
    for scenario in "AB":
        exposure[f"r18_{scenario}_eligible"] = exposure[f"{scenario}_eligible"].eq("True") & verified
    return exposure


def scope_summary(summary, exposure):
    corrected = []
    for record in summary.itertuples(index=False):
        cut = exposure.cohort.eq(record.cohort) & exposure.study.eq(record.study)
        cut &= exposure.hours.astype(float).eq(float(record.hours))
        if record.dose_uM != "all":
            cut &= exposure.dose.astype(float).eq(float(record.dose_uM))
        part = exposure.loc[cut]
        result = record._asdict()
        for scenario in "AB":
            use = part.loc[part[f"r18_{scenario}_eligible"]].drop_duplicates(
                ["verified_rna_plate", "verified_rna_well"])
            result[f"RNA_{scenario}_parents"] = use.parent.nunique()
            result[f"RNA_{scenario}_wells"] = len(use)
            result[f"RNA_{scenario}_plates"] = use.verified_rna_plate.nunique()
        corrected.append(result)
    return pd.DataFrame(corrected)


def combined_dose(exposure):
    rows = exposure.loc[exposure.cohort.eq("C2")].copy()
    rows["dose_uM"] = rows.dose.astype(float)
    columns = ["verified_rna_plate", "verified_rna_well"]
    counts = []
    for label, part in [("all", rows), *list(rows.groupby("dose_uM", sort=True))]:
        result = {"dose_uM": label}
        for scenario in "AB":
            eligible = part.loc[part[f"r18_{scenario}_eligible"]].drop_duplicates(columns)
            result[f"{scenario}_parents"] = eligible.parent.nunique()
            result[f"{scenario}_physical_wells"] = len(eligible)
            result[f"{scenario}_evaluation_status"] = (
                "SUPPORTED_100_PARENT" if eligible.parent.nunique() >= 100 else "TECHNICAL_LT_100_PARENT"
            )
        counts.append(result)
    return pd.DataFrame(counts)


def fold_and_position_support(exposure):
    base = exposure.loc[exposure.r18_A_eligible].drop_duplicates(
        ["cohort", "study", "hours", "verified_rna_plate", "verified_rna_well"])
    folds, positions = [], []
    for key, cohort in base.groupby(["cohort", "study", "hours"]):
        for scenario in "AB":
            pool = cohort.loc[cohort[f"r18_{scenario}_eligible"]]
            outer = pool.parent.map(lambda parent: fold_id(parent, "R15-outer", 3))
            for fold in range(3):
                training = pool.loc[outer.ne(fold)]
                score = pool.loc[outer.eq(fold)]
                select = training.parent.map(lambda parent: fold_id(parent, f"R15-inner-{fold}", 5)).eq(0)
                for split, rows in [("inner_fit", training.loc[~select]), ("select", training.loc[select]),
                                    ("score", score)]:
                    folds.append((*key, scenario, fold, split, rows.parent.nunique(), len(rows)))
                donors = training.groupby("verified_rna_well").verified_rna_plate.agg(set)
                support = sum(bool(donors.get(row.verified_rna_well, set()) - {row.verified_rna_plate})
                              for row in score.itertuples(index=False))
                positions.append((*key, scenario, fold, len(score), support))
    fields = ["cohort", "study", "hours", "scenario", "fold"]
    return (pd.DataFrame(folds, columns=[*fields, "split", "parents", "physical_wells"]),
            pd.DataFrame(positions, columns=[*fields, "query_wells", "same_coordinate_other_plate_fit_support"]))



def main(args):
    args.out.mkdir(parents=True, exist_ok=False)
    audit = read_json(args.audit / "receipt.json")
    for name, expected in audit["files"].items():
        if sha256(args.audit / name) != expected:
            raise ValueError(f"Changed support artifact: {name}")

    exposure = table(args.audit / "exposure_manifest.tsv")
    c3 = read_json(args.c3_fit / "receipt.json")
    for name, digest in c3["files"].items():
        if sha256(args.c3_fit / name) != digest:
            raise ValueError(f"Changed C3 fit support artifact: {name}")
    donors_c3 = table(args.c3_fit / "c3_B_hepg2_held_donor_support.tsv")
    c3_scope = exposure.cohort.eq("C3")
    eligible_c3 = pd.MultiIndex.from_arrays([exposure.hours.astype(float), exposure.parent]).isin(
        pd.MultiIndex.from_arrays([donors_c3.hours.astype(float), donors_c3.parent]))
    exposure["query_panel_only_B_eligible"] = exposure.B_eligible
    exposure.loc[c3_scope, "B_eligible"] = (
        exposure.A_eligible.eq("True") & eligible_c3
    ).loc[c3_scope].map({True: "True", False: "False"})
    rna_path = args.crosswalk / "rna_crosswalk.tsv"
    RNA_roles_path = args.crosswalk / "rna_assignments.tsv"
    RNA_roles = table(RNA_roles_path)
    wells = table(rna_path)
    checks = table(args.crosswalk / "rna_support.tsv")
    exposure = promote_rows(exposure, wells, RNA_roles)
    support = scope_summary(table(args.audit / "support.tsv"), exposure)
    summary = support.loc[support.study.eq(GEO), ["dose_uM", "RNA_A_parents", "RNA_A_wells", "RNA_A_plates"]]
    paired = summary.merge(checks, on="dose_uM", validate="one_to_one")
    for a, b in [("RNA_A_parents", "RNA_3role_parent_count"), ("RNA_A_wells", "RNA_3role_well_count"),
                 ("RNA_A_plates", "RNA_3role_plate_count")]:
        if not paired[a].eq(paired[b].astype(int)).all():
            raise ValueError("RNA support did not reconcile to independent crosswalk")
    write(support, args.out / "support.tsv")
    write(combined_dose(exposure), args.out / "c2_dose_support.tsv")
    write(exposure, args.out / "exposure_manifest.tsv")
    folds, positions = fold_and_position_support(exposure)
    write(folds, args.out / "r18_fold_support.tsv")
    write(positions, args.out / "r18_position_support.tsv")

    controls = table(args.audit / "r18_control_assignments.tsv")
    revised_roles = RNA_roles.assign(study=GEO, physical_axis="rna",
                                    plate=RNA_roles.rna_plate, well=RNA_roles.rna_well)
    control_registry = pd.concat([controls.loc[~controls.study.eq(GEO)], revised_roles[controls.columns]],
                                 ignore_index=True)
    if control_registry.duplicated(["rna_plate", "rna_well"]).any():
        raise ValueError("Multiple control assignments share one physical RNA well")
    per_role = control_registry.groupby(["study", "rna_plate", "role"]).size().unstack().reset_index()
    if per_role[list("ABC")].isna().any().any():
        raise ValueError("A retained RNA plate lacks a distinct DMSO role")
    write(control_registry, args.out / "r18_control_assignments.tsv")
    write(per_role, args.out / "r18_control_support.tsv")

    reserve_path = args.audit / "r18_reserved_manifest.tsv"
    donor_path = args.audit / "r18_B_donor_exceptions.tsv"
    reserve, donors = table(reserve_path), table(donor_path)
    donors = pd.concat([donors.loc[~donors.cohort.eq("C3")],
                        donors_c3.assign(record_type="B_nonhepatic_input_exception")], ignore_index=True).fillna("")
    if in_footprint(donors, register_footprint(reserve)).any():
        raise ValueError("A permitted B donor overlaps a reserved treated target")
    write(donors, args.out / donor_path.name)
    if reserve.DEV_fit_select_or_pretrain_allowed.ne("False").any():
        raise ValueError("Missing DEV exclusion in reserved manifest")
    if donors.CONFIRM_R18_B_source_donor_allowed.ne("True").any():
        raise ValueError("Missing B contract flag in donor manifest")
    eligible_wells = exposure.loc[exposure.r18_A_eligible, "inst_id"]
    if not set(eligible_wells).issubset(reserve.inst_id):
        raise ValueError("Eligible wells missing from reservation")
    try:
        assert_dev_allowed(exposure.loc[exposure.r18_A_eligible], reserve, donors)
    except ValueError as error:
        if "Reserved R18 assay input in DEV" not in str(error):
            raise
    else:
        raise ValueError("DEV gate accepted a reserved input")
    for name in ("r18_reserved_manifest.tsv", "c3_transfer_support.tsv"):
        shutil.copyfile(args.audit / name, args.out / name)
    shutil.copyfile(args.audit / "c3_zero_shot_fold_support.tsv", args.out / "c3_query_panel_only_fold_support.tsv")
    for path in (rna_path, RNA_roles_path, args.crosswalk / "rna_support.tsv"):
        shutil.copyfile(path, args.out / path.name)
    for name in c3["files"]:
        shutil.copyfile(args.c3_fit / name, args.out / name)
    write_json(args.out / "receipt.json", {
        "audit_sha256": sha256(args.audit / "receipt.json"), "finalizer_sha256": sha256(__file__),
        "RNA_crosswalk_input": {path.name: sha256(path) for path in (rna_path, RNA_roles_path)},
        "C3_fit_support_sha256": sha256(args.c3_fit / "receipt.json"),
        "physical_units": "RNA plate and well, no DMSO rotations counted as additional wells",
        "reserved_identifiers": {"manifest_rows": len(reserve), "B_donor_exception_rows": len(donors)},
        "R18_treated_expression_matrices_read": False, "R18_scored": False, "DEV_fit": False,
        "files": {p.name: sha256(p) for p in sorted(args.out.iterdir())},
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("audit", "crosswalk", "c3-fit", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    main(parser.parse_args())
