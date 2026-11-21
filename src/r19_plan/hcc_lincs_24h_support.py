"""Audit exact-dose adult HCC evaluation and same-assay donor support without expression."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r15_lincs.data import table
from src.r15_lincs.design import fold_id
from src.r16_lincs.audit import INST, add_units
from src.r17_lincs.inventory import ROOT, ids, in_footprint, register_footprint
from src.r19_plan.audit_r18 import prepare
from src.r19_plan.hcc_time_lincs import cmap_measured
from src.r19_plan.i_audit import structure
from src.r19_plan.i_preflight import provenance, read_r18_domains, verify_domain
from src.r19_plan.t4_scaffold_transfer import scaffold


TARGETS = ("JHH5", "JHH7")
HEPATIC = ("HEPG2", "PHH")
PHYSICAL = ["rna_plate", "rna_well"]
SCOPE = ["inst_id", "cell_id", "pert_id", *PHYSICAL, "det_plate", "det_well"]


def forbidden_signatures(path, protected):
    blocked = set(ids(protected.sig_id)) - {""}
    members = set()
    for batch in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                             usecols=["sig_id", "distil_ids"], chunksize=160_000):
        selected = batch.loc[ids(batch.sig_id).isin(blocked), "distil_ids"].str.split("|").explode()
        members.update(ids(selected))
    return members - {""}


def instance_rows(path, known):
    kept = []
    for batch in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, usecols=INST, chunksize=160_000):
        rows = batch.loc[batch.qc_pass.eq("1") & batch.cell_iname.isin(known)
                         & batch.pert_type.isin(("trt_cp", "ctl_vehicle"))]
        rows = add_units(rows)
        rows = rows.loc[rows.hours.eq(24) & (rows.pert_type.eq("ctl_vehicle") | rows.dose.between(9.999, 10.001))]
        kept.append(rows)
    rows = prepare(pd.concat(kept, ignore_index=True).rename(columns={"sample_id": "inst_id", "cell_iname": "cell_id",
                                                                                 "cmap_name": "pert_iname"}))
    rows = rows.loc[rows.rna_plate.ne("") & rows.rna_well.str.fullmatch(r"[A-Z]+\d{2}")]
    duplicates = rows.duplicated("inst_id", keep=False) | rows.duplicated(PHYSICAL, keep=False)
    invalid = int(duplicates.sum())
    rows = rows.loc[~duplicates]
    treated = rows.loc[rows.pert_type.eq("trt_cp")]
    exact = cmap_measured(treated)
    treated = treated.merge(exact.loc[exact.actual_dose_uM.eq("10")], on="inst_id", validate="one_to_one")
    controls = rows.loc[rows.pert_type.eq("ctl_vehicle") & (ids(rows.pert_id).eq("DMSO") | ids(rows.pert_iname).eq("DMSO"))]
    return treated, controls, invalid


def supported_controls(treated, controls):
    dimensions = ["cell_id", "rna_plate"]
    independent = controls.groupby(dimensions).size().rename("controls")
    result = treated.merge(independent, on=dimensions, how="left", validate="many_to_one")
    eligible = result.loc[result.controls.ge(3)].copy()
    keep = pd.MultiIndex.from_frame(controls[dimensions]).isin(pd.MultiIndex.from_frame(eligible[dimensions]))
    return eligible, controls.loc[keep].copy()


def evaluate_support(target, source, cell, name):
    grouped = source.groupby("parent")
    donor = grouped.agg(donor_wells=("inst_id", "size"), donor_cells=("cell_id", "nunique"),
                        donor_plates=("rna_plate", "nunique"))
    min_cells = 2 if name == "nonhepatic" else 1
    covered = donor.index[donor.donor_plates.ge(2) & donor.donor_cells.ge(min_cells)]
    sample = target.loc[target.parent.isin(covered)].copy()
    support, splits = [], []
    for scope, rows in (("all_exact", target), ("donor_covered", sample)):
        fold_counts = rows.groupby("fold").parent.nunique().reindex(range(3), fill_value=0)
        donor_rows = source.loc[source.parent.isin(rows.parent)]
        physical_repeat, different_repeat = set(), set()
        for parent, group in rows.groupby("parent"):
            if group.rna_plate.nunique() >= 2:
                physical_repeat.add(parent)
                if any(len(set(group.loc[group.rna_plate.ne(r.rna_plate), "rna_well"]) - {r.rna_well}) > 0
                       for r in group.itertuples()):
                    different_repeat.add(parent)
        if scope == "donor_covered":
            for (fold, plate), held in rows.groupby(["fold", "rna_plate"]):
                held_fold = rows.loc[rows.fold.eq(fold)]
                fit = rows.loc[rows.fold.ne(fold) & rows.rna_plate.ne(plate) & ~rows.scaffold.isin(held_fold.scaffold)]
                splits.append(dict(cell=cell, donor=name, fold=fold, score_plate=plate,
                    score_parents=held.parent.nunique(), score_wells=len(held), fit_parents=fit.parent.nunique(),
                    fit_wells=len(fit), fit_plates=fit.rna_plate.nunique(), held_scaffolds=held_fold.scaffold.nunique()))
        fit_min = min((r["fit_parents"] for r in splits), default=0)
        gates = rows.parent.nunique() >= 100 and fold_counts.min() >= 20 and rows.rna_plate.nunique() >= 3 and fit_min >= 30
        support.append(dict(cell=cell, donor=name, scope=scope, parents=rows.parent.nunique(),
            wells=len(rows), plates=rows.rna_plate.nunique(), fold0=fold_counts[0], fold1=fold_counts[1], fold2=fold_counts[2],
            b1_exact_wells=int(rows.b1_seen.sum()), b1_exact_parents=rows.loc[rows.b1_seen].parent.nunique(),
            different_plate_repeats=len(physical_repeat), different_position_repeats=len(different_repeat),
            donor_wells=len(donor_rows), donor_plates=donor_rows.rna_plate.nunique(), donor_cells=donor_rows.cell_id.nunique(),
            fit_parent_min=fit_min, eligible=bool(gates)))
    return support, splits, sample


def run(args):
    protected = read_r18_domains(args)
    blocked = pd.concat([provenance(rows) for rows in protected], ignore_index=True)
    registry = ROOT / "LINCS2020"
    b1 = args.base / "vcell_liver_transigen_adaptation_b1/assets"
    files = dict(samples=b1 / "raw/cmap2020/instinfo_beta.txt", signatures=registry / "siginfo_beta.txt",
                 identity=registry / "compoundinfo_beta.txt", cells=registry / "cellinfo_beta.txt",
                 b1_rows=b1 / "processed/cmap2020_v1/rows.tsv")
    b1_manifest = read_json(files["b1_rows"].parent / "build_manifest.json")
    if sha256(files["b1_rows"]) != b1_manifest["files"]["rows.tsv"]["sha256"]:
        raise ValueError("B1 row inventory changed")
    cells = table(files["cells"], usecols=["cell_iname", "cell_lineage"])
    lineage = cells.cell_lineage.str.strip().str.lower()
    nonhepatic = set(cells.loc[~lineage.isin(("", "unknown", "na")) & ~lineage.str.contains("liver|hepatic"),
                             "cell_iname"]) - set(TARGETS + HEPATIC)
    if not {"A549", "MCF7"} <= nonhepatic or set(TARGETS + HEPATIC) & nonhepatic:
        raise ValueError("The reference CMap cell lineage mapping changed")
    treated, control, ambiguous = instance_rows(files["samples"], nonhepatic | set(TARGETS + HEPATIC))
    foot = register_footprint(blocked)
    blocked_members = forbidden_signatures(files["signatures"], blocked)
    before = len(treated), len(control)
    treated = treated.loc[~in_footprint(treated, foot) & ~ids(treated.inst_id).isin(blocked_members)]
    control = control.loc[~in_footprint(control, foot) & ~ids(control.inst_id).isin(blocked_members)]
    after = len(treated), len(control)
    chem = table(files["identity"], usecols=["pert_id", "canonical_smiles"]).drop_duplicates()
    chem = chem.loc[chem.pert_id.isin(treated.pert_id)].copy()
    chem["parent"] = chem.canonical_smiles.map(structure)
    chem = chem[["pert_id", "parent"]].drop_duplicates()
    chem = chem.loc[chem.parent.ne("") & ~chem.pert_id.duplicated(keep=False)]
    treated = treated.merge(chem, on="pert_id", validate="many_to_one")
    treated, control = supported_controls(treated, control)
    targets = treated.loc[treated.cell_id.isin(TARGETS)].copy()
    seen = table(files["b1_rows"], usecols=["role", "cell", "pert_id", "det_plate"])
    seen = seen.loc[seen.role.eq("target") & seen.cell.isin(TARGETS)].rename(columns={"cell": "cell_id"})
    join = ["cell_id", "pert_id", "det_plate"]
    targets["b1_seen"] = pd.MultiIndex.from_frame(targets[join]).isin(pd.MultiIndex.from_frame(seen[join]))
    targets["fold"] = targets.parent.map(lambda p: fold_id(p, "R15-outer", 3))
    cores = {p: scaffold(p) for p in set(targets.parent)}
    targets["scaffold"] = targets.parent.map(cores)
    source = treated.loc[~treated.rna_plate.isin(targets.rna_plate) & treated.parent.isin(targets.parent)].copy()
    scopes = {"nonhepatic": source.loc[source.cell_id.isin(nonhepatic)], "hepatic": source.loc[source.cell_id.isin(HEPATIC)]}
    verify_domain(pd.concat((targets, *scopes.values()), ignore_index=True), control, *protected[:4])
    if ids(pd.concat((targets, control, *scopes.values()), ignore_index=True).inst_id).isin(blocked_members).any():
        raise ValueError("A protected signature constituent entered DEV support")
    stats, splits = [], []
    for cell in TARGETS:
        for name, donors in scopes.items():
            supports, partitions, _ = evaluate_support(targets.loc[targets.cell_id.eq(cell)], donors, cell, name)
            stats.extend(supports)
            splits.extend(partitions)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent, prefix=".hcc-24h-") as tmp:
        stage = Path(tmp)
        for name, data in (("support", stats), ("splits", splits)):
            pd.DataFrame(data).to_csv(stage / (name+".tsv"), sep="\t", index=False)
        targets.assign(study="CMap2020")[["study", *SCOPE, "actual_dose_uM", "parent", "fold", "scaffold", "b1_seen"]].to_csv(
            stage / "targets.tsv", sep="\t", index=False)
        donor_metadata = pd.concat([rows.assign(view=name) for name, rows in scopes.items()], ignore_index=True)
        donor_metadata.assign(study="CMap2020")[["study", *SCOPE, "actual_dose_uM", "parent", "view"]].to_csv(
            stage / "donors.tsv", sep="\t", index=False)
        keep_plates = set(targets.rna_plate) | set(donor_metadata.rna_plate)
        control.loc[control.rna_plate.isin(keep_plates)].assign(study="CMap2020")[["study", *SCOPE]].to_csv(
            stage / "controls.tsv", sep="\t", index=False)
        write_json(stage / "receipt.json", dict(protocol_sha256=sha256(args.protocol), code_sha256=sha256(__file__),
            metadata_only=True, matrix_opened=False, expression_read=False,
            reservation_registry_sha256=sha256(args.r18_registered / "receipt.json"),
            reservation_access_sha256=sha256(args.r18_access / "receipt.json"), registered_control_block=True,
            raw_rna_ambiguous_aliases=ambiguous, wells_before_reservation=before, wells_after_reservation=after,
            protected_signature_members=len(blocked_members),
            source_metadata_sha256={name: sha256(path) for name, path in files.items()},
            files={file.name: sha256(file) for file in stage.iterdir()}))
        stage.rename(args.out)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for field in ("base", "reports", "r18-registered", "r18-lock", "r18-access", "r18-index", "protocol", "out"):
        cli.add_argument("--"+field, type=Path, required=True)
    run(cli.parse_args())
