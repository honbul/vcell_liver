"""Lock the X0–X2 held-cell and held-parent experiment on physical LINCS wells."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from zlib import crc32

import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r15_lincs.data import asset
from src.r15_lincs.design import fold_id
from src.r17_lincs.inventory import ROOT, ids, in_footprint, register_footprint
from src.r19_plan.audit_r18 import geo_metadata, prepare
from src.r19_plan.i_preflight import provenance, read_r18_domains, verify_domain, write_plan
from src.r19_plan.i_stage import parse_landmarks, parse_tsv, pinned_bytes, slice_roles
from src.r19_plan.r18_matrix_index import GENE_IDS, SAMPLE_IDS, metadata_index
from src.r19_plan.t4_support import eligible_links, measured_dose
from src.r19_plan.x_context_support import DOSE, HOURS, STUDY, inventory

PHYSICAL = ["rna_plate", "rna_well"]
PREDICT = ("labels", "query", "fit_controls", "select_controls")
SCORE = ("labels", "controls")


def control_registry(inst, treatments, protected):
    keep = (inst.pert_type.eq("ctl_vehicle") & (ids(inst.pert_id).eq("DMSO") | ids(inst.pert_iname).eq("DMSO"))
            & inst.rna_plate.isin(treatments.rna_plate) & inst.rna_well.str.fullmatch("[A-Z]+[0-9]{2}"))
    possible = inst.loc[keep]
    rows = possible.loc[~in_footprint(possible, register_footprint(protected)),
                        ["inst_id", "cell_id", *PHYSICAL]].copy()
    # A repeated physical coordinate must refer to one cell line. Retain its canonical instance ID.
    if rows.groupby(PHYSICAL).cell_id.nunique().gt(1).any():
        raise ValueError("An X vehicle physical well maps to multiple cell lines")
    rows = rows.sort_values("inst_id").drop_duplicates(PHYSICAL)
    if rows.inst_id.duplicated().any() or (rows.groupby("rna_plate").size() < 3).any():
        raise ValueError("X requires three independent DMSOs per plate")
    if not set(rows.rna_plate) == set(treatments.rna_plate):
        raise ValueError("An X treatment plate lacks a usable DMSO")
    by_plate = treatments.groupby("rna_plate").cell_id.first()
    if not rows.cell_id.eq(rows.rna_plate.map(by_plate)).all():
        raise ValueError("X DMSO and treatment wells differ in cell context")
    rows = rows.assign(crc=rows.inst_id.map(lambda s: crc32(s.encode()))).sort_values(["rna_plate", "crc", "inst_id"])
    roles = rows.groupby("rna_plate").cumcount().mod(3).map(dict(enumerate("ABC")))
    return rows.drop(columns="crc").assign(study=STUDY, role=roles).reset_index(drop=True)


def intersect_contexts(rows, cell_order, expected):
    exact, _, parent_sets, _ = inventory(rows)
    if not set(cell_order) <= set(parent_sets) or len(cell_order) != 5:
        raise ValueError("The five registered X contexts are absent")
    shared = set.intersection(*(parent_sets[cell] for cell in cell_order))
    if len(shared) != expected:
        raise ValueError("The X2 matched parent cohort changed since its registered basal audit")
    keep = exact.loc[exact.cell_id.isin(cell_order) & exact.parent.isin(shared)].copy()
    keep = keep[["inst_id", "sig_id", "parent", "pert_id", "cell_id", *PHYSICAL, "det_plate", "det_well"]]
    return keep.assign(study=STUDY, hours=HOURS, dose=DOSE,
                       parent_fold=keep.parent.map(lambda p: fold_id(p, "R15-outer", 3)))


def split_scope(treatments, controls, cell_order, target, fold, fit_role):
    if target not in cell_order:
        raise ValueError("X query cell outside the registered context list")
    _, _, select_role, score_role = slice_roles(f"f{fold}-{fit_role}")
    validation_cell = cell_order[(cell_order.index(target) + 1) % len(cell_order)]
    source = treatments.loc[treatments.cell_id.ne(target) & treatments.parent_fold.ne(fold)].copy()
    held = treatments.loc[treatments.cell_id.eq(target) & treatments.parent_fold.eq(fold)]
    validation_parents = set(source.loc[source.parent.map(lambda p: fold_id(p, f"R15-inner-{fold}", 5) == 0), "parent"])
    source["inner_phase"] = "outer_only"
    source.loc[source.parent.isin(validation_parents) & source.cell_id.eq(validation_cell), "inner_phase"] = "select"
    source.loc[~source.parent.isin(validation_parents) & source.cell_id.ne(validation_cell), "inner_phase"] = "inner_fit"
    validation = source.loc[source.inner_phase.eq("select")]
    train = source.loc[source.inner_phase.eq("inner_fit")]
    if (source.cell_id.nunique() != 4 or train.cell_id.nunique() != 3 or validation.cell_id.nunique() != 1
            or held.cell_id.nunique() != 1 or min(train.parent.nunique(), validation.parent.nunique(), held.parent.nunique()) < 20
            or set(train.parent) & set(validation.parent) or set(source.parent) & set(held.parent)
            or set(source.rna_plate) & set(held.rna_plate)):
        raise ValueError("X lost independently held cell, plate or parent support")

    def select_controls(rows, role):
        active = set(rows.rna_plate)
        keep = controls.loc[controls.role.eq(role) & controls.rna_plate.isin(active)]
        if set(keep.rna_plate) != active:
            raise ValueError(f"X plate lacks registered DMSO role {role}")
        return keep

    fit = select_controls(pd.concat((source, held), ignore_index=True), fit_role)
    select = select_controls(validation, select_role)
    score = select_controls(held, score_role)
    if pd.concat((source, held, fit, select, score)).duplicated(PHYSICAL).any():
        raise ValueError("A physical plate well crosses the X treatment/control roles")
    split = {
        "predict": {"labels": source, "query": held.drop(columns="level3_row_index"),
                    "fit_controls": fit, "select_controls": select},
        "score": {"labels": held, "controls": score},
    }
    summary = dict(target=target, validation_cell=validation_cell, fold=fold, role=fit_role,
                   training_contexts=train.cell_id.nunique(), outer_contexts=source.cell_id.nunique(),
                   inner_fit_parents=train.parent.nunique(), inner_select_parents=validation.parent.nunique(),
                   query_parents=held.parent.nunique(), query_wells=len(held), query_plates=held.rna_plate.nunique(),
                   fit_controls=len(fit), select_controls=len(select), score_controls=len(score))
    return split, summary


def run(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    basal = read_json(args.basal_reference)
    context = read_json(args.context_reference)
    if (basal["context_support_sha256"] != sha256(args.context_reference) or not basal["metadata_only"]
            or basal["expression_values_loaded"] or context["reservation_sha256"] != sha256(args.r18_access / "receipt.json")):
        raise ValueError("X basal, matched contexts and protected-access metadata have different lineages")
    profile = parse_tsv(pinned_bytes(args.basal_profiles, basal["files"]["context_profiles.tsv"]))
    cells = basal["selected_contexts"]
    if not set(cells) <= set(profile.loc[profile.eligible.eq("True"), "cell"]):
        raise ValueError("An X context lacks its single registered basal profile")
    protected = read_r18_domains(args)[:4]
    footprint = pd.concat([provenance(frame) for frame in protected], ignore_index=True)
    sources = context["source_metadata_sha256"]
    if any(sha256(path) != digest for path, digest in sources.items()):
        raise ValueError("The eligible raw X metadata have changed")
    inst, sig = geo_metadata(STUDY, args.base / "vcell_liver_r15_lincs/v1")
    inst = prepare(inst)
    raw = asset(ROOT, STUDY, "inst_info")
    doses = measured_dose(raw, set(inst.loc[inst.pert_type.eq("trt_cp") & inst.hours.eq(HOURS), "inst_id"]))
    eligible, _ = eligible_links(inst, sig.loc[sig.hours.eq(HOURS)], footprint, doses)
    treatments = intersect_contexts(eligible.loc[eligible.actual_dose_uM.eq(DOSE)], cells, basal["common_parents"])
    controls = control_registry(inst, treatments, footprint)
    verify_domain(treatments, controls, *protected)
    matrix = read_json(args.matrix_reference)
    if not matrix["metadata_only"] or matrix["expression_read"] or matrix["study"] != STUDY:
        raise ValueError("X expected the registered GSE92742 Level 3 metadata axis")
    genes = parse_landmarks(pinned_bytes(args.landmarks, matrix["landmark_axis_sha256"]))
    expected = matrix["matrix_records"]["combined"]
    combined = pd.concat((treatments, controls), ignore_index=True).fillna("")
    indexed, landmarks, record = metadata_index(Path(expected["path"]), combined, genes.pr_gene_id.to_numpy())
    if record != expected or indexed.level3_row_index.duplicated().any():
        raise ValueError("X physical RNA indices disagree with the audited matrix axes")
    treatments = indexed.iloc[:len(treatments)].drop(columns="role")
    controls = indexed.iloc[len(treatments):][["inst_id", "study", "cell_id", *PHYSICAL, "role", "level3_row_index"]]
    treatments["parent_fold"] = treatments.parent_fold.astype(int)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent, prefix=".x-lock-") as temp:
        stage = Path(temp)
        files, results = {}, []
        for target in cells:
            for fold in range(3):
                for role in "ABC":
                    key = f"{target}-f{fold}-{role}"
                    tables, summary = split_scope(treatments, controls, cells, target, fold, role)
                    files[key] = {phase: write_plan(stage / (phase + "-plan") / key,
                                                   {name + ".tsv": cols for name, cols in data.items()})
                                  for phase, data in tables.items()}
                    results.append(summary)
        pd.DataFrame(results).to_csv(stage / "support.tsv", sep="\t", index=False)
        genes.to_csv(stage / "landmarks.tsv", sep="\t", index=False)
        landmarks.to_csv(stage / "gene_indices.tsv", sep="\t", index=False)
        write_json(stage / "receipt.json", dict(
            cells=cells, parents=treatments.parent.nunique(), treatments=len(treatments), physical_controls=len(controls),
            matrix=record, metadata_only=True, expression_read=False, permitted_datasets=[SAMPLE_IDS, GENE_IDS],
            basal_reference_sha256=sha256(args.basal_reference), context_reference_sha256=sha256(args.context_reference),
            protected_access_sha256=sha256(args.r18_access / "receipt.json"), code_sha256=sha256(__file__),
            files={path.name: sha256(path) for path in sorted(stage.glob("*.tsv"))}, slices=files,
        ))
        stage.rename(args.out)
    return sha256(args.out / "receipt.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "reports", "r18-registered", "r18-lock", "r18-access", "r18-index", "basal-reference",
                 "basal-profiles", "context-reference", "matrix-reference", "landmarks", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    print(run(parser.parse_args()))
