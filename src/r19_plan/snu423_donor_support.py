"""Pin the public, nonreserved 24-hour LINCS donors that match Tahoe SNU-423."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r11_chemical_probe.dataset import GROUP
from src.r15_lincs.data import asset, table
from src.r17_lincs.inventory import ROOT, liver_cells, normal
from src.r19_plan.audit_r18 import geo_metadata, prepare
from src.r19_plan.i_preflight import provenance, read_r18_domains, verify_domain
from src.r19_plan.r18_matrix_index import GENE_IDS, SAMPLE_IDS, metadata_index
from src.r19_plan.t4_bridge_support import target_scope
from src.r19_plan.t4_support import eligible_links, measured_dose
from src.r19_plan.x_access import PHYSICAL, control_registry


STUDY = "GSE92742"
NONHEPATIC = ("A375", "A549", "HT29", "MCF7")
TARGET = "CVCL_0366"


def donor_cells(expanded):
    if not expanded:
        return NONHEPATIC, None
    path = asset(ROOT, STUDY, "cell_info")
    annotations = table(path)
    if not annotations.cell_id.is_unique:
        raise ValueError("The official cell catalog does not have a unique cell axis")
    site = annotations.primary_site.str.strip().str.lower()
    active = annotations.cell_type.eq("cell line") & annotations.cell_id.eq(annotations.base_cell_id)
    hepatic = normal(liver_cells(path, cmap=False).cell)
    selected = annotations.loc[active & ~site.isin(["", "-666", "unknown", "na", "n/a"])
                               & ~normal(annotations.cell_id).isin(hepatic), "cell_id"]
    if not set(NONHEPATIC).issubset(selected):
        raise ValueError("The expanded cell catalog dropped a registered original context")
    return tuple(sorted(selected)), sha256(path)


def mapping(query, structure):
    pairs = query[[GROUP, "parent"]].drop_duplicates()
    ambiguous = pairs[GROUP].duplicated(keep=False) | pairs.parent.duplicated(keep=False)
    blocked = set(pairs.loc[ambiguous, GROUP])
    valid = pairs.loc[~pairs[GROUP].isin(blocked)]
    roster = structure.merge(valid, on=GROUP, how="left", validate="one_to_one")
    if len(roster) != 203 or set(roster[GROUP]) != set(pairs[GROUP]):
        raise ValueError("The new donors do not use the canonical axis of the frozen P4 target")
    return roster, len(blocked)


def audit_inputs(args, contexts):
    guard = read_json(args.support_reference)
    if (not guard["metadata_only"] or guard["expression_read"]
            or sha256(args.r18_access/"receipt.json") != guard["reservation_access_sha256"]
            or any(sha256(path) != h for path, h in guard["source_metadata_sha256"][STUDY].items())):
        raise ValueError("The original nonreserved time metadata and receipt diverged")
    footprint = read_r18_domains(args)
    protected = pd.concat([provenance(rows) for rows in footprint], ignore_index=True)
    target, lineage = target_scope(args)
    guard = read_json(args.structure_reference)
    if guard["expression_read"] or sha256(args.structure) != guard["files"]["parent_support.tsv"]:
        raise ValueError("The SNU-423 structure-only stratum changed")
    structure = table(args.structure)[[GROUP, "fold", "min_tanimoto"]]
    structure["fold"], structure["min_tanimoto"] = structure.fold.astype(int), structure.min_tanimoto.astype(float)
    roster, collisions = mapping(target.loc[target.cell_line_id.eq(TARGET)], structure)
    inst, sig = geo_metadata(STUDY, args.base/"vcell_liver_r15_lincs/v1")
    if not set(PHYSICAL) <= set(inst):
        raise ValueError("GSE92742 is missing unambiguous RNA coordinates")
    inst = prepare(inst)
    wanted = sig.cell_id.isin(contexts) & sig.hours.eq(24) & sig.parent.isin(roster.parent.dropna())
    active = inst.cell_id.isin(contexts) & inst.pert_type.eq("trt_cp") & inst.hours.eq(24)
    dose = measured_dose(asset(ROOT, STUDY, "inst_info"), set(inst.loc[active, "inst_id"]))
    multi_context = inst.groupby("rna_plate").cell_id.nunique().loc[lambda n: n > 1].index
    rows, diagnostics = eligible_links(inst, sig.loc[wanted], protected, dose)
    high = rows.actual_dose_uM.astype(float).between(5., 20.)
    rows = rows.loc[high & ~rows.rna_plate.isin(multi_context)].copy()
    verify_domain(rows, rows.iloc[:0], *footprint[:4])
    verify_domain(rows, rows.iloc[:0], footprint[-1], *[rows.iloc[:0]]*3)
    return roster, rows, inst, protected, {**lineage, **diagnostics, "ambiguous_tahoe_groups": collisions}


def strata(roster, rows):
    exact = rows.loc[rows.actual_dose_uM.eq("5")]
    by_plate = rows.groupby("parent").rna_plate.nunique()
    same_cell = rows.groupby(["parent", "cell_id"]).rna_plate.nunique()
    scopes = {
        "measured_5uM": set(exact.parent),
        "measured_5_to_20uM": set(rows.parent),
        "5_to_20uM_at_least_2_RNA_plates": set(by_plate.loc[by_plate.ge(2)].index),
        "5_to_20uM_at_least_2_RNA_plates_same_cell": set(same_cell.loc[same_cell.ge(2)].index.get_level_values("parent")),
    }
    result = []
    for name, parents in scopes.items():
        roster[name] = roster.parent.isin(parents)
        supported = roster.loc[roster[name]]
        for label, selected in (("all", supported), ("tanimoto_below_0.7", supported.loc[supported.min_tanimoto.lt(.7)]),
                                ("tanimoto_at_least_0.7", supported.loc[supported.min_tanimoto.ge(.7)])):
            donors = (exact if name == "measured_5uM" else rows).loc[lambda t: t.parent.isin(selected.parent)]
            counts = selected.groupby("fold")[GROUP].nunique().reindex(range(3), fill_value=0)
            result.append(dict(scope=name, structure=label, parents=len(selected),
                               **{f"fold{fold}": counts[fold] for fold in range(3)},
                               source_wells=donors.inst_id.nunique(), source_RNA_plates=donors.rna_plate.nunique(),
                               contexts="|".join(sorted(donors.cell_id.unique())),
                               eligible=len(selected) >= 100 and counts.ge(20).all()))
    return pd.DataFrame(result)


def pinned_axis(args, rows, inst, protected):
    if rows.empty:
        return {}
    controls = control_registry(inst, rows, protected)
    columns = ["inst_id", "sig_id", "cell_id", "parent", "actual_dose_uM", "hours", *PHYSICAL]
    admitted = rows[columns].assign(study=STUDY).sort_values(["parent", "cell_id", "actual_dose_uM", *PHYSICAL])
    axis_receipt = read_json(args.matrix_reference)
    genes = pd.read_csv(args.genes, sep="\t")
    if (axis_receipt["expression_read"] or not axis_receipt["metadata_only"] or axis_receipt["study"] != STUDY
            or sha256(args.genes) != read_json(args.gene_reference)["files"]["genes.tsv"]
            or any(not genes[c].is_unique for c in ("token_id_lincs", "token_id_tahoe", "ensembl_id")) or len(genes) != 161):
        raise ValueError("The 161-gene assay bridge or matrix receipt changed")
    expected = axis_receipt["matrix_records"]["combined"]
    indexed, axis, record = metadata_index(Path(expected["path"]), pd.concat((admitted, controls), ignore_index=True).fillna(""),
                                           genes.token_id_lincs.to_numpy())
    if record != expected or indexed.level3_row_index.duplicated().any() or len(axis) != len(genes):
        raise ValueError("Donor support could not be verified on the original physical and common gene axes")
    return {"admitted_donors.tsv": indexed.iloc[:len(admitted)].drop(columns="role"),
            "admitted_vehicles.tsv": indexed.iloc[len(admitted):][[*controls.columns, "level3_row_index"]],
            "gene_indices.tsv": axis}


def run(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    contexts, cells_sha256 = donor_cells(args.expanded)
    roster, rows, inst, protected, source = audit_inputs(args, contexts)
    summary = strata(roster, rows)
    outputs = {"support.tsv": summary, "parent_support.tsv": roster, **pinned_axis(args, rows, inst, protected)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent, prefix=".snu-donor-") as tmp:
        stage = Path(tmp)
        for name, value in outputs.items():
            value.to_csv(stage/name, sep="\t", index=False)
        write_json(stage/"receipt.json", dict(metadata_only=True, expression_read=False, study=STUDY,
            nonhepatic=list(contexts), cells_sha256=cells_sha256, source=source, allowed_datasets=[SAMPLE_IDS, GENE_IDS],
            protocol_sha256=sha256(args.protocol), code_sha256=sha256(__file__),
            protected_access_sha256=sha256(args.r18_access/"receipt.json"),
            protected_control_sha256=sha256(args.r18_lock/"receipt.json"),
            matrix_reference_sha256=sha256(args.matrix_reference), source_audit_sha256=sha256(args.support_reference),
            structure_reference_sha256=sha256(args.structure_reference),
            files={p.name: sha256(p) for p in sorted(stage.iterdir())}))
        stage.rename(args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "reports", "r18-registered", "r18-lock", "r18-access", "r18-index", "support-reference",
                 "tahoe", "tahoe-reference", "entities", "gene-reference", "matrix-reference", "genes",
                 "structure-reference", "structure", "protocol", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expanded", action="store_true")
    run(parser.parse_args())
