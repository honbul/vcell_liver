"""Admit only covered, nonreserved physical LINCS wells for the HepaRG donor test."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.goal_tracks.heparg import DATASETS, PRIMARY
from src.p123.p3_run import T3_FREEZE
from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r13_optimization.lincs_transfer import clean, structural_keys
from src.r15_lincs.data import asset
from src.r17_lincs.inventory import ROOT, ids, in_footprint, register_footprint
from src.r19_plan.audit_r18 import geo_metadata, prepare
from src.r19_plan.i_preflight import provenance, read_pinned, read_r18_domains, verify_domain
from src.r19_plan.r18_matrix_index import GENE_IDS, SAMPLE_IDS, metadata_index
from src.r19_plan.snu423_donor_support import STUDY, donor_cells
from src.r19_plan.t4_heparg_support import DATASETS as CACHE_NAMES
from src.r19_plan.t4_support import eligible_links, measured_dose
from src.r19_plan.x_access import PHYSICAL, control_registry

FIELDS = ("parent", "cell_id", "actual_dose_uM")


def target_roster(args, frozen):
    chunks = []
    for dataset in DATASETS:
        folder = args.cache / "data" / CACHE_NAMES[dataset]
        receipt = folder / "receipt.json"
        if sha256(receipt) != frozen["inputs"][dataset]:
            raise ValueError("Target cache does not match the T3 freeze")
        rows = read_pinned(folder, "rows.tsv", read_json(receipt)["files"]["rows.tsv"])
        active = rows.groupby(["outer_fold", "plate", "dose_rank"]).parent_key.transform("nunique").ge(3)
        rows = rows.loc[active & rows.dose_uM.ge(10) & rows.outer_fold.isin(range(3))]
        counts = rows.groupby(["dataset", "parent_key", "outer_fold"]).size().rename("wells").reset_index()
        structures = rows[["parent_key", "smiles"]].drop_duplicates()
        structures["parent"] = structures.smiles.map(lambda smiles: structural_keys(clean(smiles))[1])
        mapping = structures[["parent_key", "parent"]].drop_duplicates()
        chunks.append(counts.merge(mapping, on="parent_key", how="left", validate="one_to_many"))
    both = pd.concat(chunks, ignore_index=True)
    mapping = both[["parent_key", "parent"]].drop_duplicates()
    collisions = mapping.parent_key.duplicated(False) | mapping.parent.duplicated(False) | mapping.parent.eq("")
    blocked = mapping.loc[collisions, "parent_key"]
    roster = both.drop_duplicates(["dataset", "parent_key"]).copy()
    roster.loc[roster.parent_key.isin(blocked), "parent"] = ""
    return roster, blocked.nunique()


def donors(args, roster):
    footprint = read_r18_domains(args)
    protected = pd.concat([provenance(frame) for frame in footprint], ignore_index=True)
    proof = read_json(args.support_reference)
    if proof["expression_read"] or any(sha256(path) != h for path, h in proof["source_metadata_sha256"][STUDY].items()):
        raise ValueError("Official source metadata changed after the nonreserved audit")
    cells, cell_hash = donor_cells(True)
    inst, sig = geo_metadata(STUDY, args.base / "vcell_liver_r15_lincs/v1")
    inst = prepare(inst)
    seeded = sig.cell_id.isin(cells) & sig.hours.eq(24) & sig.parent.isin(roster.parent)
    active = inst.cell_id.isin(cells) & inst.pert_type.eq("trt_cp") & inst.hours.eq(24)
    measured = measured_dose(asset(ROOT, STUDY, "inst_info"), set(inst.loc[active, "inst_id"]))
    rows, diagnostics = eligible_links(inst, sig.loc[seeded], protected, measured)
    plates = inst.groupby("rna_plate").cell_id.nunique().loc[lambda x: x.eq(1)].index
    vehicles = inst.loc[inst.pert_type.eq("ctl_vehicle")
                        & (ids(inst.pert_id).eq("DMSO") | ids(inst.pert_iname).eq("DMSO"))
                        & inst.rna_well.str.fullmatch("[A-Z]+[0-9]{2}") & inst.rna_plate.isin(plates)]
    vehicles = vehicles.loc[~in_footprint(vehicles, register_footprint(protected))].drop_duplicates(PHYSICAL)
    plates = vehicles.groupby("rna_plate").size().loc[lambda x: x.ge(3)].index
    rows = rows.loc[rows.actual_dose_uM.astype(float).between(5, 20) & rows.rna_plate.isin(plates)].copy()
    rows["distance"] = rows.actual_dose_uM.astype(float).map(lambda d: abs(d-10))
    selected = rows.sort_values(["distance", "actual_dose_uM"]).drop_duplicates(["parent", "cell_id"])
    rows = rows.merge(selected[list(FIELDS)], on=list(FIELDS), validate="many_to_one").reset_index(drop=True)
    controls = control_registry(inst, rows, protected)
    verify_domain(rows, controls, *footprint[:4])
    verify_domain(rows, controls, footprint[4], *[rows.iloc[:0]]*3)
    if rows.duplicated(PHYSICAL).any():
        raise ValueError("A source physical well appears more than once")
    return rows, controls, dict(cell_catalog_sha256=cell_hash, **diagnostics)


def index_axis(args, rows, controls):
    receipt = read_json(args.p3_score)
    axis = read_pinned(args.p3_score.parent, "landmark_map.tsv", receipt["files"]["landmark_map.tsv"])
    genes = axis.loc[axis.both_temposeq_assays.eq(True), ["entrez_id", "ensembl_id"]].reset_index(drop=True)
    if len(genes) != 827 or not genes.entrez_id.is_unique or not genes.ensembl_id.is_unique:
        raise ValueError("One-to-one P3 landmark axis is missing or changed")
    audit = read_json(args.matrix_reference)
    expected = audit["matrix_records"]["combined"]
    if audit["expression_read"] or not audit["metadata_only"] or audit["study"] != STUDY:
        raise ValueError("Expected the approved LINCS metadata axis")
    metadata = rows[["inst_id", "sig_id", *FIELDS, *PHYSICAL]].assign(study=STUDY)
    combined, positions, record = metadata_index(Path(expected["path"]), pd.concat((metadata, controls)).fillna(""),
                                                  genes.entrez_id.to_numpy())
    if expected != record or not combined.level3_row_index.is_unique or not positions.gene_id.equals(genes.entrez_id):
        raise ValueError("Unexpected LINCS original matrix axes")
    genes["level3_gene_index"] = positions.level3_gene_index
    return combined.iloc[:len(metadata)].drop(columns="role"), combined.iloc[len(metadata):][
        [*controls.columns, "level3_row_index"]], genes


def run(args):
    path = args.t3 / "freeze.json"
    if sha256(path) != T3_FREEZE:
        raise ValueError("Original T3 freeze changed")
    roster, collisions = target_roster(args, read_json(path))
    source, vehicle, details = donors(args, roster)
    rows, controls, genes = index_axis(args, source, vehicle)
    roster = roster.assign(support=roster.parent.isin(rows.parent))
    support = roster.groupby(["dataset", "outer_fold"]).agg(
        parents=("parent_key", "size"), covered_parents=("support", "sum"), wells=("wells", "sum")).reset_index()
    covered = roster.loc[roster.support].groupby(["dataset", "outer_fold"]).wells.sum()
    support["covered_wells"] = [covered.get((d, f), 0) for d, f in zip(support.dataset, support.outer_fold)]
    eligible = [d for d, fold in support.groupby("dataset")
                if len(fold)==3 and fold.covered_parents.sum()>=100 and fold.covered_parents.min()>=20]
    with TemporaryDirectory(dir=args.out.parent) as temp:
        build = Path(temp)
        tables = {"support.tsv": support, "parent_roster.tsv": roster, "donors.tsv": rows,
                  "vehicle.tsv": controls, "genes.tsv": genes,
                  "source_dose.tsv": rows.groupby("actual_dose_uM").agg(
                      wells=("inst_id", "size"), parents=("parent", "nunique")).reset_index()}
        for name, frame in tables.items():
            frame.to_csv(build / name, sep="\t", index=False)
        write_json(build / "receipt.json", dict(expression_read=False, metadata_only=True, study=STUDY,
                   primary=PRIMARY, eligible=eligible, structural_collisions=collisions, source=details,
                   t3_freeze=T3_FREEZE, protocol_sha256=sha256(args.protocol), code_sha256=sha256(__file__),
                   matrix_reference_sha256=sha256(args.matrix_reference), source_target_provenance=sha256(args.p3_score),
                   protected_metadata_sha256=sha256(args.r18_access/"receipt.json"),
                   allowed_datasets=[SAMPLE_IDS, GENE_IDS], files={p.name: sha256(p) for p in build.iterdir()}))
        if args.out.exists():
            raise FileExistsError(args.out)
        build.rename(args.out)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "reports", "cache", "t3", "r18-registered", "r18-lock", "r18-access", "r18-index",
                 "support-reference", "p3-score", "matrix-reference", "protocol", "out"):
        cli.add_argument(f"--{name}", type=Path, required=True)
    run(cli.parse_args())
