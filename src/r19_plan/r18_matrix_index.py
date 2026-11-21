"""Resolve locked R18 well IDs to Level 3 metadata axes without accessing expression."""

import argparse
from pathlib import Path
from zlib import crc32
import hashlib

import h5py
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r15_lincs.data import asset, table
from src.r16_lincs.audit import DET, RNA, geo_exposure
from src.r17_lincs.inventory import CMAP, ROOT, in_footprint, register_footprint
from src.r19_plan.audit_r18 import PHYSICAL, prepare
from src.r19_plan.r18_preflight import checked_tables

STUDIES = ("GSE92742", "GSE70138", CMAP)
SAMPLE_IDS = "0/META/COL/id"
GENE_IDS = "0/META/ROW/id"


def metadata_index(path, rows, landmarks):
    with h5py.File(path, "r") as archive:
        sample_ids = archive[SAMPLE_IDS][:].astype(str)
        gene_ids = archive[GENE_IDS][:].astype(int)
    sample_axis = pd.Index(sample_ids)
    gene_axis = pd.Index(gene_ids)
    if not sample_axis.is_unique or not gene_axis.is_unique:
        raise ValueError(f"Nonunique original matrix IDs in {path.name}")
    row_indices = sample_axis.get_indexer(rows.inst_id)
    landmark_indices = gene_axis.get_indexer(landmarks)
    if (row_indices < 0).any() or (landmark_indices < 0).any():
        raise ValueError(f"Unmatched registered IDs in {path.name}: {(row_indices < 0).sum()} wells, "
                         f"{(landmark_indices < 0).sum()} landmarks")
    info = dict(
        path=str(path), bytes=path.stat().st_size, indexed_wells=len(sample_axis), indexed_genes=len(gene_axis),
        sample_axis_sha256=hashlib.sha256("\n".join(sample_axis).encode()).hexdigest(),
        gene_axis_sha256=hashlib.sha256("\n".join(map(str, gene_axis)).encode()).hexdigest(),
    )
    return rows.assign(level3_row_index=row_indices), pd.DataFrame(
        {"gene_id": landmarks, "level3_gene_index": landmark_indices}), info


def raw_controls(study, prior, cmap_dir, donors):
    plates = set(donors.plate)
    if study != CMAP:
        path = asset(ROOT, study, "inst_info")
        selected = {"inst_id", "cell_id", "pert_type", "pert_id", "pert_iname", *RNA, *DET}
        raw = table(path, usecols=lambda name: name in selected)
        raw["qc"] = True
        prepared = prepare(raw)
    else:
        path = cmap_dir / "instinfo_beta.txt"
        selected = ["sample_id", "cell_iname", "pert_type", "pert_id", "cmap_name", "qc_pass", *RNA, *DET]
        chunks = []
        for raw in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, usecols=selected, chunksize=160000):
            keep = (raw.rna_plate.str.strip().str.upper().isin(plates) & raw.pert_type.eq("ctl_vehicle")
                    & raw.qc_pass.eq("1"))
            chunks.append(raw.loc[keep])
        raw = pd.concat(chunks).rename(columns={"sample_id": "inst_id", "cell_iname": "cell_id",
                                              "cmap_name": "pert_iname"})
        raw["qc"] = raw.qc_pass.eq("1")
        prepared = prepare(raw)
    allowed = (prepared.plate.isin(plates) & prepared.qc & prepared.pert_type.eq("ctl_vehicle")
               & (prepared.pert_id.str.upper().eq("DMSO") | prepared.pert_iname.str.upper().eq("DMSO"))
               & prepared.well.str.fullmatch(r"[A-Z]+\d{2}"))
    columns = ["inst_id", "cell_id", *PHYSICAL, *RNA, *DET]
    controls = prepared.loc[allowed, columns].copy()
    if study == CMAP:
        _, ids, physical, _ = geo_exposure(prior)
        controls = controls.loc[~in_footprint(controls, (ids, physical))]
    # R16 groups CMap by cell and RNA plate. GEO supplies one cell context for each donor plate.
    contexts = donors[["plate", "cell_id"]].drop_duplicates()
    if contexts.plate.duplicated().any():
        raise ValueError("Donor physical plate has more than one registered cell context")
    controls = controls.merge(contexts, on=["plate", "cell_id"], validate="many_to_one")
    controls = controls.sort_values("inst_id").drop_duplicates(PHYSICAL)
    sizes = controls.groupby("plate").size()
    if donors.plate.map(sizes).fillna(0).lt(2).any():
        raise ValueError("A registered B donor has fewer than two physical DMSO controls")
    controls["crc"] = controls.inst_id.map(lambda key: crc32(key.encode()))
    controls = controls.sort_values(["plate", "crc", "inst_id"])
    group = controls.groupby("plate")
    controls["half"] = (group.cumcount().ge(group.inst_id.transform("size").floordiv(2))).astype(int) + 1
    return controls.drop(columns="crc").assign(study=study), path


def run(args):
    lock = checked_tables(args.lock, args.lock_reference)
    registered = checked_tables(args.registered, args.registered_reference)
    if lock["registered_support_sha256"] != sha256(args.registered_reference):
        raise ValueError("R18 split lock and reserved support differ")
    d0 = read_json(args.r17_reference)
    prior = args.base / "vcell_liver_r15_lincs/v1"
    cmap = args.base / "vcell_liver_transigen_adaptation_b1/assets/raw/cmap2020"
    reference = prior / "cache/GSE92742/landmarks.tsv"
    gene_hash = sha256(reference)
    if set(d0["gene_axis_sha256"].values()) != {gene_hash}:
        raise ValueError("R18 landmark order diverges from R17")
    genes = table(reference).pr_gene_id.astype(int)
    if len(genes) != d0["landmarks"] or not genes.is_unique:
        raise ValueError("Invalid landmark axis")

    study = args.study
    split = table(args.lock / "split_rows.tsv")
    treatment = split.loc[split.study.eq(study), ["study", "inst_id"]].drop_duplicates()
    controls = table(args.lock / "control_assignments.tsv")
    donors = table(args.registered / "r18_B_donor_exceptions.tsv")
    donors = donors.loc[donors.study.eq(study)].copy()
    if in_footprint(donors, register_footprint(controls)).any() or set(donors.inst_id) & set(treatment.inst_id):
        raise ValueError("R18 target-role and donor treatment indices overlap")
    source_controls, inst_path = raw_controls(study, prior, cmap, donors)
    if (in_footprint(source_controls, register_footprint(controls)).any()
        or (set(source_controls.inst_id) & (set(treatment.inst_id) | set(donors.inst_id)))):
        raise ValueError("Donor controls overlap the locked target-role input axis")
    target_controls = controls.loc[controls.study.eq(study), ["study", "inst_id"]].copy()
    treated = pd.concat([treatment.assign(use="target_treatment"), donors[["study", "inst_id"]].assign(use="B_donor")])
    vehicle = pd.concat([target_controls.assign(use="target_role_DMSO"),
                         source_controls[["study", "inst_id"]].assign(use="B_donor_DMSO")])
    if pd.concat([treated, vehicle]).inst_id.duplicated().any():
        raise ValueError("A single physical sample ID appears twice in the Level 3 inputs")

    if study == CMAP:
        trt_path, = cmap.glob("level3_beta_trt_cp_*.gctx")
        ctl_path, = cmap.glob("level3_beta_ctl_*.gctx")
        row_t, col_t, trt_meta = metadata_index(trt_path, treated, genes)
        row_c, col_c, ctl_meta = metadata_index(ctl_path, vehicle, genes)
        matrix_records = {"treated": trt_meta, "controls": ctl_meta}
        rows = pd.concat([row_t.assign(matrix="treated"), row_c.assign(matrix="controls")])
        columns = pd.concat([col_t.assign(matrix="treated"), col_c.assign(matrix="controls")])
    else:
        path, = (ROOT / study).glob("*Level3*.gctx")
        rows, columns, info = metadata_index(path, pd.concat([treated, vehicle]), genes)
        rows, columns = rows.assign(matrix="combined"), columns.assign(matrix="combined")
        matrix_records = {"combined": info}

    out = args.out
    out.mkdir(exist_ok=False, parents=True)
    rows.to_csv(out / "level3_rows.tsv", sep="\t", index=False)
    columns.to_csv(out / "level3_landmarks.tsv", sep="\t", index=False)
    source_controls.to_csv(out / "donor_controls.tsv", sep="\t", index=False)
    write_json(out / "receipt.json", dict(study=study, metadata_only=True, expression_read=False,
        permitted_HDF5_datasets=[SAMPLE_IDS, GENE_IDS], code_sha256=sha256(__file__),
        lock_sha256=sha256(args.lock_reference), audited_donor_sha256=registered["files"]["r18_B_donor_exceptions.tsv"],
        instance_metadata_sha256=sha256(inst_path), landmark_axis_sha256=gene_hash, landmarks=len(genes),
        support={key: int(count) for key, count in rows.use.value_counts().items()},
        donor_plates=int(donors.plate.nunique()), matrix_records=matrix_records,
        files={path.name: sha256(path) for path in sorted(out.iterdir())}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, choices=STUDIES)
    for key in ("base", "lock", "lock-reference", "registered", "registered-reference", "r17-reference", "out"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    run(parser.parse_args())
