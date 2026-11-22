"""Build protected, exact-dose, non-hepatic LINCS 6 h/24 h source pairs."""

import argparse
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold

from src.r11_chemical_probe.common import sha256, write_json
from src.r13_optimization.lincs_transfer import clean, structural_keys
from src.r15_lincs.data import DOSE, TIME, asset, measures, table
from src.r17_lincs.inventory import CMAP, ROOT, liver_cells, in_footprint, register_footprint
from src.r19_plan.r18_matrix_index import GENE_IDS, SAMPLE_IDS
from src.r19_plan.r18_input_bundle import EXPRESSION

GEO = ("GSE92742", "GSE70138")
PAIR = ["cell_id", "parent", "dose"]


def chemical_key(parent):
    structure = structural_keys(clean(parent))[1]
    if not structure:
        return None
    mol = Chem.MolFromSmiles(structure)
    identity = Chem.MolToInchiKey(mol).split("-")[0]
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    if not identity:
        return None
    return structure, identity, scaffold or f"acyclic:{identity}"


def lock(args):
    reserved = [table(args.r18 / name) for name in ("split_rows.tsv", "control_assignments.tsv")]
    reserved.append(table(args.donors))
    pair_queries = []
    for directory in (args.hepg2, args.phh):
        files = sorted((directory / "access/predict-plan").glob("*/query.tsv"))
        if not files:
            raise ValueError(f"No score axis manifest at {directory}")
        pair_queries.extend(table(path) for path in files)
    protected = pd.concat([*reserved, *pair_queries], ignore_index=True)
    molecule_roles = protected.parent.dropna().unique()
    with rdBase.BlockLogs():
        keys = [chemical_key(s) for s in molecule_roles]
    invalid = [s for s, k in zip(molecule_roles, keys) if not k and clean(s)]
    if invalid:
        raise ValueError(f"Protected parent registry contains {len(invalid)} unresolved structures: {invalid[:3]}")
    fixed = pd.DataFrame([k for k in keys if k], columns=["parent", "connectivity", "scaffold"]).drop_duplicates()
    plates = set(protected.rna_plate.dropna().str.upper()) | set(protected.det_plate.dropna().str.upper())
    plates.discard("")
    return fixed, register_footprint(protected.fillna("")), plates


def compound_map(path, blocked):
    metadata = table(path, usecols=lambda c: c in ("pert_type", "pert_id", "canonical_smiles", "inchi_key"))
    if "pert_type" in metadata:
        metadata = metadata.loc[metadata.pert_type.eq("trt_cp")]
    metadata = metadata.drop_duplicates(["pert_id", "canonical_smiles"])
    keys = {}
    with rdBase.BlockLogs():
        for smiles in metadata.canonical_smiles.unique():
            keys[smiles] = chemical_key(smiles)
    standardized = pd.DataFrame(metadata.canonical_smiles.map(keys).tolist(),
                                columns=["parent", "connectivity", "scaffold"], index=metadata.index)
    metadata = pd.concat([metadata[["pert_id"]], standardized], axis=1).dropna()
    metadata = metadata.drop_duplicates()
    metadata = metadata.loc[~metadata.pert_id.duplicated(keep=False)]
    return metadata.loc[~metadata.connectivity.isin(blocked.connectivity) &
                        ~metadata.scaffold.isin(blocked.scaffold)]


def matrix_rows(path, wanted, genes):
    data = np.empty((len(wanted), len(genes)), dtype=np.float32)
    with h5py.File(path, "r") as h5:
        samples = pd.Index(h5[SAMPLE_IDS][:].astype(str))
        row_pos = samples.get_indexer(wanted.inst_id)
        col_pos = pd.Index(h5[GENE_IDS][:].astype(int)).get_indexer(genes)
        if not samples.is_unique or (row_pos < 0).any() or (col_pos < 0).any():
            raise ValueError(f"Unmapped expression IDs in {path}")
        order = np.argsort(row_pos)
        matrix = h5[EXPRESSION]
        for offset in range(0, len(order), 512):
            positions = order[offset:offset + 512]
            data[positions] = matrix[row_pos[positions], :][:, col_pos]
    if not np.isfinite(data).all():
        raise ValueError(f"Nonfinite LINCS values in {path}")
    return data


def study_paths(study, cmap):
    if study == CMAP:
        official = ROOT / "LINCS2020"
        return (cmap / "instinfo_beta.txt", official / "compoundinfo_beta.txt",
                official / "cellinfo_beta.txt", next(cmap.glob("level3_beta_trt_cp_*.gctx")),
                next(cmap.glob("level3_beta_ctl_*.gctx")))
    matrix, = (ROOT / study).glob("*Level3*.gctx")
    return asset(ROOT, study, "inst_info"), asset(ROOT, study, "pert_info"), asset(ROOT, study, "cell_info"), matrix, matrix


def lock_metadata(study, paths, blocked, footprint, blocked_plates):
    instances, compounds, cells, *_ = paths
    chemical = compound_map(compounds, blocked)
    hepatic = set(liver_cells(cells, cmap=study == CMAP).cell.str.upper())
    aliases = {"sample_id": "inst_id", "cell_iname": "cell_id", "cmap_name": "pert_iname"}
    features = {"inst_id", "cell_id", "pert_id", "pert_type", "pert_iname", "pert_dose", "pert_dose_unit",
                "pert_time", "pert_time_unit", "rna_plate", "rna_well", "det_plate", "det_well", "qc_pass", *aliases}
    admitted = []
    for chunk in pd.read_csv(instances, sep="\t", dtype=str, keep_default_na=False,
                             usecols=lambda c: c in features, chunksize=180_000):
        rows = chunk.rename(columns=aliases)
        if "rna_plate" not in rows:
            rows = rows.assign(rna_plate=rows.det_plate, rna_well=rows.det_well)
        if "det_plate" not in rows:
            rows = rows.assign(det_plate=rows.rna_plate, det_well=rows.rna_well)
        for name in ("rna_plate", "rna_well", "det_plate", "det_well", "cell_id"):
            rows[name] = rows[name].str.strip().str.upper()
        rows["rna_well"] = rows.rna_well.str.replace(r"^([A-Z]+)(\d)$", r"\g<1>0\2", regex=True)
        control = rows.pert_type.eq("ctl_vehicle") & (rows.pert_id.str.upper().eq("DMSO") |
                                                       rows.pert_iname.str.upper().eq("DMSO"))
        treated = rows.pert_type.eq("trt_cp") & rows.pert_id.isin(chemical.pert_id)
        qc = rows.qc_pass.eq("1") if "qc_pass" in rows else True
        rows = rows.loc[(control | treated) & qc & ~rows.cell_id.isin(hepatic) &
                        rows.rna_well.str.fullmatch(r"[A-Z]+\d{2}") & rows.rna_plate.ne("") &
                        ~rows.rna_plate.isin(blocked_plates) & ~rows.det_plate.isin(blocked_plates)].copy()
        rows = rows.loc[~in_footprint(rows, footprint)].copy()
        rows["hours"] = measures(rows.pert_time, rows.pert_time_unit, TIME)
        rows["dose"] = measures(rows.pert_dose, rows.pert_dose_unit, DOSE)
        rows = rows.loc[rows.pert_type.eq("ctl_vehicle") | (rows.hours.isin((6., 24.)) & rows.dose.gt(0))]
        admitted.append(rows)
    all_rows = pd.concat(admitted).sort_values("inst_id")
    all_rows = all_rows.drop_duplicates(["rna_plate", "rna_well"], keep=False).drop_duplicates("inst_id", keep=False)
    controls = all_rows.loc[all_rows.pert_type.eq("ctl_vehicle")].copy()
    sizes = controls.groupby(["cell_id", "rna_plate"]).inst_id.size()
    two_controls = sizes[sizes.ge(2)].index
    treatments = all_rows.loc[all_rows.pert_type.eq("trt_cp")].merge(
        chemical, on="pert_id", validate="many_to_one")
    treatments = treatments.loc[pd.MultiIndex.from_frame(treatments[["cell_id", "rna_plate"]]).isin(two_controls)]
    times = treatments.groupby(PAIR).hours.transform("nunique")
    treatments = treatments.loc[times.eq(2)].reset_index(drop=True)
    controls = controls.loc[pd.MultiIndex.from_frame(controls[["cell_id", "rna_plate"]]).isin(
        pd.MultiIndex.from_frame(treatments[["cell_id", "rna_plate"]]))].reset_index(drop=True)
    return treatments, controls, set(all_rows.rna_plate) | set(all_rows.det_plate)


def build(args):
    args.out.mkdir(parents=True, exist_ok=False)
    genes = table(args.landmarks).pr_gene_id.astype(int).to_numpy()
    protected, footprint, excluded = lock(args)
    earlier_geo = set()
    receipts = []
    for study in (*GEO, CMAP):
        paths = study_paths(study, args.cmap)
        treatments, controls, all_plates = lock_metadata(
            study, paths, protected, footprint, excluded | earlier_geo)
        earlier_geo |= all_plates
        if treatments.empty:
            raise ValueError(f"No eligible 6h-24h pairs from {study}")
        treatment_path, control_path = paths[-2:]
        print(study, "eligible wells", len(treatments), "controls", len(controls), flush=True)
        treatment = matrix_rows(treatment_path, treatments, genes)
        vehicle = matrix_rows(control_path, controls, genes)
        means = {key: vehicle[indices].mean(axis=0)
                 for key, indices in controls.groupby(["cell_id", "rna_plate"]).indices.items()}
        keys = zip(treatments.cell_id, treatments.rna_plate)
        contrasts = treatment - np.stack([means[key] for key in keys])
        conditions = defaultdict(dict)
        for (*key, time), ix in treatments.groupby([*PAIR, "hours"]).indices.items():
            conditions[tuple(key)][time] = contrasts[ix].mean(axis=0)
        keys = sorted(conditions)
        frame = pd.DataFrame(keys, columns=PAIR).assign(study=study)
        x = np.stack([conditions[k][6.] for k in keys])
        y = np.stack([conditions[k][24.] for k in keys])
        output = args.out / f"{study}.npz"
        np.savez(output, x=x, y=y, genes=genes)
        frame.to_parquet(args.out / f"{study}.parquet", index=False)
        counts = {"study": study, "pairs": len(frame), "wells": len(treatments), "vehicles": len(controls),
                  "cells": frame.cell_id.nunique(), "parents": frame.parent.nunique(),
                  "protected_connectivity_overlap": len(set(treatments.connectivity) & set(protected.connectivity)),
                  "protected_scaffold_overlap": len(set(treatments.scaffold) & set(protected.scaffold)),
                  "matrix_bytes": {str(p): p.stat().st_size for p in set(paths[-2:])},
                  "metadata_sha256": {str(p): sha256(p) for p in paths[:3]},
                  "pairs_sha256": sha256(output)}
        receipts.append(counts)
        print(counts, flush=True)
    write_json(args.out / "receipt.json", {"studies": receipts, "gene_axis_sha256": sha256(args.landmarks),
                                          "locked_molecules": len(protected), "locked_plates": len(excluded)})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for field in ("hepg2", "phh", "cmap", "r18", "donors", "landmarks", "out"):
        cli.add_argument("--" + field, type=Path, required=True)
    build(cli.parse_args())
