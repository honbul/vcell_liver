"""Phase 0 metadata audit for the HCC genotype-FiLM study (plan v3).

Reads metadata only (no expression matrix values, no treated-response value, no QC metric of a
treated signature) and writes the tables listed in plan 3.7. Every rule comes from
manifests/genotype_film_v3/audit_policy.yaml, which was committed before this script ran.

L1000 releases are audited one at a time and never pooled. Tahoe is audited at treatment-well level.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

DRIVERS = ["TP53", "CTNNB1", "AXIN1", "ARID1A", "RB1", "CDKN2A", "PTEN", "PIK3CA"]
ADULT_HCC_NCIT = "Adult hepatocellular carcinoma"
INVENTORY: list[dict] = []


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def note_file(path: Path, release: str, role: str, read: str, hash_it: bool = True) -> Path:
    st = path.stat()
    INVENTORY.append(dict(
        platform_release=release, role=role, path=str(path), bytes=st.st_size,
        mtime=pd.Timestamp(st.st_mtime, unit="s").strftime("%Y-%m-%d"),
        sha256=sha256(path) if hash_it else "not_hashed_matrix_not_read", how_read=read))
    return path


def strip_name(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def sig4(x: float) -> str:
    return f"{float(x):.4g}"


# ----------------------------------------------------------------------------- cell identity
class Registry:
    def __init__(self, depmap: Path, cello: Path):
        m = pd.read_csv(note_file(depmap / "Model.csv", "DepMap_25Q3", "model annotation", "full"))
        self.model = m
        self.by_stripped = defaultdict(list)
        for _, r in m.iterrows():
            self.by_stripped[str(r.StrippedCellLineName)].append(r)
        self.by_rrid = defaultdict(list)
        for _, r in m.iterrows():
            if isinstance(r.RRID, str):
                self.by_rrid[r.RRID].append(r)
        self.cello = pd.read_csv(note_file(cello, "Cellosaurus_57.0", "disease/age/derivation", "full"),
                                 sep="\t", dtype=str).fillna("").set_index("cellosaurus_id")
        self.liver_stripped = set(m[m.OncotreeLineage == "Liver"].StrippedCellLineName.astype(str))
        self.liver_rrid = set(m[m.OncotreeLineage == "Liver"].RRID.dropna().astype(str))
        # union-find over Cellosaurus root ancestor, same-individual links and DepMap PatientID
        self.parent: dict[str, str] = {}
        for ac, r in self.cello.iterrows():
            self._union(ac, r.root_ancestor)
            for o in filter(None, r.same_individual.split(",")):
                self._union(ac, o)
        for pid, g in m[m.OncotreeLineage == "Liver"].groupby("PatientID"):
            acs = [a for a in g.RRID.dropna().astype(str)]
            for a in acs[1:]:
                self._union(acs[0], a)

    def _find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def _union(self, a: str, b: str) -> None:
        ra, rb = self._find(a), self._find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)

    def family(self, ac: str) -> str:
        return "FAM_" + self._find(ac) if ac else ""

    def resolve(self, release: str, name: str, platform_ac: str, platform_note: str) -> dict:
        """One registry row. No fuzzy matching: exact accession, else exact stripped name."""
        row = dict(platform_release=release, platform_cell_name=name, cellosaurus_id="",
                   depmap_model_id="", patient_or_origin_id="", family_id="", disease="",
                   age_category="", subtype="", is_adult_hcc=False, inclusion_status="",
                   mapping_source="", ambiguity_reason="", platform_annotation=platform_note,
                   cellosaurus_disease="", cellosaurus_age="", cellosaurus_parent="",
                   cellosaurus_problematic="")
        hits, src = [], ""
        if platform_ac:
            hits, src = self.by_rrid.get(platform_ac, []), "platform_cellosaurus_id->DepMap RRID"
            row["cellosaurus_id"] = platform_ac
        if not hits:
            hits = self.by_stripped.get(strip_name(name), [])
            src = "exact StrippedCellLineName" if hits else src
        if len(hits) > 1:
            row.update(inclusion_status="held_ambiguous",
                       ambiguity_reason=f"{len(hits)} DepMap models share this name")
            return row
        if len(hits) == 1:
            d = hits[0]
            ac = d.RRID if isinstance(d.RRID, str) else ""
            if platform_ac and ac and platform_ac != ac:
                row.update(inclusion_status="held_ambiguous",
                           ambiguity_reason=f"platform {platform_ac} vs DepMap {ac}")
                return row
            row.update(cellosaurus_id=ac or platform_ac, depmap_model_id=d.ModelID,
                       patient_or_origin_id=d.PatientID, disease=str(d.OncotreePrimaryDisease),
                       age_category=str(d.AgeCategory), subtype=str(d.OncotreeSubtype),
                       mapping_source=src)
        ac = row["cellosaurus_id"]
        if ac and ac in self.cello.index:
            c = self.cello.loc[ac]
            row.update(cellosaurus_disease=c.ncit_diseases, cellosaurus_age=c.age,
                       cellosaurus_parent=c.parent, cellosaurus_problematic=c.problematic_note[:120],
                       family_id=self.family(ac))
        if not row["depmap_model_id"] and not ac:
            row.update(inclusion_status="excluded_unmapped",
                       ambiguity_reason="no Cellosaurus accession and no exact DepMap name match")
            return row
        dep_hcc = row["subtype"] == "Hepatocellular Carcinoma" and row["age_category"] == "Adult"
        cel_hcc = ADULT_HCC_NCIT in row["cellosaurus_disease"]
        if not row["depmap_model_id"]:
            # e.g. a derived clone known to Cellosaurus only
            if cel_hcc and row["cellosaurus_parent"]:
                row.update(inclusion_status="derived_clone_of_family_no_depmap_model",
                           ambiguity_reason="counts only through its family; no own DepMap genotype")
            else:
                row.update(inclusion_status="excluded_no_depmap_model")
        elif dep_hcc and cel_hcc:
            row.update(is_adult_hcc=True, inclusion_status="included_adult_hcc")
        elif dep_hcc and not cel_hcc:
            row.update(inclusion_status="held_ambiguous",
                       ambiguity_reason="DepMap adult HCC; Cellosaurus: " + (row["cellosaurus_disease"] or "no record"))
        else:
            row.update(inclusion_status="excluded_not_adult_hcc",
                       ambiguity_reason=f"{row['subtype']} / {row['age_category']} / {row['cellosaurus_disease']}")
        return row


# ----------------------------------------------------------------------------- L1000 instance level
def audit_l1000_inst(release: str, inst: pd.DataFrame, cells: pd.DataFrame, reg: Registry,
                     gctx_l3: Path, out: dict) -> None:
    """inst columns: inst_id, plate, well, pert_id, pert_type, dose, dose_unit, time, time_unit, cell."""
    cand = []
    for _, c in cells.iterrows():
        liver = "liver" in str(c.primary_site).lower() or strip_name(c.cell_id) in reg.liver_stripped
        if liver:
            cand.append(reg.resolve(release, c.cell_id, "", f"{c.primary_site} / {c.subtype} / {c.sample_type}"))
    out["cells"] += cand
    names = {r["platform_cell_name"] for r in cand}
    sub = inst[inst.cell.isin(names)].copy()
    if "plate_id_extends_rna_plate" in sub and not sub.plate_id_extends_rna_plate.all():
        raise SystemExit(f"{release}: liver instances whose inst_id prefix does not extend rna_plate")

    # Level 3 column ids: confirm that every audited instance is a column of the local matrix
    import h5py
    with h5py.File(gctx_l3, "r") as h:
        col_ids = set(x.decode() for x in h["0/META/COL/id"][:])
        n_rows = h["0/META/ROW/id"].shape[0]
    missing = int((~sub.inst_id.isin(col_ids)).sum())
    out["checks"][release] = dict(level3_matrix=str(gctx_l3), level3_columns=len(col_ids),
                                  level3_rows=n_rows, audited_liver_instances=len(sub),
                                  audited_instances_missing_from_level3=missing)

    veh = sub[sub.pert_type == "ctl_vehicle"].groupby("plate").agg(
        n_vehicle_wells=("well", "nunique"), vehicle_pert_ids=("pert_id", lambda s: ",".join(sorted(set(s)))))
    trt = sub[sub.pert_type == "trt_cp"].copy()
    known = (~trt.dose.isin([-666])) & trt.dose.notna() & (trt.dose_unit.astype(str) != "-666") \
        & trt.time.notna() & (trt.time != -666)
    trt["dose_key"] = np.where(known, trt.dose.map(sig4) + " " + trt.dose_unit.astype(str), "unknown")
    trt["time_key"] = np.where(known, trt.time.map(sig4) + " " + trt.time_unit.astype(str), "unknown")
    trt["design"] = trt.plate.str.replace(r"_X\d+(\.\w+)?_.*$", "", regex=True)
    trt["linked"] = trt.plate.isin(veh.index)

    per_plate = trt.groupby(["cell", "plate"]).agg(n_trt_cp_wells=("well", "nunique")).reset_index()
    per_plate = per_plate.merge(veh, left_on="plate", right_index=True, how="left")
    per_plate["n_vehicle_wells"] = per_plate.n_vehicle_wells.fillna(0).astype(int)
    per_plate["vehicle_pert_ids"] = per_plate.vehicle_pert_ids.fillna("")
    per_plate.insert(0, "platform_release", release)
    per_plate["control_rule"] = "ctl_vehicle wells on the same detection plate"
    out["control_links"].append(per_plate.rename(columns={"cell": "platform_cell_name", "plate": "block_id"}))

    g = trt.groupby(["cell", "pert_id", "dose_key", "time_key"])
    cond = g.agg(n_wells=("inst_id", "nunique"), n_blocks=("plate", "nunique"),
                 n_plate_designs=("design", "nunique"),
                 n_blocks_with_same_plate_vehicle=("plate", lambda s: s[trt.loc[s.index, "linked"]].nunique()),
                 block_ids=("plate", lambda s: "|".join(sorted(set(s))))).reset_index()
    cond.insert(0, "platform_release", release)
    cond["assay_level"] = "L1000 Level 3 (instance)"
    out["conditions"].append(cond.rename(columns={"cell": "platform_cell_name", "pert_id": "compound_id"}))


def load_gse92742(d: Path):
    inst = pd.read_csv(note_file(d / "GSE92742_Broad_LINCS_inst_info.txt", "GSE92742", "instance metadata", "full"),
                       sep="\t", low_memory=False)
    inst = inst.rename(columns=dict(rna_well="well", pert_dose="dose",
                                    pert_dose_unit="dose_unit", pert_time="time",
                                    pert_time_unit="time_unit", cell_id="cell"))
    # rna_plate drops the bead-batch suffix; the detection plate id used by the matrix and by
    # LINCS2020 is the part of inst_id before ':'. Both must agree on the plate they name.
    inst["plate"] = inst.inst_id.str.split(":").str[0]
    # (mixed-cell 'XC' plates of other lineages name rna_plate differently; the audited liver
    # instances are checked in audit_l1000_inst and must all agree)
    inst["plate_id_extends_rna_plate"] = [p.startswith(r) for p, r in zip(inst.plate, inst.rna_plate)]
    cells = pd.read_csv(note_file(d / "GSE92742_Broad_LINCS_cell_info.txt", "GSE92742", "cell metadata", "full"), sep="\t")
    genes = pd.read_csv(note_file(d / "GSE92742_Broad_LINCS_gene_info.txt", "GSE92742", "gene metadata", "full"), sep="\t")
    l3 = note_file(d / "GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx", "GSE92742",
                   "Level 3 matrix", "column/row ids only", hash_it=False)
    return inst, cells, genes, l3


def load_gse70138(d: Path):
    inst = pd.read_csv(note_file(d / "GSE70138_Broad_LINCS_inst_info_2017-03-06.txt.gz", "GSE70138", "instance metadata", "full"),
                       sep="\t", low_memory=False)
    inst = inst.rename(columns=dict(det_plate="plate", det_well="well", pert_dose="dose",
                                    pert_dose_unit="dose_unit", pert_time="time",
                                    pert_time_unit="time_unit", cell_id="cell"))
    cells = pd.read_csv(note_file(d / "GSE70138_Broad_LINCS_cell_info_2017-04-28.txt.gz", "GSE70138", "cell metadata", "full"), sep="\t")
    genes = pd.read_csv(note_file(d / "GSE70138_Broad_LINCS_gene_info_2017-03-06.txt.gz", "GSE70138", "gene metadata", "full"), sep="\t")
    l3 = note_file(d / "GSE70138_Broad_LINCS_Level3_INF_mlr12k_n345976x12328_2017-03-06.gctx", "GSE70138",
                   "Level 3 matrix", "column/row ids only", hash_it=False)
    return inst, cells, genes, l3


# ----------------------------------------------------------------------------- LINCS2020 (Level 5 metadata only)
def audit_lincs2020(d: Path, reg: Registry, out: dict, older_plates: dict[str, set]) -> None:
    release = "LINCS2020"
    cells = pd.read_csv(note_file(d / "cellinfo_beta.txt", release, "cell metadata", "full"), sep="\t", dtype=str).fillna("")
    sig = pd.read_csv(note_file(d / "siginfo_beta.txt", release, "signature metadata (Level 5)", "full"),
                      sep="\t", low_memory=False)
    for f in sorted(d.glob("level5_beta_*.gctx")):
        note_file(f, release, "Level 5 matrix", "not opened", hash_it=False)
    cand = []
    for _, c in cells.iterrows():
        txt = " ".join([c.cell_lineage, c.primary_disease, c.subtype]).lower()
        liver = ("liver" in txt or "hepat" in txt or c.cellosaurus_id in reg.liver_rrid
                 or strip_name(c.cell_iname) in reg.liver_stripped)
        if liver:
            row = reg.resolve(release, c.cell_iname, c.cellosaurus_id,
                              f"{c.cell_lineage} / {c.primary_disease} / {c.subtype} / {c.cell_type}")
            cand.append(row)
    # A derived clone that Cellosaurus knows but the platform gives no accession for is resolved
    # only through the committed lookup table, by exact Cellosaurus name after stripping.
    cel_by_name = {strip_name(n): ac for ac, n in reg.cello.name.items()}
    for row in cand:
        if row["inclusion_status"] == "excluded_unmapped" and strip_name(row["platform_cell_name"]) in cel_by_name:
            ac = cel_by_name[strip_name(row["platform_cell_name"])]
            fixed = reg.resolve(release, row["platform_cell_name"], ac, row["platform_annotation"])
            fixed["mapping_source"] = "exact Cellosaurus name (stripped) in cellosaurus_lookup.tsv"
            row.update(fixed)
    out["cells"] += cand
    names = {r["platform_cell_name"] for r in cand}
    sub = sig[sig.cell_iname.isin(names)].copy()
    sub["plates"] = sub.det_plates.astype(str).str.split("|")
    veh_plates = set(p for ps in sub[sub.pert_type == "ctl_vehicle"].plates for p in ps)
    trt = sub[sub.pert_type == "trt_cp"].copy()
    known = trt.pert_dose.notna() & trt.pert_time.notna() & trt.pert_dose_unit.notna()
    trt["dose_key"] = np.where(known, trt.pert_dose.map(lambda x: sig4(x) if pd.notna(x) else "") + " " + trt.pert_dose_unit.astype(str), "unknown")
    trt["time_key"] = np.where(known, trt.pert_time.map(lambda x: sig4(x) if pd.notna(x) else "") + " " + trt.pert_time_unit.astype(str), "unknown")
    rows = []
    for (cell, pid, dk, tk), g in trt.groupby(["cell_iname", "pert_id", "dose_key", "time_key"]):
        plates = sorted(set(p for ps in g.plates for p in ps))
        designs = {re.sub(r"_X\d+(\.\w+)?_.*$", "", p) for p in plates}
        rows.append(dict(platform_release=release, platform_cell_name=cell, compound_id=pid,
                         dose_key=dk, time_key=tk, n_wells=int(g.nsample.sum()), n_blocks=len(plates),
                         n_plate_designs=len(designs),
                         n_blocks_with_same_plate_vehicle=sum(p in veh_plates for p in plates),
                         block_ids="|".join(plates),
                         assay_level="L1000 Level 5 signature metadata (Level 3 not on server)"))
    out["conditions"].append(pd.DataFrame(rows))
    # labelled diagnostic only: the platform's own `nearest_dose` column instead of the exact dose
    hccn = {r["platform_cell_name"]: r["family_id"] for r in cand
            if r["is_adult_hcc"] or r["inclusion_status"] == "derived_clone_of_family_no_depmap_model"}
    dg = trt[trt.cell_iname.isin(hccn) & trt.nearest_dose.notna() & trt.pert_time.notna()].copy()
    dg["family_id"] = dg.cell_iname.map(hccn)
    drows = []
    for (nd, t), g in dg.groupby(["nearest_dose", "pert_time"]):
        fpd = g.groupby("pert_id").family_id.nunique()
        drows.append(dict(platform_release=release, nearest_dose_uM=nd, time_h=t,
                          n_adult_hcc_families=g.family_id.nunique(), n_drugs_ge1_family=int((fpd >= 1).sum()),
                          n_drugs_ge2_families=int((fpd >= 2).sum()), n_drugs_ge3_families=int((fpd >= 3).sum()),
                          note="DIAGNOSTIC: platform nearest_dose bin; not used by any gate"))
    out["lincs2020_nearest_dose"] = pd.DataFrame(drows).sort_values(
        ["n_adult_hcc_families", "n_drugs_ge3_families", "n_drugs_ge2_families"], ascending=False)
    cl = defaultdict(lambda: [0, 0])
    for cell, ps in zip(trt.cell_iname, trt.plates):
        for p in ps:
            cl[(cell, p)][0] += 1
    out["control_links"].append(pd.DataFrame([dict(
        platform_release=release, platform_cell_name=c, block_id=p, n_trt_cp_wells=n[0],
        n_vehicle_wells=int(p in veh_plates), vehicle_pert_ids="ctl_vehicle signature exists" if p in veh_plates else "",
        control_rule="ctl_vehicle SIGNATURE lists the same detection plate (instance file not on server; "
                     "n_trt_cp_wells counts signatures, n_vehicle_wells is 0/1)") for (c, p), n in cl.items()]))
    # overlap of detection plates with the two GEO releases, per liver cell
    ov = []
    for cell, g in trt.groupby("cell_iname"):
        plates = set(p for ps in g.plates for p in ps)
        ov.append(dict(platform_cell_name=cell, n_trt_cp_signatures=len(g), n_det_plates=len(plates),
                       **{f"plates_also_in_{k}": len(plates & v) for k, v in older_plates.items()}))
    out["release_overlap"] = ov


# ----------------------------------------------------------------------------- Tahoe
def audit_tahoe(path: Path, reg: Registry, out: dict) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    release = "Tahoe-100M"
    note_file(path, release, "per-cell obs metadata", "7 columns, all row groups")
    f = pq.ParquetFile(path)
    cols = ["plate", "sample", "drugname_drugconc", "drug", "cell_line", "cell_name", "pass_filter"]
    parts = []
    for i in range(f.metadata.num_row_groups):
        t = f.read_row_group(i, columns=cols)
        t = t.set_column(t.schema.get_field_index("pass_filter"), "pass_filter",
                         pa.compute.cast(pa.compute.equal(pa.compute.cast(t["pass_filter"], pa.string()), "full"), pa.int64()))
        for c in ["cell_line", "cell_name"]:
            t = t.set_column(t.schema.get_field_index(c), c, pa.compute.cast(t[c], pa.string()))
        parts.append(t.group_by(["plate", "sample", "drugname_drugconc", "drug", "cell_line", "cell_name"])
                     .aggregate([("pass_filter", "count"), ("pass_filter", "sum")]).to_pandas())
    w = pd.concat(parts).groupby(["plate", "sample", "drugname_drugconc", "drug", "cell_line", "cell_name"],
                                 as_index=False).sum()
    w = w.rename(columns={"pass_filter_count": "n_cells", "pass_filter_sum": "n_cells_pass_full"})

    def parse(s: str):
        v = ast.literal_eval(s)
        if len(v) != 1:
            return ("combination", "unknown")
        name, dose, unit = v[0]
        return (name, f"{sig4(dose)} {unit}")
    parsed = {s: parse(s) for s in w.drugname_drugconc.unique()}
    w["dose_key"] = w.drugname_drugconc.map(lambda s: parsed[s][1])
    w["is_control"] = w.drug == "DMSO_TF"
    w["block"] = w.plate + ":" + w["sample"]
    out["tahoe_well_table_rows"] = len(w)

    lines = w.groupby(["cell_line", "cell_name"], as_index=False).n_cells.sum()
    cand = []
    for _, c in lines.iterrows():
        if c.cell_line in reg.liver_rrid or strip_name(c.cell_name) in reg.liver_stripped:
            cand.append(reg.resolve(release, c.cell_name, c.cell_line, f"Tahoe cell_line={c.cell_line}"))
    out["cells"] += cand
    out["tahoe_n_lines"] = int(lines.cell_line.nunique())
    lin = []
    for _, c in lines.iterrows():
        d = reg.by_rrid.get(c.cell_line, [])
        lin.append(dict(cell_line=c.cell_line, cell_name=c.cell_name, n_cells=int(c.n_cells),
                        depmap_model_id=d[0].ModelID if len(d) == 1 else "",
                        depmap_lineage=str(d[0].OncotreeLineage) if len(d) == 1 else "unmapped",
                        depmap_subtype=str(d[0].OncotreeSubtype) if len(d) == 1 else ""))
    out["tahoe_lines"] = pd.DataFrame(lin).sort_values(["depmap_lineage", "cell_name"])
    names = {r["platform_cell_name"] for r in cand}
    hcc = {r["platform_cell_name"]: r["family_id"] for r in cand if r["is_adult_hcc"]}

    ctl = w[w.is_control]
    ctl_cells = ctl.groupby(["cell_name", "plate"]).agg(n_vehicle_wells=("sample", "nunique"), n_vehicle_cells=("n_cells", "sum"))
    trt = w[~w.is_control]
    sub = trt[trt.cell_name.isin(names)].copy()
    sub["linked"] = [(c, p) in ctl_cells.index for c, p in zip(sub.cell_name, sub.plate)]
    cond = sub.groupby(["cell_name", "drug", "dose_key"]).agg(
        n_wells=("block", "nunique"), n_blocks=("block", "nunique"), n_plate_designs=("plate", "nunique"),
        n_blocks_with_same_plate_vehicle=("linked", "sum"), n_cells=("n_cells", "sum"),
        block_ids=("block", lambda s: "|".join(sorted(set(s))))).reset_index()
    cond.insert(0, "platform_release", release)
    cond["time_key"] = "not_in_metadata (dataset card: 24 h)"
    cond["assay_level"] = "scRNA raw counts, per treatment well"
    out["conditions"].append(cond.rename(columns={"cell_name": "platform_cell_name", "drug": "compound_id"}))
    pp = sub.groupby(["cell_name", "plate"]).agg(n_trt_cp_wells=("sample", "nunique")).reset_index()
    pp = pp.merge(ctl_cells.reset_index(), on=["cell_name", "plate"], how="left")
    pp["n_vehicle_wells"] = pp.n_vehicle_wells.fillna(0).astype(int)
    pp["vehicle_pert_ids"] = "DMSO_TF"
    pp.insert(0, "platform_release", release)
    pp["control_rule"] = "DMSO_TF wells of the same cell line on the same plate"
    out["control_links"].append(pp.rename(columns={"cell_name": "platform_cell_name", "plate": "block_id"})
                                .drop(columns=["n_vehicle_cells"]))

    # --- strict LOCO: is the same drug+dose observed in a well that does not contain the target?
    wells_of = trt.groupby(["drug", "dose_key"]).block.agg(lambda s: frozenset(s))
    lines_in_well = trt.groupby("block").cell_name.agg(lambda s: frozenset(s))
    n_lines_in_well = lines_in_well.map(len)
    out["tahoe_design"] = dict(
        n_conditions=int(len(wells_of)),
        wells_per_condition={str(k): int(v) for k, v in wells_of.map(len).value_counts().sort_index().items()},
        lines_per_treatment_well=dict(min=int(n_lines_in_well.min()), median=float(n_lines_in_well.median()),
                                      max=int(n_lines_in_well.max())),
        n_treatment_wells=int(len(lines_in_well)))
    loco = []
    for dose, gcond in wells_of.groupby(level="dose_key"):
        for t, fam in hcc.items():
            n_eval = n_any = n_hcc = 0
            src_hcc = {n for n, fm in hcc.items() if fm != fam}
            for (drug, _), wells in gcond.items():
                with_t = {b for b in wells if t in lines_in_well[b]}
                if not with_t:
                    continue
                n_eval += 1
                rest = wells - with_t
                n_any += any(len(lines_in_well[b]) > 0 for b in rest)
                n_hcc += any(lines_in_well[b] & src_hcc for b in rest)
            loco.append(dict(platform_release=release, dose_key=dose, target_cell=t, target_family=fam,
                             n_target_conditions=n_eval,
                             n_with_same_drug_dose_in_a_well_without_target_any_source_line=n_any,
                             n_with_same_drug_dose_in_a_well_without_target_adult_hcc_source=n_hcc))
    out["tahoe_loco"] = pd.DataFrame(loco)


# ----------------------------------------------------------------------------- genotype
def audit_genotype(depmap: Path, cells: pd.DataFrame) -> pd.DataFrame:
    prof = pd.read_csv(note_file(depmap / "OmicsProfiles.csv", "DepMap_25Q3", "sequencing profiles", "full"))
    models = sorted(set(cells.depmap_model_id) - {""})
    dna = prof[prof.DataType.isin(["wes", "wgs"])].groupby("ModelID").DataType.agg(lambda s: ",".join(sorted(set(s))))
    use = ["ModelID", "IsDefaultEntryForModel", "HugoSymbol", "ProteinChange", "VariantInfo", "VepImpact",
           "LikelyLoF", "TumorSuppressorHighImpact", "OncogeneHighImpact", "Hotspot", "AF"]
    keep = []
    path = note_file(depmap / "OmicsSomaticMutations.csv", "DepMap_25Q3", "somatic mutation MAF",
                     "rows of audited models x 8 driver genes")
    for ch in pd.read_csv(path, usecols=use, chunksize=500_000, low_memory=False):
        keep.append(ch[ch.ModelID.isin(models) & ch.HugoSymbol.isin(DRIVERS)])
    mut = pd.concat(keep)
    in_maf = set()
    for ch in pd.read_csv(path, usecols=["ModelID"], chunksize=2_000_000):
        in_maf |= set(ch.ModelID.unique()) & set(models)
    mut = mut[mut.IsDefaultEntryForModel.astype(str) == "Yes"]

    def truthy(x) -> bool:
        return str(x).strip().lower() in ("true", "1", "1.0", "yes")
    rows = []
    for mid in models:
        for gene in DRIVERS:
            v = mut[(mut.ModelID == mid) & (mut.HugoSymbol == gene)]
            v = v[v.ProteinChange.notna() | v.VepImpact.isin(["HIGH", "MODERATE"])]
            states = set()
            for _, r in v.iterrows():
                if truthy(r.LikelyLoF) or truthy(r.TumorSuppressorHighImpact):
                    states.add("confirmed_functional_LoF")
                elif truthy(r.OncogeneHighImpact) or truthy(r.Hotspot):
                    states.add("confirmed_activating_or_GoF")
                else:
                    states.add("other_or_unresolved_coding_variant")
            if not states:
                states = {"no_qualifying_variant_reported_with_assay_available"
                          if (mid in dna.index and mid in in_maf) else "unmeasured_or_unresolved"}
            rows.append(dict(depmap_model_id=mid, gene=gene, state="+".join(sorted(states)),
                             variants=";".join(f"{r.ProteinChange}[{r.VariantInfo}]" for _, r in v.iterrows()),
                             dna_profiles=dna.get(mid, ""), model_has_rows_in_maf=mid in in_maf))
    g = pd.DataFrame(rows)
    meta = cells.drop_duplicates("depmap_model_id").set_index("depmap_model_id")
    for c in ["family_id", "cellosaurus_id", "is_adult_hcc", "inclusion_status"]:
        g[c] = g.depmap_model_id.map(meta[c])
    g["platforms"] = g.depmap_model_id.map(cells.groupby("depmap_model_id").platform_release.agg(lambda s: ",".join(sorted(set(s)))))
    return g


# ----------------------------------------------------------------------------- coverage and gate
def coverage(cond: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    key = ["platform_release", "platform_cell_name"]
    c = cond.merge(cells[key + ["family_id", "is_adult_hcc", "inclusion_status"]], on=key, how="left")
    c = c[(c.dose_key != "unknown") & (c.time_key != "unknown") & (c.n_blocks_with_same_plate_vehicle > 0)]
    rows = []
    for (rel, dk, tk), g in c.groupby(["platform_release", "dose_key", "time_key"]):
        h = g[g.is_adult_hcc | (g.inclusion_status == "derived_clone_of_family_no_depmap_model")]
        fam_per_drug = h.groupby("compound_id").family_id.nunique()
        held = g[g.is_adult_hcc | g.inclusion_status.isin(["derived_clone_of_family_no_depmap_model", "held_ambiguous"])]
        rows.append(dict(
            platform_release=rel, dose_key=dk, time_key=tk,
            n_adult_hcc_families=h.family_id.nunique(),
            adult_hcc_families=",".join(sorted(set(h.family_id))),
            n_drugs_ge1_family=int((fam_per_drug >= 1).sum()),
            n_drugs_ge2_families=int((fam_per_drug >= 2).sum()),
            n_drugs_ge3_families=int((fam_per_drug >= 3).sum()),
            n_hcc_condition_rows=len(h),
            n_hcc_blocks=len(set(b for s in h.block_ids for b in s.split("|"))),
            n_hcc_conditions_with_ge2_blocks=int((h.n_blocks_with_same_plate_vehicle >= 2).sum()),
            n_hcc_conditions_with_ge2_plate_designs=int((h.n_plate_designs >= 2).sum()),
            sensitivity_n_families_if_held_cells_counted=held.family_id.nunique(),
            context_n_liver_candidate_families_any_status=g[g.family_id != ""].family_id.nunique()))
    cov = pd.DataFrame(rows)
    num = lambda s: s.str.extract(r"^([0-9.eE+-]+)")[0].astype(float)
    cov["_t"], cov["_d"] = num(cov.time_key).fillna(np.inf), num(cov.dose_key).fillna(np.inf)
    # policy order: families -> drugs in >=3 families -> blocks; tie-break lower time, lower dose
    cov = cov.sort_values(["platform_release", "n_adult_hcc_families", "n_drugs_ge3_families", "n_hcc_blocks", "_t", "_d"],
                          ascending=[True, False, False, False, True, True]).drop(columns=["_t", "_d"])
    cov["policy_rank_in_release"] = cov.groupby("platform_release").cumcount() + 1
    return cov


def loco_support(cond: pd.DataFrame, cov: pd.DataFrame) -> pd.DataFrame:
    """At each release's policy-selected (dose, time): for every outer target family, how many of its
    drugs are observed in >=1 / >=2 / >=3 OTHER adult-HCC families (control-linked conditions only)."""
    rows = []
    ok = cond[(cond.is_adult_hcc | (cond.inclusion_status == "derived_clone_of_family_no_depmap_model"))
              & (cond.n_blocks_with_same_plate_vehicle > 0)]
    for rel, g in cov.groupby("platform_release"):
        b = g.iloc[0]
        c = ok[(ok.platform_release == rel) & (ok.dose_key == b.dose_key) & (ok.time_key == b.time_key)]
        fams = sorted(set(c.family_id))
        by = {f: set(c[c.family_id == f].compound_id) for f in fams}
        rep_ok = {f: set(c[(c.family_id == f) & (c.n_blocks_with_same_plate_vehicle >= 2)].compound_id) for f in fams}
        for t in fams:
            n_src = {d: sum(d in by[f] for f in fams if f != t) for d in by[t]}
            rows.append(dict(platform_release=rel, dose_key=b.dose_key, time_key=b.time_key, target_family=t,
                             n_source_families=len(fams) - 1, n_target_drugs=len(by[t]),
                             n_target_drugs_with_ge2_blocks=len(rep_ok[t]),
                             n_drugs_in_ge1_source_family=sum(v >= 1 for v in n_src.values()),
                             n_drugs_in_ge2_source_families=sum(v >= 2 for v in n_src.values()),
                             n_drugs_in_ge3_source_families=sum(v >= 3 for v in n_src.values())))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lincs", default="/mnt/S1/sdata/pubdata/LINCS")
    ap.add_argument("--depmap", default="/mnt/S1/sdata/pubdata/DepMap/25Q3")
    ap.add_argument("--tahoe", default="/mnt/S1/sdata/pubdata/Virtual_Cell/Tahoe-100M/metadata/obs_metadata.parquet")
    ap.add_argument("--out", default="manifests/genotype_film_v3")
    a = ap.parse_args()
    outd = Path(a.out)
    reg = Registry(Path(a.depmap), outd / "cellosaurus_lookup.tsv")
    out: dict = dict(cells=[], conditions=[], control_links=[], checks={})

    older_plates = {}
    for rel, loader in [("GSE92742", load_gse92742), ("GSE70138", load_gse70138)]:
        inst, cells, genes, l3 = loader(Path(a.lincs) / rel)
        older_plates[rel] = set(inst.plate.unique())
        audit_l1000_inst(rel, inst, cells, reg, l3, out)
        out["checks"][rel].update(
            n_instances=len(inst), n_cells_in_release=int(cells.cell_id.nunique()),
            n_landmark_genes=int((genes.pr_is_lm == 1).sum()), n_genes_in_matrix_metadata=len(genes),
            pert_types={k: int(v) for k, v in inst.pert_type.value_counts().items()})
        del inst
    both = older_plates["GSE92742"] & older_plates["GSE70138"]
    out["checks"]["plates_shared_GSE92742_GSE70138"] = len(both)
    audit_lincs2020(Path(a.lincs) / "LINCS2020", reg, out, older_plates)
    audit_tahoe(Path(a.tahoe), reg, out)

    cells = pd.DataFrame(out["cells"])
    cond = pd.concat(out["conditions"], ignore_index=True)
    ctrl = pd.concat(out["control_links"], ignore_index=True)
    geno = audit_genotype(Path(a.depmap), cells)
    cov = coverage(cond, cells)

    key = ["platform_release", "platform_cell_name"]
    cond = cond.merge(cells[key + ["family_id", "is_adult_hcc", "inclusion_status"]], on=key, how="left")
    rep = cond[cond.is_adult_hcc | (cond.inclusion_status == "derived_clone_of_family_no_depmap_model")]
    rep = rep[key + ["family_id", "compound_id", "dose_key", "time_key", "n_blocks", "n_plate_designs",
                     "n_blocks_with_same_plate_vehicle", "block_ids"]].copy()
    rep["replicate_type"] = np.where(rep.platform_release == "Tahoe-100M", "treatment well",
                                     "detection plate (replicate plates of one design share a batch)")
    cond_out = cond.drop(columns=["block_ids"])
    cond_out = cond_out[cond_out.inclusion_status != "excluded_unmapped"] if False else cond_out
    summ = cond.groupby(key + ["inclusion_status", "family_id"], dropna=False).agg(
        n_conditions=("compound_id", "size"), n_compounds=("compound_id", "nunique"),
        n_conditions_control_linked=("n_blocks_with_same_plate_vehicle", lambda s: int((s > 0).sum())),
        n_dose_time_pairs=("dose_key", lambda s: len(set(zip(s, cond.loc[s.index, "time_key"]))))).reset_index()

    outd.mkdir(parents=True, exist_ok=True)
    w = dict(sep="\t", index=False)
    cells.to_csv(outd / "cell_registry.tsv", **w)
    cond_out.to_csv(outd / "condition_inventory.tsv", **w)
    ctrl.to_csv(outd / "control_links.tsv", **w)
    rep.to_csv(outd / "replicate_registry.tsv", **w)
    geno.to_csv(outd / "genotype_support.tsv", **w)
    cov.to_csv(outd / "coverage_by_dose_time.tsv", **w)
    out["tahoe_loco"].to_csv(outd / "tahoe_strict_loco.tsv", **w)
    out["tahoe_lines"].to_csv(outd / "tahoe_all_lines_lineage.tsv", **w)
    out["lincs2020_nearest_dose"].to_csv(outd / "lincs2020_nearest_dose_diagnostic.tsv", **w)
    loco = loco_support(cond, cov)
    loco.to_csv(outd / "loco_support_at_selected_dose_time.tsv", **w)
    summ.to_csv(outd / "cell_condition_summary.tsv", **w)
    pd.DataFrame(INVENTORY).to_csv(outd / "source_inventory.tsv", **w)

    best = {}
    for rel, g in cov.groupby("platform_release"):
        b = g.iloc[0]
        best[rel] = {k: (int(v) if isinstance(v, (np.integer,)) else v) for k, v in b.items()}
    families_any = {}
    for rel, g in cond[(cond.is_adult_hcc | (cond.inclusion_status == "derived_clone_of_family_no_depmap_model"))
                       & (cond.n_blocks_with_same_plate_vehicle > 0)].groupby("platform_release"):
        families_any[rel] = sorted(set(g.family_id))
    summary = dict(checks=out["checks"], release_overlap_LINCS2020=out["release_overlap"],
                   tahoe_design=out["tahoe_design"], tahoe_n_lines=out["tahoe_n_lines"],
                   best_dose_time_per_release=best,
                   adult_hcc_families_with_any_control_linked_condition=families_any,
                   outputs={p.name: sha256(p) for p in sorted(outd.glob("*.tsv"))})
    (outd / "phase0_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))
    print(cells[["platform_release", "platform_cell_name", "cellosaurus_id", "depmap_model_id", "family_id",
                 "is_adult_hcc", "inclusion_status"]].to_string())
    print(summ.to_string())
    print(cov.groupby("platform_release").head(8).drop(columns=["adult_hcc_families"]).to_string())
    print(out["tahoe_loco"].to_string())
    print(loco.to_string())
    print(out["lincs2020_nearest_dose"].head(6).to_string())
    print(out["tahoe_lines"].depmap_lineage.value_counts().to_string())


if __name__ == "__main__":
    main()
