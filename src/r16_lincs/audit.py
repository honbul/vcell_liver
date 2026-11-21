"""Audit exposure, coverage and physical assay position for CMap 2020."""

import argparse
from pathlib import Path
from zlib import crc32

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r13_optimization.lincs_transfer import clean, structural_keys
from src.r15_lincs.cache import picks_by_parent, sort_dose
from src.r15_lincs.data import (DOSE, TIME, STUDIES, asset, genes, hepatic_lines,
                               landmark_rows, measures, table)
from src.r15_lincs.position_audit import physical_effect, position_scores, repeat_layout, vehicle_pairs

CELLS = ("HUH7", "JHH5", "JHH7", "HEPG2")
RNA = ["rna_plate", "rna_well"]
DET = ["det_plate", "det_well"]
INST = ["sample_id", *RNA, *DET, "cell_iname", "pert_id", "pert_type", "qc_pass",
        "pert_time", "pert_time_unit", "pert_dose", "pert_dose_unit", "nearest_dose", "cmap_name"]
SIG = ["sig_id", "distil_ids", "qc_pass", "nsample", "cc_q75", "is_exemplar_sig",
       "cell_iname", "pert_id", "pert_type", "pert_time", "pert_time_unit", "pert_dose", "pert_dose_unit"]
LINCS = Path("/mnt/S1/sdata/pubdata/LINCS")


def token(x):
    return x.str.strip().str.upper().str.replace(r"^([A-Z]+)([0-9])$", r"\g<1>0\2", regex=True)


def keys(rows, columns):
    return pd.MultiIndex.from_arrays([token(rows[c]) for c in columns])


def add_units(rows):
    return rows.assign(dose=measures(rows.pert_dose, rows.pert_dose_unit, DOSE),
                       hours=measures(rows.pert_time, rows.pert_time_unit, TIME))


def geo_exposure(prior):
    sig_ids, constituent, rna_positions, det_positions, samples = set(), set(), [], [], set()
    files = {}
    for study in STUDIES:
        for kind in ("inst_info", "sig_info"):
            path = asset(LINCS, study, kind)
            files[f"{study}_{kind}"] = sha256(path)
        inst = table(asset(LINCS, study, "inst_info"))
        samples.update(token(inst.inst_id))
        if set(RNA) <= set(inst.columns):
            rna_positions.append(keys(inst,RNA))
        if set(DET) <= set(inst.columns):
            det_positions.append(keys(inst,DET))
        sig = table(asset(LINCS,study,"sig_info"), usecols=["sig_id","distil_id"])
        sig_ids.update(token(sig.sig_id))
        constituent.update(token(sig.distil_id.str.split("|").explode()))
        # R15 source and target measurements come only from the two GEO releases.
        # Its exact target aliases are checked again, separately from release filtering.
        cache = prior/"cache"/study
        files[f"R15_{study}_target_index"] = sha256(cache/"target_rows.tsv")
    both = rna_positions + det_positions
    rna = both[0].append(both[1:]).unique()
    return sig_ids, constituent | samples, rna, files


def b1_scope(base, raw, mapping):
    b1 = base/"vcell_liver_transigen_adaptation_b1/assets/processed/cmap2020_v1"
    manifest = read_json(b1/"build_manifest.json")
    rows_path = b1/"rows.tsv"
    if sha256(rows_path) != manifest["files"]["rows.tsv"]["sha256"]:
        raise ValueError("B1 matrix row registry changed")
    old = table(rows_path)
    old = old.loc[old.cell.isin(CELLS)].rename(columns={"cell":"cell_iname"})
    columns = ["cell_iname","pert_id","det_plate"]
    allowed = raw.loc[raw.nearest_dose.eq("10") & raw.qc_pass.eq("1") & raw.pert_type.eq("trt_cp")
                      & raw.hours.eq(24) & raw.pert_dose_unit.eq("uM") & raw.cell_iname.isin(CELLS)]
    exposure = allowed.merge(old,on=columns,how="inner",validate="many_to_one")
    if len(exposure) != old.n_trt_wells.astype(int).sum():
        raise ValueError("CMap B1 target well reconstruction failed")
    r8_path = base/"vcell_liver_replicate_calibrated_r8/assets/processed/store_v1/units.tsv"
    units = table(r8_path)
    units = units.loc[units.cell.isin(CELLS)]
    units = units[["cell","drug_group","plates"]].assign(det_plate=lambda x:x.plates.str.split("|")).explode("det_plate")
    r8 = exposure.merge(units.rename(columns={"cell":"cell_iname"}).drop(columns="plates"),
                        on=["cell_iname","drug_group","det_plate"],validate="many_to_one")
    r8 = pd.concat((r8,exposure.loc[exposure.role.eq("target")])).drop_duplicates("sample_id")
    return {label:(set(x.sample_id),set(zip(x.cell_iname,x.pert_id.map(mapping))))
            for label,x in (("B1",exposure),("R8",r8))}, {
                "B1_rows":sha256(rows_path),"R8_units":sha256(r8_path)}


def controls_index(vehicles, raw):
    values = {}
    for (cell,plate), group in vehicles.groupby(["cell_iname","rna_plate"]):
        group = sorted(group.itertuples(index=False), key=lambda r:(crc32(r.sample_id.encode()),r.sample_id))
        if len(group) >= 2:
            ix = [r.expr_index for r in group]
            cut = len(ix)//2
            values[cell,plate] = (raw[ix[:cut]].mean(0),raw[ix[cut:]].mean(0))
    registry = {k:i for i,k in enumerate(values)}
    return registry,np.array(list(values.values()),dtype=np.float32)


def collect(args):
    args.out.mkdir(parents=True, exist_ok=False)
    public = LINCS/"LINCS2020"
    raw_dir = args.base/"vcell_liver_transigen_adaptation_b1/assets/raw/cmap2020"
    paths = dict(inst=raw_dir/"instinfo_beta.txt", sig=public/"siginfo_beta.txt",
                 compounds=public/"compoundinfo_beta.txt", lines=public/"cellinfo_beta.txt")
    geo_sig, geo_well, geo_plate_well, old_files = geo_exposure(args.prior)
    sig = add_units(table(paths["sig"], usecols=SIG))
    sig = sig.loc[sig.pert_type.eq("trt_cp") & sig.hours.eq(24) & sig.dose.between(5,20)].copy()
    # Source lines are chosen from registered nonhepatic cells using support metadata.
    meta = table(paths["lines"])
    excluded = set.union(*(hepatic_lines(LINCS,study) for study in STUDIES))
    excluded.update(meta.loc[meta.cell_lineage.str.contains("liver|hepat",case=False),"cell_iname"])
    source_lines = set(meta.cell_iname)-excluded
    sig = sig.loc[sig.cell_iname.isin(set(CELLS)|source_lines)].copy()
    sig["quality"] = (pd.to_numeric(sig.cc_q75,errors="coerce").ge(.2)
                      & pd.to_numeric(sig.nsample,errors="coerce").ge(2) & sig.qc_pass.eq("1"))
    sig["geo_signature"] = token(sig.sig_id).isin(geo_sig)
    sig["is_exemplar"] = sig.is_exemplar_sig
    identity = table(paths["compounds"],usecols=["pert_id","canonical_smiles"]).drop_duplicates("pert_id")
    identity = identity.loc[identity.pert_id.isin(sig.pert_id)].copy()
    identity["parent"] = identity.canonical_smiles.map(lambda s:structural_keys(clean(s))[1])
    mapping = identity.set_index("pert_id").parent
    sig["parent"] = sig.pert_id.map(mapping).fillna("")
    sig = sig.loc[sig.parent.ne("")].drop_duplicates("sig_id").reset_index(drop=True)
    rna_candidates = sig.loc[sig.cell_iname.isin(CELLS)]
    wanted = set(rna_candidates.parent)
    sig = sig.loc[sig.parent.isin(wanted)]
    sigs = sig[["sig_id","distil_ids"]].assign(sample_id=lambda x:x.distil_ids.str.split("|")).explode("sample_id")
    sigs = sigs.drop_duplicates(["sig_id","sample_id"])
    ids = set(sigs.sample_id)
    raw = []
    for chunk in pd.read_csv(paths["inst"], sep="\t", dtype=str, keep_default_na=False,
                             usecols=INST, chunksize=150000):
        relevant = chunk.sample_id.isin(ids)
        hepatic = chunk.cell_iname.isin(CELLS) & chunk.pert_type.eq("trt_cp")
        control = (chunk.pert_type.eq("ctl_vehicle") & chunk.qc_pass.eq("1")
                   & (chunk.pert_id.str.upper().eq("DMSO") | chunk.cmap_name.str.upper().eq("DMSO")))
        raw.append(chunk.loc[relevant|hepatic|control])
    raw = add_units(pd.concat(raw,ignore_index=True)).drop_duplicates("sample_id")
    raw["rna_well"] = token(raw.rna_well)
    raw["det_well"] = token(raw.det_well)

    history, history_files = b1_scope(args.base,raw,mapping)
    old_files.update(history_files)
    raw["geo_well"] = (token(raw.sample_id).isin(geo_well) | keys(raw,RNA).isin(geo_plate_well)
                       | keys(raw,DET).isin(geo_plate_well))
    trt = raw.loc[raw.pert_type.eq("trt_cp")].copy()
    flags = sigs.merge(trt[["sample_id","geo_well"]],on="sample_id",how="left",validate="many_to_one")
    any_overlap = set(flags.loc[flags.geo_well.eq(True),"sig_id"]) | set(
        flags.loc[token(flags.sample_id).isin(geo_well),"sig_id"])
    sig["geo_constituent"] = sig.sig_id.isin(any_overlap)
    joined = sig.merge(sigs.drop(columns="distil_ids"),on="sig_id",validate="one_to_many").merge(
        trt[["sample_id","geo_well","rna_plate","rna_well","qc_pass","pert_id","cell_iname","dose","hours"]],
        on=["sample_id","pert_id","cell_iname"],validate="many_to_one",suffixes=("","_well"))
    hit = joined.rna_plate.ne("") & joined.rna_well.str.fullmatch(r"[A-Z]+\d\d")
    joined = joined.loc[hit & joined.hours.eq(joined.hours_well) & joined.dose.eq(joined.dose_well)]
    raw_clean = joined.loc[~joined.geo_well & ~joined.geo_signature & ~joined.geo_constituent
                           & joined.qc_pass_well.eq("1")].copy()
    r15_target_samples = set()
    for study in STUDIES:
        registered = table(args.prior/"cache"/study/"target_rows.tsv")
        r15_target_samples.update(token(registered.inst_id))
    if token(raw_clean.sample_id).isin(r15_target_samples).any():
        raise ValueError("A CMap confirmation well is an R15/R16 registered target")
    for role, (wells, parents) in history.items():
        raw_clean[f"{role}_exact"] = raw_clean.sample_id.isin(wells)
        raw_clean[f"{role}_parent"] = pd.MultiIndex.from_frame(raw_clean[["cell_iname","parent"]]).isin(parents)
    raw_clean["prior_exact"] = raw_clean.B1_exact | raw_clean.R8_exact
    raw_clean["prior_parent"] = raw_clean.B1_parent | raw_clean.R8_parent
    raw_clean["cell_id"] = raw_clean.cell_iname
    target = picks_by_parent(raw_clean.loc[raw_clean.cell_iname.isin(CELLS)],["cell_iname","parent","hours"])
    target = target.sort_values(["quality","sig_id"],ascending=[False,True],kind="stable").drop_duplicates(
        ["cell_iname","parent","sample_id"])
    good_parents = set(target.loc[target.quality,"parent"])
    inputs = raw_clean.loc[~raw_clean.cell_iname.isin(CELLS) & raw_clean.parent.isin(good_parents)]
    by_line = inputs.groupby("cell_iname").parent.nunique().sort_index().sort_values(ascending=False,kind="stable")
    source = by_line.index[:6].tolist()
    inputs = sort_dose(inputs.loc[inputs.cell_iname.isin(source)])
    best = inputs.drop_duplicates(["parent","cell_iname"])["sig_id"]
    inputs = inputs.loc[inputs.sig_id.isin(best)].drop_duplicates(["parent","cell_iname","sample_id"])

    # Same RNA-plate well, not detection plate, defines the technical context.
    selected = pd.concat((target,inputs),ignore_index=True)
    vehicles = raw.loc[raw.pert_type.eq("ctl_vehicle") & raw.qc_pass.eq("1") & ~raw.geo_well
                       & raw.rna_well.str.fullmatch(r"[A-Z]+\d\d")].copy()
    vehicles = vehicles.loc[pd.MultiIndex.from_frame(vehicles[["cell_iname","rna_plate"]]).isin(
        pd.MultiIndex.from_frame(selected[["cell_iname","rna_plate"]]))]
    vehicles = vehicles.sort_values("sample_id").drop_duplicates(["cell_iname",*RNA])
    vehicles = vehicles.reset_index(drop=True).assign(expr_index=lambda x:np.arange(len(x)))
    land = genes(LINCS,STUDIES[0])
    ctr_file, = raw_dir.glob("level3_beta_ctl_*.gctx")
    trt_file, = raw_dir.glob("level3_beta_trt_cp_*.gctx")
    Xc = landmark_rows(ctr_file,vehicles.sample_id,land.pr_gene_id)
    ctrl_index, ctl = controls_index(vehicles,Xc)
    for frame in (target,inputs):
        frame["ctrl_index"] = [ctrl_index.get(k,-1) for k in zip(frame.cell_iname,frame.rna_plate)]
    target = target.loc[target.ctrl_index.ge(0)].reset_index(drop=True)
    inputs = inputs.loc[inputs.ctrl_index.ge(0)].reset_index(drop=True)
    print("Reduced to",len(target),"target wells and",len(inputs),"source wells",flush=True)
    Xt = landmark_rows(trt_file,target.sample_id,land.pr_gene_id)
    support = set(inputs.parent)

    inventory, pos, layouts, vehicle_report = [], [], [], []
    for cell in CELLS:
        j = joined.loc[joined.cell_iname.eq(cell)]
        clean_rows = raw_clean.loc[raw_clean.cell_iname.eq(cell)]
        t = target.loc[target.cell_iname.eq(cell)]
        quality = t.loc[t.quality]
        orig = rna_candidates.loc[rna_candidates.cell_iname.eq(cell)]
        inventory.append(dict(cell=cell, signature_parents=orig.parent.nunique(), signatures=len(orig),
            linked_parents=j.parent.nunique(), linked_wells=j.sample_id.nunique(),
            overlapping_geo_signatures=orig.loc[orig.geo_signature,"sig_id"].nunique(),
            overlapping_geo_constituents=j.loc[j.geo_constituent,"sig_id"].nunique(),
            overlapping_geo_wells=j.loc[j.geo_well,"sample_id"].nunique(),
            remaining_parents=clean_rows.parent.nunique(), remaining_wells=clean_rows.sample_id.nunique(),
            dmsosplit_parents=t.parent.nunique(), dmsosplit_plates=t.rna_plate.nunique(),
            qc_parents=quality.parent.nunique(), qc_wells=len(quality), qc_plates=quality.rna_plate.nunique(),
            B_support=quality.parent.isin(support).groupby(quality.parent).any().sum(),
            overlap_R15_R16_target_well=0,
            qc_B1_exact_parents=quality.loc[quality.B1_exact,"parent"].nunique(),
            qc_R8_exact_parents=quality.loc[quality.R8_exact,"parent"].nunique(),
            qc_B1_parent=quality.loc[quality.B1_parent,"parent"].nunique(),
            qc_R8_parent=quality.loc[quality.R8_parent,"parent"].nunique()))
        qc = quality.assign(pert_type="trt_cp")
        control = vehicles.loc[vehicles.cell_iname.eq(cell) & vehicles.rna_plate.isin(t.rna_plate)]
        if len(t):
            vehicle_report.append(dict(cell=cell,**vehicle_pairs(control,Xc[control.expr_index])))
            combined = pd.concat((qc,control),ignore_index=True)
            values = np.concatenate((Xt[quality.index],Xc[control.expr_index]))
            physical = physical_effect(combined,values,3)
            if physical is not None:
                pos.extend(dict(cell=cell,**metric) for metric in position_scores(*physical))
        layouts.append(dict(cell=cell,**repeat_layout(qc)))
    pd.DataFrame(inventory).to_csv(args.out/"inventory.tsv",sep="\t",index=False)
    pd.DataFrame(pos).to_csv(args.out/"position.tsv",sep="\t",index=False)
    pd.DataFrame(layouts).to_csv(args.out/"repeat-layout.tsv",sep="\t",index=False)
    write_json(args.out/"vehicles.json",vehicle_report)
    target = target.rename(columns={"sample_id":"inst_id"})
    target.insert(0,"index", np.arange(len(target)))
    columns = ["index","inst_id","cell_id","sig_id","pert_id","parent","hours","dose",*RNA,
               "quality","is_exemplar","prior_exact","prior_parent","B1_exact","B1_parent","R8_exact","R8_parent","ctrl_index"]
    target[columns].to_csv(args.out/"target_rows.tsv",sep="\t",index=False)
    land.to_csv(args.out/"landmarks.tsv",sep="\t",index=False)
    np.save(args.out/"target_wells.npy",Xt,allow_pickle=False)
    np.save(args.out/"vehicle.npy",ctl,allow_pickle=False)
    inputs.to_csv(args.out/"source_wells.tsv",sep="\t",index=False)
    write_json(args.out/"receipt.json",dict(
        code_sha256=sha256(__file__),metadata_sha256={**{a:sha256(p) for a,p in paths.items()},**old_files},
        matrix={k:{"path":str(p),"bytes":p.stat().st_size} for k,p in (("trt",trt_file),("ctl",ctr_file))},
        nonhepatic_source=source,
        checked_geo_sample_aliases=len(geo_well),checked_geo_positions=len(geo_plate_well),
        target_in_R15_R16=False,files={p.name:sha256(p) for p in args.out.iterdir()}))


def stage(args):
    from shutil import copyfile

    audit = read_json(args.audit/"receipt.json")
    args.out.mkdir(parents=True,exist_ok=False)
    for key in ("target_rows.tsv","target_wells.npy","vehicle.npy","landmarks.tsv","source_wells.tsv"):
        if sha256(args.audit/key) != audit["files"][key]:
            raise ValueError(f"The audited asset changed: {key}")
        if key != "source_wells.tsv":
            copyfile(args.audit/key,args.out/key)
    src = table(args.audit/"source_wells.tsv")
    # The expression reader receives only the audited nonhepatic source list.
    path = Path(audit["matrix"]["trt"]["path"])
    if path.stat().st_size != audit["matrix"]["trt"]["bytes"]:
        raise ValueError("The original CMap Level 3 matrix changed")
    land = pd.read_csv(args.out/"landmarks.tsv",sep="\t")
    target = pd.read_csv(args.out/"target_rows.tsv",sep="\t",keep_default_na=False)
    target_parents = sorted(target.parent.unique())
    pd.DataFrame({"parent":target_parents}).to_csv(args.out/"chemicals.tsv",sep="\t",index=False)
    if len(src):
        ctl = np.load(args.out/"vehicle.npy",allow_pickle=False)
        X = landmark_rows(path,src.sample_id,land.pr_gene_id)-ctl[src.ctrl_index.astype(int)].mean(axis=1)
        by_line = src.groupby(["parent","cell_iname"],sort=True).indices
        meta = pd.DataFrame(list(by_line),columns=["parent","cell_iname"])
        averages = np.stack([X[ix].mean(0) for ix in by_line.values()])
        by_parent = meta.groupby("parent",sort=True).indices
        transfer = np.stack([averages[ix].mean(0) for ix in by_parent.values()])
    else:
        by_parent = {}
        transfer = np.empty((0,len(land)),dtype=np.float32)
    pd.DataFrame({"parent":list(by_parent)}).to_csv(args.out/"transfer_rows.tsv",sep="\t",index=False)
    np.save(args.out/"transfer.npy",transfer,allow_pickle=False)
    write_json(args.out/"receipt.json",dict(audit_sha256=sha256(args.audit/"receipt.json"),
        protocol_sha256=sha256(args.protocol),files={p.name:sha256(p) for p in args.out.iterdir()}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("base","prior","out","audit","protocol"):
        parser.add_argument("--"+key,type=Path,required=key in ("base","prior","out"))
    args = parser.parse_args()
    if args.audit:
        stage(args)
    else:
        collect(args)


if __name__=="__main__":
    main()
