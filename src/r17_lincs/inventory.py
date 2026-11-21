"""Inventory liver-line chemical perturbations from metadata on the server."""

import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r13_optimization.lincs_transfer import clean, structural_keys
from src.r15_lincs.data import EXCLUDE_LINES, STUDIES, asset, numbers, measures, DOSE, TIME, table


ROOT = Path("/mnt/S1/sdata/pubdata/LINCS")
CMAP = "CMap2020"
SOURCES = ("B1","R8","R13","P2","P3","R15","R16")


def normal(values):
    return values.str.upper().str.replace(r"[^A-Z0-9]","",regex=True)


def liver_cells(path, cmap):
    cells = table(path)
    key, site = ("cell_iname","cell_lineage") if cmap else ("cell_id","primary_site")
    disease = "primary_disease" if cmap else "subtype"
    available = (cells[site].str.contains("liver|hepati",case=False,regex=True)
                 | cells[disease].str.contains("hepatoma|hepatocellular|hepatoblastoma",case=False,regex=True)
                 | normal(cells[key]).isin(EXCLUDE_LINES | {"HEP3B217","LI7","SNU761","HCCLM3","MHCC97H","MHCC97L","SKHEP1"}))
    return cells.loc[available,[key,site,disease]].rename(columns={key:"cell",site:"site",disease:"annotation"})


def ids(values):
    return values.str.strip().str.upper()


def coordinates(wells, kind):
    cols = [kind+"_plate",kind+"_well"]
    if not set(cols)<=set(wells):
        return pd.MultiIndex.from_arrays([[""]*len(wells),[""]*len(wells)])
    plate, well = (ids(wells[k]) for k in cols)
    well = well.str.replace(r"^([A-Z]+)([0-9])$",r"\g<1>0\2",regex=True)
    return pd.MultiIndex.from_arrays([plate,well])


def register_footprint(wells):
    physical = coordinates(wells,"rna").append(coordinates(wells,"det")).unique()
    physical = physical[physical.get_level_values(0).str.len()>0]
    return set(ids(wells.inst_id)),physical


def in_footprint(wells, foot):
    aliases, physical = foot
    return ids(wells.inst_id).isin(aliases).to_numpy() | coordinates(wells,"rna").isin(physical) | coordinates(wells,"det").isin(physical)


def geo_frames(study):
    cell_meta = asset(ROOT,study,"cell_info")
    cells = liver_cells(cell_meta,cmap=False)
    inst_path, sig_path = asset(ROOT,study,"inst_info"),asset(ROOT,study,"sig_info")
    keep = {"inst_id","rna_plate","rna_well","det_plate","det_well","cell_id","pert_type"}
    inst = table(inst_path,usecols=lambda c:c in keep)
    inst = inst.loc[inst.cell_id.isin(cells.cell) & inst.pert_type.eq("trt_cp")].drop_duplicates("inst_id")
    sig_keep = ["sig_id","cell_id","pert_id","pert_type","distil_id","pert_idose","pert_itime"]
    sig = table(sig_path,usecols=sig_keep)
    sig = sig.loc[sig.cell_id.isin(cells.cell)&sig.pert_type.eq("trt_cp")].rename(
        columns={"distil_id":"members","cell_id":"cell"}).copy()
    sig["dose"] = numbers(sig.pert_idose,DOSE)
    sig["hours"] = numbers(sig.pert_itime,TIME)
    metric_file = asset(ROOT,study,"sig_metrics")
    metrics = table(metric_file,usecols=["sig_id","distil_cc_q75","distil_nsample"])
    sig = sig.merge(metrics,on="sig_id",how="left",validate="one_to_one")
    sig["nsample"] = pd.to_numeric(sig.distil_nsample,errors="coerce")
    sig["qc"] = sig.nsample.ge(2) & pd.to_numeric(sig.distil_cc_q75,errors="coerce").ge(.2)
    kind = "rna" if "rna_plate" in inst else "det"
    inst["physical_plate"],inst["physical_well"] = inst[kind+"_plate"],inst[kind+"_well"]
    return cells,inst,sig,dict(cells=cell_meta,inst=inst_path,sig=sig_path,quality=metric_file)


def cmap_frames(base):
    origin = ROOT/"LINCS2020"
    inst_path = base/"vcell_liver_transigen_adaptation_b1/assets/raw/cmap2020/instinfo_beta.txt"
    cells = liver_cells(origin/"cellinfo_beta.txt",cmap=True)
    columns = ["sample_id","cell_iname","pert_id","pert_type","rna_plate","rna_well","det_plate","det_well",
               "nearest_dose","pert_dose_unit","pert_time","pert_time_unit","qc_pass"]
    inst = pd.concat([chunk.loc[chunk.cell_iname.isin(cells.cell)&chunk.pert_type.eq("trt_cp")]
        for chunk in pd.read_csv(inst_path,sep="\t",dtype=str,keep_default_na=False,usecols=columns,chunksize=160000)])
    inst = inst.rename(columns={"sample_id":"inst_id"}).drop_duplicates("inst_id")
    inst["physical_plate"],inst["physical_well"] = inst.rna_plate,inst.rna_well
    sig_path = origin/"siginfo_beta.txt"
    columns = ["sig_id","cell_iname","pert_id","pert_type","distil_ids","pert_dose","pert_dose_unit",
               "pert_time","pert_time_unit","nsample","cc_q75","qc_pass"]
    sig = pd.concat([chunk.loc[chunk.cell_iname.isin(cells.cell)&chunk.pert_type.eq("trt_cp")]
        for chunk in pd.read_csv(sig_path,sep="\t",dtype=str,keep_default_na=False,usecols=columns,chunksize=160000)])
    sig = sig.rename(columns={"distil_ids":"members","cell_iname":"cell"})
    sig["dose"] = measures(sig.pert_dose,sig.pert_dose_unit,DOSE)
    sig["hours"] = measures(sig.pert_time,sig.pert_time_unit,TIME)
    sig["nsample"] = pd.to_numeric(sig.nsample,errors="coerce")
    sig["qc"] = sig.nsample.ge(2)&pd.to_numeric(sig.cc_q75,errors="coerce").ge(.2)&sig.qc_pass.eq("1")
    return cells,inst,sig,dict(cells=origin/"cellinfo_beta.txt",inst=inst_path,sig=sig_path)


def previous(base, cmap):
    paths = {}
    b1 = base/"vcell_liver_transigen_adaptation_b1/assets/processed/cmap2020_v1"
    paths["B1"] = b1/"rows.tsv"
    if sha256(paths["B1"])!=read_json(b1/"build_manifest.json")["files"]["rows.tsv"]["sha256"]:
        raise ValueError("Prior B1 rows changed")
    registry = table(paths["B1"]).rename(columns={"cell":"cell_iname"})
    registry = registry.loc[registry.cell_iname.isin(cmap.cell_iname)]
    keep = cmap.nearest_dose.eq("10") & cmap.qc_pass.eq("1") & cmap.pert_time.eq("24") & cmap.pert_time_unit.eq("h") & cmap.pert_dose_unit.eq("uM")
    b1_used = cmap.loc[keep].merge(registry,on=["cell_iname","pert_id","det_plate"],validate="many_to_one")
    if len(b1_used)!=registry.n_trt_wells.astype(int).sum():
        raise ValueError("Failed to reproduce B1 hepatic well counts")
    paths["R8"] = base/"vcell_liver_replicate_calibrated_r8/assets/processed/store_v1/units.tsv"
    r8 = table(paths["R8"],usecols=["cell","drug_group","plates"]).rename(columns={"cell":"cell_iname"})
    r8 = r8.loc[r8.cell_iname.isin(cmap.cell_iname)]
    r8 = r8.assign(det_plate=r8.plates.str.split("|")).explode("det_plate")
    r8_used = b1_used.merge(r8.drop(columns="plates"),on=["cell_iname","drug_group","det_plate"],validate="many_to_one")
    r8_used = pd.concat((r8_used,b1_used.loc[b1_used.role.eq("target")])).drop_duplicates("inst_id")
    paths["R13"] = base/"vcell-r13-lincs-cid-study.7NGzdyc8/donor/identity.tsv"
    old_brd = table(paths["R13"])
    r13_used = b1_used.loc[b1_used.role.eq("source") & b1_used.pert_id.isin(old_brd.loc[old_brd.retained.eq("True"),"pert_id"])]
    paths["P23_match"] = base/"vcell_liver_p123/v1/lincs/match_audit.tsv"
    paths["P23"] = base/"vcell_liver_p123/v1/lincs/profile_metadata.tsv"
    matches = table(paths["P23_match"])
    successful = matches.loc[matches.decision.eq("matched"),["study","pert_id","cohort"]].drop_duplicates()
    p23 = table(paths["P23"]).merge(successful,on=["study","pert_id"],validate="many_to_many")
    footprints = {name:register_footprint(w) for name,w in (("B1",b1_used),("R8",r8_used),("R13",r13_used))}
    signature = defaultdict(set)
    for name,cohort in (("P2","tahoe"),("P3","heparg")):
        for study, use in p23.loc[p23.cohort.eq(cohort)].groupby("study"):
            signature[study,name].update(use.sig_id)

    r15 = base/"vcell_liver_r15_lincs/v1/cache"
    r16 = base/"vcell_liver_r16_lincs/v1/audit"
    old = []
    for study in STUDIES:
        p = r15/study/"target_rows.tsv"
        paths[f"R15_{study}"] = p
        target = table(p)
        signature[study,"R15"].update(target.sig_id)
        geo_alias = target.rename(columns={"rna_plate":"det_plate","rna_well":"det_well"}) if study=="GSE70138" else target
        old.append(geo_alias)
    paths["R16"] = r16/"target_rows.tsv"
    r16_used = table(paths["R16"])
    signature[CMAP,"R16"].update(r16_used.sig_id)
    footprints["R15"] = register_footprint(pd.concat(old,ignore_index=True).fillna(""))
    footprints["R16"] = register_footprint(r16_used)
    return footprints,signature,paths


def repeated(links, distinct_position=False):
    use = links.loc[~links.parent_key.str.startswith("UNMAPPED:") & links.physical_plate.ne("")]
    if distinct_position:
        use = use.loc[use.physical_well.ne("")]
    groups = use.groupby(["parent_key","hours","dose"])
    n = groups.physical_plate.nunique()
    keep = n.ge(2)
    if distinct_position:
        keep &= groups.physical_well.nunique().ge(2)
    return n.loc[keep].reset_index().parent_key.nunique()


def counts(study, cells, inst, sig, identity, exposure, signatures):
    sig = sig.drop_duplicates("sig_id").copy()
    sig["parent"] = sig.pert_id.map(identity).fillna("")
    sig["parent_key"] = sig.parent.where(sig.parent.ne(""),"UNMAPPED:"+sig.pert_id)
    cols = ["sig_id","members"]
    links = sig[cols].assign(inst_id=lambda x:x.members.str.split("|")).explode("inst_id").merge(
        inst,on="inst_id",how="inner",validate="many_to_one").drop_duplicates(["sig_id","inst_id"])
    for source in SOURCES:
        p = signatures[study,source]
        overlapping = links.loc[in_footprint(links,exposure[source]),"sig_id"] if source in exposure else []
        sig[source] = sig.sig_id.isin(p) | sig.sig_id.isin(overlapping)
    sig["any_prior"] = sig[list(SOURCES)].any(axis=1)
    keep = ["sig_id","parent_key","hours","dose"]
    links = links.merge(sig[keep],on="sig_id",validate="many_to_one")
    if study==CMAP:
        links = links.loc[links.qc_pass.eq("1")]
    summaries, support = [], []
    for elem in cells.itertuples(index=False):
        available = sig.loc[sig.cell.eq(elem.cell)]
        candidate = available.loc[available.qc & ~available.any_prior & available.parent.ne("")]
        for (hour,dose), part in candidate.groupby(["hours","dose"],dropna=False):
            wells = links.loc[links.sig_id.isin(part.sig_id)]
            support.append(dict(study=study,cell=elem.cell,hours=hour,dose_uM=dose,
                canonical_groups=part.parent.nunique(),signatures=len(part),linked_wells=wells.inst_id.nunique(),
                linked_plates=wells.physical_plate.nunique(),distinct_plate_groups=repeated(wells),
                distinct_plate_position_groups=repeated(wells,True)))
        for scope, selected in (
                ("all",available),("6h_5-20uM",available.loc[available.hours.eq(6)&available.dose.between(5,20)]),
                ("24h_5-20uM",available.loc[available.hours.eq(24)&available.dose.between(5,20)])):
            qc = selected.loc[selected.qc]
            new = qc.loc[~qc.any_prior]
            wells = links.loc[links.sig_id.isin(selected.sig_id)]
            good = wells.loc[wells.sig_id.isin(qc.sig_id)]
            novel = good.loc[good.sig_id.isin(new.sig_id)]
            summaries.append(dict(study=study,cell=elem.cell,site=elem.site,annotation=elem.annotation,scope=scope,
               pert_ids=selected.pert_id.nunique(),canonical_groups=selected.loc[selected.parent.ne(""),"parent"].nunique(),
               signatures=len(selected),nrep_ge2_signatures=int(selected.nsample.ge(2).sum()),
               linked_wells=wells.inst_id.nunique(),linked_plates=wells.loc[wells.physical_plate.ne(""),"physical_plate"].nunique(),
               different_plate_groups=repeated(wells),qc_signatures=len(qc),qc_groups=qc.loc[qc.parent.ne(""),"parent"].nunique(),
               qc_linked_plates=good.physical_plate.nunique(),qc_different_plate_groups=repeated(good),
               qc_no_recorded_overlap_signatures=len(new),
               qc_no_recorded_overlap_groups=new.loc[new.parent.ne(""),"parent"].nunique(),
               qc_no_recorded_overlap_different_plate_groups=repeated(novel),
               qc_no_recorded_overlap_different_plate_position_groups=repeated(novel,True),
               **{f"{source}_overlap_signatures":int(selected[source].sum()) for source in SOURCES}))
    return summaries,support


def main(args):
    out = args.out
    out.mkdir(parents=True,exist_ok=False)
    cmap = cmap_frames(args.base)
    footprint, signatures, paths = previous(args.base,cmap[1])
    hashes = {name:sha256(path) for name,path in paths.items()}
    results, support = [], []
    prior = args.base/"vcell_liver_r15_lincs/v1/audit"
    for study in (*STUDIES,CMAP):
        cells,inst,sig,inputs = cmap if study==CMAP else geo_frames(study)
        if study==CMAP:
            inputs["chemistry"] = ROOT/"LINCS2020/compoundinfo_beta.txt"
            mapping = table(inputs["chemistry"],usecols=["pert_id","canonical_smiles"]).drop_duplicates("pert_id")
            mapping = mapping.loc[mapping.pert_id.isin(sig.pert_id)]
            mapping["parent"] = mapping.canonical_smiles.map(lambda x:structural_keys(clean(x))[1])
        else:
            inputs["chemistry"] = prior/study/"identities.tsv"
            if sha256(inputs["chemistry"])!=read_json(inputs["chemistry"].parent/"receipt.json")["files"]["identities.tsv"]:
                raise ValueError("Registered GEO identities changed")
            mapping = table(inputs["chemistry"],usecols=["pert_id","parent"])
            mapping["parent"] = mapping.parent.where(~mapping.parent.str.startswith("UNRESOLVED:"),"")
        structure = mapping.set_index("pert_id",verify_integrity=True).parent
        current, new = counts(study,cells,inst,sig,structure,footprint,signatures)
        results.extend(current)
        support.extend(new)
        hashes.update({f"{study}_{name}":sha256(path) for name,path in inputs.items()})
        print(f"Inventoried {study}: {len(cells)} liver labels and {len(sig)} compound signatures",flush=True)
    pd.DataFrame(results).to_csv(out/"lincs.tsv",sep="\t",index=False)
    pd.DataFrame(support).to_csv(out/"qc_no_recorded_overlap_conditions.tsv",sep="\t",index=False,na_rep="NA")
    write_json(out/"receipt.json",dict(code_sha256=sha256(__file__),metadata_sha256=hashes,
                    files={p.name:sha256(p) for p in out.iterdir()},expression_read=False))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    main(parser.parse_args())
