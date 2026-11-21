"""Register study DMSO wells before fitting and stage split control means."""

import argparse
from pathlib import Path
from zlib import crc32

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r15_lincs.data import STUDIES, asset, landmark_rows, table
from src.r15_lincs.design import fold_id


GEO = "GSE92742"
ROLES = "ABC"


def checked(path):
    stamp = read_json(path.parent/"receipt.json")
    if sha256(path)!=stamp["files"][path.name]:
        raise ValueError(f"Changed upstream artifact: {path}")
    return path


def eligibility(cache):
    target = pd.read_csv(checked(cache/GEO/"target_rows.tsv"),sep="\t",keep_default_na=False)
    target = target.loc[target.cell_id.eq("HEPG2") & target.hours.eq(6) & target.quality & ~target.prior_exact]
    transfer = pd.concat([pd.read_csv(checked(cache/study/"transfer_rows.tsv"),sep="\t") for study in STUDIES])
    return target, target.loc[target.parent.isin(transfer.loc[transfer.hours.eq(6),"parent"])]


def register(args):
    args.out.mkdir(parents=True,exist_ok=False)
    cache = args.cache
    targets = eligibility(cache)
    source = asset(args.lincs,GEO,"inst_info")
    inst = table(source,usecols=["inst_id","rna_plate","rna_well","pert_id","pert_iname","pert_type"])
    use = (inst.pert_type.eq("ctl_vehicle") & inst.rna_plate.isin(targets[0].rna_plate) &
           (inst.pert_iname.str.upper().eq("DMSO") | inst.pert_id.str.upper().eq("DMSO")))
    controls = inst.loc[use,["inst_id","rna_plate","rna_well"]].copy()
    if controls.inst_id.duplicated().any() or controls.duplicated(["rna_plate","rna_well"]).any():
        raise ValueError("A physical DMSO well has more than one alias")
    controls["crc"] = controls.inst_id.map(lambda v:crc32(v.encode()))
    controls = controls.sort_values(["rna_plate","crc","inst_id"]).reset_index(drop=True)
    sizes = controls.groupby("rna_plate").size()
    good = set(sizes.loc[sizes.ge(3)].index)
    plates = controls.loc[controls.rna_plate.isin(good)].copy()
    plates["role"] = plates.groupby("rna_plate").cumcount().mod(3).map(dict(enumerate(ROLES)))
    plates.drop(columns="crc").to_csv(args.out/"assignments.tsv",sep="\t",index=False)
    per_role = plates.groupby(["rna_plate","role"]).size().unstack().reindex(columns=list(ROLES)).astype(int)
    coverage = sizes.to_frame("vehicles").join(per_role).fillna(0).astype(int).reset_index()
    coverage.to_csv(args.out/"plate_support.tsv",sep="\t",index=False)
    summary = []
    by_fold = []
    for scenario,target in zip("AB",targets,strict=True):
        kept = target.loc[target.rna_plate.isin(good)]
        summary.append(dict(scenario=scenario,original_parents=target.parent.nunique(),parents=kept.parent.nunique(),
                            lost_parents=target.parent.nunique()-kept.parent.nunique(),
                            original_wells=len(target),wells=len(kept),lost_wells=len(target)-len(kept),
                            original_plates=target.rna_plate.nunique(),plates=kept.rna_plate.nunique(),
                            lost_plates=target.rna_plate.nunique()-kept.rna_plate.nunique(),
                            minimum_well_per_plate=int(sizes.reindex(kept.rna_plate.unique()).min()),
                            min_DMSO_per_role=int(per_role.reindex(kept.rna_plate.unique()).min().min())))
        assigned = kept.parent.map(lambda p:fold_id(p,"R15-outer",3))
        for fold in range(3):
            fit = kept.loc[assigned.ne(fold)]
            selection = fit.parent.map(lambda p:fold_id(p,f"R15-inner-{fold}",5)).eq(0)
            groups = [fit.loc[~selection],fit.loc[selection],kept.loc[assigned.eq(fold)]]
            by_fold.append(dict(scenario=scenario,fold=fold,**{
                f"{name}_{unit}":value for name, frame in zip(("inner_fit","select","score"),groups,strict=True)
                for unit,value in (("parents",frame.parent.nunique()),("wells",len(frame)))}))
    pd.DataFrame(summary).to_csv(args.out/"support.tsv",sep="\t",index=False)
    pd.DataFrame(by_fold).to_csv(args.out/"fold_support.tsv",sep="\t",index=False)
    write_json(args.out/"receipt.json",dict(code_sha256=sha256(__file__),instance_sha256=sha256(source),
        old_target_receipt=sha256(cache/GEO/"receipt.json"),files={p.name:sha256(p) for p in args.out.iterdir()}))


def stage(args):
    audit = read_json(args.audit/"receipt.json")
    for name,digest in audit["files"].items():
        if sha256(args.audit/name)!=digest:
            raise ValueError("The registered DMSO split changed")
    land = pd.read_csv(checked(args.cache/GEO/"landmarks.tsv"),sep="\t")
    split = table(args.audit/"assignments.tsv")
    matrix_file, = (args.lincs/GEO).glob("*Level3*.gctx")
    values = landmark_rows(matrix_file,split.inst_id,land.pr_gene_id)
    plates = sorted(split.rna_plate.unique())
    idx = split.groupby(["rna_plate","role"]).indices
    matrix = np.stack([[values[idx[plate,role]].mean(0) for role in ROLES] for plate in plates])
    args.out.mkdir(parents=True,exist_ok=False)
    pd.DataFrame({"rna_plate":plates}).to_csv(args.out/"vehicles.tsv",sep="\t",index=False)
    np.save(args.out/"vehicles.npy",matrix,allow_pickle=False)
    write_json(args.out/"receipt.json",dict(protocol_sha256=sha256(args.protocol),audit_sha256=sha256(args.audit/"receipt.json"),
        files={p.name:sha256(p) for p in args.out.iterdir()},matrix_path=str(matrix_file),matrix_bytes=matrix_file.stat().st_size))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("register","stage"))
    for name in ("cache","lincs","out","audit","protocol"):
        parser.add_argument("--"+name,type=Path,required=name in ("cache","lincs","out"))
    arguments = parser.parse_args()
    (register if arguments.action=="register" else stage)(arguments)
