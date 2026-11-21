"""Stage only eligible liver target wells and selected non-hepatic measurements."""

import argparse
from pathlib import Path
from zlib import crc32

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json

from .data import LIVER, STUDIES, asset, genes, hepatic_lines, instances, landmark_rows, signatures, table


def quality(root, study, sig):
    metrics = table(asset(root, study, "sig_metrics"))
    keys = ["sig_id", "distil_cc_q75", "distil_nsample"]
    if "is_exemplar" in metrics:
        keys.append("is_exemplar")
    sig = sig.merge(metrics[keys], on="sig_id", validate="one_to_one", how="left")
    if "is_exemplar" not in sig:
        sig["is_exemplar"] = ""
    sig["quality"] = (pd.to_numeric(sig.distil_cc_q75, errors="coerce").ge(.2)
                      & pd.to_numeric(sig.distil_nsample, errors="coerce").ge(2))
    return sig


def map_wells(frame, gen, ids):
    x = frame[["sig_id", "distil_id"]].assign(inst_id=lambda a: a.distil_id.str.split("|")).explode("inst_id")
    return x.merge(gen, on="inst_id", validate="many_to_one", how="inner")[ids]


def sort_dose(frame):
    return frame.assign(distance=np.abs(np.log2(frame.dose/10))).sort_values(["distance", "sig_id"], kind="stable")


def picks_by_parent(frame, keys):
    ordered = sort_dose(frame)
    window = ordered.groupby(keys).distance.transform("min")
    return ordered.loc[ordered.distance.eq(window)].reset_index(drop=True)


def build(args):
    # Both audits have to be complete before extracting the research cache.
    for folder in (*STUDIES, "exposure-verified"):
        if not (args.audit/folder/"receipt.json").is_file():
            raise ValueError("The exposure and position audits must finish before cache staging")
    study = args.study
    out = args.out / study
    out.mkdir(parents=True, exist_ok=False)
    sig = signatures(args.root, study)
    inst = instances(args.root, study)
    land = genes(args.root, study)
    chem = table(args.audit/study/"identities.tsv")
    mapping = chem.set_index("pert_id").parent
    liver = sig.loc[sig.cell_id.isin(LIVER) & sig.pert_type.eq("trt_cp") &
                    sig.dose.between(5,20) & sig.hours.isin([6,24])].copy()
    liver["parent"] = liver.pert_id.map(mapping)
    liver = liver.loc[liver.parent.notna() & ~liver.parent.str.startswith("UNRESOLVED:",na=True)]
    liver = quality(args.root, study, liver)
    exposure = table(args.audit/"exposure-verified/signature_exposure.tsv")
    exposure = exposure.loc[exposure.study.eq(study)]
    flags = [c for c in exposure if c.endswith(("_exact","_parent"))]
    liver = liver.merge(exposure[["sig_id", *flags]], on="sig_id", validate="one_to_one")
    liver["prior_exact"] = liver[[c for c in flags if c.endswith("_exact")]].eq("True").any(axis=1)
    liver["prior_parent"] = liver[[c for c in flags if c.endswith("_parent")]].eq("True").any(axis=1)
    target = picks_by_parent(liver, ["parent","cell_id","hours"])
    target = target[["sig_id","distil_id","parent","cell_id","hours","dose","quality","is_exemplar",
                     "distil_cc_q75","prior_exact","prior_parent"]]

    p2_receipt = read_json(args.p23/"receipt.json")
    if (p2_receipt["target_inputs"]["entities"] != sha256(args.tahoe_entities)
            or p2_receipt["target_inputs"]["split"] != sha256(args.tahoe_split)
            or p2_receipt["files"]["match_audit.tsv"] != sha256(args.p23/"match_audit.tsv")):
        raise ValueError("Tahoe identity ledger differs from the frozen P2 contract")
    audit = table(args.p23/"match_audit.tsv")
    audit = audit.loc[audit.cohort.eq("tahoe") & audit.study.eq(study)].copy()
    audit["parent"] = audit.pert_id.map(mapping)
    positive = audit.loc[audit.decision.eq("matched") & audit.parent.notna()]
    if positive.groupby("parent").parent_key.nunique().gt(1).any():
        raise ValueError("One LINCS canonical parent maps to multiple Tahoe groups")
    parents_tahoe = positive.set_index("parent").parent_key.to_dict()
    blocked = set(audit.parent.dropna())
    audit.to_csv(out/"tahoe_identity.tsv",sep="\t",index=False)
    target["tahoe_group"] = target.parent.map(parents_tahoe).fillna("")
    target["tahoe_exclude"] = target.parent.isin(blocked)

    candidates = sig.loc[~sig.cell_id.isin(hepatic_lines(args.root, study)) & sig.pert_type.eq("trt_cp") & sig.dose.between(5,20)
                         & sig.hours.isin([6,24])].copy()
    candidates["parent"] = candidates.pert_id.map(mapping)
    parent_wanted = set(target.parent) | set(parents_tahoe)
    candidates = candidates.loc[candidates.parent.isin(parent_wanted)]
    source_lines, present = {}, []
    for hour, rows in candidates.groupby("hours"):
        count = rows.groupby("cell_id").parent.nunique().sort_index().sort_values(ascending=False,kind="stable")
        source_lines[str(int(hour))] = count.index[:6].tolist()
        present.append(rows.loc[rows.cell_id.isin(source_lines[str(int(hour))])])
    present = pd.concat(present,ignore_index=True)
    present = sort_dose(present).drop_duplicates(["parent","cell_id","hours"])
    present = present[["sig_id","distil_id","parent","cell_id","hours","dose"]]
    instances_cols = ["inst_id","rna_plate","rna_well","pert_id"]
    assigned = map_wells(pd.concat((target,present)), inst[instances_cols], ["sig_id",*instances_cols])
    if assigned.duplicated(["sig_id","inst_id"]).any():
        raise ValueError("A signature repeats an instance")
    plates = pd.Index(assigned.rna_plate.unique())
    vehicle = inst.loc[inst.pert_type.eq("ctl_vehicle") & inst.rna_plate.isin(plates)].copy()
    vehicle = vehicle.loc[vehicle.pert_iname.str.upper().eq("DMSO") | vehicle.pert_id.str.upper().eq("DMSO")]
    observed = pd.concat((inst.loc[inst.inst_id.isin(assigned.inst_id)],vehicle)).drop_duplicates("inst_id")
    matrix_file, = (args.root/study).glob("*Level3*.gctx")
    X = landmark_rows(matrix_file, observed.inst_id, land.pr_gene_id)
    print(f"{study} loaded {len(X)} selected Level3 wells",flush=True)
    observed = observed.assign(idx=np.arange(len(observed))).set_index("inst_id",verify_integrity=True)
    veh = observed.loc[observed.pert_type.eq("ctl_vehicle")].reset_index()
    controls = {}
    for name, group in veh.groupby("rna_plate"):
        positions = sorted(group.itertuples(index=False),key=lambda v:(crc32(v.inst_id.encode()),v.inst_id))
        raw = X[[s.idx for s in positions]]
        if len(raw) >= 2:
            cut = len(raw)//2
            controls[name] = (raw[:cut].mean(0),raw[cut:].mean(0),len(raw))
    controls_order = sorted(controls)
    position = {p:i for i,p in enumerate(controls_order)}
    assigned["ctrl_index"] = assigned.rna_plate.map(position)
    assigned["expr_index"] = assigned.inst_id.map(observed.idx)
    assigned = assigned.dropna(subset=["ctrl_index","expr_index"]).astype({"ctrl_index":int,"expr_index":int})
    control = np.array([(controls[k][0],controls[k][1]) for k in controls_order],dtype=np.float32)
    target_rows = target.merge(assigned,on="sig_id",validate="one_to_many").drop(columns="distil_id")
    target_rows = target_rows.sort_values(["quality","prior_exact","sig_id"],ascending=[False,False,True]).drop_duplicates(
        ["parent","cell_id","hours","inst_id"],keep="first")
    target_rows = target_rows.sort_values(["hours","cell_id","parent","rna_plate","rna_well","sig_id"]).reset_index(drop=True)
    target_y = X[target_rows.pop("expr_index").to_numpy()]
    target_rows.insert(0,"index",np.arange(len(target_rows)))
    target_rows.to_csv(out/"target_rows.tsv",sep="\t",index=False)
    np.save(out/"target_wells.npy",target_y,allow_pickle=False)
    np.save(out/"vehicle.npy",control,allow_pickle=False)

    observed_source = present.merge(assigned,on="sig_id",validate="one_to_many")
    priors = X[observed_source.expr_index] - control[observed_source.ctrl_index].mean(axis=1)
    grouped = observed_source.groupby(["parent","hours","cell_id"],sort=True).indices
    sequences = pd.DataFrame(list(grouped),columns=["parent","hours","cell_id"])
    by_line = np.stack([priors[idx].mean(axis=0) for idx in grouped.values()])
    panels = sequences.groupby(["parent","hours"],sort=True).indices
    inputs = pd.DataFrame(list(panels),columns=["parent","hours"])
    inputs["lines"] = [len(idx) for idx in panels.values()]
    inputs["tahoe_group"] = inputs.parent.map(parents_tahoe).fillna("")
    inputs.to_csv(out/"transfer_rows.tsv",sep="\t",index=False)
    np.save(out/"transfer.npy",np.stack([by_line[idx].mean(0) for idx in panels.values()]),allow_pickle=False)
    land.to_csv(out/"landmarks.tsv",sep="\t",index=False)

    chem = chem.loc[chem.parent.isin(parent_wanted)].copy()
    chem["tahoe_group"] = chem.parent.map(parents_tahoe).fillna("")
    chem.to_csv(out/"chemicals.tsv",sep="\t",index=False)
    write_json(out/"receipt.json",dict(study=study,code_sha256=sha256(__file__),protocol_sha256=sha256(args.protocol),
        audit_sha256={k:sha256(args.audit/k/"receipt.json") for k in (*STUDIES,"exposure-verified")},
        matrix={"path":str(matrix_file),"bytes":matrix_file.stat().st_size},
        tahoe_inputs={"entities":sha256(args.tahoe_entities),"split":sha256(args.tahoe_split),
                      "identity":sha256(args.p23/"receipt.json")},
        transfer_cells=source_lines,controls=len(controls),target_wells=len(target_rows),
        files={p.name:sha256(p) for p in sorted(out.iterdir())}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("root","audit","protocol","tahoe-entities","tahoe-split","p23","out"):
        parser.add_argument(f"--{key}",type=Path,required=True)
    parser.add_argument("--study",choices=STUDIES,required=True)
    build(parser.parse_args())
