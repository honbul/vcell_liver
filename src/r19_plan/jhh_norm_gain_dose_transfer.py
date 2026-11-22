"""Test fit-only amplitude gain on position-disjoint JHH-5 measurements below 10 µM."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r19_plan.hcc_lincs_24h_access import CONTEXT, register
from src.r19_plan.hcc_lincs_24h_support import evaluate_support
from src.r19_plan.hcc_lincs_24h_transfer import context, effect, group_means
from src.r19_plan.hcc_off_position_pilot_freeze import old_panels
from src.r19_plan.i_score import checked_array
from src.r19_plan.jhh5_donor_attribution import shrink_consensus
from src.r19_plan.jhh5_off_position import COUNT, donor, selected_cells
from src.r19_plan.jhh5_off_position_native import references as high_reference
from src.r19_plan.jhh7_donor_transfer import controls_for
from src.r19_plan.jhh_donor_agreement_gain import model_gain, source_statistics
from src.r19_plan.jhh_dose_bridge import audited
from src.r19_plan.jhh_low_dose_transfer import finish, load_plan
from src.r19_plan.t4_position import compare
from src.r19_plan.t4_workflow import axes, scalar
from src.r19_plan.x_inputs import check_unique


DOSES = ("3.33333","1.11111")
MAIN, HIGH, LOCAL = "high_norm_gain","high_constant","local_constant"
ARMS = (MAIN,HIGH,LOCAL)
PAIRS = tuple((MAIN, r) for r in (HIGH,LOCAL,"B0")) + ((HIGH,"B0"),(LOCAL,"B0"))


def position_separated(target, donors):
    all_refs = donors.loc[donors.parent.isin(target.parent) & ~donors.rna_plate.isin(target.rna_plate)]
    locations = ["parent","rna_well"]
    forbidden = pd.MultiIndex.from_frame(target[locations].drop_duplicates())
    off = all_refs.loc[~pd.MultiIndex.from_frame(all_refs[locations]).isin(forbidden)]
    panels = selected_cells(off)
    selected = pd.DataFrame([(parent, c) for parent,cells in panels.items() for c in cells],columns=["parent","cell_id"])
    eligible = off.merge(selected,on=["parent","cell_id"],validate="many_to_one")
    enough = eligible.groupby("parent").rna_plate.nunique()
    chosen = enough.index[enough.ge(COUNT)]
    return target.loc[target.parent.isin(chosen)].copy(), eligible.loc[eligible.parent.isin(chosen)]


def access(args):
    bank=audited(args.audit,args.audit_pin)
    reports=[]
    for (cell,dose), group in bank["targets"].groupby(["cell_id","actual_dose_uM"]):
        refs=bank["donors"].loc[lambda d:d.actual_dose_uM.eq(dose)&d.view.eq("nonhepatic")]
        chosen,selected=position_separated(group,refs)
        folds=chosen.groupby("fold").parent.nunique().reindex(list("012"),fill_value=0)
        reports.append(dict(cell=cell,dose=dose,parents=chosen.parent.nunique(),wells=len(chosen),
                            plates=chosen.rna_plate.nunique(),fold_parents=folds.tolist()))
        if cell=="JHH5" and dose==args.dose:
            target,source=chosen,selected
    target["fold"]=target.fold.astype(int)
    figures, splits, _=evaluate_support(target.assign(b1_seen=False),source,"JHH5","nonhepatic")
    support=next(s for s in figures if s["scope"]=="donor_covered")
    if not support["eligible"] or min(s["score_parents"] for s in splits)<3:
        raise ValueError("Existing low-dose data do not meet the position-separated support requirement")
    target["fold"]=target.fold.astype(str)
    splits=pd.DataFrame(splits).assign(fold=lambda f:f.fold.astype(str))
    contexts=pd.concat([target[CONTEXT],source[CONTEXT]]).drop_duplicates()
    controls=bank["controls"].merge(contexts,on=CONTEXT,validate="many_to_one")
    check_unique(pd.concat((target,source,controls),ignore_index=True))
    register(args,target,source,controls,splits,dict(dose=args.dose,position_audit=reports))


def high_data(args, ref, genes, cols):
    old, prior, g, c, freeze, _=high_reference(SimpleNamespace(previous=args.high, access_pin=args.high_access_pin,
        previous_protocol=args.high_protocol, previous_pin=args.high_pin, protocol=args.high_native_protocol))
    if (ref["matrices"]!=prior["matrices"] or not np.array_equal(genes,g)
            or any(not np.array_equal(cols[k],c[k]) for k in cols)):
        raise ValueError("Cross-dose fitting requires matched CMap release and normalized gene axis")
    return old,prior,freeze


def summary(panels):
    mean=panels.mean(axis=1)
    return shrink_consensus(panels,[COUNT]*len(panels),mean),source_statistics(panels)[:,:1]


def predict(args,ref,genes,cols,stamp):
    role=args.group[-1]
    prior,high,freeze=high_data(args,ref,genes,cols)
    hi_names,hi_panels=old_panels(SimpleNamespace(previous=args.high),high,genes,freeze,role)
    hi_shrunk,hi_stat=summary(hi_panels)
    low=args.run/"donors"/role
    low_receipt=read_json(low/"receipt.json")
    archive=low/"panels.npz"
    if (low_receipt["role"]!=role or any(low_receipt[k]!=v for k,v in stamp.items())
            or sha256(archive)!=low_receipt["data_sha256"]):
        raise ValueError("The independent low-dose source panel changed")
    with np.load(archive,allow_pickle=False) as f:
        target_names=pd.Index(f["parents"])
        lo_shrunk,lo_stat=summary(checked_array(f["panels"],(ref["target_parents"],COUNT,len(genes)),"low-dose panels"))
    split_keys=[key for key in sorted(ref["slices"]) if key.startswith(args.group+"-")]
    plans={key:load_plan(args,"predict",key,ref["slices"][key]["predict"]) for key in split_keys}
    held=pd.concat([plan["query.tsv"] for plan in plans.values()])
    root=args.run/"forecasts"/args.group
    root.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=root.parent,prefix=".cross-dose-predict-") as work:
        stage=Path(work)
        files,fits={},[]
        for key,plan in plans.items():
            fold,_,_,_,plate=axes(key)
            fit,fit_ctl,query=(plan[x] for x in ("fit.tsv","fit_controls.tsv","query.tsv"))
            former=load_plan(prior,"predict",key,high["slices"][key]["predict"])
            hi_fit=former["fit.tsv"].loc[lambda r:~r.scaffold.isin(held.scaffold)&~r.parent.isin(held.parent)]
            hi_ctl=controls_for(former["fit_controls.tsv"],hi_fit,role)
            if (len(set(hi_fit.parent))<30 or len(set(fit.parent))<30 or "level3_row_index" in query
                    or set(hi_fit.parent)&set(held.parent) or set(hi_fit.scaffold)&set(held.scaffold)
                    or set(fit.scaffold)&set(held.scaffold) or hi_fit.rna_plate.eq(plate).any() or fit.rna_plate.eq(plate).any()
                    or hi_ctl.rna_plate.eq(plate).any() or fit_ctl.rna_plate.eq(plate).any()
                    or hi_fit.fold.eq(str(fold)).any() or set(pd.concat([hi_fit,fit]).inst_id)&set(query.inst_id)):
                raise ValueError("High or low fitting crossed the locked dose-transfer boundary")
            train_hi,y_hi=group_means(effect(hi_fit,hi_ctl,high,genes,cols,role),hi_fit,["parent"])
            train_lo,y_lo=group_means(effect(fit,fit_ctl,ref,genes,cols,role),fit,["parent"])
            i=hi_names.get_indexer(train_hi.parent)
            j=target_names.get_indexer(train_lo.parent)
            q=target_names.get_indexer(query.parent)
            if any((idx<0).any() for idx in (i,j,q)):
                raise ValueError("A fitting or evaluation parent has no dose-matched donor panel")
            origin=scalar(train_hi,hi_shrunk[i],y_hi)
            local=scalar(train_lo,lo_shrunk[j],y_lo)
            gains,coef=model_gain(hi_stat[i],lo_stat[q],hi_shrunk[i],y_hi,agreement=False)
            estimates={MAIN:gains[:,None]*lo_shrunk[q],HIGH:origin*lo_shrunk[q],LOCAL:local*lo_shrunk[q]}
            point=stage/(key+".npz")
            np.savez_compressed(point,**{arm:checked_array(v,(len(query),len(genes)),arm).astype("float32") for arm,v in estimates.items()})
            files[key]=sha256(point)
            fits.append(dict(slice=key,high_fit_parents=len(i),low_fit_parents=len(j),query_wells=len(q),high_constant=origin,
                local_constant=local,amplitude_slope=coef[1],high_gain_median=np.median(gains),
                fit_control_and_score_overlap=0,fit_treatment_and_score_overlap=0))
        write_json(stage/"receipt.json",dict(stamp,group=args.group,files=files,fits=fits,donor_sha256=sha256(low/"receipt.json")))
        stage.rename(root)


def score_pairs(rows,y,arrays,cell,scope):
    return compare(rows,y,arrays,PAIRS,scope)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("access","donor","predict","freeze","score"))
    parser.add_argument("--dose",choices=DOSES,required=True)
    for name in ("run","protocol","audit","audit-pin","cmap-pin","landmarks","out","access","access-pin",
                 "high","high-access-pin","high-protocol","high-native-protocol","high-pin"):
        parser.add_argument("--"+name,type=Path)
    for name in ("access-digest","digest","role","group"):
        parser.add_argument("--"+name)
    args=parser.parse_args()
    if args.phase=="access":
        access(args)
    else:
        if sha256(args.access/"receipt.json")!=args.access_digest:
            raise ValueError("The newly frozen access receipt changed")
        ref,genes,cols,stamp=context(args)
        if ref["dose"]!=args.dose:
            raise ValueError("Dose differs from the registered assay")
        stamp.update(transfer_code_sha256=sha256(__file__),high_freeze_sha256=sha256(args.high_pin),
            gain_code_sha256=sha256(Path(__file__).with_name("jhh_donor_agreement_gain.py")),
            donor_code_sha256=sha256(Path(__file__).with_name("jhh5_off_position.py")))
        if args.phase=="donor":
            donor(args,ref,genes,cols,stamp,dose=args.dose)
        elif args.phase=="predict":
            predict(args,ref,genes,cols,stamp)
        else:
            finish(args,ref,genes,cols,stamp,arms=ARMS,score_pairs=score_pairs,panel_count=0)
