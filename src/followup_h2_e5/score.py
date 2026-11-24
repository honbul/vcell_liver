"""Score E5 three tracks once, then produce diagnostic curves from the same labels."""
import argparse
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from followup_h2.common import panel_hash, sha256
from followup_h2.inference import measure, summarize
from followup_h2_e4.data import OUTCOMES,V1_ASSETS,cpu_threads,registry,train_hash,utc_now
from followup_h2_e4.forecast import licob18
from followup_h2_e4.model import LOSO,POOLED
from followup_h2_e4.selection import InnerMetrics
from nextgen_h2.assets import KEY
from nextgen_h2.run import META,TRAIN,read_json
from nextgen_h2.score import external_labels
from vc_build.external import native_sensitivity
from vc_build.io import publish_json

from .forecast import E4_ARM,P1_PERSONAL,PILLARS,TRACK_ARMS,V2,choice_lock,read_frozen
from .kernel import BALANCED,DEBIASED,TAUS
from .registration import ROOT
from .select import COMBINED,INSCREEN,P1,P2,SHARED
from .test_pipeline import aggregates

TRACK_TAG = {"T1":"E5-Nested-CrossScreen","T2":"E5-DRP-Nested","T3":"E5-External"}
ANCHOR_ONLY={P1:P1_PERSONAL,P2:P1_PERSONAL}
EXTERNAL=("GDSC1","LIMORE","LICOB","LICOB18")
CORE=("DRP","CrossScreen")


def planned_pairs(track):
    order=TRACK_ARMS[track]
    anchors={"C1",V2,E4_ARM,POOLED,LOSO}
    cases=[(x,y) for x in order if x not in anchors and x!="H2-v1" for y in order if y in anchors]
    cases.extend((a,b) for a,b in combinations((x for x in order if x in anchors),2))
    cases.append((P2,P1))
    if track=="T2":
        cases.extend([(INSCREEN,SHARED),(COMBINED,"C2"),(COMBINED,"C3")])
    return [(f"{x} - {y}",{x:1.,y:-1.}) for x,y in cases]


def score_panel(panels,groups,track,cohort):
    order=TRACK_ARMS[track]
    pairs=planned_pairs(track)
    out,ci,support=[],[],{}
    for scope in ("all","lincs_observed","lincs_unobserved"):
        selected=[(i,x if scope=="all" else x.loc[x.lincs_observed.eq(scope=="lincs_observed")]) for i,x in panels]
        pooled=pd.concat([x for _,x in selected],ignore_index=True)
        if pooled.empty:
            continue
        unique={"rows":len(pooled),"parents":pooled[KEY].nunique(),"cells":pooled.cell_id.nunique(),
                "scaffolds":pooled[KEY].map(groups).fillna(pooled[KEY]).nunique(),
                "lineages":pooled.lineage_group.nunique()}
        support[scope]=unique
        tag={"track":TRACK_TAG[track],"cohort":cohort,"support":scope,
             "exposure":"previously_exposed_external_audit" if track=="T3" else "held_under_E5_nested"}
        blocks=measure(selected,groups,order,components=ANCHOR_ONLY)
        arm_meta={a:{"arm":a} for a in order}
        points, intervals=summarize(blocks,tag,arm_meta,pairs)
        for row in points:
            if row["screen"]=="macro":
                row.update(unique)
        out.extend(points)
        ci.extend(row|{"screen":"macro","fold":"all"} for row in intervals)
        for b in blocks:
            _, local=summarize([b],tag,arm_meta,pairs)
            ci.extend(row|{"screen":b.screen,"fold":b.fold} for row in local)
    print(TRACK_TAG[track],cohort,support,flush=True)
    return out,ci,support


def adjudicate(intervals):
    table=intervals[intervals.screen.eq("macro")].set_index(["cohort","support","metric","contrast"])

    def effect(cohort,metric,a,b):
        scope="lincs_observed" if metric=="spearman" else "all"
        row=table.loc[(cohort,scope,metric,f"{a} - {b}")]
        return {k:float(row[k]) for k in ("delta","lo95","hi95")}

    verdict={}
    tc=effect("CrossScreen","spearman",COMBINED,POOLED)
    tn=effect("CrossScreen","ndcg_at_5",COMBINED,POOLED)
    tl=effect("CrossScreen","spearman",COMBINED,LOSO)
    external_n=effect("GDSC1","ndcg_at_5",COMBINED,V2)
    external_r=effect("GDSC1","spearman",COMBINED,V2)
    external_e4=effect("GDSC1","spearman",COMBINED,E4_ARM)
    lim=effect("LIMORE","spearman",COMBINED,V2)
    dr=effect("DRP","spearman",COMBINED,V2)
    dn=effect("DRP","ndcg_at_5",COMBINED,V2)
    cn={a:effect("DRP","spearman",COMBINED,a) for a in ("C1","C2","C3")}
    a=(tc["lo95"]>0 or (tc["lo95"]>=-.003 and tn["lo95"]>0)) and tl["lo95"]>=-.005
    b=all(v["lo95"]>0 for v in cn.values()) and dr["delta"]>=-.010 and dn["delta"]>=-.015
    c=(external_n["delta"]>=.05 and external_e4["delta"]>0 and external_r["lo95"]>=-.008
       and lim["delta"]>.015)
    verdict["track1"]={"pass":a,"rho_vs_pooled":tc,"ndcg_vs_pooled":tn,"rho_vs_E4":tl}
    verdict["track2"]={"pass":b,"rho_vs_controls":cn,"rho_vs_v2":dr,"ndcg_vs_v2":dn}
    verdict["track3"]={"pass":c,"gdsc1_ndcg_vs_v2":external_n,"gdsc1_rho_vs_E4":external_e4,
                         "gdsc1_rho_vs_v2":external_r,"limore_rho_vs_v2":lim}
    winners=[]
    for role in CORE:
        arms=(P1,P2,DEBIASED,SHARED,BALANCED) if role=="CrossScreen" else (P1,P2,DEBIASED,SHARED,INSCREEN,BALANCED)
        for arm in arms:
            r=effect(role,"spearman",arm,E4_ARM)
            n=effect(role,"ndcg_at_5",arm,E4_ARM)
            if min(r["lo95"],n["lo95"])>=0. and max(r["lo95"],n["lo95"])>0.:
                winners.append({"cohort":role,"arm":arm,"rho_vs_E4":r,"ndcg_vs_E4":n})
    verdict["partial_pillar_winners"]=winners
    verdict["decision"]=("GO-V3-PARETO-UPGRADE" if a and b and c else
                          "PARTIAL-PILLAR-WIN" if winners else "NO-GO-KEEP-E4-DUAL")
    return verdict


def diagnostic(panels,inner,filename):
    stamp=utc_now()
    points=pd.read_csv(inner,sep="\t")
    frontiers=[]
    for _,data in points.groupby(["outer","comparison"],sort=False):
        coords=data[["pooled","loso"]].to_numpy()
        flags=[]
        for x in coords:
            flags.append(not np.any((coords>=x).all(axis=1) & (coords>x).any(axis=1)))
        frontiers.append(data.assign(pareto=flags,kind="internal_pooled_loso"))
    curves=[]
    for track,cohort,cohort_panels,groups in panels:
        for support in ("all","lincs_observed"):
            for mode in ("shared","in_screen") if track=="T2" else ("shared",):
                vals=defaultdict(list)
                for fold,frame in cohort_panels:
                    mask=frame.lincs_observed.to_numpy() if support=="lincs_observed" else None
                    ev=InnerMetrics(frame,groups,mask)
                    for tau in TAUS:
                        for screen, value,n in ev.rank(frame[f"tau/{tau:g}/{mode}"].to_numpy()):
                            vals[tau].append({"screen":screen,"value":value,"n":n,"fold":fold})
                for tau,data in vals.items():
                    d=pd.DataFrame(data)
                    curves.append({"kind":"tau_curve","track":TRACK_TAG[track],"cohort":cohort,"support":support,
                                   "mode":mode,"tau":tau,"spearman":float(d.groupby("screen").value.mean().mean()),
                                   "eligible_drug_screen_folds":int(d.n.sum())})
    pd.concat([*frontiers,pd.DataFrame(curves)],ignore_index=True).to_csv(filename,sep="\t",index=False,na_rep="NA")
    return stamp


def run(preflight,selected,kernels,frozen,tests,out):
    cpu_threads()
    split,choices,digest=choice_lock(preflight,selected)
    t=tests/"test_receipt.json"
    test=read_json(t)
    assert test["status"]=="PASS" and test["selection_sha256"]==digest
    assert test["queried_outcomes"]==test["intercept_within_drug_delta"]==0
    assert datetime.fromisoformat(test["passed_at_utc"])>datetime.fromisoformat(choices["locked_at_utc"])
    objective=kernels/"kernel_objectives.tsv"
    assert choices["kernel_objectives_sha256"]==sha256(objective)
    role_receipts=[frozen/str(i)/"receipt.json" for i in range(7)]
    prepared=[]
    for i,path in enumerate(role_receipts):
        assert test["forecast_receipt_sha256"][str(i)]==sha256(path)
        prepared.append(read_frozen(path.parent,digest))
    assert sha256(TRAIN)==train_hash()
    reg=registry()

    out.mkdir(exist_ok=False,parents=True)
    guard=ROOT/"official-outcome-access.json"
    first=utc_now()
    publish_json(guard,{"score_passes":1,"started_at_utc":first,
                        "selection_sha256":digest,"selection_locked_at_utc":choices["locked_at_utc"],
                        "test_sha256":sha256(t),"planned_reads":["TRAIN","GDSC1","LIMORE","LICOB"]})
    with TRAIN.open("rb") as h:
        source=pd.read_parquet(h,columns=[*META,*OUTCOMES]).sort_values("row_id",ignore_index=True)
    assert panel_hash(source)==split["source_roster_sha256"]
    values=pd.Series(native_sensitivity(source).to_numpy(),index=source.row_id)
    del source
    by_track=defaultdict(list)
    for p,ev in prepared[:6]:
        by_track[ev["track"]].append((ev["fold"],p.assign(truth=p.row_id.map(values))))
    labels,license_table,ext_groups=external_labels(V1_ASSETS,read_json(V1_ASSETS/"data_receipt.json"))
    ext=prepared[6][0]
    label_map=labels.set_index("row_id").truth
    assert set(label_map.index)==set(ext.row_id) and np.isfinite(label_map).all()
    print("One TRAIN read and one read of each external outcome",len(values),len(labels),flush=True)
    cohorts={}
    for name,block in ext.assign(truth=ext.row_id.map(label_map)).groupby("source_id",sort=True):
        if name=="LICOB":
            block=aggregates(block,license_table)
            cohorts["LICOB18"]=licob18(block)
        cohorts[name]=block
    runs=[("T1","CrossScreen",by_track["T1"],reg.scaffold_block),
          ("T2","DRP",by_track["T2"],reg.scaffold_block),
          *(("T3",cohort,[(0,cohorts[cohort])],ext_groups) for cohort in EXTERNAL)]
    point,paired,support=[],[],{}
    for track,cohort,panels,groups in runs:
        m,p,sd=score_panel(panels,groups,track,cohort)
        point.extend(m)
        paired.extend(p)
        support[cohort]=sd
    metrics,ci=pd.DataFrame(point),pd.DataFrame(paired)
    decision=adjudicate(ci)
    metrics.to_csv(out/"E5_METRICS.tsv",index=False,sep="\t",na_rep="NA")
    ci.to_csv(out/"E5_PAIRED_95CI.tsv",index=False,sep="\t",na_rep="NA")
    (metrics[metrics.screen.eq("macro") & metrics.arm.isin((*PILLARS,INSCREEN,*("C1",V2,E4_ARM,POOLED,LOSO)))]
     .to_csv(out/"E5_PILLAR_ABLATION.tsv",index=False,sep="\t",na_rep="NA"))
    metrics_at=utc_now()
    diagnostic_at=diagnostic(runs,objective,out/"E5_TAU_AND_PARETO_DIAGNOSTIC.tsv")
    files={f.name:sha256(f) for f in out.glob("*.tsv")}
    publish_json(out/"score_receipt.json",{"score_passes":1,"selection_sha256":digest,
                 "selection_locked_at_utc":choices["locked_at_utc"],"tests_passed_at_utc":test["passed_at_utc"],
                 "test_sha256":sha256(t),"started_at_utc":first,"metrics_sealed_at_utc":metrics_at,
                 "diagnostic_started_at_utc":diagnostic_at,"finished_at_utc":utc_now(),
                 "guard_sha256":sha256(guard),"native_reads":{"TRAIN":1,"GDSC1":1,"LICOB":1,"LIMORE":1},
                 "LICOB18":"subset of the same LICOB read","supports":support,"comparison":decision,
                 "output_sha256":files,"outcome_diagnostics_used_for_selection":False,
                 "forecast_receipt_sha256":test["forecast_receipt_sha256"]})
    print(decision["decision"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("preflight","selected","kernels","frozen","tests","out"):
        p.add_argument(name,type=Path)
    a=p.parse_args()
    run(a.preflight,a.selected,a.kernels,a.frozen,a.tests,a.out)
