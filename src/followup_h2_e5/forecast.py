"""Refit sealed E5 policies, freeze outcome-blind forecasts and numerical test inputs."""
import argparse
from pathlib import Path
import os

import numpy as np
import pandas as pd
import torch

from followup_h2.common import context, panel_hash, sha256
from followup_h2_e4.data import CPU, cpu_threads, internal_only, registry, source_rows, utc_now
from followup_h2_e4.forecast import (
    FULL_OFFSET as E4_OFFSET, PERSONAL as E4_PERSONAL, frozen_components, outcomes_locked,
    read_forecast as read_e4,
)
from followup_h2_e4.model import FULL as E4_FULL, LOSO, POOLED, v2_policy
from followup_h2_e4.selection import role_data
from nextgen_h2.assets import KEY
from nextgen_h2.run import META, read_json
from nextgen_h2_v2.freeze import gram_receipt
from vc_build.io import publish_json

from .anchors import fit_personal, predict_anchor
from .kernel import BALANCED, DEBIASED, TASK, TAUS, ScreenKernel
from .registration import E4, E4_SELECTED, PLAN, read_split
from .select import COMBINED, INSCREEN, P1, P2, REL, SHARED

V2 = "H2-v2"
E4_ARM = "H2-LOSO-Full"
P1_PERSONAL, P1_OFFSET, P2_OFFSET = "__p1_personal", "__p1_offset", "__p2_offset"
COMMON = ("C1","H2-v1",V2,E4_ARM)
PILLARS = (P1,REL,P2,DEBIASED,SHARED,BALANCED,COMBINED)
TRACK_ARMS = {"T1":(*COMMON,POOLED,LOSO,*PILLARS),
              "T2":(*COMMON,*PILLARS,INSCREEN,"C2","C3"),
              "T3":(*COMMON,*PILLARS)}


def choice_lock(preflight, selected):
    split=read_split(preflight)
    file=selected/"e5_selection_receipt.json"
    proof=read_json(file)
    assert proof["plan_sha256"]==sha256(PLAN) and proof["split_sha256"]==sha256(preflight/"split.json")
    assert sha256(selected/"anchor_grid.tsv")==proof["anchor_grid_sha256"]
    assert proof["E4_internal_selection_sha256"]==E4_SELECTED
    assert proof["geometry_candidates_each"]==160 and proof["hierarchical_tau_points"]==list(TAUS)
    assert proof["external_cohort_access"]==proof["external_score_access"]==0
    return split,proof,sha256(file)


def pinned_e4(index):
    area=E4/"forecasts"/str(index)
    test=read_json(E4/"tests/test_receipt.json")
    assert test["selection_sha256"]==E4_SELECTED and test["status"]=="PASS"
    assert test["forecast_receipt_sha256"][str(index)]==sha256(area/"receipt.json")
    data,note=read_e4(area,E4_SELECTED)
    return data, note


def old_scores(index, track, fold, query, engine):
    old,note=pinned_e4(index)
    assert old[META].equals(query[META])
    base=old[[*META,"C1","H2-v1","lincs_observed"]].copy()
    if track=="T1":
        np.testing.assert_array_equal(old[LOSO],old[E4_FULL])
        base[LOSO],base[POOLED],base[E4_ARM]=old[LOSO],old[POOLED],old[E4_FULL]
        base[V2]=engine.extract_rows(engine.solve(v2_policy(engine.role),75204))
        replay={"E4_max_abs_diff":float(np.max(np.abs(base[E4_ARM]-old[LOSO]))),
                "rows":len(old),"E4_replay":"registered LOSO and FULL arms share the sealed refit"}
        missing=engine.baseline_residual[engine.selector]
    else:
        pieces,replay=frozen_components(fold)
        assert pieces[META].equals(query[META])
        base[V2]=pieces[V2]
        np.testing.assert_array_equal(old[V2],pieces[V2])
        base[E4_ARM]=old[E4_PERSONAL].to_numpy(float)+old[E4_OFFSET].to_numpy(float)
        replay["E4_max_abs_diff"]=float(np.max(np.abs(base[E4_ARM]-old[E4_FULL])))
        assert replay["E4_max_abs_diff"]<1e-6
        replay["E4_replay"]="sealed E4 personal residual plus recorded drug anchor"
        missing=pieces.rc.to_numpy(float)
    replay["E4_forecast_sha256"]=note["predictions_sha256"]
    return base,missing,replay


def active_task(s,sigma):
    gram=np.ones((s,s))+sigma*np.eye(s)
    vals=np.linalg.eigvalsh(gram)
    active=vals if sigma else vals[-1:]
    note={"tasks":s,"tau":sigma,"raw_minimum_eigenvalue":float(vals[0]),
          "active_minimum_eigenvalue":float(active[0]),"rank":len(active)}
    assert note["active_minimum_eigenvalue"]>0 and note["raw_minimum_eigenvalue"] > -1e-10
    return note


def tau0_numerical_inputs(engine, gram, out):
    """Save fitting-only Grams and the same finite grid for direct tau=0 tests."""
    screens, mass, weighted=engine.screen_grids()
    # Include the best supported 28 cells and 26 compounds deterministically.
    c=np.argsort(-mass.sum(axis=(0,2)),kind="stable")[:28]
    d=np.argsort(-mass.sum(axis=(0,1)),kind="stable")[:26]
    md=mass[:,c][:,:,d]
    wy=weighted[:,c][:,:,d]
    np.savez(out, cell=engine.cell.numpy()[c][:,c],drug=gram[:engine.stop,:engine.stop].numpy()[d][:,d],
             task_mass=md,task_y=wy,
             orig_mass=engine.weights.numpy()[c][:,d],
             orig_response=engine.response.numpy()[c][:,d],screens=np.asarray(screens))


def freeze(preflight,selected,out,index):
    split,choice,choice_hash=choice_lock(preflight,selected)
    reg=registry()
    last=index==len(split["evaluations"])
    if not last:
        internal_only()
        name=split["evaluations"][index]
        track,f,_=name.split("/")
        fold=int(f)
        data,fit,query=role_data(split["roles"][name],reg,inner=False)
    else:
        outcomes_locked()
        track,fold="T3",3
        data,other,query=context(CPU,external=True)
        assert reg.equals(other)
        screens=split["roles"]["T2/0/outer"]["fit_screens"]
        fit=source_rows(sorted(reg.index),screens,labels=True)
        assert panel_hash(fit)==split["source_roster_sha256"]
        query=query[META].sort_values("row_id",ignore_index=True)
    role=f"{track}/{fold}"
    policies=choice["policies"][role]
    engine=ScreenKernel(data,fit,query,reg,role)
    panel,fallback,replay=old_scores(index,track,fold,query,engine)
    dest=out/str(index)
    dest.mkdir(exist_ok=False,parents=True)
    cached,grids,details={}, {}, {}
    supported=panel.lincs_observed.to_numpy()

    def model(rec,control="real"):
        key=tuple(rec[n] for n in ("family","tau","base","shape","weight","penalty"))+(control,)
        if key not in cached:
            both=rec["family"]==TASK and track=="T2"
            cached[key]=fit_personal(engine,rec,control=control,mode="both" if both else "shared")
        residual, gram, available, diagnostics=cached[key]
        assert np.array_equal(supported,available[engine.selector[1]])
        p=predict_anchor(engine,rec["intercept"],gram,available)[engine.selector[1]]
        return residual,p,gram,diagnostics

    def record(arm,residual,offset,gram,diag,policy):
        a=np.where(supported,residual,fallback)
        if arm==P1:
            panel[P1_PERSONAL],panel[P1_OFFSET]=a,offset
        if arm==P2:
            np.testing.assert_array_equal(panel[P1_PERSONAL],a)
            panel[P2_OFFSET]=offset
        panel[arm]=a+offset
        if arm not in grids:
            fn=dest/f"gram-{len(grids)}.npy"
            arr=gram[:engine.stop,:engine.stop]
            np.save(fn,arr.numpy(),allow_pickle=False)
            ev=gram_receipt(arr)
            assert ev["minimum_eigenvalue"]>0
            grids[arm]=ev|{"file":fn.name,"sha256":sha256(fn)}
        details[arm]=diag|{"task_gram":active_task(len(engine.screen_grids()[0]),policy["tau"])
                           if policy["family"]==TASK else None}
    for policy_arm,rec in policies.items():
        z,p,gram,diag=model(rec)
        if policy_arm==TASK:
            if track=="T2":
                record(SHARED,z["shared"],p,gram,diag,rec)
                record(INSCREEN,z["in_screen"],p,gram,diag,rec)
            else:
                record(SHARED,z,p,gram,diag,rec)
            fname=dest/"tau0-inputs.npz"
            tau0_numerical_inputs(engine,gram,fname)
        else:
            z=z["in_screen"] if isinstance(z,dict) else z
            record(policy_arm,z,p,gram,diag,rec)
    if track=="T2":
        rec=policies[COMBINED]
        for arm,control in (("C2","mask"),("C3","shuffle")):
            z,p,gram,diag=model(rec,control)
            z=z["in_screen"] if isinstance(z,dict) else z
            record(arm,z,p,gram,diag,rec)

    original=policies[TASK]
    for tau in TAUS:
        rec=original|{"tau":tau}
        z,_,_,_=model(rec)
        if isinstance(z,dict):
            for m,y in z.items():
                panel[f"tau/{tau:g}/{m}"]=np.where(supported,y,fallback)
        else:
            panel[f"tau/{tau:g}/shared"]=np.where(supported,z,fallback)
    file=dest/"predictions.parquet"
    panel.to_parquet(file,index=False)
    publish_json(dest/"receipt.json",{"track":track,"fold":fold,"created_at_utc":utc_now(),
                 "selection_sha256":choice_hash,"predictions_sha256":sha256(file),"rows":len(panel),
                 "queried_outcomes":0,"fit_rows":len(fit),"fit_roster_sha256":panel_hash(fit),
                 "query_roster_sha256":panel_hash(query),"selected_grams":grids,"details":details,
                 "replays":replay,"tau0_input_sha256":sha256(fname)})
    print(track,fold,len(panel),list(grids),flush=True)


def read_frozen(folder,selected_hash):
    from followup_h2.common import parquet_lock
    a=read_json(folder/"receipt.json")
    b=parquet_lock(folder/"predictions.parquet",a["predictions_sha256"])
    assert a["selection_sha256"]==selected_hash and a["queried_outcomes"]==0
    assert b.row_id.is_unique and len(b)==a["rows"] and panel_hash(b)==a["query_roster_sha256"]
    assert np.isfinite(b[list(TRACK_ARMS[a["track"]])]).all().all()
    return b,a


if __name__=="__main__":
    cpu_threads()
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("preflight","selected","out"):
        p.add_argument(name,type=Path)
    a=p.parse_args()
    freeze(a.preflight,a.selected,a.out,int(os.environ["SLURM_ARRAY_TASK_ID"]))
