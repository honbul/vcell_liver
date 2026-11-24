"""Fit and select pooled or cross-screen anchors on each already-selected E5 kernel."""
import argparse
from pathlib import Path
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import torch

from followup_h2.common import panel_hash, parquet_lock, sha256
from followup_h2_e4.data import cpu_threads, registry, utc_now
from followup_h2_e4.model import ANCHORS, BLENDS, SHRINKS, Candidate, anchor_key, settings
from followup_h2_e4.selection import InnerMetrics, anchor_scores, macros, role_data
from nextgen_h2.assets import KEY
from nextgen_h2.run import META, read_json
from vc_build.external import SOURCE, native_sensitivity
from vc_build.io import publish_json

from .grid import all_roles
from .kernel import STRUCTURES, ScreenKernel
from .registration import PLAN, read_split, source_only
from .select import COMBINED, P1, P2, REL

EXTRA = ("screen_median_centered_shrinkage", "rank_robust_blend")


def candidate(record):
    return Candidate(*(record[k] for k in ("base","shape","weight","penalty")))


def task_args(record):
    return record["family"], record["tau"]


def fit_personal(engine, record, *, mode="in_screen", control="real"):
    cfg = candidate(record)
    values, gram, support, proof = engine.solutions(settings(engine.role,cfg,control),*task_args(record),
                                                    [cfg.penalty],mode=mode)
    y, n, e, solver = values[cfg.penalty]
    return y, gram, support, {"iterations":n,"relative_residual":e,"solver":solver,"transform":proof}


def transformed_targets(engine):
    means = engine.rows.groupby([*SOURCE,KEY],sort=True).model_value.mean().rename("value").reset_index()
    center = means.groupby(SOURCE).value.transform("median")
    width = means.groupby(SOURCE).value.transform(lambda x: float(np.quantile(x,.75)-np.quantile(x,.25)))
    r = means.groupby(SOURCE).value.transform(lambda x: (rankdata(x)-.5)/len(x))
    scales = {EXTRA[0]: means.value-center, EXTRA[1]: center + (r-.5)*width}
    targets = {k:torch.as_tensor(means.assign(y=v).groupby(KEY).y.mean().reindex(engine.fit_drugs).to_numpy(np.float32))
               for k,v in scales.items()}
    return targets,float(means.assign(c=center).groupby(SOURCE).c.first().median())


def custom_anchor(engine, strategy, y, gram, supported):
    family,n=strategy
    stop=engine.stop
    use_lincs=family.startswith("lincs")
    matrix=gram if use_lincs else engine.chemical if family!="chem_top5" else engine.drug.tanimoto
    fit,cross=matrix[:stop,:stop],matrix[stop:,:stop]
    if family.endswith("krr"):
        value=y.mean()+cross @ torch.linalg.solve(fit+n*torch.eye(len(y)),y-y.mean())
    else:
        scores,ix=torch.topk(cross.clamp_min(0),min(int(n),len(y)),dim=1,sorted=True)
        weights=scores.pow(4)
        value=torch.where(weights.sum(1)>0,(weights*y[ix]).sum(1)/weights.sum(1).clamp_min(1e-12),y.mean())
    if use_lincs:
        fallback=custom_anchor(engine,("chem_top5",5.),y,gram,supported)
        return np.where(supported,value.numpy(),fallback)
    return value.numpy()


def predict_anchor(engine, spec, gram, keep):
    family=spec["family"]
    if family not in EXTRA:
        return engine.selected_anchor(spec,gram,keep)
    targets, med=transformed_targets(engine)
    y=targets[family]
    c=custom_anchor(engine,spec["chem"],y,gram,keep)
    l=custom_anchor(engine,spec["lincs"],y,gram,keep)
    blend=.5*(c.astype(float)+l)
    if family==EXTRA[0]:
        blend=med+.5*blend
    return np.where(engine.warm,engine.baseline_anchor,blend)


def fit_anchors(preflight, kernels, out, index):
    source_only()
    split=read_split(preflight)
    role=all_roles(split)[index]
    group=role.rsplit("/",2)[0]
    proof=read_json(kernels/"kernels.json")
    assert proof["registration_sha256"]==sha256(preflight/"split.json") and not proof["external_cohort_access"]
    policy=proof["choices"][group]
    reg=registry()
    data,fit,query=role_data(split["roles"][role],reg,inner=True)
    engine=ScreenKernel(data,fit,query[META],reg,role)
    loso=split["roles"][role]["kind"]=="loso"
    arms=(P2,COMBINED) if loso else (P1,REL,*STRUCTURES)
    selected={}
    dest=out/f"{index:02d}"
    dest.mkdir(exist_ok=False,parents=True)
    for arm in arms:
        personal,gram,keep,_=fit_personal(engine,policy[arm],mode="shared" if loso else "in_screen")
        panel=query[META].assign(truth=native_sensitivity(query).to_numpy(),residual=personal,
                                 warm=engine.warm[engine.selector[1]])
        values={anchor_key(*k):v for k,v in ((k,engine.anchor_strategy(k,gram,keep)) for k in ANCHORS)}
        meta={"grand":float(engine.target_mean.mean()),"median":None}
        if loso:
            targets,med=transformed_targets(engine)
            meta["median"]=med
            for family,y in targets.items():
                for spec in ANCHORS:
                    values[f"{family}:{anchor_key(*spec)}"]=custom_anchor(engine,spec,y,gram,keep)
        panel=panel.assign(**{key:val[engine.selector[1]] for key,val in values.items()})
        file=dest/f"{arm}.parquet"
        panel.to_parquet(file,index=False)
        selected[arm]=meta|{"sha256":sha256(file)}
    publish_json(dest/"receipt.json",{"role":role,"group":group,"arms":selected,"external_cohort_access":0,
                 "kernel_selection_sha256":sha256(kernels/"kernels.json"),"query_roster_sha256":panel_hash(query)})
    print(role,len(query),list(selected),flush=True)


def select_anchors(evs, *, extras):
    configs={anchor_key(f,p):{"family":f,"parameter":p} for f,p in ANCHORS}
    simple=list(configs)
    rows=[]
    for key in simple:
        rows.extend(anchor_scores(evs,key,lambda d,_:d.residual+d[key]))
    means=macros(pd.DataFrame(rows))
    chem=max([k for k in simple if k.startswith("chem")],key=means.get)
    lincs=max([k for k in simple if k.startswith("lincs")],key=means.get)
    chemistry=list(ANCHORS[simple.index(chem)])
    lincs_spec=list(ANCHORS[simple.index(lincs)])

    def mix(family,alpha,gamma):
        return {"family":family,"alpha":alpha,"gamma":gamma,"chem":chemistry,"lincs":lincs_spec}

    def evaluate(alpha,gamma,d,meta):
        grand=meta["grand"]
        raw=(1.-alpha)*d[chem]+alpha*d[lincs]
        return d.residual+np.where(d.warm,d["chem_top5/5"],grand+gamma*(raw-grand))

    for a in BLENDS:
        key=f"convex_blend/{a:g}"
        configs[key]=mix("convex_blend",a,1.)
        rows.extend(anchor_scores(evs,key,lambda d,g:evaluate(a,1.,d,g)))
    diagnostic=[]
    for g in SHRINKS:
        key=f"shrinkage/gamma={g:g}"
        diagnostic.extend(anchor_scores(evs,key,lambda d,x:evaluate(.5,g,d,x)))
    gamma_scores=macros(pd.DataFrame(diagnostic))
    gamma=max(SHRINKS,key=lambda g:(gamma_scores[f"shrinkage/gamma={g:g}"],g))
    configs["shrinkage"]=mix("shrinkage",.5,gamma)
    rows.extend(r|{"candidate":"shrinkage"} for r in diagnostic if r["candidate"]==f"shrinkage/gamma={gamma:g}")
    if extras:
        for family in EXTRA:
            configs[family]=mix(family,.5,.5 if family==EXTRA[0] else 1.)
            def evaluate_custom(d,meta):
                val=.5*(d[f"{family}:{chem}"].astype(float)+d[f"{family}:{lincs}"].astype(float))
                if family==EXTRA[0]:
                    val=meta["median"]+.5*val
                return d.residual+np.where(d.warm,d["chem_top5/5"],val)
            rows.extend(anchor_scores(evs,family,evaluate_custom))
    scores=macros(pd.DataFrame(rows))
    assert len(scores)==len(configs)==(18 if extras else 16)
    best=max(configs,key=scores.get)
    table=pd.DataFrame(rows).assign(tier="screen")
    aggregate=pd.DataFrame({"candidate":list(configs),"value":[scores[k] for k in configs],"tier":"macro"})
    variants=pd.DataFrame(diagnostic).assign(tier="shrinkage_subgrid")
    return {"intercept_candidate":best,"intercept":configs[best],"inner_ndcg_at_5":float(scores[best]),
            "best_chem":chem,"best_lincs":lincs,"anchor_candidates":len(configs)},pd.concat([table,aggregate,variants])


def seal(preflight,kernels,work,out):
    source_only()
    split=read_split(preflight)
    kfile=kernels/"kernels.json"
    kern=read_json(kfile)
    assert kern["registration_sha256"]==sha256(preflight/"split.json")
    assert kern["kernel_objectives_sha256"]==sha256(kernels/"kernel_objectives.tsv")
    reg=registry()
    names=all_roles(split)
    policies,rows,changes={},{},{}
    for parent,arms in kern["choices"].items():
        placements={}
        for arm,chosen in arms.items():
            loso=arm in (P2,COMBINED)
            method="Sel-LOSO" if loso else "Sel-Pooled"
            evs=[]
            for role in split["selections"][f"{parent}/{method}"]:
                folder=work/f"{names.index(role):02d}"
                proof=read_json(folder/"receipt.json")
                assert proof["role"]==role and not proof["external_cohort_access"]
                assert proof["kernel_selection_sha256"]==sha256(kfile)
                assert proof["query_roster_sha256"]==split["roles"][role]["query_roster_sha256"]
                changes[role]=sha256(folder/"receipt.json")
                note=proof["arms"][arm]
                frame=parquet_lock(folder/f"{arm}.parquet",note["sha256"])
                assert panel_hash(frame)==proof["query_roster_sha256"]
                evs.append((role,InnerMetrics(frame,reg.scaffold_block),note))
            summary,table=select_anchors(evs,extras=loso)
            placements[arm]=chosen|summary|{"anchor_selection":method}
            rows[parent,arm]=table.assign(outer=parent,arm=arm,method=method,
                                          chosen=table.candidate.eq(summary["intercept_candidate"]))
        assert all(placements[P1][x]==placements[P2][x] for x in ("candidate","family","tau"))
        policies[parent]=placements
    out.mkdir(exist_ok=False,parents=True)
    file=out/"anchor_grid.tsv"
    pd.concat(rows.values(),ignore_index=True).to_csv(file,sep="\t",index=False,na_rep="NA")
    publish_json(out/"e5_selection_receipt.json",{
        "locked_at_utc":utc_now(),"external_cohort_access":0,"external_score_access":0,
        "plan_sha256":sha256(PLAN),"split_sha256":sha256(preflight/"split.json"),
        "E4_internal_selection_sha256":kern["E4_internal_selection_sha256"],"kernel_selection_sha256":sha256(kfile),
        "kernel_objectives_sha256":kern["kernel_objectives_sha256"],"anchor_grid_sha256":sha256(file),
        "anchor_role_receipts":changes,"policies":policies,
        "geometry_candidates_each":160,"hierarchical_tau_points":kern["hierarchical_tau_points"],
    })
    print("Sealed seven research roles at",utc_now(),flush=True)


if __name__=="__main__":
    cpu_threads()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("fit","seal"))
    p.add_argument("preflight",type=Path)
    p.add_argument("kernels",type=Path)
    p.add_argument("paths",type=Path,nargs="+")
    a=p.parse_args()
    if a.phase=="fit":
        fit_anchors(a.preflight,a.kernels,*a.paths,int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        seal(a.preflight,a.kernels,*a.paths)
