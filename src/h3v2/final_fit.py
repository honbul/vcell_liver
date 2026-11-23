"""Refit only frozen family settings and emit sealed forecasts on the registered rosters."""
import argparse
from collections import Counter
import pickle

import numpy as np
import pandas as pd
from scipy.linalg import solve

from nextgen_h2.assets import COLD, KEY, LIMORE, parquet_lock
from nextgen_h2.run import META, read_json
from vc_build.io import publish_json, sha256, verify_forecast

from .atlas import read, save
from .data import ASSETS, ROOT, SCREENS, SPLITS, STUDY, V1, read_data, register, store, table_hash
from .features import SourceFeatures
from .fit import run_fit
from .grid import FAMILIES, SEEDS
from .kernel import oof_fit, read_bank


def external_data():
    h2=read_json(ASSETS/"data_receipt.json")
    lc=read_json(COLD/"receipt.json")
    lim=read_json(LIMORE/"receipt.json")["artifacts"]
    limore=parquet_lock(LIMORE/"test_inputs/rows.parquet",lim["test_inputs/rows.parquet"]["sha256"])
    limore=limore.rename(columns={"record_id":"row_id","cell_group":"lineage_group"}).assign(source_id="LIMORE")
    licob=parquet_lock(COLD/"query.parquet",lc["files"]["query.parquet"]).assign(source_id="LICOB",endpoint_id="AUC")
    gdsc=parquet_lock(ASSETS/"gdsc_query.parquet",h2["files"]["gdsc_query.parquet"])
    columns=[*META,"isomeric_smiles"]
    query=pd.concat([limore[columns],licob[columns],gdsc[columns]],ignore_index=True).sort_values("row_id",ignore_index=True)
    if query.row_id.duplicated().any() or len(query)!=46213:
        raise ValueError("Unexpected H2 external query identities")
    source=parquet_lock(COLD/"train.parquet",lc["files"]["train.parquet"]).sort_values("row_id")
    return source, query


def prepare_external():
    if not (ROOT/"locks/frozen.json").exists():
        raise ValueError("External fit must wait for the frozen operating points")
    fit,query=external_data()
    source=SourceFeatures.read(extra=query)
    save(fit,query,source,ROOT/"features/external")


def old_h2(part):
    target=STUDY/"forecasts"/str(part)
    meta=read_json(target/"receipt.json")
    if meta["partition"]!=part or meta["query_outcomes_loaded"]!=0:
        raise ValueError("Unexpected registered H2 forecast")
    raw=parquet_lock(target/"predictions.parquet",meta["prediction_sha256"])
    if raw.row_id.duplicated().any():
        raise ValueError("Ambiguous H2 reference row")
    return raw.set_index("row_id"),meta["prediction_sha256"]


def exact_join(query, ref):
    new=query.set_index("row_id")
    if set(new.index)!=set(ref.index) or len(new)!=len(ref):
        raise ValueError("Reference forecast has a different test roster")
    existing=ref.loc[new.index]
    for key in [x for x in META if x!="row_id"]:
        if key in existing and not new[key].equals(existing[key]):
            raise ValueError(f"Baseline test identity differs in {key}")
    return existing


def external_operating_point(lock,family):
    if family==lock["external"]["champion"]:
        return lock["external"]["operating_point"]
    modes=[lock["splits"][f"drug-{i}"]["families"][family] for i in range(3)]
    count=Counter(x["id"] for x in modes)
    return min(modes,key=lambda x:(-count[x["id"]],-x["utility"],x["id"]))


def rig(split,family=None):
    lock=read_json(ROOT/"locks/frozen.json")
    if lock["outer_labels_opened"]:
        raise ValueError("Final fitting may not use the outer response files")
    if split=="external":
        fit,query=external_data()
        atlas_path=ROOT/"features"/split
        pick=None if family is None else external_operating_point(lock,family)
    else:
        fit,query=read_data(split,"fit"),read_data(split,"query")
        atlas_path=ROOT/"features"/split/"outer"
        pick=None if family is None else lock["splits"][split]["families"][family]
    atlas,meta=read(atlas_path,fit,query)
    if sha256(atlas_path/"schema.pkl")!=meta["schema_sha256"]:
        raise ValueError("Fitting schema changed")
    with (atlas_path/"schema.pkl").open("rb") as f:
        schema=pickle.load(f)
    return lock, fit, query, atlas, meta, schema, pick


def fixed_kernel(split,fit,query):
    registry=register()
    if split=="external":
        q,sig=old_h2(3)
        join=exact_join(query,q)
        train=pd.concat([read_bank(f"drug-{part}","outer") for part in range(3)],verify_integrity=True)
        if set(train.index)!=set(fit.row_id):
            raise ValueError("External stacking must cross-fit all source observations")
        return train.loc[fit.row_id,"H2"].to_numpy(np.float32),join.champion.to_numpy(np.float32),sig
    bank=oof_fit(fit,split,registry)
    test=read_bank(split,"outer")
    return bank.H2.to_numpy(np.float32),test.loc[query.row_id,"H2"].to_numpy(np.float32),sha256(
        ROOT/"kernels"/split/"receipt.json")


def final_candidate(split,family,seed):
    lock,fit,query,atlas,meta,schema,point=rig(split,family)
    config=point["config"]
    root=ROOT/"final"/split/family/str(seed)
    root.mkdir(parents=True,exist_ok=False)
    train_base=val_base=kernel_stamp=None
    if family=="hybrid":
        train_base,val_base,kernel_stamp=fixed_kernel(split,fit,query)
    prediction,state,curve,stamp=run_fit(config,atlas,schema,fit,query,register(),seed,
                                      base_fit=train_base,base_valid=val_base,
                                      fixed=point["selected_budget"],
                                      progress=lambda d:print(split,family,seed,d,flush=True))
    if not np.isfinite(prediction).all() or len(prediction)!=len(query):
        raise ValueError("Final package did not forecast the entire registered test roster")
    file=root/"forecast.parquet"
    store(query[META].assign(predicted=prediction),file)
    model=root/"state.pkl"
    with model.open("xb") as stream:
        pickle.dump({"model":state,"config":config,"seed":seed,"scale":float(atlas["scale"]),
                     "source_atlas_sha256":meta["atlas_sha256"],"kernel_base_sha256":kernel_stamp},stream,protocol=5)
    publish_json(root/"receipt.json",{
        "split":split,"family":family,"seed":seed,"training":stamp,"operating_point":point["id"],
        "lock_sha256":sha256(ROOT/"locks/frozen.json"),"atlas_sha256":meta["atlas_sha256"],
        "forecast_sha256":sha256(file),"checkpoint_sha256":sha256(model),"query_id_sha256":table_hash(query),
        "query_rows":len(query),"outer_labels_loaded":False,"schema_sha256":meta["schema_sha256"],
        "kernel_base_sha256":kernel_stamp,
    })


def ridge_extrapolate(atlas,penalty=64.):
    """Fixed per-parent Ridge control. A cold parent uses its five closest fitting parents."""
    x=atlas["rna"][:,:256]
    rows,drugs=atlas["fit_cells"],atlas["fit_drugs"]
    m,n=len(x),len(atlas["bits"])
    response=np.zeros((m,n),np.float32)
    count=np.zeros((m,n),np.float32)
    np.add.at(response,(rows,drugs),atlas["labels"]-atlas["mu"][drugs])
    np.add.at(count,(rows,drugs),1.)
    coef=np.zeros((n,x.shape[1]),np.float32)
    seen=np.unique(drugs)
    for d in seen:
        measured=count[:,d]>0
        zd=x[measured]
        yy=response[measured,d]/count[measured,d]
        xt=zd.T
        gram=xt@zd+np.eye(zd.shape[1],dtype=np.float32)*penalty
        coef[d]=solve(gram,xt@yy,assume_a="pos",check_finite=False)
    unknown=np.setdiff1d(np.arange(n),seen)
    if len(unknown):
        bits=atlas["bits"]
        overlap=bits[unknown]@bits[seen].T
        fraction=overlap/(bits[unknown].sum(1)[:,None]+bits[seen].sum(1)[None,:]-overlap).clip(min=1)
        neighbors=np.argsort(-fraction,axis=1,kind="stable")[:,:5]
        w=np.take_along_axis(fraction,neighbors,axis=1)**4
        coef[unknown]=(coef[seen[neighbors]]*w[:,:,None]).sum(1)/w.sum(1)[:,None].clip(min=1e-8)
    row,col=atlas["query_cells"],atlas["query_drugs"]
    return atlas["mu"][col]+np.einsum("ij,ij->i",x[row],coef[col])


def final_references(split):
    lock, fit, query, atlas, meta, schema, point=rig(split)
    assets=read_json(ASSETS/"data_receipt.json")
    avail=parquet_lock(ASSETS/"coverage.parquet",assets["files"]["coverage.parquet"])
    avail=query[KEY].map(avail.lincs_modz24).fillna(False)
    mode,idx=(split.rsplit("-",1) if split!="external" else ("external","3"))
    readings={}
    target=ROOT/"final"/split/"references"
    target.mkdir(parents=True,exist_ok=False)
    if mode in ("drug","external"):
        old,digest=old_h2(int(idx))
        old=exact_join(query,old)
        c1=old.C1.to_numpy()
        h2=old.champion.to_numpy()
        if not np.array_equal(avail.to_numpy(),old.champion_observed.to_numpy()):
            raise ValueError("Exact 24-hour LINCS support differs from the published H2 roster")
        readings["h2_archive_forecast_sha256"]=digest
    else:
        bank=read_bank(split,"outer")
        bank=bank.loc[query.row_id]
        c1,h2=bank.C1.to_numpy(),bank.H2.to_numpy()
        readings["h2_refit_digest"]=sha256(ROOT/"kernels"/split/"receipt.json")
    result=query[META].assign(C1=c1,H2=h2,drug_mean=atlas["mu"][atlas["query_drugs"]],
                             drug_ridge=ridge_extrapolate(atlas),lincs=avail)
    if mode in SCREENS:
        v1=V1/"outer-forecast"/split/"forecast.parquet"
        raw,proof=verify_forecast(v1)
        readings["h3_v1_forecast_sha256"]=proof["sha256"]
        old=raw.pivot(index="row_id",columns="method",values="predicted_value")
        known=exact_join(query,old)
        result["structured_v1"]=known.structured.to_numpy()
        result["drug_ridge_v1"]=known.drug_ridge.to_numpy()
        result["drug_mean_v1"]=known.drug_mean.to_numpy()
        for seed in SEEDS:
            result[f"H3v1__{seed}"]=known[f"G2__{seed}"].to_numpy()
    path=target/"forecast.parquet"
    store(result,path)
    publish_json(target/"receipt.json",{
        "split":split,"lock_sha256":sha256(ROOT/"locks/frozen.json"),"forecast_sha256":sha256(path),
        "rows":len(result),"roster_sha256":table_hash(query),"outer_labels_loaded":False,"read_only_sources":readings,
    })


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("operation",choices=("external-atlas","fit","references"))
    p.add_argument("--split",choices=[*SPLITS,"external"])
    p.add_argument("--family",choices=FAMILIES)
    p.add_argument("--seed",type=int,choices=SEEDS)
    a=p.parse_args()
    if a.operation=="external-atlas":
        prepare_external()
    elif a.operation=="fit":
        final_candidate(a.split,a.family,a.seed)
    else:
        final_references(a.split)
