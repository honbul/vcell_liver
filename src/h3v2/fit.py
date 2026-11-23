"""Train and retain full-panel inner evidence without reading assessment labels."""
import argparse
from datetime import datetime,timezone
import pickle

import numpy as np
import pandas as pd

from vc_build.io import publish_json,sha256
from nextgen_h2.run import read_json
from .atlas import read
from .data import ROOT,SPLITS,nested,read_data,register,store
from .grid import FAMILIES,SEEDS,identify,initial
from .kernel import oof_fit,read_bank
from .quality import quality,utility


def backends(name):
    return ("gbdt","torch") if name=="hybrid" else ("gbdt",) if name=="gbdt" else ("torch",)


def run_fit(config, atlas, schema, train, valid, registry, seed, *, base_fit=None,base_valid=None,
            fixed=None,progress=None):
    if config["backend"]=="gbdt":
        from .fit_tree import train as fit
    else:
        from .fit_neural import train as fit
    return fit(atlas,schema,train,valid,registry,config,seed,
               train_base=base_fit,val_base=base_valid,fixed_epochs=fixed,progress=progress)


def options(stage,split,family,backend):
    if stage=="r1":
        configs=initial(family)
    else:
        known=read_json(ROOT/"locks"/f"{stage}.json")["splits"][split][family]
        configs=known
    return [c for c in configs if c["backend"]==backend]


def fold_fit(stage, split, family, inner, backend, *, smoke=False):
    config_list=options(stage,split,family,backend)
    if not config_list:
        print(f"{stage}/{split}/{family}/{inner}: no {backend} configs",flush=True)
        return
    registry=register()
    all_rows=read_data(split,"fit")
    train,valid=nested(all_rows,split,registry,inner)
    root=ROOT/"features"/split/f"inner-{inner}"
    atlas,receipt=read(root,train,valid)
    with (root/"schema.pkl").open("rb") as stream:
        schema=pickle.load(stream)
    base_train=base_val=None
    # Smoke validates training handles against one real registered fold separately from selection evidence.
    if family=="hybrid":
        bank=oof_fit(train,split,registry,inner)
        base_train=bank.H2.to_numpy(np.float32)
        base_val=read_bank(split,f"two-{inner}").loc[valid.row_id,"H2"].to_numpy(np.float32)
    folder="smoke" if smoke else stage
    if smoke:
        c=config_list[0]
        config_list=[c|({"trees":25} if backend=="gbdt" else {"epochs":2})]
    for config in config_list:
        identity=identify(config)
        dest=ROOT/"fits"/folder/split/f"inner-{inner}"/family/identity
        if (dest/"receipt.json").is_file():
            previous=read_json(dest/"receipt.json")
            if (previous["config_id"],previous["forecast_sha256"],previous["curve_sha256"]) != (
                    identity,sha256(dest/"predictions.parquet"),sha256(dest/"curves.parquet")):
                raise ValueError("Previously finished candidate was modified")
            continue
        if dest.is_dir():
            recovery=dest.parent/"partial-attempts"
            recovery.mkdir(exist_ok=True)
            dest.rename(recovery/(dest.name+"-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")))
        dest.mkdir(parents=True,exist_ok=False)
        def show(point):
            print(stage,split,inner,family,identity,{k:point[k] for k in (
                "iteration","branch","utility","seconds") if k in point},flush=True)
        pred,state,curves,stamp=run_fit(config,atlas,schema,train,valid,registry,SEEDS[0],
                                       base_fit=base_train,base_valid=base_val,progress=show)
        score=quality(valid,pred,registry)
        pd.DataFrame(curves).to_parquet(dest/"curves.parquet",index=False)
        store(valid[["row_id"]].assign(predicted=pred),dest/"predictions.parquet")
        publish_json(dest/"receipt.json", {
            "stage":folder,"split":split,"inner":inner,"family":family,"config":config,"config_id":identity,
            "seed":SEEDS[0],"score":score,"utility":utility(score),
            "atlas_sha256":receipt["atlas_sha256"],"query_has_outer_labels":False,"training":stamp,
            "curve_sha256":sha256(dest/"curves.parquet"),"forecast_sha256":sha256(dest/"predictions.parquet"),
            "finished":datetime.now(timezone.utc).isoformat(),
        })


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("stage")
    p.add_argument("split",choices=SPLITS)
    p.add_argument("family",choices=FAMILIES)
    p.add_argument("inner",type=int,choices=range(3))
    p.add_argument("backend",choices=("torch","gbdt"))
    p.add_argument("--smoke",action="store_true")
    x=p.parse_args()
    fold_fit(x.stage,x.split,x.family,x.inner,x.backend,smoke=x.smoke)
