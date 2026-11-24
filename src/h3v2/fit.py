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
from .selection import NEURAL_FAMILIES, PRIMARY_TREE_SPLITS, read_neural_expansion, read_primary_tree_expansion


def backends(name):
    return ("gbdt","torch") if name=="hybrid" else ("gbdt",) if name=="gbdt" else ("torch",)


def run_fit(config, atlas, schema, train, valid, registry, seed, *, base_fit=None,base_valid=None,
            fixed=None,progress=None,tree_device="cpu"):
    options = dict(train_base=base_fit, val_base=base_valid, fixed_epochs=fixed, progress=progress)
    if config["backend"]=="gbdt":
        from .fit_tree import train as fit
        options["tree_device"] = tree_device
    else:
        if tree_device != "cpu":
            raise ValueError("The LightGBM device option applies only to tree fits")
        from .fit_neural import train as fit
    return fit(atlas,schema,train,valid,registry,config,seed,**options)


def options(stage,split,family,backend):
    if stage=="r1":
        configs=initial(family)
    else:
        known=read_json(ROOT/"locks"/f"{stage}.json")["splits"][split][family]
        configs=known
    return [c for c in configs if c["backend"]==backend]


def fold_fit(stage, split, family, inner, backend, *, smoke=False, tree_device="cpu",
             candidate_id=None, diagnostic=False, early_neural=False, early_primary_tree=False,
             candidate_ids=None, fit_mask=None, artifact_root=None):
    if tree_device != "cpu" and backend != "gbdt":
        raise ValueError("The LightGBM device option applies only to tree fits")
    if early_neural and early_primary_tree:
        raise ValueError("Choose one registered early expansion")
    if early_neural:
        if backend != "torch" or family not in NEURAL_FAMILIES or smoke or diagnostic:
            raise ValueError("The early expansion applies only to registered neural fits")
        config_list = read_neural_expansion(stage)["splits"][split][family]
    elif early_primary_tree:
        if (backend, family, tree_device) != ("gbdt", "gbdt", "cpu") or split not in PRIMARY_TREE_SPLITS or smoke or diagnostic:
            raise ValueError("The early primary expansion applies only to registered CPU GBDT fits")
        config_list = read_primary_tree_expansion(stage)["splits"][split][family]
    else:
        config_list=options(stage,split,family,backend)
    if candidate_id is not None:
        config_list = [config for config in config_list if identify(config) == candidate_id]
        if len(config_list) != 1:
            raise ValueError(f"Expected registered candidate {candidate_id} in {stage}/{split}/{family}/{inner}")
    if candidate_ids is not None:
        if (backend, tree_device) != ("gbdt", "cpu") or (
            candidate_id is not None or smoke or diagnostic or early_neural
        ):
            raise ValueError("A candidate batch applies only to regular CPU tree fits")
        registered = {identify(config): config for config in config_list}
        requested = set(candidate_ids)
        if not requested or len(requested) != len(candidate_ids) or not requested <= registered.keys():
            raise ValueError(f"Unregistered candidate batch in {stage}/{split}/{family}/{inner}")
        config_list = [registered[identity] for identity in candidate_ids]
    if diagnostic and (tree_device not in {"cuda", "opencl"} or candidate_id is None or smoke):
        raise ValueError("The GPU diagnostic must replay one unchanged registered candidate")
    if not config_list:
        print(f"{stage}/{split}/{family}/{inner}: no {backend} configs",flush=True)
        return
    registry=register()
    all_rows=read_data(split,"fit")
    if fit_mask is None:
        train,valid=nested(all_rows,split,registry,inner)
    else:
        fit_mask = np.asarray(fit_mask)
        if candidate_ids is None or fit_mask.dtype != np.bool_ or fit_mask.shape != (len(all_rows),):
            raise ValueError("An imported inner row mask requires a registered CPU candidate batch")
        train, valid = all_rows.loc[fit_mask], all_rows.loc[~fit_mask]
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
    fit_root = ROOT / (f"{tree_device}-validation" if diagnostic else "fits")
    if artifact_root is not None:
        fit_root = artifact_root.resolve()
        if candidate_ids is None or not fit_root.is_relative_to((ROOT / "external-cpu").resolve()):
            raise ValueError("Staged artifacts require registered CPU tree candidates")
    if smoke:
        c=config_list[0]
        config_list=[c|({"trees":25} if backend=="gbdt" else {"epochs":2})]
    for config in config_list:
        identity=identify(config)
        dest=fit_root/folder/split/f"inner-{inner}"/family/identity
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
                "iteration","branch","phase","limit","utility","seconds") if k in point},flush=True)
        pred,state,curves,stamp=run_fit(config,atlas,schema,train,valid,registry,SEEDS[0],
                                       base_fit=base_train,base_valid=base_val,progress=show,tree_device=tree_device)
        score=quality(valid,pred,registry)
        pd.DataFrame(curves).to_parquet(dest/"curves.parquet",index=False)
        store(valid[["row_id"]].assign(predicted=pred),dest/"predictions.parquet")
        publish_json(dest/"receipt.json", {
            "stage":folder,"split":split,"inner":inner,"family":family,"config":config,"config_id":identity,
            "seed":SEEDS[0],"score":score,"utility":utility(score),
            "atlas_sha256":receipt["atlas_sha256"],"model_input_sha256":receipt["model_input_sha256"],
            "lincs_full_unclipped":receipt["lincs_full_unclipped"],"query_has_outer_labels":False,"training":stamp,
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
    early=p.add_mutually_exclusive_group()
    early.add_argument("--early-neural",action="store_true")
    early.add_argument("--early-primary-tree",action="store_true")
    p.add_argument("--tree-device",choices=("cpu","cuda","opencl"),default="cpu")
    x=p.parse_args()
    if x.tree_device != "cpu" and x.backend != "gbdt":
        p.error("--tree-device requires the gbdt backend for GPU fits")
    if x.tree_device == "cuda":
        from .fit_tree import g3_cuda_library
        g3_cuda_library()
    fold_fit(x.stage,x.split,x.family,x.inner,x.backend,smoke=x.smoke,tree_device=x.tree_device,
             early_neural=x.early_neural,early_primary_tree=x.early_primary_tree)
