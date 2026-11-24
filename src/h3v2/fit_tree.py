"""LightGBM regression and LambdaRank with fit-only alignment and OOF offsets."""
from ctypes import CDLL, byref, c_char, c_int, c_size_t, c_uint, c_ulong, c_void_p
from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from nextgen_h2.run import read_json
from vc_build.external import native_sensitivity
from vc_build.io import sha256
from .data import ROOT
from .features import tabular
from .fit_neural import aligned_values
from .quality import quality, utility


def g3_a40():
    import torch

    if (os.environ.get("SLURMD_NODENAME") != "g3" or torch.cuda.device_count() != 1
            or "A40" not in torch.cuda.get_device_name(0)):
        raise ValueError("GPU LightGBM requires one assigned G3 A40 per task")


@cache
def g3_cuda_library():
    g3_a40()
    overlay = ROOT / "deps/lightgbm-cuda-4.6.0/runtime"
    library = overlay / "lightgbm/lib/lib_lightgbm.so"
    if (lgb.__version__ != "4.6.0" or Path(lgb.basic._LIB._name).resolve() != library.resolve()
            or sha256(library) != (overlay / "library.sha256").read_text().strip()):
        raise ValueError("Expected the verified isolated CUDA LightGBM runtime")
    return library


@cache
def g3_opencl_library():
    g3_a40()
    library = ROOT / "deps/lightgbm/lib/lib_lightgbm.so"
    verified = read_json(ROOT / "step0_verification.json")["versions"]["lightgbm_so_sha256"]
    if (lgb.__version__ != "4.6.0" or Path(lgb.basic._LIB._name).resolve() != library.resolve()
            or sha256(library) != verified):
        raise ValueError("Expected the registered OpenCL LightGBM library")
    driver = CDLL("libcuda.so.1")
    assigned, cuda_uuid = c_int(), (c_char * 16)()
    if driver.cuInit(0) or driver.cuDeviceGet(byref(assigned), 0) or driver.cuDeviceGetUuid(cuda_uuid, assigned):
        raise ValueError("Unable to identify the allocated A40")
    opencl = CDLL("libOpenCL.so.1")
    platform, device = c_void_p(), c_void_p()
    platforms, devices = c_uint(), c_uint()
    all_devices = c_ulong(0xffffffff)
    if (opencl.clGetPlatformIDs(1, byref(platform), byref(platforms)) or platforms.value != 1
            or opencl.clGetDeviceIDs(platform, all_devices, 1, byref(device), byref(devices)) or devices.value != 1):
        raise ValueError("Expected one OpenCL device visible to the Slurm task")
    opencl_uuid = (c_char * 16)()
    device_uuid_khr = 0x106A
    if (opencl.clGetDeviceInfo(device, device_uuid_khr, c_size_t(len(opencl_uuid)), opencl_uuid, None)
            or bytes(opencl_uuid) != bytes(cuda_uuid)):
        raise ValueError("OpenCL UUID differs from the allocated A40")
    return library


def ranking(rows):
    n=len(rows)
    codes=pd.factorize(rows.source_id.astype(str)+"/"+rows.endpoint_id.astype(str)+"/"+rows.cell_id.astype(str),sort=True)[0]
    order=np.argsort(codes,kind="stable")
    groups=np.bincount(codes)
    y=native_sensitivity(rows).to_numpy()
    starts=np.r_[0,np.cumsum(groups)[:-1]]
    lo=np.minimum.reduceat(y[order],starts)
    hi=np.maximum.reduceat(y[order],starts)
    relevance=np.rint(63*(y-lo[codes])/(hi[codes]-lo[codes]).clip(min=1e-12)).astype(np.int32)
    if groups.sum()!=n:
        raise ValueError("Ranking groups must cover all observed rows")
    return order,groups,relevance


def train(atlas,schema,fit,query,registry,config,seed,*,train_base=None,val_base=None,
          fixed_epochs=None,progress=None,tree_device="cpu"):
    start=perf_counter()
    train_a,train_b=atlas["fit_cells"],atlas["fit_drugs"]
    check_a,check_b=atlas["query_cells"],atlas["query_drugs"]
    scale=float(atlas["scale"])
    fit_offset=atlas["mu"][train_b] if train_base is None else np.asarray(train_base,np.float32)
    val_offset=atlas["mu"][check_b] if val_base is None else np.asarray(val_base,np.float32)
    fit_y=atlas["labels"]/scale
    val_y=None if fixed_epochs else aligned_values(query,scale,schema)
    params={
        "max_depth":config["depth"],"num_leaves":config["leaves"],"learning_rate":config["learning_rate"],
        "feature_fraction":config["colsample"],"min_data_in_leaf":config["min_child"],
        "lambda_l2":config["lambda_l2"],"max_bin":63,"bagging_fraction":.90,"bagging_freq":1,
        "num_threads":int(os.environ.get("SLURM_CPUS_PER_TASK","8")),"seed":seed,
        "verbosity":-1,"force_col_wise":True,"deterministic":True,"feature_pre_filter":False,
    }
    if tree_device != "cpu":
        params.pop("force_col_wise")
        params.pop("deterministic")
    if tree_device == "cuda":
        params.update(device_type="cuda", gpu_device_id=0, num_gpu=1)
    elif tree_device == "opencl":
        g3_opencl_library()
        params.update(device_type="gpu", gpu_platform_id=0, gpu_device_id=0, gpu_use_dp=True)
    elif tree_device != "cpu":
        raise ValueError(f"Unsupported LightGBM device {tree_device}")
    requested=config["objective"]
    stages=("l2","lambdarank") if requested=="l2_lambdarank" else (requested,)
    if fixed_epochs is None:
        limit = config["trees"] if len(stages) == 1 else max(160, config["trees"] // 2)
        budgets = [limit] * len(stages)
    else:
        budgets = [fixed_epochs] if isinstance(fixed_epochs, int) else fixed_epochs
        if len(budgets) != len(stages) or any(budget < 1 for budget in budgets):
            raise ValueError("Frozen tree budgets do not match the boosting stages")
    curves=[]
    boosters=[]
    with TemporaryDirectory(prefix="tree-map-") as temp:
        path=Path(temp)
        for stage, this_limit in zip(stages, budgets):
            rank=stage=="lambdarank"
            train_order,valid_order=np.arange(len(fit)),np.arange(len(query))
            group_train=group_val=None
            if rank:
                train_order,group_train,y=ranking(fit)
                if not fixed_epochs:
                    valid_order,group_val,v=ranking(query)
            else:
                y=fit_y-fit_offset/scale
                v=None if fixed_epochs else val_y-val_offset/scale
            train_x=tabular(atlas,train_a[train_order],train_b[train_order],
                            base=None if train_base is None else train_base[train_order],out=path/"train")
            valid_x=tabular(atlas,check_a[valid_order],check_b[valid_order],
                            base=None if val_base is None else val_base[valid_order],out=path/"valid")
            treedata=lgb.Dataset(train_x,label=y[train_order],group=group_train,
                                 init_score=fit_offset[train_order]/scale if rank else None,
                                 free_raw_data=True)
            preddata=None if fixed_epochs else lgb.Dataset(valid_x,label=v[valid_order],group=group_val,
                                              init_score=val_offset[valid_order]/scale if rank else None,
                                              reference=treedata,free_raw_data=True)
            metric={"l2":"l2","huber":"huber","lambdarank":"ndcg"}[stage]
            choice=params|{"objective":stage,"metric":metric}
            if rank:
                choice|={"ndcg_eval_at":[5],"lambdarank_truncation_level":5,"label_gain":list(range(64))}
            watch={}
            checks=[lgb.record_evaluation(watch)]
            if preddata is not None:
                checks.append(lgb.early_stopping(130,first_metric_only=True,verbose=False))
            train_sets=[treedata] if preddata is None else [treedata,preddata]
            model=lgb.train(choice,treedata,num_boost_round=this_limit,valid_sets=train_sets,
                           valid_names=["train","inner_val"][:len(train_sets)],callbacks=checks)
            best=model.best_iteration or model.current_iteration()
            p=model.predict(valid_x,num_iteration=best).astype(np.float32)
            ordered=np.empty(len(query),np.float32)
            ordered[valid_order]=p
            applied=1.
            if rank and len(stages)==2:
                # Additive ranking residual on top of the fold-local L2-corrected kernel.
                pt=model.predict(train_x,num_iteration=best).astype(np.float32)
                factor=float(np.clip(np.std(fit_y-fit_offset/scale)/max(np.std(pt),.1),.03,1.))
                applied=factor*.35
                ordered*=applied
            val_offset=val_offset+ordered*scale
            if len(stages)==2 and not rank:
                pt=model.predict(train_x,num_iteration=best).astype(np.float32)
                fit_offset=fit_offset+pt*scale
            metric_name=next(iter(watch["train"]))
            for i,m in enumerate(watch["train"][metric_name],1):
                if i==1 or i%25==0 or i==best or i==model.current_iteration():
                    stats={"iteration":i,"branch":stage,"train_objective":m,"metric":metric_name,
                           "validation_objective":None if preddata is None else watch["inner_val"][metric_name][i-1],
                           "seconds":perf_counter()-start}
                    curves.append(stats)
            boosters.append({"objective":stage,"iteration":best,"multiplier":applied,"model":model.model_to_string(num_iteration=best)})
            if progress is not None:
                progress({"branch":stage,"iteration":best,"seconds":perf_counter()-start})
            del model,treedata,preddata,train_x,valid_x
    if not fixed_epochs:
        record=quality(query,val_offset,registry)
        curves.append({"iteration":sum(b["iteration"] for b in boosters),"branch":"selected",
                        "seconds":perf_counter()-start,"utility":utility(record)}|record)
    info={"best_iteration":sum(b["iteration"] for b in boosters),"branch_iterations":[b["iteration"] for b in boosters],
          "wall_seconds":perf_counter()-start,"train_rows":len(fit),"device":tree_device,"library":lgb.__version__}
    return val_offset,boosters,curves,info
