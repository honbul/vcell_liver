"""Full-scale neural fits with observed-only block batches and early stopping."""
from copy import deepcopy
from time import perf_counter
import math

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from vc_build.external import SOURCE, native_sensitivity
from .network import CrossAttention, Film, on_device
from .objectives import balanced_weights, batches, component_targets, objectives
from .quality import quality, utility


def aligned_values(rows,scale,schema):
    cols=pd.DataFrame(schema["transforms"]).set_index(SOURCE)
    rows=rows.join(cols[["slope","intercept"]],on=SOURCE,validate="many_to_one")
    return ((native_sensitivity(rows)*rows.slope+rows.intercept)/scale).to_numpy(dtype=np.float32)


def expect(model,bank,a,b,baseline,device,batch_size,dtype):
    model.eval()
    results=[]
    with torch.inference_mode(), torch.autocast(device_type=device.type,dtype=dtype,enabled=device.type=="cuda"):
        for start in range(0,len(a),batch_size):
            t=slice(start,start+batch_size)
            guess=model(bank,a[t],b[t],None if baseline is None else baseline[t])[0].float()
            results.append(guess.cpu().numpy())
    return np.concatenate(results)


def train(atlas, schema, fit, query, registry, config, seed, *, train_base=None, val_base=None,
          fixed_epochs=None, device="cuda", progress=None):
    device=torch.device(device)
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    begin=perf_counter()
    if device.type=="cuda":
        torch.set_float32_matmul_precision("high")
    dtype=torch.bfloat16
    cfg=config
    architecture=CrossAttention if cfg.get("arch",cfg["family"])=="attention" else Film
    A=on_device(atlas,device)
    scale=float(atlas["scale"])
    fa,fb,qa,qb=[A[k] for k in ("fit_cells","fit_drugs","query_cells","query_drugs")]
    ncell,ndrug=len(A["rna"]),len(A["bits"])
    bt=A["mu"][fb] if train_base is None else torch.as_tensor(train_base,device=device,dtype=torch.float32)
    bq=A["mu"][qb] if val_base is None else torch.as_tensor(val_base,device=device,dtype=torch.float32)
    feature_bt=bt if train_base is not None else None
    feature_bq=bq if val_base is not None else None
    y=A["labels"]/scale
    sc=pd.Categorical(fit.source_id, categories=sorted(fit.source_id.unique())).codes.astype(np.int64)
    group=torch.as_tensor(sc,device=device)
    yval=None if fixed_epochs is not None else aligned_values(query,scale,schema)
    model=architecture(A,cfg).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"],
                                fused=device.type=="cuda")
    limit=fixed_epochs or cfg["epochs"]
    schedule_limit=max(limit,cfg["epochs"])
    warmup=min(8,math.ceil(schedule_limit*.08))
    def rate(epoch):
        if epoch < warmup:
            return (epoch+1)/warmup
        t=min(1.,(epoch-warmup)/max(1,schedule_limit-warmup))
        return .08+.92*(1+math.cos(math.pi*t))/2
    schedule=torch.optim.lr_scheduler.LambdaLR(optimizer,rate)
    rows_a,rows_b=atlas["fit_cells"],atlas["fit_drugs"]
    parts=None
    if cfg.get("parts"):
        alpha,interaction=component_targets(rows_a,rows_b,atlas["labels"]/scale-bt.cpu().numpy()/scale,ncell,ndrug)
        parts=torch.as_tensor(alpha,device=device),torch.as_tensor(interaction,device=device)
    bcell,bdrug=cfg["batch"]
    eval_batch=cfg.get("eval_batch",8192)
    trial=0
    best,best_epoch,best_state=-np.inf,0,None
    curves=[]
    weight=cfg["loss"]
    for epoch in range(1,limit+1):
        model.train()
        huber,total=0.,0.
        magnitudes=None
        for index in batches(rows_a,rows_b,sc,rng,ncell,ndrug,bcell,bdrug):
            i=torch.as_tensor(index,device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=dtype,enabled=device.type=="cuda"):
                score,alpha,delta=model(A,fa[i],fb[i],None if feature_bt is None else feature_bt[i])
            pred=score.float()+bt[i]/scale
            if any(cfg["loss"][1:]) or parts is not None:
                subs=None if parts is None else (parts[0][fa[i]],parts[1][i])
                ls=objectives(pred,y[i],fb[i]+group[i]*ndrug,fa[i]+group[i]*ncell,alpha,delta,subs)
                if cfg.get("balance") and magnitudes is None:
                    weight=balanced_weights(ls,pred,cfg["loss"])
                    magnitudes=weight
                w=torch.as_tensor(weight,device=device,dtype=torch.float32)
                loss=ls @ w
            else:
                loss=F.smooth_l1_loss(pred,y[i],beta=.45)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),4.)
            optimizer.step()
            with torch.no_grad():
                huber=huber+F.smooth_l1_loss(pred,y[i],beta=.45,reduction="sum")
            total+=len(i)
            trial+=1
        if int(total)!=len(fit):
            raise ValueError("Different measured row budget across model families")
        schedule.step()
        monitor=(epoch==1 or epoch%cfg.get("check_every",4)==0 or epoch==limit)
        if monitor:
            values={"iteration":epoch,"updates":trial,"train_huber":float(huber.cpu().item()/total),
                    "learning_rate":optimizer.param_groups[0]["lr"],"seconds":perf_counter()-begin,
                    "pearson_weight":weight[1],"listnet_weight":weight[2]}
            if fixed_epochs is None:
                q=expect(model,A,qa,qb,feature_bq,device,eval_batch,dtype)+bq.cpu().numpy()/scale
                diff=np.abs(q-yval)
                values["val_huber"]=float(np.where(diff<.45,.5*diff**2/.45,diff-.225).mean())
                record=quality(query,q*scale,registry)
                values.update(record)
                merit=utility(record)
                values["utility"]=merit
                if merit>best+2e-5:
                    best,best_epoch,best_state=merit,epoch,deepcopy(model.state_dict())
            curves.append(values)
            if progress is not None:
                progress(values)
            if fixed_epochs is None and epoch>=cfg.get("min_epochs",36) and epoch>best_epoch+cfg.get("patience",24):
                break
    if fixed_epochs is None:
        if best_state is None:
            raise ValueError("No finite inner monitoring result")
        model.load_state_dict(best_state)
    else:
        best_epoch=limit
    output=expect(model,A,qa,qb,feature_bq,device,eval_batch,dtype)*scale+bq.cpu().numpy()
    state={k:v.cpu() for k,v in model.state_dict().items()}
    return output,state,curves,{"best_iteration":best_epoch,"total_epochs":epoch,"updates":trial,
                               "train_rows":len(fit),"wall_seconds":perf_counter()-begin,
                               "parameter_count":sum(x.numel() for x in model.parameters()),"device":str(device)}
