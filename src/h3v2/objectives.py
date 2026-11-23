"""Observed-only regression, Pearson and low-noise listwise ranking."""
import numpy as np
import torch
from torch.nn import functional as F

from .graphs import aggregate, segment_softmax


def batches(a,b,source,rng,nc,nd,cell_width,drug_width):
    """All measured rows once per epoch; each tile spans multiple cell and drug groups."""
    cells=rng.permutation(nc)
    drugs=rng.permutation(nd)
    cg=np.empty(nc,dtype=np.int64)
    dg=np.empty(nd,dtype=np.int64)
    cg[cells]=np.arange(nc)//cell_width
    dg[drugs]=np.arange(nd)//drug_width
    packed=cg[a]*int(dg.max()+1)+dg[b]
    order=np.argsort(packed,kind="stable")
    ends=np.flatnonzero(np.diff(packed[order]))+1
    tiles=np.split(order,ends)
    for key in rng.permutation(len(tiles)):
        yield tiles[key]


def component_targets(a,b,centered,nc,nd):
    """Estimate additive effects by alternating masked least squares on training rows."""
    count_c=np.bincount(a,minlength=nc).clip(1)
    count_d=np.bincount(b,minlength=nd).clip(1)
    d=np.zeros(nd)
    c=np.zeros(nc)
    for _ in range(50):
        c=np.bincount(a,weights=centered-d[b],minlength=nc)/count_c
        updated=np.bincount(b,weights=centered-c[a],minlength=nd)/count_d
        if abs(updated-d).max()<1e-10:
            d=updated
            break
        d=updated
    mean=np.average(c,weights=count_c)
    c-=mean
    d+=mean
    return c.astype(np.float32), (centered-c[a]-d[b]).astype(np.float32)


def objectives(pred,truth,group_drug,group_cell,alpha,delta,parts=None):
    p=pred.float()
    y=truth.float()
    regression=F.smooth_l1_loss(p,y,beta=.45)
    if parts is not None:
        regression=regression+.35*F.smooth_l1_loss(alpha.float(),parts[0],beta=.45)
        regression=regression+.35*F.smooth_l1_loss(delta.float(),parts[1],beta=.45)
    _,g=torch.unique(group_drug,return_inverse=True)
    _,c=torch.unique(group_cell,return_inverse=True)
    ones=torch.ones_like(y)
    nd=aggregate(ones[:,None],g,len(y)).flatten().clamp_min(1)
    u=y-(aggregate(y[:,None],g,len(y)).flatten()/nd)[g]
    v=p-(aggregate(p[:,None],g,len(y)).flatten()/nd)[g]
    su=aggregate(u.square()[:,None],g,len(y)).flatten()
    sv=aggregate(v.square()[:,None],g,len(y)).flatten()
    d=(nd>=5)&(su>1e-8)
    pearson=(aggregate((u*v)[:,None],g,len(y)).flatten()/(su*sv).clamp_min(1e-8).sqrt()*d).sum()/d.sum().clamp_min(1)
    count=aggregate(ones[:,None],c,len(y)).flatten()
    use=count[c]>=5
    target=segment_softmax(y[:,None]/.25,c,len(y)).flatten()
    pred=segment_softmax(p[:,None]/.25,c,len(y)).flatten().clamp_min(1e-12).log()
    listnet=-(target*pred*use).sum()/(count>=5).sum().clamp_min(1)
    return torch.stack([regression,1-pearson,listnet])


def balanced_weights(loss,pred,ratios):
    magnitudes=[]
    for weight,part in zip(ratios,loss):
        if weight==0:
            magnitudes.append(0.)
        else:
            g=torch.autograd.grad(part,pred,retain_graph=True,allow_unused=False)[0]
            magnitudes.append(g.float().norm().detach().item())
    return [float(np.clip(r*magnitudes[0]/max(m,1e-8),.01,5.)) if r else 0.
            for r,m in zip(ratios,magnitudes)]
