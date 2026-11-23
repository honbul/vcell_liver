"""Full-information FiLM, cross-attention, and graph-enhanced models."""
import numpy as np
import torch
from torch import nn

from .graphs import MoleculeNet


def block(inp, out, dropout):
    return nn.Sequential(nn.Linear(inp,out),nn.LayerNorm(out),nn.GELU(),nn.Dropout(dropout))


def observed_branch(layer,value,mask):
    clean = torch.where(mask[:,None],value,torch.zeros_like(value))
    return layer(clean)*mask[:,None]


class Views(nn.Module):
    def __init__(self, atlas, cfg):
        super().__init__()
        w,drop=cfg["width"],cfg["dropout"]
        cell_dim=sum(atlas[k].shape[1] for k in ("rna","activity","proxy","paths"))
        self.cell=block(cell_dim,w,drop)
        self.fingerprint=block(atlas["bits"].shape[1],w,drop)
        self.props=block(atlas["props"].shape[1],w,drop)
        self.target=block(atlas["targets"].shape[1],w,drop)
        self.effect=block(atlas["lincs"].shape[1]+atlas["perturb_paths"].shape[1],w,drop)
        self.availability=nn.Linear(1,w,bias=False)
        self.pair=block(55+int(cfg.get("hybrid",False)),w,drop)
        self.graph=None
        if cfg.get("family")=="gnn":
            self.graph=MoleculeNet(atlas["atoms"].shape[-1],atlas["bonds"].shape[-1],w,cfg["message_depth"],drop,cfg["graph"])
        self.fm_key=cfg.get("fm")
        if self.fm_key:
            self.pretrained=block(atlas[self.fm_key].shape[-1],w,drop)
        self.dropout=nn.Dropout(drop)

    def cell_features(self,a,ids):
        return self.cell(torch.cat([a[k][ids] for k in ("rna","activity","proxy","paths")],dim=1))

    def drug_features(self,a,ids):
        mask=a["lincs_mask"][ids]
        chem=self.fingerprint(a["bits"][ids])
        props=self.props(a["props"][ids])
        tgt=self.target(a["targets"][ids])
        effects=torch.cat((a["lincs"][ids],a["perturb_paths"][ids]),dim=1)
        effect=observed_branch(self.effect,effects,mask)
        chem=chem+self.availability(mask[:,None].to(chem.dtype))
        if self.graph is not None:
            chem=chem+self.graph(a,ids)
        if self.fm_key:
            chem=chem+observed_branch(self.pretrained,a[self.fm_key][ids],a[self.fm_key+"_mask"][ids])
        return chem,props,tgt,effect,mask

    def pair_features(self,a,i,j,baseline):
        valid=a["lincs_mask"][j]
        lp=torch.where(valid[:,None],a["perturb_paths"][j],0.)
        cp=a["paths"][i]
        product=cp*lp
        similarity=product.sum(1)/(cp.norm(dim=1)*lp.norm(dim=1)).clamp_min(1e-8)
        interaction=torch.stack((product.mean(1),product.abs().amax(1),similarity,(product>0).float().mean(1)),dim=1)
        columns=[a["target_match"][i,j],product,interaction,valid[:,None],a["mu"][j,None]]
        if baseline is not None:
            columns.append(baseline[:,None])
        return self.pair(torch.cat(columns,dim=1))


class FilmBlock(nn.Module):
    def __init__(self,width,dropout):
        super().__init__()
        self.mod=nn.Linear(width,width*2)
        self.norm=nn.LayerNorm(width)
        self.ff=nn.Sequential(nn.Linear(width,width*2),nn.GELU(),nn.Dropout(dropout),
                              nn.Linear(width*2,width),nn.Dropout(dropout))

    def forward(self,x,d):
        gamma,beta=self.mod(d).chunk(2,dim=-1)
        return x+self.ff(self.norm(x)*(1+.5*gamma.tanh())+beta)


class Film(nn.Module):
    def __init__(self,atlas,cfg):
        super().__init__()
        w,drop=cfg["width"],cfg["dropout"]
        self.views=Views(atlas,cfg)
        self.drug=block(w*4,w,drop)
        self.layers=nn.ModuleList([FilmBlock(w,drop) for _ in range(cfg["depth"])])
        self.head=nn.Sequential(nn.LayerNorm(w),nn.Linear(w,1))
        self.alpha=nn.Linear(w,1) if cfg.get("parts",False) else None

    def forward(self,a,i,j,baseline=None):
        c,ci=torch.unique(i,sorted=True,return_inverse=True)
        d,di=torch.unique(j,sorted=True,return_inverse=True)
        cell=self.views.cell_features(a,c)[ci]
        drug=self.drug(torch.cat(self.views.drug_features(a,d)[:4],dim=1))[di]
        z=cell+self.views.pair_features(a,i,j,baseline)
        for layer in self.layers:
            z=layer(z,drug)
        p=self.head(z).squeeze(-1)
        alpha=torch.zeros_like(p) if self.alpha is None else self.alpha(cell).squeeze(-1)
        return p+alpha,alpha,p


class CrossAttention(nn.Module):
    def __init__(self,atlas,cfg):
        super().__init__()
        w,drop=cfg["width"],cfg["dropout"]
        self.views=Views(atlas,cfg)
        # One token per signed pathway plus a token per grouped pathway/TF module and global RNA.
        chunks=12
        channels=atlas["activity"].shape[1]
        self.chunks=chunks
        self.segment=(channels+chunks-1)//chunks
        self.module=nn.Linear(self.segment,w)
        self.path=nn.Linear(1,w)
        self.module_id=nn.Parameter(torch.randn(44+chunks,w)*.02)
        self.cross=nn.ModuleList([nn.MultiheadAttention(w,cfg["heads"],dropout=drop,batch_first=True)
                                  for _ in range(cfg["depth"])])
        self.within=nn.MultiheadAttention(w,cfg["heads"],dropout=drop,batch_first=True)
        self.norm=nn.ModuleList([nn.LayerNorm(w) for _ in self.cross])
        self.ff=nn.ModuleList([nn.Sequential(nn.Linear(w,w*2),nn.GELU(),nn.Dropout(drop),nn.Linear(w*2,w))
                              for _ in self.cross])
        self.out=nn.Sequential(nn.LayerNorm(w),nn.Linear(w,1))
        self.alpha=nn.Linear(w,1) if cfg.get("parts",False) else None

    def forward(self,a,i,j,baseline=None):
        c,ci=torch.unique(i,sorted=True,return_inverse=True)
        d,di=torch.unique(j,sorted=True,return_inverse=True)
        cell=self.views.cell_features(a,c)
        scores=a["activity"][c]
        n=self.chunks*self.segment-scores.shape[1]
        scores=torch.nn.functional.pad(scores,(0,n)).reshape(len(c),self.chunks,self.segment)
        modules=torch.cat([self.path(a["paths"][c,:,None]),self.module(scores)],dim=1)+self.module_id
        ctx=torch.cat((cell[:,None,:],modules),dim=1)[ci]
        *drug,mask=self.views.drug_features(a,d)
        q=torch.stack(drug,dim=1)[di]
        pair=self.views.pair_features(a,i,j,baseline)
        q=torch.cat([pair[:,None],q],dim=1)
        key_padding=torch.cat([torch.zeros((len(q),4),device=q.device,dtype=torch.bool),~mask[di,None]],dim=1)
        z=q+self.within(q,q,q,key_padding_mask=key_padding, need_weights=False)[0]
        # The missing perturbation position cannot participate in attention or readout.
        z=z.masked_fill(key_padding[...,None],0.)
        for cross,norm,ff in zip(self.cross,self.norm,self.ff):
            z=norm(z+cross(z,ctx,ctx,need_weights=False)[0])
            z=(z+ff(z)).masked_fill(key_padding[...,None],0.)
        p=self.out(z[:,0]).squeeze(-1)
        alpha=torch.zeros_like(p) if self.alpha is None else self.alpha(cell[ci]).squeeze(-1)
        return p+alpha,alpha,p


def on_device(array,device):
    return {name:torch.as_tensor(block,device=device,dtype=torch.float32 if block.dtype.kind=="f" else None)
            for name,block in array.items()
            if isinstance(block,np.ndarray) and block.dtype.kind not in "UOS"}
