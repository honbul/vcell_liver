"""Reviewer exploratory diagnostic 6 (re-implementation; INNER folds of Codex's manifest only; not an official result).
Per dose (0.05 / 0.5 / 5 uM): B0, B_pos-all, chemistry kernel on position-corrected residuals, MoA neighbours,
combined arm, and the plate 6<->14 same-position replicate ceiling at 5 uM."""
import numpy as np, pandas as pd, scipy.sparse as sp, json, os, sys
from rdkit import Chem, DataStructs, RDLogger; RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import AllChem
R='/mnt/S1/sdata/workbench/Virtual_Cell/gssong/'
MINC=int(os.environ.get('MINC','20'))
def load(d, name):
    n=np.load(d+name+'_counts.npz'); m=sp.csr_matrix((n['data'],n['indices'],n['indptr']),shape=tuple(n['shape']))
    r=pd.read_csv(d+name+'_rows.tsv',sep='\t'); assert m.shape[0]==len(r); return m,r
d05=R+'vcell_liver_r11_chemical_probe/assets/source/'; dxx=R+'vcell_liver_r12_multiaxis/assets/doses/source/'
m05,r05=load(d05,'treated'); mxx,rxx=load(dxx,'treated')
def lognorm(m, genes=None):
    lib=np.asarray(m.sum(1)).ravel().astype(np.float64); lib[lib==0]=1
    if genes is not None: m=m[:,genes]
    return np.log1p(m.toarray().astype(np.float64)/lib[:,None]*1e4).astype(np.float32)
ok05=r05.eligible_cell_count.values.astype(bool)
xa=lognorm(m05[ok05]); var=xa.var(0); genes=np.sort(np.argsort(-var)[:2000]); del xa
rows=pd.concat([r05,rxx],ignore_index=True); X=np.vstack([lognorm(m05,genes),lognorm(mxx,genes)])
keep=rows.eligible_cell_count.values.astype(bool)&(rows.n_cells.values>=MINC); rows=rows[keep].reset_index(drop=True); X=X[keep]
rows['pool']=rows.plate_id+'|'+rows.cell_line_id+'|'+rows.dose_uM.astype(str)
keep=(rows.groupby('pool').compound_group_id.transform('nunique')>=3).values; rows=rows[keep].reset_index(drop=True); X=X[keep]
G=X.shape[1]
sm=pd.read_parquet('/mnt/S1/sdata/pubdata/Tahoe/Tahoe-100M/metadata/sample_metadata.parquet')
sm['num']=sm['sample'].str.extract(r'(\d+)').astype(int); sm['pos']=sm.groupby('plate').num.rank().astype(int)
rows['pos']=rows.physical_well_id.map(dict(zip(sm['sample'],sm.pos))).values
man=pd.read_csv(R+'review_claude_20261121/split_manifest.tsv',sep='\t')
of=man[man.usage=='outer_query'].drop_duplicates('compound_group_id').set_index('compound_group_id').outer_fold
rows['fold']=rows.compound_group_id.map(of).values; assert rows.fold.notna().all()
inner={}
for f in range(3):
    s=man[(man.evaluation_outer==f)&(man.inner_fold>=0)].drop_duplicates('compound_group_id').set_index('compound_group_id').inner_fold
    inner[f]=rows.compound_group_id.map(s).fillna(-1).astype(int).values
fold=rows.fold.astype(int).values; gi=rows.compound_group_id.values; pl=rows.plate_id.values; ds=rows.dose_uM.values
ln=rows.cell_line_id.values; pos=rows.pos.values; pool=rows.pool.values; well=rows.physical_well_id.values
# chemistry
gs=rows.drop_duplicates('compound_group_id').set_index('compound_group_id')
gids=np.array(sorted(gs.index)); gix={g:i for i,g in enumerate(gids)}
fps=[AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(gs.loc[g,'smiles']),2,2048) for g in gids]
TAN=np.array([DataStructs.BulkTanimotoSimilarity(f,fps) for f in fps],np.float64)
dm=pd.read_parquet('/mnt/S1/sdata/pubdata/Tahoe/Tahoe-100M/metadata/drug_metadata.parquet').set_index('drug')
moa=np.array([str(dm['moa-fine'].get(gs.loc[g,'compound_entity_id'],'unclear')) for g in gids])
MOA=((moa[:,None]==moa[None,:])&(moa[:,None]!='unclear')).astype(np.float64)
LIVER={'CVCL_0366':'SNU-423','CVCL_1098':'HepG2/C3A'}
print('rows',len(rows),'groups',len(gids),'minc',MINC,{d:int((ds==d).sum()) for d in np.unique(ds)},flush=True)

def rm(a): return np.sqrt((a**2).mean(1))
def center(idx):
    k=pd.Series(pool[idx]); o=np.zeros((len(idx),G),np.float32); v=np.zeros(len(idx),bool)
    for p,ii in k.groupby(k).indices.items():
        if len(set(gi[idx[ii]]))>=3: o[ii]=X[idx[ii]]-X[idx[ii]].mean(0); v[ii]=True
    return o,v
def recentre(P,idx):
    k=pd.Series(pool[idx]); o=P.copy()
    for p,ii in k.groupby(k).indices.items(): o[ii]=P[ii]-P[ii].mean(0)
    return o
def bpos(te,dn,D,own_excl):
    """plate-equal mean of donor contrasts at the same (position, line) on other plates (any dose)."""
    key=pd.DataFrame({'p':pos[dn],'l':ln[dn]}).groupby(['p','l']).indices
    P=np.zeros((len(te),G),np.float32); has=np.zeros(len(te),bool)
    for n,i in enumerate(te):
        kk=key.get((pos[i],ln[i]))
        if kk is None: continue
        j=dn[kk]; m=pl[j]!=pl[i]
        if own_excl: m&=gi[j]!=gi[i]
        kk=kk[m]
        if len(kk)==0: continue
        pj=pl[dn[kk]]; up=np.unique(pj); P[n]=np.mean([D[kk[pj==u]].mean(0) for u in up],0); has[n]=True
    return P,has
def group_line_mean(idx,Y,d):
    """mean target per (group,line) at dose d among rows idx -> dict line -> (group index array, matrix)"""
    m=ds[idx]==d; ii=np.where(m)[0]; df=pd.DataFrame({'l':ln[idx[ii]],'g':gi[idx[ii]]}); o={}
    for l,sub in df.groupby('l'):
        gg=sub.groupby('g').indices; ks=list(gg); o[l]=(np.array([gix[g] for g in ks]),np.stack([Y[ii[sub.index.get_indexer(sub.index[v])]].mean(0) if False else Y[ii[sub.index.values[v]-0]].mean(0) for v in gg.values()]))
    return o
def group_line_mean(idx,Y,d):
    ii=np.where(ds[idx]==d)[0]; o={}
    ll=ln[idx[ii]]; gg=gi[idx[ii]]
    for l in np.unique(ll):
        a=ii[ll==l]; ga=gg[ll==l]; ug=np.unique(ga)
        o[l]=(np.array([gix[g] for g in ug]),np.stack([Y[a[ga==g]].mean(0) for g in ug]))
    return o
def kernel_pred(te,tr,Ytr,d,K,lam,shared=False):
    """kernel ridge per line (or on the line-averaged well response if shared) at dose d."""
    P=np.zeros((len(te),G),np.float32); sel=np.where(ds[te]==d)[0]
    if shared:
        ii=np.where(ds[tr]==d)[0]; ga=gi[tr[ii]]; ug=np.unique(ga); gx=np.array([gix[g] for g in ug]); Y=np.stack([Ytr[ii[ga==g]].mean(0) for g in ug])
        A=np.linalg.solve(K[np.ix_(gx,gx)]+lam*np.eye(len(gx)),Y)
        tg_=np.array([gix[g] for g in gi[te[sel]]]); P[sel]=K[np.ix_(tg_,gx)]@A; return P
    gl=group_line_mean(tr,Ytr,d)
    for l in np.unique(ln[te[sel]]):
        if l not in gl: continue
        gx,Y=gl[l]; A=np.linalg.solve(K[np.ix_(gx,gx)]+lam*np.eye(len(gx)),Y)
        s2=sel[ln[te[sel]]==l]; tg_=np.array([gix[g] for g in gi[te[s2]]]); P[s2]=K[np.ix_(tg_,gx)]@A
    return P
def nn_pred(te,tr,Ytr,d,W):
    """mean of training groups flagged by W (e.g. same MoA), same line & dose."""
    P=np.zeros((len(te),G),np.float32); sel=np.where(ds[te]==d)[0]; gl=group_line_mean(tr,Ytr,d)
    for l in np.unique(ln[te[sel]]):
        if l not in gl: continue
        gx,Y=gl[l]; s2=sel[ln[te[sel]]==l]; tg_=np.array([gix[g] for g in gi[te[s2]]]); w=W[np.ix_(tg_,gx)]; n=w.sum(1,keepdims=True)
        P[s2]=np.where(n>0,(w@Y)/np.maximum(n,1),0)
    return P

# ---------------- diag11 body: robust position estimate, sharp neighbours, energy-weighted endpoint ----------------
def bpos_robust(te,dn,D,mode):
    """same (pos,line), other plates; mode: 'median' over donor plates, or 'trim' = drop donors whose norm > 1.5x median donor norm of that line."""
    key=pd.DataFrame({'p':pos[dn],'l':ln[dn]}).groupby(['p','l']).indices
    nrm=np.sqrt((D.astype(np.float64)**2).sum(1)); med=pd.Series(nrm).groupby(ln[dn]).median().to_dict()
    P=np.zeros((len(te),G),np.float32)
    for n,i in enumerate(te):
        kk=key.get((pos[i],ln[i]))
        if kk is None: continue
        kk=kk[pl[dn[kk]]!=pl[i]]
        if mode=='trim': kk=kk[nrm[kk]<=1.5*med[ln[i]]]
        if len(kk)==0: continue
        pj=pl[dn[kk]]; up=np.unique(pj); V=np.stack([D[kk[pj==u]].mean(0) for u in up])
        P[n]=np.median(V,0) if (mode=='median' and len(up)>=3) else V.mean(0)
    return P
TOX=None
def nn1(te,tr,Ytr,d,S,thr):
    """sharp neighbour: copy the residual of the single most similar training group (same line, dose) if similarity >= thr."""
    P=np.zeros((len(te),G),np.float32); sel=np.where(ds[te]==d)[0]; gl=group_line_mean(tr,Ytr,d)
    for l in np.unique(ln[te[sel]]):
        if l not in gl: continue
        gx,Y=gl[l]; s2=sel[ln[te[sel]]==l]; tg_=np.array([gix[g] for g in gi[te[s2]]]); w=S[np.ix_(tg_,gx)]; j=w.argmax(1); okk=w[np.arange(len(j)),j]>=thr
        P[s2]=np.where(okk[:,None],Y[j],0)
    return P
# ---- diag15 (exploratory, inner folds): trimmed position + sibling prediction split into a line-shared and a line-specific component, with strong-sibling gating ----
DOSES=[0.05,0.5,5.0]; store=[]
def bpos_trim(te,dn,D,own_excl,thr=1.5):
    key=pd.DataFrame({'p':pos[dn],'l':ln[dn]}).groupby(['p','l']).indices
    nrm=np.sqrt((D.astype(np.float64)**2).sum(1)); med=pd.Series(nrm).groupby(ln[dn]).median().to_dict()
    P=np.zeros((len(te),G),np.float32)
    for n,i in enumerate(te):
        kk=key.get((pos[i],ln[i]))
        if kk is None: continue
        j=dn[kk]; m=(pl[j]!=pl[i])&(nrm[kk]<=thr*med[ln[i]])
        if own_excl: m&=gi[j]!=gi[i]
        kk=kk[m]
        if len(kk)==0: continue
        pj=pl[dn[kk]]; up=np.unique(pj); P[n]=np.mean([D[kk[pj==u]].mean(0) for u in up],0)
    return P
def sib(te,tr,Ytr,d,S,thr):
    Psh=np.zeros((len(te),G),np.float32); Psp=np.zeros_like(Psh); q=np.zeros(len(te),np.float32)
    sel=np.where(ds[te]==d)[0]; ii=np.where(ds[tr]==d)[0]
    if len(sel)==0 or len(ii)==0: return Psh,Psp,q
    ga=gi[tr[ii]]; ug=np.unique(ga); SH=np.zeros((len(gids),G),np.float32); nr=np.full(len(gids),np.nan)
    for g in ug: SH[gix[g]]=Ytr[ii[ga==g]].mean(0); nr[gix[g]]=np.sqrt((SH[gix[g]].astype(np.float64)**2).sum())
    rk=pd.Series(nr).rank(pct=True).values; gl=group_line_mean(tr,Ytr,d)
    for l in np.unique(ln[te[sel]]):
        if l not in gl: continue
        gx,Y=gl[l]; s2=sel[ln[te[sel]]==l]; tg_=np.array([gix[g] for g in gi[te[s2]]]); w=S[np.ix_(tg_,gx)]; j=w.argmax(1); ok=w[np.arange(len(j)),j]>=thr
        Psh[s2]=np.where(ok[:,None],SH[gx[j]],0); Psp[s2]=np.where(ok[:,None],Y[j]-SH[gx[j]],0); q[s2]=np.where(ok,rk[gx[j]],0)
    return Psh,Psp,q
for f in range(3):
    for k in (0,1):
        tr=np.where((fold!=f)&(inner[f]>=0)&(inner[f]!=k))[0]; va=np.where((fold!=f)&(inner[f]==k))[0]
        Dtr,v=center(tr); tr=tr[v]; Dtr=Dtr[v]; T,v=center(va); va=va[v]; T=T[v]
        arms={'pos':recentre(bpos_trim(va,tr,Dtr,False),va)}
        Ptr=bpos_trim(tr,tr,Dtr,True); s_in=float((Dtr*Ptr).sum()/((Ptr**2).sum()+1e-30)); Rtr=Dtr-s_in*Ptr
        names=['nn_sh','nn_sp','g50_sh','g50_sp','g75_sh','g75_sp','k_sh','k_sp']
        for nm in names: arms[nm]=np.zeros((len(va),G),np.float32)
        for d in DOSES:
            a,b,q=sib(va,tr,Rtr,d,TAN,0.5); arms['nn_sh']+=a; arms['nn_sp']+=b
            for t,nm in ((0.5,'g50'),(0.75,'g75')): m=(q>=t)[:,None]; arms[nm+'_sh']+=a*m; arms[nm+'_sp']+=b*m
            ks=kernel_pred(va,tr,Rtr,d,TAN,0.3,shared=True); kl=kernel_pred(va,tr,Rtr,d,TAN,0.3); arms['k_sh']+=ks; arms['k_sp']+=kl-ks
        arms={a:(P if a=='pos' else recentre(P,va)) for a,P in arms.items()}
        store.append(dict(va=va,T=T,arms=arms)); print('split',f,k,len(tr),len(va),'s_in',round(s_in,3),flush=True)
def evalset(mask_fn,label):
    res={}
    for d in DOSES:
        parts=[(s,(ds[s['va']]==d)&mask_fn(s['va'])) for s in store]
        gg=np.concatenate([gi[s['va']][m] for s,m in parts]); T2=np.concatenate([(s['T'][m].astype(np.float64)**2).sum(1) for s,m in parts]); b0=np.sqrt(T2/G)
        def gm(v): return float(pd.Series(v).groupby(gg).mean().mean())
        B0=gm(b0); SST=T2.sum(); o={'rows':int(len(gg)),'groups':int(len(set(gg))),'B0':round(B0,5)}
        sub=[(s['T'][m].astype(np.float64),{a:P[m].astype(np.float64) for a,P in s['arms'].items()},gi[s['va']][m]) for s,m in parts]
        def cross(names,grid,crit):
            tot=[]
            for i,(Tm,A,g1) in enumerate(sub):
                Ts,As,g2=sub[i^1]
                def sc(p):
                    e=((Ts-sum(c*As[n] for c,n in zip(p,names)))**2).sum(1)
                    return float(pd.Series(np.sqrt(e/G)).groupby(g2).mean().mean()) if crit=='rmse' else float(e.sum())
                best=min(grid,key=sc); tot.append(((Tm-sum(c*A[n] for c,n in zip(best,names)))**2).sum(1))
            return np.concatenate(tot)
        rng=np.random.default_rng(130113); ug=pd.Series(gg).groupby(gg).ngroup().values; ng=ug.max()+1; ix=rng.integers(0,ng,(4000,ng))
        def gsum(v): return np.bincount(ug,weights=v,minlength=ng)
        tS=gsum(T2); cnt=np.bincount(ug,minlength=ng)
        def rep(name,names,grid,ref=None):
            e_r=cross(names,grid,'rmse'); e_s=cross(names,grid,'sse'); eS=gsum(e_s)
            r={'rmse_%':round(100*(gm(np.sqrt(e_r/G))/B0-1),3),'R2_%':round(100*(1-e_s.sum()/SST),3)}
            if ref is not None:
                rs,rr=ref; rS=gsum(rs); gain=rS-eS; bs=100*(gain[ix].sum(1))/tS[ix].sum(1)
                r['R2_gain']=round(100*gain.sum()/SST,3); r['R2_gain_ci95']=[round(float(x),3) for x in np.percentile(bs,[2.5,97.5])]; r['R2_gain_ci99']=[round(float(x),3) for x in np.percentile(bs,[0.5,99.5])]
                dd=gsum(np.sqrt(e_r/G)-np.sqrt(rr/G))/cnt; bm=dd[ix].mean(1)/B0*100
                r['rmse_diff_%']=round(float(dd.mean())/B0*100,3); r['rmse_diff_ci95']=[round(float(x),3) for x in np.percentile(bm,[2.5,97.5])]; r['rmse_diff_ci99']=[round(float(x),3) for x in np.percentile(bm,[0.5,99.5])]
                r['frac_groups_improved_sse']=round(float((gain>0).mean()),3); r['frac_groups_improved_rmse']=round(float((dd<0).mean()),3)
                top=np.argsort(-gain)[:5]; posg=gain[gain>0].sum(); r['top5_share_pos_%']=round(float(100*gain[top].sum()/max(posg,1e-30)),1)
                keep=np.ones(ng,bool); keep[top]=False; r['R2_gain_excl_top5']=round(float(100*gain[keep].sum()/tS[keep].sum()),3); r['rmse_diff_excl_top5_%']=round(float(dd[keep].mean())/B0*100,3)
            o[name]=r; return e_s,e_r
        ps=(0.25,0.5,0.75,1.0); ab=[(x,a,b) for x in ps for a in (0,0.5,1.0) for b in (0,0.25,0.5,1.0)]
        trim=rep('trim',['pos'],[(x,) for x in (0,)+ps])
        for tag,(sh,sp) in {'nn':('nn_sh','nn_sp'),'g50':('g50_sh','g50_sp'),'g75':('g75_sh','g75_sp'),'k':('k_sh','k_sp')}.items():
            rep(f'trim+{tag}(one scale)',['pos',sh,sp],[(x,a,a) for x in ps for a in (0,0.25,0.5,0.75,1.0)],trim)
            rep(f'trim+{tag}(shared only)',['pos',sh],[(x,a) for x in ps for a in (0,0.25,0.5,0.75,1.0)],trim)
            rep(f'trim+{tag}(shared,specific)',['pos',sh,sp],ab,trim)
        res[str(d)]=o
    print('EVAL',label,json.dumps(res),flush=True); return res
liv=lambda v: np.isin(ln[v],list(LIVER))
out={'all_lines':evalset(lambda v: np.ones(len(v),bool),'all_lines'),'liver':evalset(liv,'liver')}
json.dump(out,open(R+'review_claude_20261121/diag15.json','w'),indent=1)
