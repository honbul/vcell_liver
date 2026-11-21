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

# ---------------- diag12 body: LINCS L1000 signature of the compound as a prior (exploratory, inner folds) ----------------
z=np.load(R+'review_claude_20261121/lincs_sigs.npz',allow_pickle=True); LM=z['M']; li=pd.read_csv(R+'review_claude_20261121/lincs_sigs.tsv',sep='\t')
def consensus(mask):
    F=np.zeros((len(gids),LM.shape[1]),np.float64); cov=np.zeros(len(gids),bool)
    for g,sub in li[mask].groupby('g'):
        if g not in gix: continue
        s24=sub[sub.time==24]; use=s24 if len(s24)>=2 else sub
        hi=use[use.dose>=use.dose.quantile(0.5)] if use.dose.notna().sum()>=4 else use     # upper half of tested doses
        F[gix[g]]=LM[hi.index.values].mean(0); cov[gix[g]]=True
    return F,cov
def coskern(F,cov):
    n=np.sqrt((F**2).sum(1,keepdims=True)); n[n==0]=1; K=(F/n)@(F/n).T; K=np.clip(K,0,None); K[~cov]=0; K[:,~cov]=0; return K
F_all,cov_all=consensus(np.ones(len(li),bool)); F_hep,cov_hep=consensus((li.cell_id=='HEPG2').values)
KL=coskern(F_all,cov_all); KH=coskern(F_hep,cov_hep)
print('LINCS coverage groups',int(cov_all.sum()),'HEPG2',int(cov_hep.sum()),'median offdiag KL',round(float(np.median(KL[np.ix_(cov_all,cov_all)])),3),flush=True)
covg_all=set(gids[cov_all]); covg_hep=set(gids[cov_hep])
def nn1(te,tr,Ytr,d,S,thr):
    P=np.zeros((len(te),G),np.float32); sel=np.where(ds[te]==d)[0]; gl=group_line_mean(tr,Ytr,d)
    for l in np.unique(ln[te[sel]]):
        if l not in gl: continue
        gx,Y=gl[l]; s2=sel[ln[te[sel]]==l]; tg_=np.array([gix[g] for g in gi[te[s2]]]); w=S[np.ix_(tg_,gx)]; j=w.argmax(1); okk=w[np.arange(len(j)),j]>=thr
        P[s2]=np.where(okk[:,None],Y[j],0)
    return P
DOSES=[0.05,0.5,5.0]; store=[]
ARMS={'tan_l0.3':(TAN,0.3),'lincs_l0.1':(KL,0.1),'lincs_l0.3':(KL,0.3),'lincs_l1.0':(KL,1.0),'lincsHEP_l0.3':(KH,0.3),'tan+lincs_l0.3':(TAN+KL,0.3),'tan+lincs_l1.0':(TAN+KL,1.0),'max(tan,lincs)_l0.3':(np.maximum(TAN,KL),0.3)}
for f in range(3):
    for k in (0,1):
        tr=np.where((fold!=f)&(inner[f]>=0)&(inner[f]!=k))[0]; va=np.where((fold!=f)&(inner[f]==k))[0]
        Dtr,v=center(tr); tr=tr[v]; Dtr=Dtr[v]; T,v=center(va); va=va[v]; T=T[v]
        Pp,has=bpos(va,tr,Dtr,False); arms={'pos':recentre(Pp,va)}
        Ptr,_=bpos(tr,tr,Dtr,True); s_in=float((Dtr*Ptr).sum()/((Ptr**2).sum()+1e-30)); Rtr=Dtr-s_in*Ptr
        for nm in list(ARMS)+['nn_lincs0.3','nn_lincs0.5']: arms[nm]=np.zeros((len(va),G),np.float32)
        for d in DOSES:
            for nm,(K,lam) in ARMS.items(): arms[nm]+=kernel_pred(va,tr,Rtr,d,K,lam)
            arms['nn_lincs0.3']+=nn1(va,tr,Rtr,d,KL,0.3); arms['nn_lincs0.5']+=nn1(va,tr,Rtr,d,KL,0.5)
        arms={a:(P if a=='pos' else recentre(P,va)) for a,P in arms.items()}
        store.append(dict(va=va,T=T,arms=arms)); print('split',f,k,len(tr),len(va),flush=True)
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
        rng=np.random.default_rng(130113); ug=pd.Series(gg).groupby(gg).ngroup().values; ng=ug.max()+1; ix=rng.integers(0,ng,(2000,ng))
        def gsum(v): return np.bincount(ug,weights=v,minlength=ng)
        tS=gsum(T2)
        def rep(name,names,grid,refs={}):
            e_r=cross(names,grid,'rmse'); e_s=cross(names,grid,'sse'); eS=gsum(e_s)
            r={'rmse_%':round(100*(gm(np.sqrt(e_r/G))/B0-1),3),'pooled_R2_%':round(100*(1-e_s.sum()/SST),3)}
            for rn,(rs,rr) in refs.items():
                rS=gsum(rs); bs=100*(rS[ix].sum(1)-eS[ix].sum(1))/tS[ix].sum(1); r[f'R2_gain_vs_{rn}']=round(100*(rs.sum()-e_s.sum())/SST,3); r[f'R2_gain_vs_{rn}_ci95']=[round(float(x),3) for x in np.percentile(bs,[2.5,97.5])]
                dd=gsum(np.sqrt(e_r/G)-np.sqrt(rr/G))/np.bincount(ug,minlength=ng); r[f'rmse_diff_vs_{rn}_ci95_%']=[round(float(x)/B0*100,3) for x in np.percentile(dd[ix].mean(1),[2.5,97.5])]
            o[name]=r; return e_s,e_r
        g2d=[(x,y) for x in (0.25,0.5,0.75) for y in (0,0.25,0.5,0.75,1.0)]
        pos_=rep('pos',['pos'],[(x,) for x in (0,0.25,0.5,0.75,1.0)]); tan_=rep('pos+tan_l0.3',['pos','tan_l0.3'],g2d,{'pos':pos_})
        for a in list(ARMS)[1:]+['nn_lincs0.3','nn_lincs0.5']: rep('pos+'+a,['pos',a],g2d,{'pos':pos_,'tan':tan_})
        rep('pos+tan+lincs(separate scales)',['pos','tan_l0.3','lincs_l0.3'],[(x,y,z_) for x in (0.25,0.5) for y in (0,0.5,1.0) for z_ in (0,0.5,1.0)],{'pos':pos_,'tan':tan_})
        res[str(d)]=o
    print('EVAL',label,json.dumps(res),flush=True); return res
liv=lambda v: np.isin(ln[v],list(LIVER))
out={'all_lines':evalset(lambda v: np.ones(len(v),bool),'all_lines'),'liver':evalset(liv,'liver'),
     'all_lines|LINCS-covered groups':evalset(lambda v: np.isin(gi[v],list(covg_all)),'covered'),'liver|LINCS-covered groups':evalset(lambda v: liv(v)&np.isin(gi[v],list(covg_all)),'liver covered'),
     'liver|HEPG2-covered groups':evalset(lambda v: liv(v)&np.isin(gi[v],list(covg_hep)),'liver hep-covered')}
json.dump(out,open(R+'review_claude_20261121/diag12.json','w'),indent=1)
for lab,res in out.items():
    for d,o in res.items():
        print('==',lab,d,{k:o[k] for k in ('rows','groups','B0')})
        for k,v in o.items():
            if isinstance(v,dict): print('   %-34s'%k, v)
