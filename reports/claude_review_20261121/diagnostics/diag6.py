"""Reviewer exploratory diagnostic 6 (re-implementation; INNER folds of Codex's manifest only; not an official result).
Per dose (0.05 / 0.5 / 5 uM): B0, B_pos-all, chemistry kernel on position-corrected residuals, MoA neighbours,
combined arm, and the plate 6<->14 same-position replicate ceiling at 5 uM."""
import numpy as np, pandas as pd, scipy.sparse as sp, json, os, sys
from rdkit import Chem, DataStructs
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

DOSES=[0.05,0.5,5.0]; LAMS=[0.3,1.0,3.0,10.0]
store=[]   # per split dict of val arrays
for f in range(3):
    for k in (0,1):
        tr=np.where((fold!=f)&(inner[f]>=0)&(inner[f]!=k))[0]; va=np.where((fold!=f)&(inner[f]==k))[0]
        Dtr,v=center(tr); tr=tr[v]; Dtr=Dtr[v]; T,v=center(va); va=va[v]; T=T[v]
        Pp,has=bpos(va,tr,Dtr,False); Pp=recentre(Pp,va)
        Ptr,_=bpos(tr,tr,Dtr,True)
        s_in=float((Dtr*Ptr).sum()/((Ptr**2).sum()+1e-30)); Rtr=Dtr-s_in*Ptr
        arms={'pos':Pp}
        for d in DOSES:
            for lam in LAMS:
                for nm,Y,sh in (('kres',Rtr,False),('kraw',Dtr,False),('kres_shared',Rtr,True)):
                    key=f'{nm}_l{lam}'; arms.setdefault(key,np.zeros((len(va),G),np.float32)); arms[key]+=kernel_pred(va,tr,Y,d,TAN,lam,sh)
            arms.setdefault('moa_res',np.zeros((len(va),G),np.float32)); arms['moa_res']+=nn_pred(va,tr,Rtr,d,MOA)
        # dose-pooled kernel: training residual averaged over the three doses per (group,line), applied to every dose
        for lam in LAMS:
            key=f'kres_alldose_l{lam}'; A=np.zeros((len(va),G),np.float32)
            ug=np.unique(gi[tr]); gx=np.array([gix[g] for g in ug])
            for l in np.unique(ln[va]):
                a=np.where(ln[tr]==l)[0]; ga=gi[tr[a]]; u2=np.unique(ga)
                if len(u2)<5: continue
                Y=np.stack([Rtr[a[ga==g]].mean(0) for g in u2]); g2=np.array([gix[g] for g in u2])
                Al=np.linalg.solve(TAN[np.ix_(g2,g2)]+lam*np.eye(len(g2)),Y); s2=np.where(ln[va]==l)[0]
                A[s2]=TAN[np.ix_(np.array([gix[g] for g in gi[va[s2]]]),g2)]@Al
            arms[key]=A
        arms={a:(recentre(P,va) if a!='pos' else P) for a,P in arms.items()}
        store.append(dict(f=f,k=k,va=va,T=T,arms=arms,has=has,s_in=s_in))
        print('split',f,k,'train',len(tr),'val',len(va),'s_in',round(s_in,3),'support',round(float(has.mean()),3),flush=True)

def gmean(v,g): return float(pd.Series(v).groupby(g).mean().mean())
def table(mask_fn,label):
    """group-equal contrast RMSE over all 6 inner splits; scale picked on the sibling inner split."""
    res={}
    for d in DOSES:
        parts=[]
        for s in store:
            m=(ds[s['va']]==d)&mask_fn(s['va'])
            parts.append((s,m))
        b0=np.concatenate([rm(s['T'][m]) for s,m in parts]); gg=np.concatenate([gi[s['va']][m] for s,m in parts]); B0=gmean(b0,gg)
        o={'rows':int(len(b0)),'groups':int(len(set(gg))),'B0':B0}
        SC=[0,0.25,0.5,0.75,1.0]
        def err(s,m,pred): return rm(s['T'][m]-pred[m])
        def cross(build,grid):
            """build(s,param)->pred ; choose param on sibling split, evaluate on own."""
            tot=[]
            for i,(s,m) in enumerate(parts):
                sib,ms=parts[i^1]
                best=min(grid,key=lambda p: gmean(err(sib,ms,build(sib,p)),gi[sib['va']][ms]))
                tot.append(err(s,m,build(s,best)))
            return np.concatenate(tot)
        def rel(e): return round(100*(gmean(e,gg)/B0-1),3)
        def boot(e1,e2,n=2000):
            a=pd.DataFrame({'d':e1-e2,'g':gg}).groupby('g').d.mean().values; rng=np.random.default_rng(130113)
            bs=a[rng.integers(0,len(a),(n,len(a)))].mean(1); return [round(float(x)/B0*100,3) for x in np.percentile(bs,[0.5,2.5,97.5,99.5])]
        e_pos=cross(lambda s,p: p*s['arms']['pos'],SC); o['B_pos_all_%']=rel(e_pos); o['B_pos_all_ci(0.5,2.5,97.5,99.5)%']=boot(e_pos,b0)
        o['B_pos_fixed_scale_%']={p:rel(np.concatenate([err(s,m,p*s['arms']['pos']) for s,m in parts])) for p in SC}
        for fam in ('kres','kraw','kres_shared','kres_alldose'):
            grid=[(lam,p) for lam in LAMS for p in (0.25,0.5,1.0)]+[(LAMS[0],0.0)]
            e=cross(lambda s,q: q[1]*s['arms'][f'{fam}_l{q[0]}'],grid); o[fam+'_alone_%']=rel(e)
            grid2=[(lam,p,sp_) for lam in LAMS for p in (0,0.25,0.5,1.0) for sp_ in (0.25,0.5,0.75,1.0)]
            e2=cross(lambda s,q: q[2]*s['arms']['pos']+q[1]*s['arms'][f'{fam}_l{q[0]}'],grid2)
            o[fam+'+pos_%']=rel(e2); o[fam+'+pos_minus_pos_ci%']=boot(e2,e_pos)
            o[fam+'_fixed(lam1,scale1)+pos0.5_%']=rel(np.concatenate([err(s,m,0.5*s['arms']['pos']+s['arms'][f'{fam}_l1.0']) for s,m in parts]))
        e=cross(lambda s,q: q[1]*s['arms']['pos']+q[0]*s['arms']['moa_res'],[(a,b) for a in (0,0.25,0.5,1.0) for b in (0.25,0.5,0.75,1.0)])
        o['moa_res+pos_%']=rel(e); o['moa_res+pos_minus_pos_ci%']=boot(e,e_pos)
        # cosine of kernel prediction with position-corrected target
        cs=[]
        for s,m in parts:
            t=s['T'][m]-0.5*s['arms']['pos'][m]; p=s['arms']['kres_l1.0'][m]; cs.append((t*p).sum(1)/np.sqrt((t**2).sum(1)*(p**2).sum(1)+1e-30))
        o['cos(kres, T-0.5pos)_median']=round(float(np.median(np.concatenate(cs))),4)
        res[str(d)]=o
    print('TABLE',label,json.dumps(res),flush=True); return res
out={}
out['all_lines']=table(lambda v: np.ones(len(v),bool),'all_lines')
for c,n in LIVER.items(): out[n]=table(lambda v,c=c: ln[v]==c,n)
out['cells>=200']=table(lambda v: rows.n_cells.values[v]>=200,'cells>=200')

# ---- plate 6 <-> 14 same-position replicate ceiling at 5 uM (inner validation rows only) ----
rep={}
acc={k:[] for k in ('b0','pos','rep','g','line')}; A=[];Bm=[];Tt=[]
for s in store:
    va=s['va']; T=s['T']; Pp=s['arms']['pos']; Km=s['arms']['kres_l1.0']
    idx={(pl[i],pos[i],ln[i],gi[i]):n for n,i in enumerate(va) if ds[i]==5.0 and pl[i] in ('plate6','plate14')}
    for (p,q,l,g),n in idx.items():
        o=idx.get(('plate14' if p=='plate6' else 'plate6',q,l,g))
        if o is None: continue
        Tt.append(T[n]); A.append(T[o]); Bm.append(Pp[n]); acc['g'].append(g); acc['line'].append(l); acc.setdefault('km',[]).append(Km[n])
Tt=np.array(Tt);A=np.array(A);Bm=np.array(Bm);Km=np.array(acc['km']);gg=np.array(acc['g']);ll=np.array(acc['line'])
def rep_table(m):
    b0=gmean(rm(Tt[m]),gg[m]); r=lambda P: round(100*(gmean(rm(Tt[m]-P[m]),gg[m])/b0-1),3)
    o={'rows':int(m.sum()),'groups':int(len(set(gg[m]))),'B0':b0}
    o['replicate_x(0.25,0.5,0.75,1)_%']=[r(c*A) for c in (0.25,0.5,0.75,1)]
    o['B_pos_x(0.25,0.5,0.75,1)_%']=[r(c*Bm) for c in (0.25,0.5,0.75,1)]
    o['B_pos0.5+kres_%']=r(0.5*Bm+Km)
    Z=np.stack([A[m].ravel(),Bm[m].ravel()],1).astype(np.float64); co,_,_,_=np.linalg.lstsq(Z,Tt[m].ravel().astype(np.float64),rcond=None)
    o['LS a*rep+b*pos coef']=[round(float(x),3) for x in co]; o['LS_combo_%']=r(co[0]*A+co[1]*Bm)
    # position-controlled: correct both target and replicate by 0.5*B_pos (same position => same B_pos up to plate exclusion)
    Tc=Tt-0.5*Bm; Ac=A-0.5*Bm; bc=gmean(rm(Tc[m]),gg[m])
    o['pos-corrected: replicate_x(0.25,0.5,1)_% vs corrected zero']=[round(100*(gmean(rm(Tc[m]-c*Ac[m]),gg[m])/bc-1),3) for c in (0.25,0.5,1)]
    o['pos-corrected: kres_% vs corrected zero']=round(100*(gmean(rm(Tc[m]-Km[m]),gg[m])/bc-1),3)
    cs=lambda U,V: round(float(np.median((U[m]*V[m]).sum(1)/np.sqrt((U[m]**2).sum(1)*(V[m]**2).sum(1)+1e-30))),4)
    o['cos(T,rep)']=cs(Tt,A); o['cos(T,pos)']=cs(Tt,Bm); o['cos(T-0.5pos,rep-0.5pos)']=cs(Tc,Ac); o['cos(T-0.5pos,kres)']=cs(Tc,Km)
    return o
rep['all_lines']=rep_table(np.ones(len(gg),bool))
for c,n in LIVER.items(): rep[n]=rep_table(ll==c)
out['replicate_6_14']=rep; print('REP',json.dumps(rep),flush=True)
json.dump(out,open(R+f'review_claude_20261121/diag6_minc{MINC}.json','w'),indent=1,default=str)
