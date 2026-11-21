"""Reviewer diagnostic 9 (descriptive): share of single-well contrast variance that is shared between wells, by pair type.
Estimator: ratio  sum<Ta,Tb> / sqrt(sum|Ta|^2 * sum|Tb|^2)  over row pairs (same line, different plates); bootstrap over compound groups."""
import numpy as np, pandas as pd, scipy.sparse as sp, json
R='/mnt/S1/sdata/workbench/Virtual_Cell/gssong/'
def load(d, name):
    n=np.load(d+name+'_counts.npz'); m=sp.csr_matrix((n['data'],n['indices'],n['indptr']),shape=tuple(n['shape']))
    r=pd.read_csv(d+name+'_rows.tsv',sep='\t'); return m,r
m05,r05=load(R+'vcell_liver_r11_chemical_probe/assets/source/','treated'); mxx,rxx=load(R+'vcell_liver_r12_multiaxis/assets/doses/source/','treated')
def lognorm(m, genes=None):
    lib=np.asarray(m.sum(1)).ravel().astype(np.float64); lib[lib==0]=1
    if genes is not None: m=m[:,genes]
    return np.log1p(m.toarray().astype(np.float64)/lib[:,None]*1e4).astype(np.float32)
ok05=r05.eligible_cell_count.values.astype(bool)
xa=lognorm(m05[ok05]); genes=np.sort(np.argsort(-xa.var(0))[:2000]); del xa
rows=pd.concat([r05,rxx],ignore_index=True); X=np.vstack([lognorm(m05,genes),lognorm(mxx,genes)])
keep=rows.eligible_cell_count.values.astype(bool); rows=rows[keep].reset_index(drop=True); X=X[keep]
rows['pool']=rows.plate_id+'|'+rows.cell_line_id+'|'+rows.dose_uM.astype(str)
keep=(rows.groupby('pool').compound_group_id.transform('nunique')>=3).values; rows=rows[keep].reset_index(drop=True); X=X[keep]
C=np.empty_like(X)
for p,idx in rows.groupby('pool').indices.items(): C[idx]=X[idx]-X[idx].mean(0)
sm=pd.read_parquet('/mnt/S1/sdata/pubdata/Tahoe/Tahoe-100M/metadata/sample_metadata.parquet')
sm['num']=sm['sample'].str.extract(r'(\d+)').astype(int); sm['pos']=sm.groupby('plate').num.rank().astype(int)
rows['pos']=rows.physical_well_id.map(dict(zip(sm['sample'],sm.pos))).values
dsetf=lambda p: 1 if p=='plate14' else (int(p.replace('plate',''))-1)//3
gi=rows.compound_group_id.values.astype(int); pl=rows.plate_id.values; ds=rows.dose_uM.values; ln=rows.cell_line_id.values; pos=rows.pos.values; st=rows.plate_id.map(dsetf).values
pln=np.array([int(p.replace('plate','')) for p in pl]); nc=rows.n_cells.values
n2=(C.astype(np.float64)**2).sum(1)
LIV=np.isin(ln,['CVCL_0366','CVCL_1098'])
# well-mean (line-averaged) vectors
wk=pd.Series(rows.physical_well_id.values); wid=wk.astype('category').cat.codes.values; nw=wid.max()+1
Wm=np.zeros((nw,C.shape[1]),np.float64); np.add.at(Wm,wid,C); Wm/=np.bincount(wid)[:,None]
wfirst=pd.Series(np.arange(len(rows))).groupby(wid).first().values
W=dict(g=gi[wfirst],pl=pln[wfirst],ds=ds[wfirst],pos=pos[wfirst],st=st[wfirst],n2=(Wm**2).sum(1))
def stat(a,b,V,N2,grp,nb=1000,seed=130113):
    """ratio-of-sums shared fraction with group bootstrap; V rows indexed by a,b."""
    if len(a)==0: return None
    ip=np.einsum('ij,ij->i',V[a].astype(np.float64),V[b].astype(np.float64)); A=N2[a]; B=N2[b]
    df=pd.DataFrame({'ip':ip,'A':A,'B':B,'g':grp}).groupby('g').sum(); v=df.values
    est=v[:,0].sum()/np.sqrt(v[:,1].sum()*v[:,2].sum()); rng=np.random.default_rng(seed)
    ix=rng.integers(0,len(v),(nb,len(v))); s=v[ix].sum(1); bs=s[:,0]/np.sqrt(s[:,1]*s[:,2])
    return {'pairs':int(len(a)),'units':int(len(v)),'shared':round(float(est),4),'ci95':[round(float(x),4) for x in np.percentile(bs,[2.5,97.5])]}
def allpairs(mask,keycols):
    df=pd.DataFrame({'i':np.where(mask)[0]}); 
    for k,v in keycols.items(): df[k]=v[mask]
    m=df.merge(df,on=list(keycols),suffixes=('','_b')); return m.i.values,m.i_b.values
out={}
def run(level,V,N2,g_,pl_,ds_,pos_,st_,extra,lab):
    res={}
    def add(name,a,b,grp): 
        r=stat(a,b,V,N2,grp); res[name]=r; print(lab,'|',name,r,flush=True)
    base=np.ones(len(g_),bool) if extra is None else extra
    kc={'g':g_} if level=='well' else {'g':g_,'l':ln}
    a,b=allpairs(base,kc); k=pl_[a]<pl_[b]; a=a[k]; b=b[k]
    sp_=pos_[a]==pos_[b]; ss=st_[a]==st_[b]; sd=ds_[a]==ds_[b]
    add('same drug, same set, same pos, same dose (plate 6<->14)',a[sp_&ss&sd],b[sp_&ss&sd],g_[a[sp_&ss&sd]])
    for d1,d2 in [(0.05,0.5),(0.5,5.0),(0.05,5.0)]:
        k=sp_&ss&(((ds_[a]==d1)&(ds_[b]==d2))|((ds_[a]==d2)&(ds_[b]==d1))); add(f'same drug, same set, same pos, doses {d1}/{d2}',a[k],b[k],g_[a[k]])
    k=~ss&~sp_; add('same drug, DIFFERENT set & pos, any dose',a[k],b[k],g_[a[k]])
    k=~ss&~sp_&sd; add('same drug, DIFFERENT set & pos, same dose',a[k],b[k],g_[a[k]])
    for d in (0.05,0.5,5.0):
        k=~ss&~sp_&(ds_[a]==d)&(ds_[b]==d); add(f'same drug, DIFFERENT set & pos, both {d}',a[k],b[k],g_[a[k]])
    k=~ss&~sp_&(g_[a]!=16); add('same drug, DIFFERENT set & pos, any dose, excl. trametinib',a[k],b[k],g_[a[k]])
    # same position, different drug, different set
    kc={'p':pos_} if level=='well' else {'p':pos_,'l':ln}
    a,b=allpairs(base,kc); k=(pl_[a]<pl_[b])&(st_[a]!=st_[b])&(g_[a]!=g_[b]); a=a[k]; b=b[k]
    add('DIFFERENT drug, same pos, different set, any dose',a,b,pos_[a])
    for d in (0.05,0.5,5.0):
        k=(ds_[a]==d)&(ds_[b]==d); add(f'DIFFERENT drug, same pos, different set, both {d}',a[k],b[k],pos_[a[k]])
    # null: different drug, different pos, different set (random sample)
    rng=np.random.default_rng(0); idx=np.where(base)[0]
    if level=='well':
        a=rng.choice(idx,200000); b=rng.choice(idx,200000)
    else:
        df=pd.DataFrame({'i':idx,'l':ln[idx]}); a=[];b=[]
        for l,sub in df.groupby('l'):
            x=sub.i.values; a.append(rng.choice(x,4000)); b.append(rng.choice(x,4000))
        a=np.concatenate(a); b=np.concatenate(b)
    k=(st_[a]!=st_[b])&(pos_[a]!=pos_[b])&(g_[a]!=g_[b]); add('null: different drug, pos, set',a[k],b[k],g_[a[k]])
    k=(pl_[a]==pl_[b])&(pos_[a]!=pos_[b])&(g_[a]!=g_[b])
    return res
out['line_level_all']=run('line',C,n2,gi,pln,ds,pos,st,None,'LINE all')
out['line_level_liver']=run('line',C,n2,gi,pln,ds,pos,st,LIV,'LINE liver')
out['line_level_cells>=200']=run('line',C,n2,gi,pln,ds,pos,st,nc>=200,'LINE >=200 cells')
out['well_mean']=run('well',Wm,W['n2'],W['g'],W['pl'],W['ds'],W['pos'],W['st'],None,'WELL mean')
json.dump(out,open(R+'review_claude_20261121/diag9.json','w'),indent=1)
