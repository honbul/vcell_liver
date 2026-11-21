"""Reviewer diagnostic 8 (descriptive): is there a drug-set-specific well effect beyond the generic position effect?
(A) DMSO(A)-DMSO(B) contrast per plate x line: cosine between plates of the same drug set vs different drug sets.
(B) compounds present in two drug sets: same-position (within set, sister dose) vs different-position (across set) agreement on the SAME groups."""
import numpy as np, pandas as pd, scipy.sparse as sp, json, itertools
R='/mnt/S1/sdata/workbench/Virtual_Cell/gssong/'
def load(d, name):
    n=np.load(d+name+'_counts.npz'); m=sp.csr_matrix((n['data'],n['indices'],n['indptr']),shape=tuple(n['shape']))
    r=pd.read_csv(d+name+'_rows.tsv',sep='\t'); return m,r
d1=R+'vcell_liver_r11_chemical_probe/assets/source/'; d2=R+'vcell_liver_r12_multiaxis/assets/doses/source/'
m05,r05=load(d1,'treated'); mxx,rxx=load(d2,'treated')
def lognorm(m, genes=None):
    lib=np.asarray(m.sum(1)).ravel().astype(np.float64); lib[lib==0]=1
    if genes is not None: m=m[:,genes]
    return np.log1p(m.toarray().astype(np.float64)/lib[:,None]*1e4).astype(np.float32)
ok05=r05.eligible_cell_count.values.astype(bool)
xa=lognorm(m05[ok05]); genes=np.sort(np.argsort(-xa.var(0))[:2000]); del xa
sm=pd.read_parquet('/mnt/S1/sdata/pubdata/Tahoe/Tahoe-100M/metadata/sample_metadata.parquet')
sm['num']=sm['sample'].str.extract(r'(\d+)').astype(int); sm['pos']=sm.groupby('plate').num.rank().astype(int); pmap=dict(zip(sm['sample'],sm.pos))
dset=lambda p: 1 if p=='plate14' else (int(p.replace('plate',''))-1)//3
out={}
# ---- (A) DMSO ----
mA,rA=load(d2,'A'); mB,rB=load(d2,'B'); XA=lognorm(mA,genes); XB=lognorm(mB,genes)
print('A cols',rA.columns.tolist()); print(rA.groupby('plate_id').physical_well_id.first().map(pmap).to_dict()); print(rB.groupby('plate_id').physical_well_id.first().map(pmap).to_dict())
okA=rA.n_cells.values>=50; okB=rB.n_cells.values>=50
a=pd.DataFrame({'p':rA.plate_id,'l':rA.cell_line_id,'i':np.arange(len(rA))})[okA]; b=pd.DataFrame({'p':rB.plate_id,'l':rB.cell_line_id,'j':np.arange(len(rB))})[okB]
ab=a.merge(b,on=['p','l']); Dm=XA[ab.i.values]-XB[ab.j.values]
plates=sorted(ab.p.unique(),key=lambda s:int(s.replace('plate','')))
vec={(p,l):Dm[n] for n,(p,l) in enumerate(zip(ab.p,ab.l))}
wm={p:Dm[(ab.p==p).values].mean(0) for p in plates}
cos=lambda u,v: float((u*v).sum()/np.sqrt((u*u).sum()*(v*v).sum()+1e-30))
recs=[]
for p,q in itertools.combinations(plates,2):
    ls=[l for l in ab.l.unique() if (p,l) in vec and (q,l) in vec]
    recs.append(dict(p=p,q=q,same_set=dset(p)==dset(q),line_cos_median=float(np.median([cos(vec[(p,l)],vec[(q,l)]) for l in ls])),wellmean_cos=cos(wm[p],wm[q]),n=len(ls)))
df=pd.DataFrame(recs); out['dmso_pairs']=recs
print(df.groupby('same_set')[['line_cos_median','wellmean_cos']].agg(['median','mean','count']).to_string())
print(df[df.same_set].to_string())
print('rms of DMSO A-B per plate', {p: round(float(np.sqrt((Dm[(ab.p==p).values]**2).mean())),4) for p in plates})
# ---- (B) duplicated compounds ----
rows=pd.concat([r05,rxx],ignore_index=True); X=np.vstack([lognorm(m05,genes),lognorm(mxx,genes)])
keep=rows.eligible_cell_count.values.astype(bool); rows=rows[keep].reset_index(drop=True); X=X[keep]
rows['pool']=rows.plate_id+'|'+rows.cell_line_id+'|'+rows.dose_uM.astype(str)
keep=(rows.groupby('pool').compound_group_id.transform('nunique')>=3).values; rows=rows[keep].reset_index(drop=True); X=X[keep]
C=np.empty_like(X)
for p,idx in rows.groupby('pool').indices.items(): C[idx]=X[idx]-X[idx].mean(0)
rows['pos']=rows.physical_well_id.map(pmap).values; rows['set']=rows.plate_id.map(dset)
gi=rows.compound_group_id.values; pl=rows.plate_id.values; ds=rows.dose_uM.values; ln=rows.cell_line_id.values; pos=rows.pos.values; st=rows.set.values
ns=rows.groupby('compound_group_id').set.nunique(); dup=set(ns[ns>1].index)
print('groups in >1 drug set',len(dup)); print(rows[rows.compound_group_id.isin(dup)].drop_duplicates(['compound_group_id','plate_id'])[['compound_entity_id','compound_group_id','plate_id','dose_uM','pos']].sort_values(['compound_group_id','plate_id']).to_string())
df=pd.DataFrame({'i':np.arange(len(rows)),'g':gi,'l':ln}); m=df.merge(df,on=['g','l'],suffixes=('','_b')); a_=m.i.values; b_=m.i_b.values
k=pl[a_]<pl[b_]; a_=a_[k]; b_=b_[k]
cs=(C[a_]*C[b_]).sum(1)/np.sqrt((C[a_]**2).sum(1)*(C[b_]**2).sum(1)+1e-30)
pr=pd.DataFrame({'g':gi[a_],'cos':cs,'same_set':st[a_]==st[b_],'same_pos':pos[a_]==pos[b_],'same_dose':ds[a_]==ds[b_],'dup':np.isin(gi[a_],list(dup)),'liver':np.isin(ln[a_],['CVCL_0366','CVCL_1098'])})
def summ(d): 
    g=d.groupby('g').cos.mean(); return {'pairs':int(len(d)),'groups':int(len(g)),'group_mean_cos':round(float(g.mean()),4),'group_median_cos':round(float(g.median()),4)}
B={}
for name,q in [('dup groups | same set, same pos (sister dose / 6-14)','dup & same_set & same_pos'),('dup groups | different set (different pos)','dup & ~same_set & ~same_pos'),('dup groups | different set but same pos','dup & ~same_set & same_pos'),('dup groups | same set, different pos','dup & same_set & ~same_pos'),
               ('non-dup groups | same set, same pos','~dup & same_set & same_pos'),('all | same set same pos same dose (6<->14)','same_set & same_pos & same_dose'),('dup | different set, same dose','dup & ~same_set & same_dose'),('dup | different set, different dose','dup & ~same_set & ~same_dose')]:
    B[name]=summ(pr.query(q)); B[name+' [liver lines]']=summ(pr.query(q+' & liver')); print(name,B[name],'| liver',B[name+' [liver lines]'])
out['B']=B
# per dup group table
t=pr[pr.dup].groupby(['g','same_set']).cos.mean().unstack(); t.columns=['diff_set','same_set']; print(t.round(3).to_string()); print('paired mean diff same-diff',round(float((t.same_set-t.diff_set).mean()),4),'n',int(t.dropna().shape[0]))
json.dump(out,open(R+'review_claude_20261121/diag8.json','w'),indent=1,default=str)
